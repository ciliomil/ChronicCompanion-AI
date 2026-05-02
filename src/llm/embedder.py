"""Embedder protocol with a deterministic offline fallback.

Two implementations are provided:

- :class:`HashTfidfEmbedder` — pure-Python char n-gram + hash-trick TF-IDF.
  Deterministic, no extra deps, suitable for offline / CI usage.
- :class:`BgeM3Embedder` — sentence-transformers + ``bge-m3`` (matches the
  ``MEM_MODEL.embedder`` in ``conf.yaml``). Requires ``sentence_transformers``;
  imports lazily so this module is safe to import without the heavy dep
  installed.

:func:`get_default_embedder` picks based on ``conf.yaml`` and silently falls
back to :class:`HashTfidfEmbedder` if BGE-M3 cannot be loaded — including when
the configured model path does not exist or the dependency is missing. Force
the offline fallback regardless via ``LONGMEM_FORCE_HASH_EMBEDDER=1``.
"""

from __future__ import annotations

import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Protocol

import yaml

_logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONF_PATH = _PROJECT_ROOT / "conf.yaml"


class Embedder(Protocol):
    @property
    def dim(self) -> int:
        ...

    def embed_text(self, text: str) -> list[float]:
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Length-tolerant cosine similarity for two embedding vectors.

    Returns 0.0 when either vector is empty / zero-norm; never raises.
    """
    if not a or not b:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    # Tolerate small length mismatches by zipping; callers are expected to
    # pass vectors of equal length.
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


# ---------------------------------------------------------------------------
# Hash-trick character n-gram TF-IDF (deterministic offline fallback)
# ---------------------------------------------------------------------------


class HashTfidfEmbedder:
    """Deterministic char n-gram hash-TF-IDF, no external dependencies.

    Suitable as a fallback when sentence-transformers / numpy aren't around.
    Output is a length-``dim`` L2-normalised list of floats. Note that the
    hash trick uses Python's built-in ``hash``: with ``PYTHONHASHSEED`` not
    fixed across processes, embeddings are stable within a process but may
    differ across runs. Pin ``PYTHONHASHSEED`` if cross-process determinism
    matters.
    """

    def __init__(self, dim: int = 256, ngram_range: tuple[int, int] = (2, 3)) -> None:
        self._dim = dim
        self._ngram_range = ngram_range

    @property
    def dim(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        if not text:
            return vec
        normalised = re.sub(r"\s+", " ", text.strip())
        n_lo, n_hi = self._ngram_range
        for n in range(n_lo, n_hi + 1):
            if len(normalised) < n:
                continue
            for i in range(len(normalised) - n + 1):
                gram = normalised[i : i + n]
                idx = hash(("ng", n, gram)) % self._dim
                vec[idx] += 1.0
        # Sub-linear TF scaling: 1 + log(tf).
        for i, v in enumerate(vec):
            if v > 0:
                vec[i] = 1.0 + math.log(v)
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


# ---------------------------------------------------------------------------
# BGE-M3 via sentence-transformers (lazy import)
# ---------------------------------------------------------------------------


class BgeM3Embedder:
    """sentence-transformers / BGE-M3 embedder; lazy model load.

    The model is loaded on first use (or via :meth:`warmup`) so importing
    this module is cheap even when ``sentence_transformers`` isn't present.
    """

    def __init__(self, model_path: str, *, device: str | None = None) -> None:
        self._model_path = model_path
        self._device = device
        self._model: Any = None
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._ensure_loaded()
        return self._dim or 0

    def warmup(self) -> None:
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except Exception as err:  # pragma: no cover - dep missing path
            raise RuntimeError(
                f"BgeM3Embedder requires sentence_transformers (import failed: {err})"
            ) from err
        self._model = SentenceTransformer(
            self._model_path,
            trust_remote_code=True,
            device=self._device,
        )
        try:
            self._dim = int(self._model.get_sentence_embedding_dimension() or 0)
        except Exception:
            self._dim = 0

    def embed_text(self, text: str) -> list[float]:
        self._ensure_loaded()
        vec = self._model.encode([text], normalize_embeddings=True)[0]
        return [float(x) for x in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_loaded()
        arr = self._model.encode(texts, normalize_embeddings=True)
        return [[float(x) for x in row] for row in arr]


# ---------------------------------------------------------------------------
# Default factory (conf-driven)
# ---------------------------------------------------------------------------


def _read_embedder_conf() -> dict[str, Any]:
    if not _CONF_PATH.exists():
        return {}
    try:
        with _CONF_PATH.open("r", encoding="utf-8") as fh:
            conf = yaml.safe_load(fh) or {}
    except Exception:
        return {}
    mem = conf.get("MEM_MODEL", {}) or {}
    embedder_cfg = (mem.get("embedder") or {}).get("config", {}) or {}
    return embedder_cfg


_DEFAULT_EMBEDDER: Embedder | None = None


def get_default_embedder() -> Embedder:
    """Return a process-wide default :class:`Embedder`.

    Tries :class:`BgeM3Embedder` (using ``MEM_MODEL.embedder.config.model``
    from ``conf.yaml``) first, falling back to :class:`HashTfidfEmbedder`
    when:

    - ``LONGMEM_FORCE_HASH_EMBEDDER`` is truthy,
    - ``conf.yaml`` doesn't define a model path,
    - the model path doesn't exist on disk, or
    - ``sentence_transformers`` import / model load fails on warm-up.
    """
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is not None:
        return _DEFAULT_EMBEDDER

    cfg = _read_embedder_conf()
    model_path = str(cfg.get("model", "")).strip()
    use_hash = os.getenv("LONGMEM_FORCE_HASH_EMBEDDER", "").strip() not in {
        "",
        "0",
        "false",
        "False",
    }

    if model_path and not use_hash and Path(model_path).is_dir():
        try:
            embedder: Embedder = BgeM3Embedder(model_path=model_path)
            # Force lazy load now so we fail fast (and hit the fallback path)
            # if the model can't actually be loaded.
            embedder.embed_text("ping")
            _logger.info("Using BgeM3Embedder at %s", model_path)
            _DEFAULT_EMBEDDER = embedder
            return embedder
        except Exception as err:
            _logger.warning(
                "BgeM3Embedder unavailable (%s); falling back to HashTfidfEmbedder.",
                err,
            )

    fallback = HashTfidfEmbedder()
    _logger.info("Using HashTfidfEmbedder (dim=%d) as fallback.", fallback.dim)
    _DEFAULT_EMBEDDER = fallback
    return fallback


def reset_default_embedder() -> None:
    """Drop the cached default embedder (useful in tests)."""
    global _DEFAULT_EMBEDDER
    _DEFAULT_EMBEDDER = None
