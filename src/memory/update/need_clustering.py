"""Mem-PAL-style clustering for the top-layer ``need_preferences`` memory.

Implements two operations:

- :meth:`NeedClusterer.initialize` — bootstrap: encode all ``(need,
  preference)`` samples, KMeans them into ``n_clusters``, then ask the LLM to
  label each cluster with ``(γ, ρ)``.
- :meth:`NeedClusterer.assign_and_refine` — incremental: for each new sample,
  find the closest existing cluster (running-mean centroid update), then ask
  the LLM to refine the cluster's ``(γ, ρ)`` given the new sample.

Backend choices:

- KMeans: prefers ``sklearn.cluster.KMeans``; falls back to a tiny pure-Python
  k-means++ implementation when sklearn isn't around (sufficient for the
  prototype-scale data we cluster here).
- Embedder: any object satisfying :class:`src.llm.embedder.Embedder`.
"""

from __future__ import annotations

import logging
import math
import random
import uuid
from collections import Counter
from typing import Any, Sequence

from src.llm.embedder import Embedder, cosine_similarity
from src.llm.llm import LLMClient
from src.memory.schemas import NeedCluster, NeedSolutionItem
from src.memory.update.prompts import (
    NEED_CLUSTER_LABEL_SYSTEM,
    NEED_CLUSTER_REFINE_SYSTEM,
    build_need_cluster_label_prompt,
    build_need_cluster_refine_prompt,
)

_logger = logging.getLogger(__name__)

# Maximum number of representative (need, preference) samples kept per
# cluster; bounds prompt size for the labelling / refining LLM calls.
_SAMPLE_LIMIT_PER_CLUSTER = 5


# ---------------------------------------------------------------------------
# Item accessor helpers (NeedSolutionItem | dict[str, Any])
# ---------------------------------------------------------------------------


def _item_id(item: NeedSolutionItem | dict[str, Any]) -> str:
    return item.item_id if isinstance(item, NeedSolutionItem) else str(item.get("item_id", ""))


def _item_need(item: NeedSolutionItem | dict[str, Any]) -> str:
    if isinstance(item, NeedSolutionItem):
        return item.inferred_need.strip()
    return str(item.get("inferred_need", "")).strip()


def _item_pref(item: NeedSolutionItem | dict[str, Any]) -> str:
    if isinstance(item, NeedSolutionItem):
        return item.preference.strip()
    return str(item.get("preference", "")).strip()


def _item_text(item: NeedSolutionItem | dict[str, Any]) -> str:
    """Compose the embedding input ``f(r_i, p_i)`` for a single item."""
    need = _item_need(item)
    pref = _item_pref(item)
    text = need
    if pref:
        text = f"{text} | {pref}" if text else pref
    return text or "（无内容）"


# ---------------------------------------------------------------------------
# Vector math helpers (pure-Python for portability; the data is small)
# ---------------------------------------------------------------------------


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _mean_normalize(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    out = [0.0] * dim
    for v in vectors:
        for d in range(dim):
            out[d] += v[d]
    inv = 1.0 / len(vectors)
    out = [x * inv for x in out]
    return _l2_normalize(out)


def _running_mean(centroid: list[float], vector: list[float], current_size: int) -> list[float]:
    """Update an L2-normalised centroid after adding ``vector`` to a cluster
    that previously held ``current_size`` members.

    We treat ``centroid`` as the mean of normalised member vectors, perform
    the running-mean update, then re-normalise to keep cosine-distance
    comparisons sane.
    """
    new_size = current_size + 1
    inv = 1.0 / new_size
    new_c = [(c * current_size + v) * inv for c, v in zip(centroid, vector)]
    return _l2_normalize(new_c)


# ---------------------------------------------------------------------------
# KMeans (sklearn-preferred, pure-Python fallback)
# ---------------------------------------------------------------------------


def _kmeans(
    vectors: list[list[float]],
    n_clusters: int,
    *,
    max_iter: int = 50,
    seed: int = 17,
) -> list[int]:
    """Return per-vector cluster labels (length == ``len(vectors)``)."""
    if not vectors or n_clusters <= 0:
        return []
    n_clusters = min(n_clusters, len(vectors))
    if n_clusters == 1:
        return [0] * len(vectors)

    try:
        import numpy as np  # type: ignore[import-not-found]
        from sklearn.cluster import KMeans  # type: ignore[import-not-found]

        arr = np.asarray(vectors, dtype="float64")
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=seed,
            max_iter=max_iter,
        )
        labels = kmeans.fit_predict(arr)
        return [int(x) for x in labels]
    except Exception as err:  # pragma: no cover - sklearn missing path
        _logger.info("Falling back to pure-Python KMeans (%s)", err)

    return _pure_python_kmeans(vectors, n_clusters, max_iter=max_iter, seed=seed)


def _pure_python_kmeans(
    vectors: list[list[float]],
    n_clusters: int,
    *,
    max_iter: int,
    seed: int,
) -> list[int]:
    """Cosine-flavour k-means++; sufficient for tens-to-hundreds of vectors."""
    rng = random.Random(seed)
    dim = len(vectors[0])

    first_idx = rng.randrange(len(vectors))
    centroids: list[list[float]] = [list(vectors[first_idx])]
    while len(centroids) < n_clusters:
        weights: list[float] = []
        for v in vectors:
            min_dist = min(1.0 - cosine_similarity(v, c) for c in centroids)
            weights.append(max(min_dist, 0.0) ** 2)
        total = sum(weights) or 1.0
        r = rng.random() * total
        acc = 0.0
        chosen_idx = 0
        for idx, w in enumerate(weights):
            acc += w
            if acc >= r:
                chosen_idx = idx
                break
        centroids.append(list(vectors[chosen_idx]))

    labels = [0] * len(vectors)
    for _ in range(max_iter):
        changed = False
        for i, v in enumerate(vectors):
            best_j = 0
            best_sim = -1.0
            for j, c in enumerate(centroids):
                sim = cosine_similarity(v, c)
                if sim > best_sim:
                    best_sim = sim
                    best_j = j
            if labels[i] != best_j:
                labels[i] = best_j
                changed = True
        if not changed:
            break
        for j in range(n_clusters):
            members = [vectors[i] for i in range(len(vectors)) if labels[i] == j]
            if not members:
                continue
            new_c = [0.0] * dim
            for m in members:
                for d in range(dim):
                    new_c[d] += m[d]
            inv = 1.0 / len(members)
            new_c = [x * inv for x in new_c]
            centroids[j] = _l2_normalize(new_c)
    return labels


def _coerce_label_response(resp: Any) -> tuple[str, str] | None:
    if not isinstance(resp, dict):
        return None
    nt = str(resp.get("need_type", "")).strip()
    pp = str(resp.get("preference_principle", "")).strip()
    if not nt:
        return None
    return nt[:24], pp[:160]


# ---------------------------------------------------------------------------
# Clusterer
# ---------------------------------------------------------------------------


class NeedClusterer:
    """Mem-PAL ``M_P`` maintainer.

    Args:
        embedder: any object satisfying :class:`Embedder`.
        llm_client: client used for cluster labelling and refining.
        n_clusters: target K for the bootstrap KMeans.
        min_items_to_cluster: when fewer items than this are available at
            initialisation time, every item becomes its own singleton cluster
            (so retrieval can still filter by ``cluster_id``).
        now_iso: ISO-8601 string used for the ``updated_at`` field on every
            cluster mutated by this instance. Defaults to wall-clock time;
            session-replay callers should pass the sample's
            ``dialogue_timestamp`` so cluster history matches the dataset.
    """

    def __init__(
        self,
        embedder: Embedder,
        llm_client: LLMClient,
        *,
        n_clusters: int = 5,
        min_items_to_cluster: int = 5,
        now_iso: str | None = None,
    ) -> None:
        self._embedder = embedder
        self._llm = llm_client
        self._n_clusters = n_clusters
        self._min_items_to_cluster = min_items_to_cluster
        self._now_iso = now_iso

    def _now(self) -> str:
        """Return the timestamp for ``updated_at`` writes.

        Resolved at call-time so swapping the wrapping clock between
        ``initialize`` / ``assign_and_refine`` calls works as expected.
        """
        if self._now_iso:
            return self._now_iso
        from src.utils.clock import RealClock

        return RealClock().now_iso()

    # -- public API --------------------------------------------------------

    def initialize(
        self,
        items: Sequence[NeedSolutionItem | dict[str, Any]],
    ) -> tuple[list[NeedCluster], dict[str, str]]:
        """Cluster everything from scratch and label each cluster via LLM."""
        items = list(items)
        if not items:
            return [], {}

        if len(items) < self._min_items_to_cluster:
            return self._singleton_clusters(items)

        texts = [_item_text(it) for it in items]
        vectors = self._embedder.embed_batch(texts)
        labels = _kmeans(vectors, self._n_clusters)

        clusters: list[NeedCluster] = []
        item_id_to_cluster: dict[str, str] = {}

        groups: dict[int, list[int]] = {}
        for idx, lab in enumerate(labels):
            groups.setdefault(lab, []).append(idx)

        for member_indices in groups.values():
            cluster_id = f"cluster-{uuid.uuid4().hex[:8]}"
            members = [items[i] for i in member_indices]
            centroid = _mean_normalize([vectors[i] for i in member_indices])
            need_type, preference_principle = self._llm_label_cluster(members)

            sample_needs = [_item_need(it) for it in members[:_SAMPLE_LIMIT_PER_CLUSTER]]
            sample_prefs = [_item_pref(it) for it in members[:_SAMPLE_LIMIT_PER_CLUSTER]]

            cluster = NeedCluster(
                cluster_id=cluster_id,
                need_type=need_type,
                preference_principle=preference_principle,
                centroid=centroid,
                member_item_ids=[_item_id(it) for it in members],
                sample_needs=sample_needs,
                sample_preferences=sample_prefs,
                size=len(members),
                updated_at=self._now(),
            )
            clusters.append(cluster)
            for it in members:
                item_id_to_cluster[_item_id(it)] = cluster_id

        return clusters, item_id_to_cluster

    def assign_and_refine(
        self,
        clusters: list[NeedCluster],
        new_items: Sequence[NeedSolutionItem | dict[str, Any]],
    ) -> tuple[list[NeedCluster], dict[str, str]]:
        """Assign each new item to the closest cluster and refine ``(γ, ρ)``."""
        new_items = list(new_items)
        if not new_items:
            return clusters, {}
        if not clusters:
            return self.initialize(new_items)

        texts = [_item_text(it) for it in new_items]
        vectors = self._embedder.embed_batch(texts)

        item_id_to_cluster: dict[str, str] = {}
        for it, vec in zip(new_items, vectors):
            best_idx = 0
            best_sim = -1.0
            for i, c in enumerate(clusters):
                sim = cosine_similarity(vec, c.centroid)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i

            cluster = clusters[best_idx]
            new_need = _item_need(it)
            new_pref = _item_pref(it)

            # Capture the pre-update sample history before we append the new
            # sample, so the refine prompt can compare the previous principle
            # against just the new sample (Mem-PAL's "previous γ_i + new r"
            # pattern).
            prev_history = list(zip(cluster.sample_needs, cluster.sample_preferences))

            cluster.centroid = _running_mean(cluster.centroid, vec, cluster.size)
            iid = _item_id(it)
            cluster.member_item_ids.append(iid)
            cluster.size += 1
            cluster.sample_needs = (cluster.sample_needs + [new_need])[-_SAMPLE_LIMIT_PER_CLUSTER:]
            cluster.sample_preferences = (
                cluster.sample_preferences + [new_pref]
            )[-_SAMPLE_LIMIT_PER_CLUSTER:]

            refined = self._llm_refine_cluster(
                cluster,
                prev_history=prev_history,
                new_need=new_need,
                new_preference=new_pref,
            )
            if refined is not None:
                cluster.need_type, cluster.preference_principle = refined
            cluster.updated_at = self._now()
            item_id_to_cluster[iid] = cluster.cluster_id

        return clusters, item_id_to_cluster

    # -- helpers -----------------------------------------------------------

    def _singleton_clusters(
        self,
        items: list[NeedSolutionItem | dict[str, Any]],
    ) -> tuple[list[NeedCluster], dict[str, str]]:
        """Bootstrap path when ``len(items) < min_items_to_cluster``.

        Every item becomes its own cluster: ``need_type`` falls back to the
        item's ``inferred_need`` phrase (already free-form) and
        ``preference_principle`` mirrors its ``preference``. This keeps
        cluster_id-based retrieval functional from day one without needing
        an LLM call.
        """
        clusters: list[NeedCluster] = []
        mapping: dict[str, str] = {}
        for it in items:
            need_text = _item_need(it) or "未分类需求"
            pref_text = _item_pref(it)
            vec = self._embedder.embed_text(_item_text(it))
            cluster_id = f"cluster-{uuid.uuid4().hex[:8]}"
            clusters.append(
                NeedCluster(
                    cluster_id=cluster_id,
                    need_type=need_text[:24],
                    preference_principle=pref_text[:160],
                    centroid=vec,
                    member_item_ids=[_item_id(it)],
                    sample_needs=[need_text] if need_text else [],
                    sample_preferences=[pref_text] if pref_text else [],
                    size=1,
                    updated_at=self._now(),
                )
            )
            mapping[_item_id(it)] = cluster_id
        return clusters, mapping

    def _llm_label_cluster(
        self,
        members: list[NeedSolutionItem | dict[str, Any]],
    ) -> tuple[str, str]:
        samples = [
            (_item_need(m), _item_pref(m))
            for m in members[:_SAMPLE_LIMIT_PER_CLUSTER]
            if _item_need(m)
        ]
        if not samples:
            return "未分类需求", ""
        try:
            prompt = build_need_cluster_label_prompt(samples)
            resp = self._llm.generate_json(prompt, system_prompt=NEED_CLUSTER_LABEL_SYSTEM)
            parsed = _coerce_label_response(resp)
            if parsed is not None:
                return parsed
        except Exception as err:
            _logger.warning(
                "Cluster labelling failed (%s); using rule fallback summary.", err,
            )

        # Rule fallback: most-common need phrase as need_type, first non-empty
        # preference as principle.
        counter = Counter(need for need, _ in samples if need)
        need_type = counter.most_common(1)[0][0] if counter else "未分类需求"
        preference = next((p for _, p in samples if p), "")
        return need_type[:24], preference[:160]

    def _llm_refine_cluster(
        self,
        cluster: NeedCluster,
        *,
        prev_history: list[tuple[str, str]],
        new_need: str,
        new_preference: str,
    ) -> tuple[str, str] | None:
        try:
            prompt = build_need_cluster_refine_prompt(
                prev_need_type=cluster.need_type,
                prev_preference_principle=cluster.preference_principle,
                history_samples=prev_history,
                new_need=new_need,
                new_preference=new_preference,
            )
            resp = self._llm.generate_json(prompt, system_prompt=NEED_CLUSTER_REFINE_SYSTEM)
            return _coerce_label_response(resp)
        except Exception as err:
            _logger.warning(
                "Cluster refine failed (%s); keeping previous (γ, ρ).", err,
            )
            return None
