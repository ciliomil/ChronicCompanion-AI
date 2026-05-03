"""Mem-PAL-style clustering for the top-layer ``need_preferences`` memory.

Implements two operations:

- :meth:`NeedClusterer.initialize` — bootstrap: embed inferred needs only ,
then partition with **HDBSCAN** (no fixed cluster count ``K``), then ask
  the LLM to label each cluster with ``(γ, ρ)``. Depends on the PyPI package
  ``hdbscan``; falls back to heuristic KMeans if it is unavailable.
- :meth:`NeedClusterer.assign_and_refine` — incremental: for each new sample,
  find the closest existing cluster (running-mean centroid update), then ask
  the LLM to refine the cluster's ``(γ, ρ)`` given the new sample.

Backend:

- Density clustering via ``hdbscan.HDBSCAN`` (vectors in Euclidean space;
  cosine-friendly when embeddings are normalised).
- Fallback: sklearn / pure-Python KMeans with a heuristic ``K``.
- Embedding text uses only inferred need.

Embedder: any object satisfying :class:`src.llm.embedder.Embedder`.

Install optionally: ``pip install hdbscan``
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
from src.memory.schemas import NeedCluster, NeedClusterSample, NeedItem
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
_MIN_REPRESENTATIVE_CONFIDENCE = 0.55


# ---------------------------------------------------------------------------
# Item accessor helpers (NeedItem | dict[str, Any])
# ---------------------------------------------------------------------------


def _item_id(item: NeedItem | dict[str, Any]) -> str:
    return item.item_id if isinstance(item, NeedItem) else str(item.get("item_id", ""))


def _item_need(item: NeedItem | dict[str, Any]) -> str:
    if isinstance(item, NeedItem):
        return item.inferred_need.strip()
    return str(item.get("inferred_need", "")).strip()


def _item_pref(item: NeedItem | dict[str, Any]) -> str:
    if isinstance(item, NeedItem):
        for v in item.solutions:
            p = v.preference.strip()
            if p:
                return p
        return ""
    for sol in item.get("solutions") or []:
        if isinstance(sol, dict):
            p = str(sol.get("preference", "")).strip()
            if p:
                return p
    return ""


def _item_text(item: NeedItem | dict[str, Any]) -> str:
    """Compose the embedding input for clustering: inferred need only."""
    need = _item_need(item)
    return need or "（无内容）"


def _samples_from_item(
    item: NeedItem | dict[str, Any],
    vector: list[float] | None = None,
) -> list[NeedClusterSample]:
    """Expand one need item into one :class:`NeedClusterSample` per solution proposal.

    Clustering still keys on need embedding once per item; preference-side
    labelling treats each ``(need, preference)`` pair independently.
    """
    out: list[NeedClusterSample] = []

    if isinstance(item, NeedItem):
        iid = item.item_id
        need = item.inferred_need.strip()
        ts = item.timestamp
        proposals = item.solutions
        if not proposals:
            return [
                NeedClusterSample(
                    item_id=iid,
                    need=need,
                    preference="",
                    timestamp=ts,
                    confidence=None,
                    vector=vector,
                ),
            ]
        for v in proposals:
            conf = v.confidence
            if conf is not None:
                try:
                    conf = float(conf)
                except (TypeError, ValueError):
                    conf = None
            out.append(
                NeedClusterSample(
                    item_id=iid,
                    need=need,
                    preference=v.preference.strip(),
                    timestamp=ts,
                    confidence=conf,
                    vector=vector,
                ),
            )
        return out
    else:
        iid = str(item.get("item_id", ""))
        need = str(item.get("inferred_need", "")).strip()
        ts = str(item.get("timestamp", ""))
        raw_sols = item.get("solutions")
        if isinstance(raw_sols, list) and raw_sols:
            for sol in raw_sols:
                if not isinstance(sol, dict):
                    continue
                pref = str(sol.get("preference", "")).strip()
                conf_raw = sol.get("confidence")
                conf: float | None = None
                if isinstance(conf_raw, (int, float)):
                    conf = float(conf_raw)
                elif conf_raw is not None:
                    try:
                        conf = float(conf_raw)
                    except (TypeError, ValueError):
                        conf = None
                out.append(
                    NeedClusterSample(
                        item_id=iid,
                        need=need,
                        preference=pref,
                        timestamp=ts,
                        confidence=conf,
                        vector=vector,
                    ),
                )
        if out:
            return out
        return [
            NeedClusterSample(
                item_id=iid,
                need=need,
                preference="",
                timestamp=ts,
                confidence=None,
                vector=vector,
            ),
        ]


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

def _confidence_value(conf: float | None) -> float:
        return 0.5 if conf is None else max(0.0, min(1.0, float(conf)))

# ---------------------------------------------------------------------------
# Sample selection helpers
# ---------------------------------------------------------------------------

def _sample_pairs(
    samples: Sequence[NeedClusterSample],
    *,
    limit: int = _SAMPLE_LIMIT_PER_CLUSTER,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    for s in samples[:limit]:
        need = str(s.need or "").strip()
        if not need:
            continue

        pref = str(s.preference or "").strip()
        pairs.append((need, pref))

    return pairs

def _refresh_representative_samples(
        cluster: NeedCluster,
        candidate_samples: list[NeedClusterSample],
        *,
        limit: int = _SAMPLE_LIMIT_PER_CLUSTER,
        min_confidence: float = 0.0,
    ) -> None:
        if not candidate_samples:
            cluster.representative_samples = []
            return

        filtered = [
            s for s in candidate_samples
            if s.confidence is None or s.confidence >= min_confidence
        ]
        if not filtered:
            filtered = candidate_samples

        def score(s: NeedClusterSample) -> tuple[float, str]:
            conf = _confidence_value(s.confidence)

            sim = 0.0
            if s.vector is not None and cluster.centroid:
                sim = cosine_similarity(s.vector, cluster.centroid)

            # Stable cluster: prefer central, grounded samples.
            if cluster.status == "stable":
                main_score = 0.70 * sim + 0.30 * conf
            else:
                # Pending cluster: prefer recent, grounded samples.
                # Timestamp is used as tie-breaker by tuple's second field.
                main_score = 0.50 * sim + 0.50 * conf

            return main_score, s.timestamp or ""

        cluster.representative_samples = sorted(
            filtered,
            key=score,
            reverse=True,
        )[:limit]

# ---------------------------------------------------------------------------
# KMeans (heuristic fallback when HDBSCAN is unavailable)
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

# ---------------------------------------------------------------------------
# HDBSCAN 
# ---------------------------------------------------------------------------

def _partition_density_labels(labels: list[int]) -> list[tuple[list[int], str]]:
    """Partition item indices from HDBSCAN labels.

    Returns:
        [(member_indices, source)]

    source:
        - "hdbscan_cluster": stable density cluster
        - "hdbscan_noise": outlier singleton
    """
    buckets: dict[int, list[int]] = {}
    groups: list[tuple[list[int], str]] = []

    for i, lab in enumerate(labels):
        if lab == -1:
            groups.append(([i], "hdbscan_noise"))
        else:
            buckets.setdefault(lab, []).append(i)

    stable_groups = [
        (indices, "hdbscan_cluster")
        for indices in sorted(buckets.values(), key=min)
    ]

    noise_groups = sorted(
        [g for g in groups if g[1] == "hdbscan_noise"],
        key=lambda x: x[0][0],
    )

    return stable_groups + noise_groups


def _try_hdbscan(
    vectors: list[list[float]],
    *,
    min_cluster_size: int,
    min_samples: int | None,
) -> list[int] | None:
    try:
        import numpy as np  # type: ignore[import-not-found]
        import hdbscan  # type: ignore[import-not-found]

        arr = np.asarray(vectors, dtype="float64")
        ms = (
            min_samples
            if min_samples is not None
            else max(1, min(min_cluster_size - 1, len(vectors)))
        )
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=ms,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labs = clusterer.fit_predict(arr)
        return [int(x) for x in labs]
    except Exception as err:  # pragma: no cover - optional dependency
        _logger.info("HDBSCAN unavailable or failed (%s); using heuristic KMeans.", err)
        return None


def _heuristic_k_partition(vectors: list[list[float]], *, seed: int = 17) -> list[int]:
    """Fallback batch partition with a data-driven K (not user-specified)."""
    if not vectors:
        return []
    n = len(vectors)
    k_guess = int(max(2, round(math.sqrt(n))))
    return _kmeans(vectors, min(k_guess, n), seed=seed)


def _density_or_fallback_labels(
    vectors: list[list[float]],
    *,
    min_cluster_size: int,
    min_samples: int | None,
    seed: int = 17,
) -> list[int]:
    n = len(vectors)
    mcs = max(2, min(min_cluster_size, n))
    labs = _try_hdbscan(vectors, min_cluster_size=mcs, min_samples=min_samples)
    if labs is not None:
        return labs
    return _heuristic_k_partition(vectors, seed=seed)


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
    """
    Args:
        embedder: any object satisfying :class:`Embedder`.
        llm_client: client used for cluster labelling and refining.
        cluster_min_size: HDBSCAN ``min_cluster_size`` (smallest group to treat
            as a cluster; **not** a fixed ``K``). Ignored by the KMeans fallback.
        hdbscan_min_samples: optional HDBSCAN ``min_samples``; ``None`` picks a
            conservative default from ``cluster_min_size``.
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
        cluster_min_size: int = 3,
        hdbscan_min_samples: int | None = None,
        min_items_to_cluster: int = 5,
        stable_threshold: float = 0.62,
        pending_threshold: float = 0.52,
        now_iso: str | None = None,
    ) -> None:
        self._embedder = embedder
        self._llm = llm_client
        self._cluster_min_size = max(2, cluster_min_size)
        self._hdbscan_min_samples = hdbscan_min_samples
        self._min_items_to_cluster = min_items_to_cluster
        self._stable_threshold = stable_threshold
        self._pending_threshold = pending_threshold
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
        items: Sequence[NeedItem | dict[str, Any]],
    ) -> tuple[list[NeedCluster], dict[str, str]]:
        """Cluster everything from scratch and label each cluster via LLM."""
        items = list(items)
        if not items:
            return [], {}

        if len(items) < self._min_items_to_cluster:
            return self._singleton_clusters(items)

        texts = [_item_text(it) for it in items]
        vectors = self._embedder.embed_batch(texts)
        labels = _density_or_fallback_labels(
            vectors,
            min_cluster_size=self._cluster_min_size,
            min_samples=self._hdbscan_min_samples,
        )
        groups = _partition_density_labels(labels)

        clusters: list[NeedCluster] = []
        item_id_to_cluster: dict[str, str] = {}

        for member_indices,source in groups:
            is_noise = source == "hdbscan_noise"

            cluster_id = f"cluster-{uuid.uuid4().hex[:8]}"
            centroid = _mean_normalize([vectors[i] for i in member_indices])
            members = [items[i] for i in member_indices]
            all_samples: list[NeedClusterSample] = []
            for i in member_indices:
                all_samples.extend(_samples_from_item(items[i], vector=vectors[i]))

            if is_noise:
                first = all_samples[0]
                need_type = first.need[:24] if first.need else "未分类需求"
                preference_principle = first.preference[:160] if first.preference else ""
            else:
                temp_cluster = NeedCluster(
                    cluster_id=cluster_id,
                    need_type="",
                    preference_principle="",
                    centroid=centroid,
                    member_item_ids=[_item_id(it) for it in members],
                    representative_samples=[],
                    size=len(members),
                    updated_at=self._now(),
                    status="stable",
                )

                _refresh_representative_samples(
                    temp_cluster,
                    all_samples,
                    min_confidence=_MIN_REPRESENTATIVE_CONFIDENCE,
                )

                need_type, preference_principle = self._llm_label_cluster(
                    temp_cluster.representative_samples,
                )

            cluster = NeedCluster(
                cluster_id=cluster_id,
                need_type=need_type,
                preference_principle=preference_principle,
                centroid=centroid,
                member_item_ids=[_item_id(it) for it in members],
                representative_samples=[],
                size=len(members),
                updated_at=self._now(),
                status="pending" if is_noise else "stable",
            )

            _refresh_representative_samples(
                cluster,
                all_samples,
                min_confidence=_MIN_REPRESENTATIVE_CONFIDENCE,
            )
            for it in members:
                item_id_to_cluster[_item_id(it)] = cluster_id
            clusters.append(cluster)

        return clusters, item_id_to_cluster
    
    def assign_and_refine(
        self,
        clusters: list[NeedCluster],
        new_items: Sequence[NeedItem | dict[str, Any]],
    ) -> tuple[list[NeedCluster], dict[str, str]]:
        """Assign each new item to a sufficiently similar cluster.

        If no existing cluster is similar enough, keep the item as a pending
        singleton instead of forcing it into the nearest cluster.
        """
        new_items = list(new_items)
        if not new_items:
            return clusters, {}
        if not clusters:
            return self.initialize(new_items)

        texts = [_item_text(it) for it in new_items]
        vectors = self._embedder.embed_batch(texts)

        item_id_to_cluster: dict[str, str] = {}

        for it, vec in zip(new_items, vectors):
            # 1. Try stable clusters first.
            stable_idx, stable_sim = self._closest_cluster(
                clusters,
                vec,
                status="stable",
            )

            if stable_sim >= self._stable_threshold:
                cluster = clusters[stable_idx]
                new_samples = _samples_from_item(it, vector=vec)
                if not new_samples:
                    continue
                iid = new_samples[0].item_id or f"item-{uuid.uuid4().hex[:8]}"

                prev_history = _sample_pairs(cluster.representative_samples)

                cluster.centroid = _running_mean(cluster.centroid, vec, cluster.size)
                cluster.member_item_ids.append(iid)
                cluster.size += 1

                candidate_samples = list(cluster.representative_samples or []) + new_samples
                _refresh_representative_samples(
                    cluster,
                    candidate_samples,
                    min_confidence=_MIN_REPRESENTATIVE_CONFIDENCE,
                )

                for ns in new_samples:
                    refined = self._llm_refine_cluster(
                        cluster,
                        prev_history=prev_history,
                        new_need=ns.need,
                        new_preference=ns.preference,
                    )
                    if refined is not None:
                        cluster.need_type, cluster.preference_principle = refined

                cluster.updated_at = self._now()
                item_id_to_cluster[iid] = cluster.cluster_id
                continue

            # 2. Try pending clusters next.
            pending_idx, pending_sim = self._closest_cluster(
                clusters,
                vec,
                status="pending",
            )

            if pending_sim >= self._pending_threshold:
                cluster = clusters[pending_idx]
                new_samples = _samples_from_item(it, vector=vec)
                if not new_samples:
                    continue
                iid = new_samples[0].item_id or f"item-{uuid.uuid4().hex[:8]}"

                cluster.centroid = _running_mean(cluster.centroid, vec, cluster.size)
                cluster.member_item_ids.append(iid)
                cluster.size += 1

                candidate_samples = list(cluster.representative_samples or []) + new_samples

                _refresh_representative_samples(
                    cluster,
                    candidate_samples,
                    min_confidence=_MIN_REPRESENTATIVE_CONFIDENCE,
                )
                cluster.updated_at = self._now()
                item_id_to_cluster[iid] = cluster.cluster_id

                if cluster.size >= self._cluster_min_size:
                    cluster.need_type, cluster.preference_principle = self._llm_label_cluster(
                        cluster.representative_samples,
                    )
                    cluster.status = "stable"
                continue
            
            # 3. If no cluster is similar enough, create a new singleton.
            singleton, mapping = self._singleton_cluster_from_item(
                it,
                vec,
                source="low_similarity",
            )
            clusters.append(singleton)
            item_id_to_cluster.update(mapping)
                    
        return clusters, item_id_to_cluster

    # -- helpers -----------------------------------------------------------
    
    def _closest_cluster(
        self,
        clusters: list[NeedCluster],
        vec: list[float],
        *,
        status: str,
    ) -> tuple[int, float]:
        best_idx = -1
        best_sim = -1.0

        for i, c in enumerate(clusters):
            if c.status != status:
                continue
            if not c.centroid:
                continue

            sim = cosine_similarity(vec, c.centroid)
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        return best_idx, best_sim

    def _singleton_cluster_from_item(
        self,
        item: NeedItem | dict[str, Any],
        vector: list[float],
        *,
        source: str = "singleton",
    ) -> tuple[NeedCluster, dict[str, str]]:
        samples = _samples_from_item(item, vector=vector)

        need_text = _item_need(item) or "未分类需求"
        pref_text = _item_pref(item)
        iid = _item_id(item) or f"item-{uuid.uuid4().hex[:8]}"
        cluster_id = f"cluster-{uuid.uuid4().hex[:8]}"

        cluster = NeedCluster(
            cluster_id=cluster_id,
            need_type=need_text[:24],
            preference_principle=pref_text[:160],
            centroid=vector,
            member_item_ids=[iid],
            representative_samples=samples,
            size=1,
            updated_at=self._now(),
            status="pending",
        )

        return cluster, {iid: cluster_id}

    def _singleton_clusters(
        self,
        items: list[NeedItem | dict[str, Any]],
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
            vec = self._embedder.embed_text(_item_text(it))
            cluster, one_mapping = self._singleton_cluster_from_item(
                it,
                vec,
                source="small_bootstrap",
            )
            clusters.append(cluster)
            mapping.update(one_mapping)

        return clusters, mapping

    def _llm_label_cluster(
        self,
        samples: Sequence[NeedClusterSample],
    ) -> tuple[str, str]:
        pairs = _sample_pairs(samples)
        if not pairs:
            return "未分类需求", ""

        try:
            prompt = build_need_cluster_label_prompt(pairs)
            resp = self._llm.generate_json(
                prompt,
                system_prompt=NEED_CLUSTER_LABEL_SYSTEM,
            )
            parsed = _coerce_label_response(resp)
            if parsed is not None:
                return parsed
        except Exception as err:
            _logger.warning(
                "Cluster labelling failed (%s); using rule fallback summary.",
                err,
            )

        # Rule fallback: most-common need phrase as need_type, first non-empty
        # preference as principle.
        counter = Counter(need for need, _ in pairs if need)
        need_type = counter.most_common(1)[0][0] if counter else "未分类需求"
        preference = next((p for _, p in pairs if p), "")
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
