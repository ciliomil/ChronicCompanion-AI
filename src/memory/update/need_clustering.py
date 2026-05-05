"""Need preference clustering by domain.

Current strategy (domain-first, no HDBSCAN):
- Each ``need_domain`` corresponds to one cluster.
- Cluster assignment is deterministic by domain.
- For each domain cluster, summarise a ``preference_principle`` from that
  domain's need/preference samples.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Sequence

from src.llm.embedder import Embedder
from src.llm.llm import LLMClient
from src.memory.ontology import NEED_DOMAINS
from src.memory.schemas import NeedCluster, NeedItem
from src.memory.update.prompts import (
    PREFERENCE_PRINCIPLE_UPDATE_SYSTEM,
    build_preference_principle_update_prompt,
)

_logger = logging.getLogger(__name__)

_SAMPLE_LIMIT_PER_CLUSTER = 8
_DEFAULT_NEED_DOMAIN = "other"


def _item_id(item: NeedItem | dict[str, Any]) -> str:
    return item.item_id if isinstance(item, NeedItem) else str(item.get("item_id", ""))


def _item_need(item: NeedItem | dict[str, Any]) -> str:
    if isinstance(item, NeedItem):
        return item.inferred_need.strip()
    return str(item.get("inferred_need", "")).strip()


def _item_domain(item: NeedItem | dict[str, Any]) -> str:
    raw = item.need_domain if isinstance(item, NeedItem) else item.get("need_domain")
    d = str(raw or "").strip()
    if d in NEED_DOMAINS:
        return d
    return _DEFAULT_NEED_DOMAIN


def _item_pref(item: NeedItem | dict[str, Any]) -> str:
    if isinstance(item, NeedItem):
        for v in item.solutions:
            p = str(getattr(v, "revealed_preference", "") or "").strip()
            if p:
                return p
        return ""
    for sol in item.get("solutions") or []:
        if isinstance(sol, dict):
            p = str(sol.get("revealed_preference", "") or sol.get("preference", "")).strip()
            if p:
                return p
    return ""


def _sample_pairs_from_items(
    items: Sequence[NeedItem | dict[str, Any]],
    *,
    limit: int = _SAMPLE_LIMIT_PER_CLUSTER,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for it in items[:limit]:
        need = _item_need(it)
        if not need:
            continue
        pairs.append((need, _item_pref(it)))
    return pairs


def _coerce_label_response(resp: Any) -> str | None:
    if not isinstance(resp, dict):
        return None
    pp = str(
        resp.get("updated_preference_principle", "")
        or resp.get("preference_principle", "")
    ).strip()
    if pp:
        return pp[:160]
    return None


class NeedClusterer:
    """Group by ``need_domain`` and summarise per-domain preferences."""

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
        # Keep signature for compatibility; some args are no longer used.
        self._embedder = embedder
        self._llm = llm_client
        self._cluster_min_size = max(1, cluster_min_size)
        self._hdbscan_min_samples = hdbscan_min_samples
        self._min_items_to_cluster = min_items_to_cluster
        self._stable_threshold = stable_threshold
        self._pending_threshold = pending_threshold
        self._now_iso = now_iso

    def _now(self) -> str:
        if self._now_iso:
            return self._now_iso
        from src.utils.clock import RealClock

        return RealClock().now_iso()

    def initialize(
        self,
        items: Sequence[NeedItem | dict[str, Any]],
    ) -> tuple[list[NeedCluster], dict[str, str]]:
        """Build one cluster per domain from scratch."""
        items = list(items)
        if not items:
            return [], {}

        buckets: dict[str, list[NeedItem | dict[str, Any]]] = defaultdict(list)
        for it in items:
            buckets[_item_domain(it)].append(it)

        clusters: list[NeedCluster] = []
        mapping: dict[str, str] = {}

        for domain, members in buckets.items():
            cluster_id = f"cluster-{domain}"
            member_item_ids = [_item_id(it) for it in members if _item_id(it)]
            preference_principle = self._summarize_domain_preference(domain, members)
            clusters.append(
                NeedCluster(
                    cluster_id=cluster_id,
                    need_domain=domain,
                    preference_principle=preference_principle,
                    member_item_ids=member_item_ids,
                    size=len(member_item_ids),
                    updated_at=self._now(),
                )
            )
            for iid in member_item_ids:
                mapping[iid] = cluster_id

        return clusters, mapping

    def assign_and_refine(
        self,
        clusters: list[NeedCluster],
        new_items: Sequence[NeedItem | dict[str, Any]],
    ) -> tuple[list[NeedCluster], dict[str, str]]:
        """Assign new items by domain and refresh affected principles."""
        new_items = list(new_items)
        if not new_items:
            return clusters, {}
        if not clusters:
            return self.initialize(new_items)

        by_domain: dict[str, NeedCluster] = {c.need_domain: c for c in clusters}
        domain_new_items: dict[str, list[NeedItem | dict[str, Any]]] = defaultdict(list)
        mapping: dict[str, str] = {}

        for it in new_items:
            domain = _item_domain(it)
            iid = _item_id(it)
            if not iid:
                continue
            domain_new_items[domain].append(it)
            mapping[iid] = f"cluster-{domain}"

        for domain, items_for_domain in domain_new_items.items():
            cluster = by_domain.get(domain)
            if cluster is None:
                cluster = NeedCluster(
                    cluster_id=f"cluster-{domain}",
                    need_domain=domain,
                    preference_principle="",
                    member_item_ids=[],
                    size=0,
                    updated_at=self._now(),
                )
                clusters.append(cluster)
                by_domain[domain] = cluster

            old_member_ids = set(cluster.member_item_ids)
            for it in items_for_domain:
                iid = _item_id(it)
                if iid and iid not in old_member_ids:
                    cluster.member_item_ids.append(iid)
                    old_member_ids.add(iid)
            cluster.size = len(cluster.member_item_ids)
            cluster.preference_principle = self._summarize_domain_preference(
                domain,
                items_for_domain,
                old_principle=cluster.preference_principle,
            )
            cluster.updated_at = self._now()

        return clusters, mapping

    def _summarize_domain_preference(
        self,
        domain: str,
        items: Sequence[NeedItem | dict[str, Any]],
        *,
        old_principle: str = "",
    ) -> str:
        item_rows: list[dict[str, Any]] = []
        for it in items:
            if isinstance(it, NeedItem):
                item_rows.append(it.to_dict())
            elif isinstance(it, dict):
                item_rows.append(it)

        if not item_rows:
            return old_principle[:160]

        principle = old_principle[:160]
        try:
            for row in item_rows:
                prompt = build_preference_principle_update_prompt(
                    need_domain=domain,
                    existing_preference_principle=principle,
                    need_item=row,
                )
                resp = self._llm.generate_json(
                    prompt,
                    system_prompt=PREFERENCE_PRINCIPLE_UPDATE_SYSTEM,
                )
                parsed = _coerce_label_response(resp)
                if parsed:
                    principle = parsed
            if principle:
                return principle
        except Exception as err:  # noqa: BLE001
            _logger.warning(
                "Domain preference summarisation failed (%s); using fallback.",
                err,
            )

        # Rule fallback: keep first non-empty preference; else keep previous.
        pairs = _sample_pairs_from_items(items)
        pref = next((p for _, p in pairs if p), "")
        if pref:
            return pref[:160]
        return old_principle[:160]
