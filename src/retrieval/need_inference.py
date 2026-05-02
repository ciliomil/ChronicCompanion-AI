"""Need inference based on query and profile.
"""

from __future__ import annotations

from typing import Any

from src.memory.update.need_solution_extractor import infer_need


def _profile_top_need_phrases(profile: dict[str, Any]) -> list[str]:
    """Extract the existing top-need phrases from a profile dict.

    Tolerates both the old ``dict[str, list[str]]`` shape and the new
    ``list[NeedCluster.to_dict()]`` shape so callers don't crash mid-migration.
    """
    prefs = profile.get("need_preferences", []) or []
    if isinstance(prefs, dict):
        return [str(k) for k in prefs.keys() if str(k).strip()]
    if isinstance(prefs, list):
        out: list[str] = []
        for cluster in prefs:
            if not isinstance(cluster, dict):
                continue
            need_type = str(cluster.get("need_type", "")).strip()
            if need_type:
                out.append(need_type)
        return out
    return []


def infer_current_need(query: str, profile: dict) -> dict[str, list[str] | str]:
    primary = infer_need(query)
    candidates: list[str] = [primary]

    for need in _profile_top_need_phrases(profile):
        if need != primary:
            candidates.append(need)
        if len(candidates) >= 3:
            break

    return {"primary_need": primary, "candidate_needs": candidates}
