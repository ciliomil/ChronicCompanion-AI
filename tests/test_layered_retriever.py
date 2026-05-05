"""Tests for layered memory retrieval."""

from __future__ import annotations

import os
import unittest

from src.llm.embedder import HashTfidfEmbedder
from src.retrieval.memory_query_builder import (
    MEMORY_QUERY_BUILD_QUERY_KEY,
    build_memory_query,
)
from src.retrieval.layered_retriever import build_memory_pack, _rank_need_items


class TestRankNeeds(unittest.TestCase):
    def test_rank_prefers_need_and_tag_overlap(self) -> None:
        os.environ["LONGMEM_FORCE_HASH_EMBEDDER"] = "1"
        emb = HashTfidfEmbedder()
        items = [
            {
                "item_id": "a",
                "inferred_need": "饮食与加餐选择",
                "context": "晚间饥饿与血糖顾虑",
                "related_tags": ["diet", "glucose"],
                "source_turn_ids": [],
                "cluster_id": "c1",
            },
            {
                "item_id": "b",
                "inferred_need": "运动计划",
                "context": "每天散步",
                "related_tags": ["activity"],
                "source_turn_ids": [],
                "cluster_id": None,
            },
        ]
        ranked = _rank_need_items(
            "晚上饿了怕血糖高能吃啥",
            "用户关注夜间加餐",
            ["diet", "glucose"],
            items,
            emb,
            top_k=2,
        )
        self.assertEqual(ranked[0]["item_id"], "a")


class TestBuildMemoryPack(unittest.TestCase):
    def test_pack_clusters_and_events(self) -> None:
        os.environ["LONGMEM_FORCE_HASH_EMBEDDER"] = "1"
        profile = {
            "recent_status": {
                "health_status": "近期血糖波动",
                "self_management_status": "",
                "mental_status": "",
                "family_social_status": "",
                "interest_changes": [],
                "risk_flags": [],
                "field_source_event_ids": {},
            },
            "basic_info": {
                "health": {
                    "summary": "二型糖尿病多年",
                    "claims": [
                        {
                            "claim_id": "claim-h1",
                            "claim_type": "stable_fact",
                            "content": "二型糖尿病多年",
                            "source_event_ids": ["ev_from_claim"],
                        }
                    ],
                },
                "medical_care": {"summary": "", "claims": []},
                "family": {"summary": "", "claims": []},
                "leisure": {"summary": "", "claims": []},
            },
            "need_preferences": [
                {
                    "cluster_id": "c1",
                    "preference_principle": "优先低升糖指数与定量分餐",
                    "need_type": "饮食",
                    "member_item_ids": ["a"],
                    "centroid": [],
                }
            ],
        }
        items = [
            {
                "item_id": "a",
                "inferred_need": "饮食与加餐选择",
                "context": "晚间饥饿与血糖",
                "related_tags": ["diet"],
                "source_turn_ids": ["t1"],
                "context_event_ids": ["ev1"],
                "cluster_id": "c1",
            }
        ]
        events = [
            {
                "event_id": "ev1",
                "event_type": "self_management",
                "timestamp": "2026-01-01T00:00:00Z",
                "source_turn_ids": ["t1"],
                "event_summary": "夜里饿但怕升糖",
                "tags": ["diet", "glucose"],
            },
            {
                "event_id": "ev_from_claim",
                "event_type": "health_medical",
                "timestamp": "2025-12-01T00:00:00Z",
                "source_turn_ids": ["t0"],
                "event_summary": "确诊慢病长期管理",
                "tags": ["glucose"],
            },
        ]

        class _FakeJson:
            def generate_json(self, prompt: str, system_prompt: str | None = None) -> dict:
                return {
                    "memory_query": {
                        "current_need": "夜间加餐不升糖",
                        "current_context": "担心血糖",
                        "current_tags": ["diet", "glucose"],
                        "basic_info_claim_ids": ["claim-h1"],
                        "relevant_claims": [
                            {
                                "category": "health",
                                "content": "血糖波动",
                                "use_role": "context",
                                "source_event_ids": [],
                            }
                        ],
                    },
                }

        pack = build_memory_pack(
            session_id="s1",
            turn_id="u1",
            user_query="晚上饿了能吃啥不升糖",
            dialogue_context="",
            profile=profile,
            need_solution_items=items,
            events=events,
            llm=_FakeJson(),
            embedder=HashTfidfEmbedder(),
            top_k_needs=4,
            top_k_events=5,
        )
        self.assertIn("优先低升糖", "".join(pack.preference_principles))
        self.assertTrue(any(e.get("event_id") == "ev1" for e in pack.relevant_events))
        self.assertEqual(pack.relevant_needs[0]["item_id"], "a")
        self.assertEqual(pack.safety_notes, [])


class TestGenerateQueryFallback(unittest.TestCase):
    def test_invalid_llm_json_uses_rules(self) -> None:
        class _Bad:
            def generate_json(self, prompt: str, system_prompt: str | None = None) -> dict:
                return {"not": "valid"}

        r = build_memory_query(
            "今天饭后想散步半小时可以吗",
            "",
            {"recent_status": {}, "basic_info": {}},
            llm=_Bad(),
        )
        self.assertTrue(r[MEMORY_QUERY_BUILD_QUERY_KEY].current_need)


if __name__ == "__main__":
    unittest.main()
