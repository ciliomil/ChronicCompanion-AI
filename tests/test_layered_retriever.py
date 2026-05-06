"""Tests for the three-stage retrieval pipeline."""

from __future__ import annotations

import os
import unittest

from src.llm.embedder import HashTfidfEmbedder
from src.retrieval import (
    CurrentQueryFrame,
    LLMRetrievalPlanner,
    LayeredRetriever,
    MemoryPack,
    RuleRetrievalPlanner,
    ResponseGenerator,
    build_memory_pack,
    build_retrieval_plan,
    generate_response,
)
from src.retrieval.layered_retriever import _rank_need_items


class _StubLLM:
    """Tiny stub that returns a fixed JSON dict on ``generate_json``."""

    def __init__(self, response: dict) -> None:
        self._response = response
        self.calls = 0

    def generate_json(self, prompt: str, system_prompt: str | None = None):
        self.calls += 1
        return self._response

    def generate_text(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.2) -> str:
        return "已经收到，您说得对。"


class _ErrorLLM:
    def generate_json(self, prompt, system_prompt=None):
        raise RuntimeError("boom")

    def generate_text(self, prompt, system_prompt=None, temperature: float = 0.2):
        raise RuntimeError("boom")


def _make_profile() -> dict:
    return {
        "basic_info": {
            "medical_care": {
                "summary": "用户在社区医院定期复诊。",
                "claims": [
                    {
                        "claim_id": "m-1",
                        "category": "medical_care",
                        "claim_type": "stable_life_background",
                        "content": "用户主要在社区医院定期复诊。",
                        "tags": ["medical_visit"],
                        "source_event_ids": ["e-101"],
                        "source_need_item_ids": [],
                        "source_turn_ids": [],
                        "status": "active",
                        "confidence": 0.9,
                    }
                ],
                "summary_source_claim_ids": ["m-1"],
                "updated_at": "2026-03-01T00:00:00Z",
            },
            "family": {
                "summary": "女儿长期参与控糖饮食。",
                "claims": [
                    {
                        "claim_id": "f-1",
                        "category": "family",
                        "claim_type": "care_context",
                        "content": "女儿长期参与用户的控糖饮食与复诊管理。",
                        "tags": ["family", "diet"],
                        "source_event_ids": ["e-200"],
                        "source_need_item_ids": [],
                        "source_turn_ids": [],
                        "status": "active",
                        "confidence": 0.85,
                    }
                ],
                "summary_source_claim_ids": ["f-1"],
                "updated_at": "2026-03-01T00:00:00Z",
            },
            "health": {
                "summary": "用户长期患糖尿病。",
                "claims": [
                    {
                        "claim_id": "h-1",
                        "category": "health",
                        "claim_type": "clinical_background",
                        "content": "用户长期患糖尿病，需要持续血糖管理。",
                        "tags": ["glucose"],
                        "source_event_ids": ["e-300"],
                        "source_need_item_ids": [],
                        "source_turn_ids": [],
                        "status": "active",
                        "confidence": 0.95,
                    }
                ],
                "summary_source_claim_ids": ["h-1"],
                "updated_at": "2026-03-01T00:00:00Z",
            },
            "leisure": {
                "summary": "",
                "claims": [],
                "summary_source_claim_ids": [],
                "updated_at": "2026-03-01T00:00:00Z",
            },
        },
        "recent_status": {
            "health_status": "近期饭后血糖偏高。",
            "self_management_status": "饮食记录不稳定。",
            "mental_status": "",
            "family_social_status": "",
            "interest_changes": [],
            "risk_flags": [],
            "field_source_event_ids": {},
        },
        "need_preferences": [
            {
                "cluster_id": "c1",
                "need_domain": "diet_glucose_management",
                "preference_principle": "优先低升糖、家常食材、容易执行的建议。",
                "member_item_ids": ["need-a"],
            }
        ],
    }


def _make_need_items() -> list[dict]:
    return [
        {
            "item_id": "need-a",
            "inferred_need": "夜间加餐如何避免血糖升高",
            "context": "用户晚上饿，怕影响血糖。",
            "need_domain": "diet_glucose_management",
            "need_object": "夜间加餐",
            "related_tags": ["diet", "glucose"],
            "context_event_ids": ["e-200"],
            "source_turn_ids": ["t1"],
            "cluster_id": "c1",
            "solutions": [
                {
                    "ai_solution_summary": "建议晚间少量低升糖食物，比如黄瓜或半个鸡蛋。",
                    "feedback_turn_ids": ["t2"],
                    "fit_score": 0.8,
                    "revealed_preference": "更喜欢家常、容易准备的食物。",
                    "confidence": 0.85,
                }
            ],
        },
        {
            "item_id": "need-b",
            "inferred_need": "膝盖不好怎么继续散步",
            "context": "用户膝盖一走久就酸。",
            "need_domain": "activity_safety",
            "need_object": "散步替代活动",
            "related_tags": ["activity", "mobility"],
            "context_event_ids": ["e-400"],
            "source_turn_ids": ["t5"],
            "cluster_id": "c2",
            "solutions": [],
        },
    ]


def _make_events() -> list[dict]:
    return [
        {
            "event_id": "e-101",
            "timestamp": "2026-02-10T09:00:00Z",
            "event_summary": "社区医院糖尿病复诊。",
            "tags": ["medical_visit"],
            "source_turn_ids": ["t10"],
        },
        {
            "event_id": "e-200",
            "timestamp": "2026-02-12T19:00:00Z",
            "event_summary": "女儿提醒晚饭少吃主食。",
            "tags": ["family", "diet"],
            "source_turn_ids": ["t11"],
        },
        {
            "event_id": "e-300",
            "timestamp": "2026-01-05T08:00:00Z",
            "event_summary": "确诊糖尿病多年，长期服药。",
            "tags": ["glucose", "medication"],
            "source_turn_ids": ["t12"],
        },
        {
            "event_id": "e-extra",
            "timestamp": "2026-02-20T11:00:00Z",
            "event_summary": "饭后血糖偏高记录。",
            "tags": ["glucose", "diet"],
            "source_turn_ids": ["t13"],
        },
    ]


_STAGE1_RESPONSE = {
    "current_query_frame": {
        "user_query": "我最近饭后血糖有点高，晚上还能不能吃水果？",
        "query_summary": "用户担心饭后血糖偏高，问晚间能否吃水果。",
        "inferred_need": "晚间水果摄入与血糖控制",
        "need_domain": "diet_glucose_management",
        "need_object": "晚间水果",
        "tags": ["glucose", "diet"],
        "current_context": "用户长期患糖尿病，近期饭后血糖偏高。",
        "intent_type": "self_management",
        "medical_relevance": "medium",
        "risk_level": "normal",
        "risk_triggers": [],
    },
    "memory_retrieval_plan": {
        "selected_basic_info_claims": [
            {"claim_id": "h-1", "assigned_role": "background_context"},
            {"claim_id": "f-1", "assigned_role": "support_network"},
            # bogus id should be dropped by the parser
            {"claim_id": "ghost", "assigned_role": "background_context"},
        ],
        "target_need_domains": ["diet_glucose_management"],
        "need_top_k": 2,
        "event_top_k": 4,
        "include_recent_status_fields": ["health_status", "self_management_status"],
        "safety_sensitive": False,
    },
}


# ---------------------------------------------------------------------------
# Stage 1: planner
# ---------------------------------------------------------------------------


class TestRetrievalPlanner(unittest.TestCase):
    def test_llm_path_parses_frame_and_plan(self) -> None:
        stub = _StubLLM(_STAGE1_RESPONSE)
        frame, plan = build_retrieval_plan(
            "我最近饭后血糖有点高，晚上还能不能吃水果？",
            "",
            _make_profile(),
            llm=stub,
        )
        self.assertEqual(frame.intent_type, "self_management")
        self.assertEqual(frame.need_domain, "diet_glucose_management")
        self.assertIn("glucose", frame.tags)
        self.assertEqual(plan.need_top_k, 2)
        self.assertEqual(plan.event_top_k, 4)
        self.assertEqual(plan.target_need_domains, ["diet_glucose_management"])

        # bogus ghost id was filtered out
        ids = [c["claim_id"] for c in plan.selected_basic_info_claims]
        self.assertEqual(ids, ["h-1", "f-1"])
        roles = {c["claim_id"]: c["assigned_role"] for c in plan.selected_basic_info_claims}
        self.assertEqual(roles["h-1"], "background_context")
        self.assertEqual(roles["f-1"], "support_network")

    def test_invalid_response_uses_rule_fallback(self) -> None:
        frame, plan = build_retrieval_plan(
            "今天有点不舒服",
            "",
            _make_profile(),
            llm=_ErrorLLM(),
        )
        self.assertEqual(frame.intent_type, "casual_chat")
        self.assertEqual(plan.selected_basic_info_claims, [])
        self.assertEqual(plan.need_top_k, 0)
        self.assertEqual(plan.event_top_k, 0)
        # rule fallback passes through nonempty recent_status fields
        self.assertIn("health_status", plan.include_recent_status_fields)

    def test_rule_planner_produces_safe_defaults(self) -> None:
        frame, plan = RuleRetrievalPlanner().plan(
            "随便聊聊",
            "",
            {"basic_info": {}, "recent_status": {}},
        )
        self.assertEqual(frame.risk_level, "normal")
        self.assertEqual(plan.target_need_domains, [])
        self.assertEqual(plan.include_recent_status_fields, [])

    def test_urgent_frame_forces_safety_sensitive(self) -> None:
        urgent_response = {
            "current_query_frame": {
                **_STAGE1_RESPONSE["current_query_frame"],
                "risk_level": "urgent",
                "risk_triggers": ["疑似低血糖"],
            },
            "memory_retrieval_plan": {
                **_STAGE1_RESPONSE["memory_retrieval_plan"],
                "safety_sensitive": False,  # planner should still flip this on
            },
        }
        _, plan = build_retrieval_plan(
            "我现在心慌出汗手抖",
            "",
            _make_profile(),
            llm=_StubLLM(urgent_response),
        )
        self.assertTrue(plan.safety_sensitive)


# ---------------------------------------------------------------------------
# Stage 2: deterministic retriever
# ---------------------------------------------------------------------------


class TestNeedItemRanking(unittest.TestCase):
    def test_rank_prefers_matching_domain_and_tags(self) -> None:
        os.environ["LONGMEM_FORCE_HASH_EMBEDDER"] = "1"
        emb = HashTfidfEmbedder()
        frame = CurrentQueryFrame(
            user_query="晚上饿了能吃啥不升糖",
            query_summary="夜间加餐",
            inferred_need="夜间加餐与血糖",
            need_domain="diet_glucose_management",
            need_object="夜间加餐",
            tags=["diet", "glucose"],
            current_context="用户晚上饿，怕血糖升高。",
        )
        ranked = _rank_need_items(
            frame,
            domains={"diet_glucose_management"},
            need_items=_make_need_items(),
            embedder=emb,
            top_k=2,
        )
        self.assertEqual(ranked[0]["item_id"], "need-a")


class TestBuildMemoryPack(unittest.TestCase):
    def test_pack_contains_role_tagged_claims_and_seeded_events(self) -> None:
        os.environ["LONGMEM_FORCE_HASH_EMBEDDER"] = "1"
        pack = build_memory_pack(
            session_id="s1",
            turn_id="u1",
            user_query="我最近饭后血糖有点高，晚上还能不能吃水果？",
            dialogue_context="",
            profile=_make_profile(),
            need_solution_items=_make_need_items(),
            events=_make_events(),
            llm=_StubLLM(_STAGE1_RESPONSE),
            embedder=HashTfidfEmbedder(),
        )
        self.assertIsInstance(pack, MemoryPack)

        # role-tagged claims
        roles_by_id = {c["claim_id"]: c["assigned_role"] for c in pack.selected_basic_info_claims}
        self.assertEqual(roles_by_id, {"h-1": "background_context", "f-1": "support_network"})

        # preference principle pulled by need_domain
        principles = [p["preference_principle"] for p in pack.preference_principles]
        self.assertTrue(any("低升糖" in p for p in principles))

        # ranked needs honors top_k=2
        self.assertLessEqual(len(pack.relevant_needs), 2)
        self.assertEqual(pack.relevant_needs[0]["item_id"], "need-a")

        # events seeded by selected_claims.source_event_ids first
        event_ids = [e["event_id"] for e in pack.relevant_events]
        self.assertIn("e-300", event_ids)  # from h-1
        self.assertIn("e-200", event_ids)  # from f-1 / need-a
        self.assertLessEqual(len(event_ids), 4)

        # recent_status sliced
        self.assertIn("health_status", pack.recent_status_slice)
        self.assertNotIn("mental_status", pack.recent_status_slice)


# ---------------------------------------------------------------------------
# Stage 3: response generator
# ---------------------------------------------------------------------------


class TestResponseGeneration(unittest.TestCase):
    def _sample_pack(self) -> MemoryPack:
        os.environ["LONGMEM_FORCE_HASH_EMBEDDER"] = "1"
        return build_memory_pack(
            session_id="s1",
            turn_id="u1",
            user_query="我最近饭后血糖有点高，晚上还能不能吃水果？",
            dialogue_context="",
            profile=_make_profile(),
            need_solution_items=_make_need_items(),
            events=_make_events(),
            llm=_StubLLM(_STAGE1_RESPONSE),
            embedder=HashTfidfEmbedder(),
        )

    def test_llm_path_returns_text(self) -> None:
        pack = self._sample_pack()
        text = generate_response(pack, llm=_StubLLM({}))  # generate_text is stubbed
        self.assertTrue(text)
        self.assertIn("收到", text)

    def test_fallback_on_error_returns_safe_message(self) -> None:
        pack = self._sample_pack()
        text = ResponseGenerator(client=_ErrorLLM()).generate(pack)
        self.assertTrue(text)

    def test_urgent_pack_falls_back_to_emergency_phrasing(self) -> None:
        pack = self._sample_pack()
        pack.current_query.risk_level = "urgent"
        text = ResponseGenerator(client=_ErrorLLM()).generate(pack)
        self.assertIn("家人", text)


if __name__ == "__main__":
    unittest.main()
