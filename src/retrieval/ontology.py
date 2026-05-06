"""Controlled vocabularies for retrieval-time planning + reply control."""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Application roles — how a basic_info claim should be used by the reply LLM.
# ---------------------------------------------------------------------------

APPLICATION_ROLES: tuple[str, ...] = (
    "background_context",
    "hard_constraint",
    "soft_preference",
    "support_network",
    "safety_risk_signal",
    "continuity_anchor",
    "engagement_hook",
    "answer_style_hint",
)

APPLICATION_ROLE_DESCRIPTIONS: dict[str, str] = {
    "background_context": "背景理解，不一定显式提及",
    "hard_constraint": "不能违反的硬约束",
    "soft_preference": "倾向性偏好，尽量照顾",
    "support_network": "家庭/照护资源，可用于建议求助或协作",
    "safety_risk_signal": "与安全风险相关，需要提高谨慎程度",
    "continuity_anchor": "用于承接之前聊过的事",
    "engagement_hook": "用兴趣爱好增强陪伴感",
    "answer_style_hint": "影响表达方式、语气、复杂度",
}


# ---------------------------------------------------------------------------
# CurrentQueryFrame controlled vocabularies.
# ---------------------------------------------------------------------------

INTENT_TYPES: tuple[str, ...] = (
    "companionship",
    "health_advice",
    "self_management",
    "emotional_support",
    "logistics",
    "emergency_risk",
    "casual_chat",
)

INTENT_TYPE_DESCRIPTIONS: dict[str, str] = {
    "companionship": "纯陪伴/闲聊，不带具体诉求",
    "health_advice": "医疗与健康知识、用药、检查、就医建议",
    "self_management": "饮食/运动/睡眠/记录/监测等长期管理",
    "emotional_support": "情绪疏导、倾听、鼓励",
    "logistics": "挂号/跑腿/工具操作/家庭事务等生活事务",
    "emergency_risk": "紧急/急性风险/明显症状",
    "casual_chat": "无明确意图的日常问候",
}

MEDICAL_RELEVANCE_LEVELS: tuple[str, ...] = ("none", "low", "medium", "high")

RISK_LEVELS: tuple[str, ...] = ("normal", "caution", "urgent")


# ---------------------------------------------------------------------------
# recent_status — fixed key set used by include_recent_status_fields.
# ---------------------------------------------------------------------------

RECENT_STATUS_FIELD_KEYS: tuple[str, ...] = (
    "health_status",
    "self_management_status",
    "mental_status",
    "family_social_status",
    "interest_changes",
    "risk_flags",
)
