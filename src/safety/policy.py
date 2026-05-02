"""Safety response policy."""

from __future__ import annotations

from src.safety.classifier import HIGH, URGENT


def apply_safety_policy(risk_level: str) -> str | None:
    if risk_level == URGENT:
        return "我无法提供紧急医疗判断。请立即联系急救电话或尽快前往医院。"
    if risk_level == HIGH:
        return "这个问题涉及高风险医疗决策，我不能给出诊断或用药调整建议。建议尽快咨询医生。"
    return None
