"""Compose safe and personalized prompts then generate a response."""

from __future__ import annotations


def build_layered_prompt(query: str, profile: dict, context: dict) -> str:
    return (
        "你是一个面向糖尿病老年用户的陪伴助手，请提供温和、非诊断性的建议。\n"
        f"用户提问: {query}\n"
        f"用户画像: {profile}\n"
        f"检索上下文: {context}\n"
        "请输出简洁、可执行、风险受控的回复。"
    )


def build_baseline_prompt(query: str, retrieved_turns: list[dict] | None = None) -> str:
    memory_text = ""
    if retrieved_turns:
        memory_text = f"\n相关历史: {retrieved_turns}"
    return (
        "你是一个陪伴助手，不提供诊断或高风险医疗建议。\n"
        f"当前提问: {query}{memory_text}\n"
        "请给出温和、日常化建议。"
    )
