"""Risk classification for user messages."""

from __future__ import annotations


LOW = "low"
MEDIUM = "medium"
HIGH = "high"
URGENT = "urgent"


def classify_risk(text: str) -> str:
    high_markers = ("胰岛素加量", "停药", "替代治疗", "我该吃多少药")
    urgent_markers = ("昏迷", "胸痛", "呼吸困难", "急救")
    medium_markers = ("血糖波动", "头晕", "乏力", "睡不好")

    if any(marker in text for marker in urgent_markers):
        return URGENT
    if any(marker in text for marker in high_markers):
        return HIGH
    if any(marker in text for marker in medium_markers):
        return MEDIUM
    return LOW
