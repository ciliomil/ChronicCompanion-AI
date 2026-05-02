"""Dataset loader compatibility checks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.dataset.loader import (
    build_history_chunks,
    iter_query_topic_tasks,
    load_mempal_style_dataset,
)


class DatasetLoaderTest(unittest.TestCase):
    def test_mempal_shape_parsing(self) -> None:
        sample = {
            "u0001": {
                "history": [
                    {
                        "sample_id": "u0001_h0",
                        "dialogue_timestamp": "2026-04-19 10:00:00",
                        "logs": [{"timestamp": "2026-04-19 09:00", "content": "吃了清淡早餐"}],
                        "dialogue": {
                            "turn_1": {
                                "user": {"content": "今天有点累"},
                                "assistant": {"content": "可以先休息一下"},
                            }
                        },
                        "topics": {},
                    }
                ],
                "query": [
                    {
                        "sample_id": "u0001_q0",
                        "dialogue_timestamp": "2026-04-20 10:00:00",
                        "dialogue": {},
                        "topics": {
                            "topic-1": {
                                "user_query": "最近睡不好怎么办",
                                "requirement": "希望改善睡眠",
                                "candidate_solutions": [{"solution": "固定作息"}],
                            }
                        },
                    }
                ],
            }
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.json"
            path.write_text(json.dumps(sample, ensure_ascii=False), encoding="utf-8")
            parsed = load_mempal_style_dataset(path)
            user = parsed["u0001"]

            chunks = build_history_chunks(user)
            tasks = iter_query_topic_tasks(user)

            self.assertTrue(len(chunks) >= 2)
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].topic_id, "topic-1")


if __name__ == "__main__":
    unittest.main()
