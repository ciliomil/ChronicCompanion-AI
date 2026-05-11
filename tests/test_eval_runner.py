"""End-to-end eval runner test using stub LLM + a stub memory provider."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from src.evaluation.dataset import iter_topic_samples
from src.evaluation.memory_context import NoMemoryProvider
from src.evaluation.runner import RunConfig, run_evaluation
from src.evaluation.tasks import TASK_NAMES


class _StubLLM:
    """Minimal LLM stub that returns canned outputs per task family.

    ``generate_json`` is reused both by the retrieval planner (when
    memory_strategy=pack) and by solution_selection. The router below routes
    by inspecting the system prompt to decide which canned payload to return.
    """

    def __init__(self) -> None:
        self.text_calls = 0
        self.json_calls = 0

    def generate_text(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.0) -> str:
        self.text_calls += 1
        sys = system_prompt or ""
        if "完整的需求描述" in sys:
            return (
                "用户在长期慢病背景下，希望就当前关心的具体场景获得一份"
                "可由本人独立执行、贴近生活习惯并兼顾家人协助的需求说明，"
                "并希望该说明能呈现出操作简单、便于咀嚼、不依赖复杂电子设备等期望特性。"
            )
        if "唯一一个" in sys:
            return "您可以晚饭少盛半碗白米饭，换成同样大小的一小份杂粮饭，搭配一掌大小的鱼肉和一小碗清炒青菜。"
        return "好的，我记住了。"

    def generate_json(self, prompt: str, system_prompt: str | None = None) -> dict:
        self.json_calls += 1
        sys = system_prompt or ""
        if "current_query_frame" in sys:
            # retrieval planner path
            return {
                "current_query_frame": {
                    "user_query": "test",
                    "query_summary": "测试",
                    "inferred_need": "测试需求",
                    "need_domain": "other",
                    "need_object": "",
                    "tags": [],
                    "current_context": "",
                    "intent_type": "casual_chat",
                    "medical_relevance": "none",
                    "risk_level": "normal",
                    "risk_triggers": [],
                },
                "memory_retrieval_plan": {
                    "selected_basic_info_claims": [],
                    "target_need_domains": [],
                    "need_top_k": 0,
                    "event_top_k": 0,
                    "include_recent_status_fields": [],
                    "safety_sensitive": False,
                },
            }
        if "selected_indices" in sys:
            return {"selected_indices": [0, 3]}
        return {}


def _write_minimal_input(path: Path) -> None:
    payload = {
        "9999": {
            "history": [],
            "query": [
                {
                    "sample_id": "9999_sample0",
                    "dialogue_timestamp": "2026-04-01 09:00:00",
                    "dialogue": {},
                    "topics": {
                        "topic-1": {
                            "user_query": "我最近饭后血糖有点高，晚上还能不能吃水果？",
                            "implicit_needs": [],
                            "requirement": (
                                "用户长期患有糖尿病，近期反映饭后血糖偏高，希望获得"
                                "一份可由本人独立执行、贴近家庭饮食习惯、不依赖复杂"
                                "电子设备的晚间饮食指导，并兼顾家人协助。"
                            ),
                            "solution": {
                                "pos": [
                                    "晚饭把白米饭换成一半杂粮饭，搭配一掌鱼肉和一小碗清炒青菜。",
                                    "饭后散步十分钟，让女儿帮忙记一下第二天空腹血糖。",
                                ],
                                "neg": [
                                    "请用手机 App 每餐拍照打卡并自动比对历史血糖。",
                                    "晚间空腹喝一大杯无糖奇亚籽柠檬水增加饱腹感。",
                                ],
                            },
                            "candidate_solutions": [
                                {
                                    "solution": "晚饭把白米饭换成一半杂粮饭，搭配一掌鱼肉和一小碗清炒青菜。",
                                    "feedback": "pos",
                                },
                                {"solution": "candidate 1", "feedback": "neu"},
                                {"solution": "candidate 2", "feedback": "neu"},
                                {
                                    "solution": "饭后散步十分钟，让女儿帮忙记一下第二天空腹血糖。",
                                    "feedback": "pos",
                                },
                                {"solution": "candidate 4", "feedback": "neu"},
                                {"solution": "candidate 5", "feedback": "neu"},
                                {"solution": "candidate 6", "feedback": "neu"},
                                {
                                    "solution": "请用手机 App 每餐拍照打卡并自动比对历史血糖。",
                                    "feedback": "neg",
                                },
                            ],
                        }
                    },
                }
            ],
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class TestEvalRunnerEndToEnd(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["LONGMEM_FORCE_HASH_EMBEDDER"] = "1"
        self.tmp = Path(tempfile.mkdtemp(prefix="cc-eval-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_full_run_no_memory_writes_predictions_and_metrics(self) -> None:
        input_path = self.tmp / "input.json"
        _write_minimal_input(input_path)

        cfg = RunConfig(
            input_path=input_path,
            memory_path=self.tmp / "memory",  # unused in no_memory
            output_dir=self.tmp / "outputs",
            api_file=Path("conf.yaml"),
            tasks=TASK_NAMES,
            user_ids=["9999"],
            memory_strategy="no_memory",
            model=None,
            temperature=0.0,
            max_samples=None,
            resume=False,
        )

        # Sanity: dataset iter yields exactly one topic sample
        samples = list(iter_topic_samples({"9999": json.loads(input_path.read_text())["9999"]}))
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].gold_pos_indices, [0, 3])

        llm = _StubLLM()
        provider = NoMemoryProvider()
        summary = run_evaluation(cfg, llm=llm, provider=provider)

        # All three task jsonls written
        for task in TASK_NAMES:
            path = cfg.predictions_path(task, "9999")
            self.assertTrue(path.is_file(), f"missing: {path}")
            rows = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(rows), 1)
            row = json.loads(rows[0])
            self.assertEqual(row["user_id"], "9999")
            self.assertEqual(row["topic_id"], "topic-1")
            self.assertIsNone(row.get("error"))

        # Selection got the gold answer right (stub returned [0, 3])
        sel_rows = json.loads(
            cfg.predictions_path("solution_selection", "9999").read_text().strip().splitlines()[0]
        )
        self.assertEqual(sel_rows["selected_indices"], [0, 3])
        self.assertTrue(sel_rows["exact_match"])
        self.assertEqual(sel_rows["selection_score"], 100.0)

        # Metrics + summary written
        self.assertTrue(cfg.metrics_path("requirement_restatement").is_file())
        self.assertTrue(cfg.metrics_path("solution_generation").is_file())
        self.assertTrue(cfg.metrics_path("solution_selection").is_file())
        self.assertTrue(cfg.summary_path.is_file())

        sel_metrics = json.loads(cfg.metrics_path("solution_selection").read_text())
        self.assertEqual(sel_metrics["mean_selection_score"], 100.0)
        self.assertEqual(sel_metrics["exact_match_rate"], 1.0)

        # No retrieval planner calls under no_memory
        self.assertEqual(llm.text_calls, 2)  # task1 + task2
        self.assertEqual(llm.json_calls, 1)  # task3

        self.assertEqual(summary["n_topics"], 1)
        self.assertEqual(summary["n_errors_per_task"], {t: 0 for t in TASK_NAMES})

    def test_resume_skips_existing_topics(self) -> None:
        input_path = self.tmp / "input.json"
        _write_minimal_input(input_path)
        cfg = RunConfig(
            input_path=input_path,
            memory_path=self.tmp / "memory",
            output_dir=self.tmp / "outputs",
            api_file=Path("conf.yaml"),
            tasks=("solution_selection",),
            user_ids=["9999"],
            memory_strategy="no_memory",
            model=None,
            temperature=0.0,
            max_samples=None,
            resume=False,
        )
        llm = _StubLLM()
        provider = NoMemoryProvider()
        run_evaluation(cfg, llm=llm, provider=provider)
        first_calls = llm.json_calls

        cfg.resume = True
        run_evaluation(cfg, llm=llm, provider=provider)
        # No additional LLM calls because the topic was already in the jsonl.
        self.assertEqual(llm.json_calls, first_calls)


if __name__ == "__main__":
    unittest.main()
