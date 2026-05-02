"""Layered pipeline smoke test."""

from __future__ import annotations

import os
import tempfile
import unittest

from src.experiments.run_layered_memory import run


class SmokePipelineTest(unittest.TestCase):
    def test_layered_pipeline_runs(self) -> None:
        original_data_dir = os.environ.get("DATA_DIR")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["DATA_DIR"] = tmp
            reply = run(session_id="test", query="我最近睡不好，有点担心")
            self.assertTrue(len(reply) > 0)
        if original_data_dir is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = original_data_dir


if __name__ == "__main__":
    unittest.main()
