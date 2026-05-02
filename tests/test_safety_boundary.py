"""Safety boundary checks."""

from __future__ import annotations

import unittest

from src.safety.classifier import HIGH, LOW, classify_risk
from src.safety.policy import apply_safety_policy


class SafetyBoundaryTest(unittest.TestCase):
    def test_high_risk_message_is_blocked(self) -> None:
        level = classify_risk("我该不该自己把胰岛素加量")
        self.assertEqual(level, HIGH)
        self.assertIsNotNone(apply_safety_policy(level))

    def test_low_risk_message_passes(self) -> None:
        level = classify_risk("今天散步了半小时")
        self.assertEqual(level, LOW)
        self.assertIsNone(apply_safety_policy(level))


if __name__ == "__main__":
    unittest.main()
