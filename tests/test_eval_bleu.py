"""Tests for char-level BLEU implementation."""

from __future__ import annotations

import unittest

from src.evaluation.bleu import (
    CorpusBleuAccumulator,
    corpus_bleu,
    sentence_bleu,
    tokenize,
)


class TestTokenization(unittest.TestCase):
    def test_chinese_chars_split_one_per_token(self) -> None:
        self.assertEqual(tokenize("你好世界"), ["你", "好", "世", "界"])

    def test_whitespace_dropped(self) -> None:
        self.assertEqual(tokenize("a b\tc\nd"), ["a", "b", "c", "d"])

    def test_empty_input(self) -> None:
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize("   "), [])


class TestSentenceBleu(unittest.TestCase):
    def test_perfect_match_yields_one(self) -> None:
        bleu = sentence_bleu("你好世界", ["你好世界"])
        for n in range(1, 5):
            self.assertAlmostEqual(bleu[f"bleu_{n}"], 1.0, places=6)

    def test_completely_disjoint_yields_small_smoothed(self) -> None:
        bleu = sentence_bleu("一二三四五六七八", ["abcdefgh"])
        for n in range(1, 5):
            # NIST method-1 smoothed precision is small but nonzero;
            # brevity penalty stays close to 1.
            self.assertGreaterEqual(bleu[f"bleu_{n}"], 0.0)
            self.assertLess(bleu[f"bleu_{n}"], 0.15)

    def test_partial_overlap_orders_decrease(self) -> None:
        # "今天天气很好" vs "今天天气真好" — 5/6 unigrams match but only 1 trigram
        bleu = sentence_bleu("今天天气很好", ["今天天气真好"])
        self.assertGreater(bleu["bleu_1"], 0.7)
        self.assertGreater(bleu["bleu_1"], bleu["bleu_2"])
        self.assertGreater(bleu["bleu_2"], bleu["bleu_4"])

    def test_multi_reference_uses_max_count(self) -> None:
        # cand "ABCD" vs refs ["ABEF", "CDGH"] — each char in cand appears in some ref
        bleu = sentence_bleu("ABCD", ["ABEF", "CDGH"])
        # All 4 unigrams covered → bleu_1 ≈ 1.0
        self.assertGreater(bleu["bleu_1"], 0.99)

    def test_empty_candidate_returns_zero(self) -> None:
        bleu = sentence_bleu("", ["你好"])
        for n in range(1, 5):
            self.assertEqual(bleu[f"bleu_{n}"], 0.0)


class TestCorpusBleu(unittest.TestCase):
    def test_empty_corpus(self) -> None:
        out = corpus_bleu([], [])
        for n in range(1, 5):
            self.assertEqual(out[f"bleu_{n}"], 0.0)
        self.assertEqual(out["n_samples"], 0)

    def test_single_perfect_sample(self) -> None:
        out = corpus_bleu(["你好"], [["你好"]])
        for n in range(1, 5):
            self.assertAlmostEqual(out[f"bleu_{n}"], 1.0, places=6)
        self.assertEqual(out["n_samples"], 1)
        self.assertEqual(out["n_with_refs"], 1)

    def test_streaming_accumulator_matches_one_shot(self) -> None:
        cands = ["今天天气真好", "我喜欢吃苹果", "随便聊聊"]
        refs = [["今天天气很好"], ["我喜欢苹果", "苹果好吃"], ["闲聊一会儿"]]
        a = CorpusBleuAccumulator()
        for c, r in zip(cands, refs):
            a.add(c, r)
        streamed = a.result()
        oneshot = corpus_bleu(cands, refs)
        for n in range(1, 5):
            self.assertAlmostEqual(streamed[f"bleu_{n}"], oneshot[f"bleu_{n}"])

    def test_no_refs_sample_is_skipped_in_scoring(self) -> None:
        a = CorpusBleuAccumulator()
        a.add("你好", ["你好"])
        a.add("世界", [])  # no references — counted in n_samples but not scored
        out = a.result()
        self.assertEqual(out["n_samples"], 2)
        self.assertEqual(out["n_with_refs"], 1)


if __name__ == "__main__":
    unittest.main()
