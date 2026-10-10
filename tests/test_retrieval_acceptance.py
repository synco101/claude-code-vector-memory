#!/usr/bin/env python3
"""
Retrieval acceptance: real questions Alexander asks → the right page must be in the top 3.

Why this exists (2026-09-05): the store held the right pages, yet six real questions found the
expected transcript at rank 4–7 or not at all, while an exact-token lookup found it instantly.
Pure embedding search is weak on proper nouns and rare terms ("Забродин" → DistroKid). This suite
is the bar for the hybrid (lexical + vector, RRF-fused) retrieval; it was written FIRST and failed
on 5 of 6 before the change. Runs against the live store — it is an acceptance test, not a unit test.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.memory_search import MemorySearcher  # noqa: E402

CASES = [
    ("что решили с Сергеем про воронку и SLA", "kira-serge-sales-funnel"),
    ("условия сотрудничества с Gray после discovery-сессии", "em-x-gray-conditions"),
    ("Забродин зарплатные ожидания собеседование", "readai-em-pm-interview"),
    ("что Покок говорит про PRD и grill перед кодом", "pocock-full-walkthrough"),
    ("что Патрик и Саша обсуждали на стендапе про AI Academy pilot и B2B", "standups-patrick-sasha"),
    ("кто такой Иван Курочкин и что ему поручили", "b2b-lidogeneratsiya"),
]
MIN_PASS = 5  # of 6 — the Gray case may stay hard (page says Gray/Grey, colour word collides)


class TestRetrievalAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.searcher = MemorySearcher()
        cls.report = []

    def _rank(self, query, expected):
        res = self.searcher.search(query, n_results=3)
        for i, r in enumerate(res, 1):
            if expected in r.get("filename", ""):
                return i, res
        return None, res

    def test_expected_page_in_top3(self):
        passed = 0
        for q, exp in CASES:
            rank, res = self._rank(q, exp)
            top = res[0]["filename"] if res else "-"
            self.report.append(f"{'OK ' if rank else 'MISS'} rank={rank} | {q[:48]} → top1={top[:55]}")
            if rank:
                passed += 1
        print("\n" + "\n".join(self.report))
        self.assertGreaterEqual(passed, MIN_PASS, f"{passed}/{len(CASES)} questions found the right page in top 3")

    def test_proper_noun_is_top1(self):
        """A rare surname that appears in exactly one transcript must win outright."""
        rank, res = self._rank("Забродин зарплатные ожидания собеседование", "readai-em-pm-interview")
        self.assertEqual(rank, 1, f"top1={res[0]['filename'] if res else '-'}")


if __name__ == "__main__":
    unittest.main()
