"""Budget-curve prep, run-row merging, and score-vs-accuracy correlation tests."""

import json
import os
import tempfile
import unittest

from data_selection_swap.llm import correlation, plotting


class BudgetCurveTest(unittest.TestCase):
    ROWS = [
        {"task": "bbh", "method": "random", "fraction": 0.05, "accuracy": 30.0},
        {"task": "bbh", "method": "random", "fraction": 0.10, "accuracy": 32.0},
        {"task": "bbh", "method": "less", "fraction": 0.05, "accuracy": 40.0},
        {"task": "bbh", "setting": "targeted", "rule": "ranked", "fraction": 0.05, "accuracy": 43.0},
        {"task": "tydiqa", "method": "less", "fraction": 0.05, "accuracy": 35.0},
    ]

    def test_label(self):
        self.assertEqual(plotting.row_label({"method": "less"}), "less")
        self.assertEqual(plotting.row_label({"setting": "targeted", "rule": "ranked"}), "targeted-ranked")
        self.assertEqual(plotting.row_label({"setting": "faithful"}), "faithful")

    def test_curve_groups_and_sorts(self):
        curve = plotting.budget_curve(self.ROWS, value_key="accuracy")
        self.assertEqual(set(curve), {"bbh", "tydiqa"})
        self.assertEqual(curve["bbh"]["random"], [(0.05, 30.0), (0.10, 32.0)])
        self.assertIn("targeted-ranked", curve["bbh"])

    def test_missing_value_skipped(self):
        curve = plotting.budget_curve([{"task": "x", "method": "m", "fraction": 0.1}], "accuracy")
        self.assertEqual(curve, {})


class ReadRunRowsTest(unittest.TestCase):
    def test_merges_diagnostics_and_accuracy(self):
        with tempfile.TemporaryDirectory() as root:
            run = os.path.join(root, "runs", "swap_t1")
            os.makedirs(run)
            with open(os.path.join(run, "diagnostics.json"), "w") as fh:
                json.dump({"setting": "targeted", "rule": "ranked", "task": "t1",
                           "fraction": 0.05, "swap": {"value": -0.2}}, fh)
            with open(os.path.join(run, "accuracy.json"), "w") as fh:
                json.dump({"accuracy": 41.5}, fh)
            rows = plotting.read_run_rows(root)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["value"], -0.2)
        self.assertEqual(rows[0]["accuracy"], 41.5)


class CorrelationTest(unittest.TestCase):
    def test_perfect_monotonic(self):
        rows = [{"method": "m", "value": v, "accuracy": v} for v in range(6)]
        res = correlation.score_vs_accuracy(rows, "value", "accuracy", ("method",))
        self.assertAlmostEqual(res["all"]["spearman"], 1.0)
        self.assertEqual(res[("m",)]["n"], 6)

    def test_grouping_and_missing(self):
        rows = [
            {"method": "a", "value": 1, "accuracy": 10}, {"method": "a", "value": 2, "accuracy": 20},
            {"method": "b", "value": 1, "accuracy": 20}, {"method": "b", "value": 2, "accuracy": 10},
            {"method": "b", "value": 3},  # missing accuracy -> skipped
        ]
        res = correlation.score_vs_accuracy(rows, "value", "accuracy", ("method",))
        self.assertEqual(res["all"]["n"], 4)
        self.assertAlmostEqual(res[("a",)]["spearman"], 1.0)
        self.assertAlmostEqual(res[("b",)]["spearman"], -1.0)


if __name__ == "__main__":
    unittest.main()
