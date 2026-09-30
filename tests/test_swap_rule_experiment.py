"""Planner tests for the paired Algorithm 2/Algorithm 3 experiment."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "scripts" / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

SCRIPT = EXPERIMENTS / "compare_swap_rules.py"
EXAMPLE = EXPERIMENTS / "swap_rules.example.json"
SPEC = importlib.util.spec_from_file_location("compare_swap_rules", SCRIPT)
swap_rules = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(swap_rules)


class SwapRulePlanTest(unittest.TestCase):
    def setUp(self):
        with EXAMPLE.open(encoding="utf-8") as handle:
            self.config = json.load(handle)

    def test_example_plan_counts(self):
        swap_rules.validate_config(self.config)
        self.assertEqual(len(swap_rules.reference_tasks(self.config)), 5)
        self.assertEqual(len(swap_rules.selection_tasks(self.config)), 10)
        self.assertEqual(len(swap_rules.retrain_tasks(self.config)), 10)

    def test_only_rule_changes_within_each_pair(self):
        selections = swap_rules.selection_tasks(self.config)
        grouped = {}
        for task in selections:
            key = (task["dataset"], task["architecture"], task["fraction"], task["seed"])
            grouped.setdefault(key, {})[task["swap_rule"]] = task
        self.assertTrue(grouped)
        for pair in grouped.values():
            self.assertEqual(set(pair), {"ranked", "best"})
            ranked, best = pair["ranked"], pair["best"]
            self.assertEqual(ranked["reference_id"], best["reference_id"])
            self.assertEqual(ranked["score_objective"], best["score_objective"])
            self.assertEqual(ranked["selection"]["max_pairs"], 0)
            self.assertEqual(best["selection"]["max_pairs"], 0)
            ranked_settings = {**ranked["selection"], "rule": None}
            best_settings = {**best["selection"], "rule": None}
            self.assertEqual(ranked_settings, best_settings)

    def test_truncated_best_search_is_rejected(self):
        self.config["selection"]["max_pairs"] = 1000
        with self.assertRaisesRegex(ValueError, "full best-swap"):
            swap_rules.validate_config(self.config)

    def test_summary_emits_paired_diagnostics(self):
        selections = [task for task in swap_rules.selection_tasks(self.config)
                      if task["seed"] == 42]
        retrains = [task for task in swap_rules.retrain_tasks(self.config)
                    if task["seed"] == 42]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for offset, task in enumerate(selections):
                target = swap_rules.core.task_dir(root, task)
                target.mkdir(parents=True)
                swap_rules.core._write_json(target / "_SUCCESS.json", {"task_id": task["id"]})
                swap_rules.core._write_json(target / "objective.json", [
                    {"objective": 2.0, "D": 3.0},
                    {"objective": 1.0 - offset, "D": 2.0 - offset,
                     "certificate": 0.1 - 0.05 * offset},
                ])
                swap_rules.core._write_json(target / "timing.json", {
                    "phases_seconds": {"scoring": 4.0, "selection": 2.0 + offset},
                    "selection_seconds": 6.0 + offset,
                    "swaps": 5 + offset, "pair_searches": 6 + offset,
                    "certificate": 0.1 - 0.05 * offset, "certificate_valid": True,
                    "algorithm_2_exact": task["swap_rule"] == "best",
                    "reason": "synthetic",
                })
            for offset, task in enumerate(retrains):
                target = swap_rules.core.task_dir(root, task)
                target.mkdir(parents=True)
                swap_rules.core._write_json(target / "_SUCCESS.json", {"task_id": task["id"]})
                swap_rules.core._write_json(target / "metrics.json", {
                    "final_test_accuracy": 70.0 + offset,
                    "initial_state_sha256": "same-init",
                })
            _, _, pair_path = swap_rules.summarize(self.config, root)
            result = swap_rules.core._read_json(pair_path)
            self.assertEqual(result["n_complete_pairs"], 1)
            self.assertTrue(result["pairs"][0]["same_reference"])
            self.assertTrue(result["pairs"][0]["same_initialization"])
            self.assertTrue(result["pairs"][0]["best_algorithm_2_exact"])
            self.assertEqual(len(result["paired_aggregates"]), 1)
            self.assertTrue(result["paired_aggregates"][0]["all_best_searches_exact"])


if __name__ == "__main__":
    unittest.main()
