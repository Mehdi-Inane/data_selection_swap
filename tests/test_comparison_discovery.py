"""Artifact-discovery guards for comparison aggregation."""

import json
import tempfile
import unittest
from pathlib import Path

from data_selection_swap.analysis import comparison


class ComparisonDiscoveryTest(unittest.TestCase):
    method = "kl_faithful_resnet18"
    budget = 100

    @staticmethod
    def _write_run(root, method, budget, k, run_id):
        prefix = root / f"{method}_{budget}_K{k}_cfg{run_id}"
        (root / f"{prefix.name}_metrics.json").write_text(
            json.dumps({"1": 50.0}), encoding="utf-8")
        (root / f"{prefix.name}_run.json").write_text(json.dumps({
            "run_id": run_id,
            "config": {"selector": "kl_factored", "arguments": {"max_iters": 10}},
        }), encoding="utf-8")

    def test_distinct_checkpoint_counts_are_separate_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_run(root, self.method, self.budget, 5, "aaaaaaaaaaaa")
            self._write_run(root, self.method, self.budget, 10, "bbbbbbbbbbbb")
            runs = comparison.discover_runs(str(root), self.method, self.budget)
            self.assertEqual(sorted(run[0] for run in runs), [5, 10])

    def test_two_configs_in_one_seed_cell_fail_with_run_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = "aaaaaaaaaaaa", "bbbbbbbbbbbb"
            self._write_run(root, self.method, self.budget, 10, first)
            self._write_run(root, self.method, self.budget, 10, second)

            with self.assertRaises(RuntimeError) as caught:
                comparison.discover_runs(str(root), self.method, self.budget)

            message = str(caught.exception)
            self.assertIn("configuration reruns", message)
            self.assertIn(first, message)
            self.assertIn(second, message)
            self.assertIn("K=10", message)


if __name__ == "__main__":
    unittest.main()
