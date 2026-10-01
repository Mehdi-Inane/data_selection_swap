"""Epoch-averaged test error and discovery of baseline plus KL artifacts."""

import json
import tempfile
import unittest
from pathlib import Path

from data_selection_swap.analysis import result_plots


class ResultPlotDiscoveryTest(unittest.TestCase):
    def test_mean_test_error_averages_logged_epochs(self):
        self.assertIsNone(result_plots.mean_test_error(None))
        self.assertIsNone(result_plots.mean_test_error({}))
        self.assertAlmostEqual(
            result_plots.mean_test_error({"50": 80.0, "100": 90.0}), 15.0)

    def test_baseline_and_kl_filenames_become_series(self):
        budget = 4500  # 0.1 * 45000, the CIFAR-10 pool
        with tempfile.TemporaryDirectory() as scratch:
            baseline = (Path(scratch) / "data_selection_baselines" / "random"
                        / "cifar10" / "resnet18" / str(budget) / "seed_42")
            baseline.mkdir(parents=True)
            (baseline / f"random_{budget}_metrics.json").write_text(
                json.dumps({"50": 80.0, "100": 90.0}), encoding="utf-8")
            (baseline / f"random_{budget}_timing.json").write_text(
                json.dumps({"selection_seconds": 12.5}), encoding="utf-8")

            kl_dir = (Path(scratch) / "gradmatch_swap" / "cifar10" / "resnet18"
                      / str(budget) / "seed_42")
            kl_dir.mkdir(parents=True)
            kl_name = (f"kl_faithful_full_resnet18_jl4096_{budget}"
                       f"_K300_cfgabc")
            (kl_dir / f"{kl_name}_metrics.json").write_text(
                json.dumps({"50": 70.0, "100": 60.0, "300": 50.0}),
                encoding="utf-8")
            (kl_dir / f"{kl_name}_timing.json").write_text(
                json.dumps({"selection_seconds": 90.0}), encoding="utf-8")

            rows = result_plots.collect_runs(
                scratch, "cifar10", "resnet18", [0.1], [42])
            by_method = {row["method"]: row for row in rows}
            self.assertEqual(set(by_method), {
                "random", "kl_faithful_full_resnet18_jl4096"})
            self.assertAlmostEqual(by_method["random"]["error"], 15.0)
            self.assertIsNone(by_method["random"]["k"])
            self.assertAlmostEqual(by_method["random"]["selection_s"], 12.5)
            self.assertAlmostEqual(
                by_method["kl_faithful_full_resnet18_jl4096"]["error"], 40.0)
            self.assertEqual(by_method["kl_faithful_full_resnet18_jl4096"]["k"], 300)

            labels = {item["label"] for item in result_plots.group_by_budget(rows, "error")}
            self.assertEqual(labels, {
                "random", "kl_faithful_full_resnet18_jl4096, K=300"})

    def test_d_cache_roundtrip_keeps_only_the_matching_scoring_id(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = str(Path(scratch) / "d_exact.csv")
            scoring = result_plots.scoring_id(
                "cifar10", "resnet18", "sgd", "auto", 0, "spread", "point",
                "test", None)
            rows = [dict(fraction=0.1, budget=4500, method="random", k=None,
                         seed=42, error=15.0, selection_s=12.5,
                         indices_path=str(Path(scratch) / "indices.pt"), d=1.25)]
            result_plots.write_table(path, rows, scoring, "cifar10", "resnet18")
            rows[0]["d"] = None
            result_plots.attach_cached_d(rows, path, scoring)
            self.assertAlmostEqual(rows[0]["d"], 1.25)
            result_plots.attach_cached_d(rows, path, scoring + "|other")
            self.assertIsNone(rows[0]["d"])


if __name__ == "__main__":
    unittest.main()
