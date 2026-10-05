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
            rows = [dict(fraction=0.1, budget=4500, method="kl_faithful", k=None,
                         seed=42, error=15.0, selection_s=12.5,
                         indices_path=str(Path(scratch) / "indices.pt"), d=1.25)]
            result_plots.write_table(path, rows, scoring, "cifar10", "resnet18")
            tex = Path(result_plots.latex_path_for(path)).read_text(encoding="utf-8")
            self.assertIn(r"\begin{tabular}", tex)
            self.assertIn(r"\textbf{85.00}", tex)
            self.assertIn("1.25", tex)
            self.assertIn(r"kl\_faithful", tex)
            self.assertNotIn("kl_faithful", tex)
            self.assertNotIn("Seed", tex)
            rows[0]["d"] = None
            result_plots.attach_cached_d(rows, path, scoring)
            self.assertAlmostEqual(rows[0]["d"], 1.25)
            result_plots.attach_cached_d(rows, path, scoring + "|other")
            self.assertIsNone(rows[0]["d"])

    def test_latex_averages_seeds_and_bolds_the_best_accuracy(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = str(Path(scratch) / "d_exact.csv")
            scoring = "cifar10|resnet18|unused"
            rows = [
                dict(fraction=0.1, budget=4500, method="random", k=None,
                     seed=42, error=20.0, selection_s=1.0, indices_path="a", d=None),
                dict(fraction=0.1, budget=4500, method="random", k=None,
                     seed=43, error=10.0, selection_s=3.0, indices_path="b", d=None),
                dict(fraction=0.1, budget=4500, method="kl_faithful", k=10,
                     seed=42, error=30.0, selection_s=5.0, indices_path="c", d=None),
                dict(fraction=0.1, budget=4500, method="kl_faithful", k=10,
                     seed=43, error=20.0, selection_s=7.0, indices_path="d", d=None),
            ]
            result_plots.write_table(path, rows, scoring, "cifar10", "resnet18")
            tex = Path(result_plots.latex_path_for(path)).read_text(encoding="utf-8")
            self.assertIn(r"\textbf{85.00}", tex)
            self.assertIn("75.00", tex)
            self.assertNotIn(r"\textbf{75.00}", tex)
            self.assertNotIn("& 42 &", tex)
            self.assertNotIn("& 43 &", tex)

            kl_path = result_plots.latex_path_for(path, "_kl")
            result_plots.write_latex(
                kl_path, [row for row in rows if result_plots.is_kl_method(row["method"])],
                "cifar10", "resnet18", kl_only=True)
            kl_tex = Path(kl_path).read_text(encoding="utf-8")
            self.assertIn(r"kl\_faithful", kl_tex)
            self.assertNotIn("random", kl_tex)
            self.assertIn(r"\textbf{75.00}", kl_tex)
            self.assertIn("KL selectors only", kl_tex)

    def test_accuracy_curves_follow_the_baseline_comparison_rules(self):
        rows = [
            dict(method="kl_faithful", k=1, fraction=0.1, error=20.0, seed=42),
            dict(method="kl_faithful", k=1, fraction=0.1, error=10.0, seed=43),
            dict(method="kl_faithful", k=10, fraction=0.1, error=40.0, seed=42),
            dict(method="kl_faithful", k=10, fraction=0.2, error=30.0, seed=42),
            dict(method="random", k=None, fraction=0.1, error=25.0, seed=42),
            dict(method="el2n", k=None, fraction=0.1, error=50.0, seed=42),
            dict(method="el2n", k=None, fraction=0.2, error=15.0, seed=42),
        ]
        compared = result_plots.comparison_accuracy_series(
            result_plots.baseline_comparison_rows(result_plots.with_accuracy(rows)))
        labels = [item["label"] for item in compared]
        self.assertEqual(labels, [
            result_plots.KL_DISPLAY_NAME, "random", "el2n"])
        ours = compared[0]["points"]
        self.assertEqual(len(ours), 1)
        self.assertEqual(ours[0][0], 0.1)
        self.assertAlmostEqual(ours[0][1], 85.0)
        self.assertAlmostEqual(ours[0][2], 7.0710678118654755)
        self.assertEqual(ours[0][3], 2)
        el2n = compared[-1]["points"]
        self.assertEqual([point[0] for point in el2n], [0.2])
        self.assertAlmostEqual(el2n[0][1], 85.0)

        k_series = result_plots.k_accuracy_series(result_plots.with_accuracy(rows))
        self.assertEqual([item["label"] for item in k_series], ["K=1", "K=10"])
        self.assertEqual([point[0] for point in k_series[1]["points"]], [0.1, 0.2])
        self.assertTrue(result_plots.emphasizes_k1("K=1"))
        self.assertTrue(result_plots.emphasizes_k1("kl_faithful, K=1"))
        self.assertFalse(result_plots.emphasizes_k1("K=10"))

    def test_checkpoint_ablation_csv_matches_the_baseline_table(self):
        from data_selection_swap.analysis import checkpoint_ablation

        budget = 4500
        with tempfile.TemporaryDirectory() as scratch:
            for k, acc in ((1, 80.0), (10, 70.0)):
                directory = (Path(scratch) / "gradmatch_swap" / "cifar10" / "resnet18"
                             / str(budget) / "seed_42")
                directory.mkdir(parents=True, exist_ok=True)
                name = f"kl_faithful_{budget}_K{k}"
                (directory / f"{name}_metrics.json").write_text(
                    json.dumps({"50": acc, "100": acc}), encoding="utf-8")
                (directory / f"{name}_timing.json").write_text(
                    json.dumps({"selection_seconds": 3.0}), encoding="utf-8")

            rows = checkpoint_ablation.collect_ablation_rows(
                scratch, "cifar10", "resnet18", [0.1], [42], [1, 10])
            self.assertEqual(sorted(row["k"] for row in rows), [1, 10])
            series = result_plots.k_accuracy_series(result_plots.with_accuracy(rows))
            by_k = {item["label"]: item["points"][0][1] for item in series}
            self.assertAlmostEqual(by_k["K=1"], 80.0)
            self.assertAlmostEqual(by_k["K=10"], 70.0)

            out = Path(scratch) / "d_exact_cifar10_resnet18.csv"
            result_plots.write_table(
                str(out), rows, "checkpoint_ablation|cifar10|resnet18",
                "cifar10", "resnet18")
            header = out.read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(header, ",".join(result_plots.CACHE_FIELDS))

            pdf = Path(scratch) / "resnet18_cifar10.pdf"
            result_plots.plot_accuracy_vs_fraction(series, str(pdf),
                                                   emphasize=result_plots.emphasizes_k1)
            self.assertGreater(pdf.stat().st_size, 0)

    def test_k_latex_table_reports_accuracy_by_architecture_and_k(self):
        from data_selection_swap.analysis import checkpoint_ablation

        budget = 4500
        with tempfile.TemporaryDirectory() as scratch:
            for seed, acc in ((42, 80.0), (43, 90.0)):
                directory = (Path(scratch) / "gradmatch_swap" / "cifar10" / "resnet18"
                             / str(budget) / f"seed_{seed}")
                directory.mkdir(parents=True)
                name = f"kl_faithful_full_resnet18_sgd_{budget}_K1"
                (directory / f"{name}_metrics.json").write_text(
                    json.dumps({"50": acc, "100": acc}), encoding="utf-8")
            path = Path(scratch) / "cifar10_k_accuracy.tex"
            checkpoint_ablation.write_k_latex(
                str(path), scratch, "cifar10", ["resnet18", "vgg16"],
                [0.1], [42, 43], [1, 4])
            text = path.read_text(encoding="utf-8")
            self.assertIn(r"\multicolumn{2}{c}{10\%}", text)
            self.assertIn(r"Mean $\pm$ std & Var.", text)
            self.assertIn(r"ResNet-18 & 1 & 85.00 $\pm$ 7.07 & 50.00 \\", text)
            self.assertIn(r"ResNet-18 & 4 & -- & -- \\", text)
            self.assertIn(r"VGG-16 & 1 & -- & -- \\", text)
            self.assertIn(r"\end{table}", text)

    def test_jl_ablation_keeps_exact_and_sketched_runs(self):
        from data_selection_swap.analysis import jl_ablation

        budget = 4500
        with tempfile.TemporaryDirectory() as scratch:
            for seed, exact, sketched in ((42, 80.0, 70.0), (43, 90.0, 70.0)):
                directory = (Path(scratch) / "gradmatch_swap" / "cifar10" / "resnet18"
                             / str(budget) / f"seed_{seed}")
                directory.mkdir(parents=True)
                exact_name = f"kl_faithful_full_resnet18_sgd_{budget}_K1"
                (directory / f"{exact_name}_metrics.json").write_text(
                    json.dumps({"50": exact, "100": exact}), encoding="utf-8")
                sketched_name = f"kl_faithful_full_resnet18_sgd_jl1024_{budget}_K1"
                (directory / f"{sketched_name}_metrics.json").write_text(
                    json.dumps({"50": sketched, "100": sketched}), encoding="utf-8")
                dropped = f"kl_faithful_full_resnet18_sgd_jl1024_{budget}_K10"
                (directory / f"{dropped}_metrics.json").write_text(
                    json.dumps({"50": 10.0, "100": 10.0}), encoding="utf-8")

            rows = jl_ablation.collect_ablation_rows(
                scratch, "cifar10", "resnet18", [0.1], [42, 43], [1, 4], [0, 1024])
            self.assertEqual(
                sorted((row["k"], row["proj_dim"], row["seed"]) for row in rows),
                [(1, 0, 42), (1, 0, 43), (1, 1024, 42), (1, 1024, 43)])
            series = jl_ablation.jl_accuracy_series(result_plots.with_accuracy(rows))
            self.assertEqual([item["label"] for item in series],
                             ["K=1, p=0", "K=1, p=1024"])
            by_label = {item["label"]: item["points"][0][1] for item in series}
            self.assertAlmostEqual(by_label["K=1, p=0"], 85.0)
            self.assertAlmostEqual(by_label["K=1, p=1024"], 70.0)
            self.assertTrue(jl_ablation.emphasizes_exact("K=1, p=0"))
            self.assertFalse(jl_ablation.emphasizes_exact("K=1, p=1024"))

            path = Path(scratch) / "cifar10_jl_accuracy.tex"
            jl_ablation.write_jl_latex(
                str(path), scratch, "cifar10", ["resnet18"],
                [0.1], [42, 43], [1, 4], [0, 1024])
            text = path.read_text(encoding="utf-8")
            self.assertIn(r"\multicolumn{2}{c}{10\%}", text)
            self.assertIn(r"Architecture & $K$ & $p$", text)
            self.assertIn(r"ResNet-18 & 1 & 0 & 85.00 $\pm$ 7.07 & 50.00 \\", text)
            self.assertIn(r"ResNet-18 & 1 & 1024 & 70.00 $\pm$ 0.00 & 0.00 \\", text)
            self.assertIn(r"ResNet-18 & 4 & 0 & -- & -- \\", text)
            self.assertIn(r"ResNet-18 & 4 & 1024 & -- & -- \\", text)


if __name__ == "__main__":
    unittest.main()
