"""Panel grouping, point selection, and the correlation reported on each figure."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_selection_swap.analysis import d_accuracy_scatter as scatter
from data_selection_swap.analysis import result_plots
from data_selection_swap.analysis.comparison import RANDOM_METHOD

FIELDS = ['dataset', 'architecture', 'fraction', 'budget', 'method',
          'num_checkpoints_tag', 'seed', 'acc', 'd_alpha', 'd_rel', 'e_u_d',
          'indices_path', 'scoring_trajectory']


def row(dataset='cifar10', architecture='resnet18', fraction=0.1, budget=4500,
        method='random', k='', seed=42, acc=80.0, d_alpha=1.0, e_u_d=2.0,
        trajectory='all 39 checkpoints, view=test'):
    return {
        'dataset': dataset, 'architecture': architecture, 'fraction': fraction,
        'budget': budget, 'method': method, 'num_checkpoints_tag': k, 'seed': seed,
        'acc': '' if acc is None else acc, 'd_alpha': d_alpha,
        'd_rel': d_alpha / e_u_d if isinstance(d_alpha, float) and e_u_d else '',
        'e_u_d': '' if e_u_d is None else e_u_d, 'indices_path': 'x.pt',
        'scoring_trajectory': f"{trajectory}, seed={seed}",
    }


def write_csv(path, rows):
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


class ReadAndGroupTest(unittest.TestCase):
    def test_one_panel_per_dataset_architecture_fraction(self):
        rows = [
            row(dataset='cifar10', architecture='resnet18', fraction=0.1),
            row(dataset='cifar10', architecture='resnet18', fraction=0.2),
            row(dataset='cifar10', architecture='vgg16', fraction=0.1),
            row(dataset='cifar100', architecture='resnet18', fraction=0.1),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(Path(tmp) / 'd_exact.csv', rows)
            panels = scatter.group_by_panel(scatter.read_rows([path]))
        self.assertEqual(list(panels), [
            ('cifar10', 'resnet18', 0.1),
            ('cifar10', 'resnet18', 0.2),
            ('cifar10', 'vgg16', 0.1),
            ('cifar100', 'resnet18', 0.1),
        ])

    def test_points_exclude_random_references_and_accuracyless_runs(self):
        rows = [
            row(method='random', seed=42),
            row(method='kl_faithful', seed=42),
            row(method='herding', seed=42, acc=None),       # --selection_only
            row(method=RANDOM_METHOD, seed=42, acc=None, d_alpha=2.5),
            row(method=RANDOM_METHOD, seed=42, acc=None, d_alpha=3.5),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(Path(tmp) / 'd_exact.csv', rows)
            parsed = scatter.read_rows([path])
        points, reference = scatter.split_points_and_reference(parsed)
        self.assertEqual([p['method'] for p in points], ['kl_faithful', 'random'])
        self.assertEqual(sorted(r['d'] for r in reference), [2.5, 3.5])

    def test_rows_without_a_finite_d_are_dropped(self):
        rows = [row(method='random', d_alpha=''), row(method='herding', d_alpha=1.5)]
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(Path(tmp) / 'd_exact.csv', rows)
            parsed = scatter.read_rows([path])
        self.assertEqual([r['method'] for r in parsed], ['herding'])

    def test_missing_identity_columns_fall_back_then_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'legacy.csv'
            legacy_fields = [f for f in FIELDS if f not in ('dataset', 'architecture')]
            with open(path, 'w', newline='') as fh:
                writer = csv.DictWriter(fh, fieldnames=legacy_fields)
                writer.writeheader()
                writer.writerow({k: v for k, v in row().items() if k in legacy_fields})

            parsed = scatter.read_rows([str(path)], 'cifar100', 'vgg16')
            self.assertEqual(parsed[0]['dataset'], 'cifar100')
            self.assertEqual(parsed[0]['architecture'], 'vgg16')
            with self.assertRaises(ValueError):
                scatter.read_rows([str(path)])

    def test_the_result_plots_cache_is_refused_not_silently_read(self):
        """Its 'error' column is epoch-averaged, not a final accuracy."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'd_exact_cifar10_resnet18.csv'
            with open(path, 'w', newline='') as fh:
                writer = csv.DictWriter(fh, fieldnames=result_plots.CACHE_FIELDS)
                writer.writeheader()
                writer.writerow({
                    'scoring_id': 'cifar10|resnet18', 'dataset': 'cifar10',
                    'architecture': 'resnet18', 'fraction': 0.1, 'budget': 4500,
                    'method': 'random', 'k': '', 'seed': 42, 'error': 20.0,
                    'selection_seconds': 1.0, 'indices_path': 'x.pt', 'd_alpha': 1.0,
                })
            with self.assertRaises(ValueError) as caught:
                scatter.read_rows([str(path)])
            self.assertIn('score_subsets.py', str(caught.exception))

    def test_trajectory_label_drops_the_per_seed_suffix(self):
        rows = [row(seed=42), row(seed=43)]
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(Path(tmp) / 'd_exact.csv', rows)
            parsed = scatter.read_rows([path])
        self.assertEqual(scatter.trajectory_label(parsed),
                         'all 39 checkpoints, view=test')


class CorrelationTest(unittest.TestCase):
    def test_perfectly_inverted_ranking_is_minus_one_within_each_seed(self):
        points = []
        for seed in (42, 43):
            offset = 0.0 if seed == 42 else 50.0   # a per-seed accuracy offset
            for i, d in enumerate([1.0, 2.0, 3.0, 4.0]):
                points.append(dict(seed=seed, d=d, acc=90.0 - 2.0 * i + offset))
        stats = scatter.correlation(points)
        self.assertEqual(stats['n_seeds'], 2)
        self.assertAlmostEqual(stats['within_mean'], -1.0)
        self.assertAlmostEqual(stats['within_std'], 0.0)
        # Centring inside each seed removes the offset, so the pooled value
        # agrees; without centring the offset alone would dominate.
        self.assertAlmostEqual(stats['pooled_centred'], -1.0)
        self.assertEqual(stats['n_points'], 8)

    def test_seeds_with_too_few_methods_are_skipped(self):
        points = [
            dict(seed=42, d=1.0, acc=80.0),
            dict(seed=42, d=2.0, acc=70.0),
            dict(seed=42, d=3.0, acc=60.0),
            dict(seed=43, d=1.0, acc=90.0),
            dict(seed=43, d=2.0, acc=50.0),
        ]
        stats = scatter.correlation(points)
        self.assertEqual(stats['n_seeds'], 1)
        self.assertEqual(stats['n_points'], 3)
        self.assertAlmostEqual(stats['within_mean'], -1.0)

    def test_a_seed_with_constant_d_or_accuracy_is_skipped(self):
        constant_accuracy = [dict(seed=42, d=float(i), acc=80.0) for i in range(4)]
        self.assertEqual(scatter.correlation(constant_accuracy)['n_seeds'], 0)

        constant_d = [dict(seed=42, d=1.0, acc=80.0 - i) for i in range(4)]
        self.assertEqual(scatter.correlation(constant_d)['n_seeds'], 0)

        mixed = constant_d + [dict(seed=43, d=float(i), acc=80.0 - i) for i in range(4)]
        stats = scatter.correlation(mixed)
        self.assertEqual(stats['n_seeds'], 1)
        self.assertEqual(stats['n_points'], 4)   # only seed 43 contributed
        self.assertAlmostEqual(stats['within_mean'], -1.0)

    def test_no_usable_seed_reports_nan(self):
        stats = scatter.correlation([dict(seed=42, d=1.0, acc=80.0)])
        self.assertEqual(stats['n_seeds'], 0)
        self.assertNotEqual(stats['within_mean'], stats['within_mean'])  # NaN

    def test_correlation_uses_the_accuracy_gap_when_present(self):
        # Lower D, smaller gap: Spearman of D against y is +1.
        points = [dict(seed=42, d=1.0 + i, acc=80.0 - i, y=float(i))
                  for i in range(4)]
        stats = scatter.correlation(points)
        self.assertEqual(stats['n_seeds'], 1)
        self.assertAlmostEqual(stats['within_mean'], 1.0)


class PlotTest(unittest.TestCase):
    def test_one_figure_per_panel_named_for_its_configuration(self):
        rows = []
        for seed in (42, 43, 44):
            for i, method in enumerate(['random', 'herding', 'kl_faithful']):
                rows.append(row(method=method, seed=seed,
                                d_alpha=1.0 + i, acc=80.0 - i))
            rows.append(row(method=RANDOM_METHOD, seed=seed, acc=None, d_alpha=4.0))
        for i, method in enumerate(['random', 'herding', 'kl_faithful']):
            rows.append(row(dataset='cifar100', method=method, seed=42,
                            d_alpha=1.0 + i, acc=70.0 - i))

        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(Path(tmp) / 'd_exact.csv', rows)
            out_dir = Path(tmp) / 'plots'
            results = scatter.plot_all(scatter.read_rows([path]), str(out_dir))

            self.assertEqual(set(results), {
                ('cifar10', 'resnet18', 0.1), ('cifar100', 'resnet18', 0.1)})
            self.assertTrue(
                (out_dir / 'cifar10_resnet18_f0.1_sqrtd_vs_accuracy.pdf').is_file())
            self.assertTrue(
                (out_dir / 'cifar100_resnet18_f0.1_sqrtd_vs_accuracy.pdf').is_file())

        cifar10 = results[('cifar10', 'resnet18', 0.1)]
        self.assertEqual(cifar10['n_seeds'], 3)
        self.assertEqual(cifar10['n_points'], 9)
        self.assertAlmostEqual(cifar10['within_mean'], -1.0)

    def test_panel_without_any_accuracy_draws_nothing(self):
        rows = [row(method='random', acc=None),
                row(method=RANDOM_METHOD, acc=None, d_alpha=2.0)]
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(Path(tmp) / 'd_exact.csv', rows)
            out_dir = Path(tmp) / 'plots'
            results = scatter.plot_all(scatter.read_rows([path]), str(out_dir))
        self.assertEqual(results, {})


class FullAccuracyGapTest(unittest.TestCase):
    def test_as_percent_converts_a_fraction_and_leaves_percent_alone(self):
        self.assertAlmostEqual(scatter.as_percent(0.7549), 75.49)
        self.assertAlmostEqual(scatter.as_percent(75.49), 75.49)
        self.assertAlmostEqual(scatter.as_percent(0.0), 0.0)
        self.assertAlmostEqual(scatter.as_percent(1.0), 100.0)

    def test_load_full_accuracy_reads_final_test_acc_as_percent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'metrics.json'
            path.write_text('{"final_test_acc": 0.7549, "val_acc": [0.5, 0.6]}',
                            encoding='utf-8')
            self.assertAlmostEqual(scatter.load_full_accuracy(path), 75.49)

    def test_load_full_accuracy_falls_back_to_last_val_acc(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'metrics.json'
            path.write_text('{"val_acc": [0.50, 0.72]}', encoding='utf-8')
            self.assertAlmostEqual(scatter.load_full_accuracy(path), 72.0)

    def test_attach_accuracy_gap_uses_the_shared_checkpoints_metrics(self):
        rows = [
            row(method='random', acc=70.0, d_alpha=2.0),
            row(method='kl_faithful', acc=74.0, d_alpha=1.0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            metrics = (Path(tmp) / 'data_selection_swap' / 'cifar10' / 'resnet18'
                       / 'checkpoints' / 'metrics.json')
            metrics.parent.mkdir(parents=True)
            metrics.write_text('{"final_test_acc": 0.7549}', encoding='utf-8')
            parsed = scatter.read_rows([write_csv(Path(tmp) / 'd_exact.csv', rows)])
            scatter.attach_accuracy_gap(parsed, tmp)
        self.assertAlmostEqual(parsed[0]['full_acc'], 75.49)
        self.assertAlmostEqual(parsed[0]['y'], abs(75.49 - 70.0))
        self.assertAlmostEqual(parsed[1]['y'], abs(75.49 - 74.0))
        self.assertEqual(parsed[0]['full_metrics_path'], str(metrics))

    def test_per_seed_metrics_override_the_shared_file(self):
        rows = [row(seed=42, acc=70.0), row(seed=43, acc=70.0)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'data_selection_swap' / 'cifar10' / 'resnet18' / 'checkpoints'
            (root).mkdir(parents=True)
            (root / 'metrics.json').write_text('{"final_test_acc": 0.80}', encoding='utf-8')
            seed_dir = root / 'seed_42'
            seed_dir.mkdir()
            (seed_dir / 'metrics.json').write_text(
                '{"final_test_acc": 0.90}', encoding='utf-8')
            parsed = scatter.read_rows([write_csv(Path(tmp) / 'd_exact.csv', rows)])
            scatter.attach_accuracy_gap(parsed, tmp)
        by_seed = {str(r['seed']): r for r in parsed}
        self.assertAlmostEqual(by_seed['42']['full_acc'], 90.0)
        self.assertAlmostEqual(by_seed['43']['full_acc'], 80.0)

    def test_missing_metrics_file_fails_closed(self):
        rows = [row(acc=70.0)]
        with tempfile.TemporaryDirectory() as tmp:
            parsed = scatter.read_rows([write_csv(Path(tmp) / 'd_exact.csv', rows)])
            with self.assertRaises(FileNotFoundError):
                scatter.attach_accuracy_gap(parsed, tmp)

    def test_gap_figure_uses_the_gap_filename(self):
        rows = []
        for seed in (42, 43, 44):
            for i, method in enumerate(['random', 'herding', 'kl_faithful']):
                rows.append(row(method=method, seed=seed,
                                d_alpha=1.0 + i, acc=74.0 - i))
        with tempfile.TemporaryDirectory() as tmp:
            metrics = (Path(tmp) / 'data_selection_swap' / 'cifar10' / 'resnet18'
                       / 'checkpoints' / 'metrics.json')
            metrics.parent.mkdir(parents=True)
            metrics.write_text('{"final_test_acc": 0.80}', encoding='utf-8')
            parsed = scatter.read_rows([write_csv(Path(tmp) / 'd_exact.csv', rows)])
            scatter.attach_accuracy_gap(parsed, tmp)
            out_dir = Path(tmp) / 'plots'
            results = scatter.plot_all(parsed, str(out_dir))
            self.assertTrue(
                (out_dir / 'cifar10_resnet18_f0.1_sqrtd_vs_accuracy_gap.pdf').is_file())
        # Lower D, higher subset acc, smaller gap -> Spearman +1.
        self.assertAlmostEqual(results[('cifar10', 'resnet18', 0.1)]['within_mean'], 1.0)


class InputExpansionTest(unittest.TestCase):
    def test_directories_globs_and_files_all_resolve_to_csv_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = write_csv(Path(tmp) / 'd_exact_cifar10_resnet18_f0.1.csv', [row()])
            second = write_csv(Path(tmp) / 'd_exact_cifar10_vgg16_f0.1.csv', [row()])
            (Path(tmp) / 'notes.txt').write_text('ignored', encoding='utf-8')
            (Path(tmp) / 'comparison_cifar100.csv').write_text(
                'method,acc\nrandom,80\n', encoding='utf-8')

            expected = sorted({first, second})
            self.assertEqual(scatter.expand_inputs([tmp]), expected)
            self.assertEqual(
                scatter.expand_inputs([str(Path(tmp) / 'd_exact_*.csv')]), expected)
            self.assertEqual(scatter.expand_inputs([first, second]), expected)
            with self.assertRaises(FileNotFoundError):
                scatter.expand_inputs([str(Path(tmp) / 'missing.csv')])


class EstimateAndFitTest(unittest.TestCase):
    def test_axis_labels_mark_the_risk_and_objective_as_estimates(self):
        self.assertIn(r'\widehat{G}_T', scatter.X_LABEL)
        self.assertEqual(scatter.Y_LABEL.count(r'\widehat{R}_P'), 2)

    def test_line_of_best_fit_recovers_a_line(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        slope, intercept = scatter.line_of_best_fit(x, 0.5 * x + 0.1)
        self.assertAlmostEqual(slope, 0.5)
        self.assertAlmostEqual(intercept, 0.1)
        self.assertIsNone(scatter.line_of_best_fit(np.ones(4), x))

    def test_excluding_el2n_restores_a_perfect_pearson(self):
        points = [dict(method='herding', seed=42, x=float(i), y=float(i))
                  for i in range(1, 5)]
        points.append(dict(method='el2n', seed=42, x=1.0, y=10.0))
        self.assertAlmostEqual(scatter.correlation(scatter.without_el2n(points))['pearson'], 1.0)
        self.assertLess(scatter.correlation(points)['pearson'], 0.5)
        self.assertNotIn('spearman', scatter.correlation(points))


if __name__ == '__main__':
    unittest.main()
