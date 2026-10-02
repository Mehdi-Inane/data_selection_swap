"""Panel grouping, point selection, and the correlation reported on each figure."""

import csv
import tempfile
import unittest
from pathlib import Path

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

    def test_no_usable_seed_reports_nan_and_a_readable_message(self):
        stats = scatter.correlation([dict(seed=42, d=1.0, acc=80.0)])
        self.assertEqual(stats['n_seeds'], 0)
        self.assertNotEqual(stats['within_mean'], stats['within_mean'])  # NaN
        self.assertIn('no correlation defined', scatter.correlation_text(stats))


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
                (out_dir / 'cifar10_resnet18_f0.1_d_vs_accuracy.png').is_file())
            self.assertTrue(
                (out_dir / 'cifar100_resnet18_f0.1_d_vs_accuracy.png').is_file())

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


class InputExpansionTest(unittest.TestCase):
    def test_directories_globs_and_files_all_resolve_to_csv_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = write_csv(Path(tmp) / 'd_exact_cifar10_resnet18_f0.1.csv', [row()])
            second = write_csv(Path(tmp) / 'd_exact_cifar10_vgg16_f0.1.csv', [row()])
            (Path(tmp) / 'notes.txt').write_text('ignored', encoding='utf-8')

            expected = sorted({first, second})
            self.assertEqual(scatter.expand_inputs([tmp]), expected)
            self.assertEqual(
                scatter.expand_inputs([str(Path(tmp) / 'd_exact_*.csv')]), expected)
            self.assertEqual(scatter.expand_inputs([first, second]), expected)
            with self.assertRaises(FileNotFoundError):
                scatter.expand_inputs([str(Path(tmp) / 'missing.csv')])


if __name__ == '__main__':
    unittest.main()
