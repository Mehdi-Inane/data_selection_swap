"""Agreement metrics, pool mapping, and indices-first discovery."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from data_selection_swap.analysis import comparison
from data_selection_swap.analysis import selection_overlap as overlap

DATASET = 'cifar10'
POOL = comparison.N_TRAIN[DATASET]


def subset(architecture='resnet18', method='kl_faithful', k=10, seed=42,
           fraction=0.1, indices=(0, 1, 2), budget=5000):
    return dict(dataset=DATASET, architecture=architecture, fraction=fraction,
                budget=budget, method=method, k=k, seed=seed,
                indices_path=f'{method}.pt',
                indices=np.asarray(sorted(indices), dtype=np.int64))


def write_indices(directory, name, values):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    torch.save(torch.as_tensor(list(values), dtype=torch.long), path)
    return path


class KappaTest(unittest.TestCase):
    def test_identical_subsets_score_one(self):
        self.assertAlmostEqual(overlap.kappa(100, 100, 100, 1000), 1.0)

    def test_chance_overlap_scores_zero(self):
        # Two independent 100-of-1000 draws share 10 examples on average.
        self.assertAlmostEqual(overlap.kappa(100, 100, 10, 1000), 0.0)

    def test_disjoint_subsets_score_below_chance(self):
        self.assertLess(overlap.kappa(100, 100, 0, 1000), 0.0)

    def test_equal_sizes_match_the_closed_form(self):
        n, m, o = 50000, 5000, 2000
        f = m / n
        self.assertAlmostEqual(overlap.kappa(m, m, o, n), (o / m - f) / (1 - f))

    def test_raw_overlap_is_not_comparable_across_budgets_but_kappa_is(self):
        # Half the pool: independent draws already share 50% of their examples.
        self.assertAlmostEqual(overlap.kappa(500, 500, 250, 1000), 0.0)
        self.assertAlmostEqual(overlap.kappa(100, 100, 10, 1000), 0.0)

    def test_unequal_sizes_use_the_reachable_ceiling(self):
        # The smaller set sits entirely inside the larger one: as equal as possible.
        self.assertAlmostEqual(overlap.kappa(50, 100, 50, 1000), 1.0)

    def test_jaccard_ranks_pairs_the_same_way_as_kappa(self):
        overlaps = [0, 25, 100, 400, 1000]
        kappas = [overlap.kappa(1000, 1000, o, 10000) for o in overlaps]
        jaccards = [overlap.jaccard(1000, 1000, o) for o in overlaps]
        self.assertEqual(np.argsort(kappas).tolist(), np.argsort(jaccards).tolist())

    def test_hypergeometric_sd_is_the_measurement_noise_scale(self):
        # ~20 examples at the production pool and budget, as the docstring claims.
        self.assertAlmostEqual(overlap.hypergeometric_sd(5000, 5000, 50000), 20.1,
                               delta=0.5)


class StratifiedTest(unittest.TestCase):
    labels = np.array([0] * 50 + [1] * 50, dtype=np.int64)

    def test_matching_class_histograms_still_leave_room_to_disagree(self):
        # Both take 10 from each class, but no example in common.
        a = np.concatenate([np.arange(10), np.arange(50, 60)])
        b = np.concatenate([np.arange(10, 20), np.arange(60, 70)])
        self.assertAlmostEqual(
            overlap.class_histogram_tv(a, b, self.labels, 2), 0.0)
        self.assertLess(
            overlap.stratified_kappa(a, b, 0, self.labels, 2), 0.0)

    def test_class_tv_sees_a_composition_difference_plain_kappa_hides(self):
        a = np.arange(20)                      # all of class 0
        b = np.arange(50, 70)                  # all of class 1
        self.assertAlmostEqual(
            overlap.class_histogram_tv(a, b, self.labels, 2), 1.0)

    def test_identical_subsets_score_one_within_classes(self):
        a = np.concatenate([np.arange(10), np.arange(50, 60)])
        self.assertAlmostEqual(
            overlap.stratified_kappa(a, a, a.size, self.labels, 2), 1.0)


class PoolMappingTest(unittest.TestCase):
    def test_pool_is_the_full_training_set_in_order(self):
        total = overlap.N_TOTAL[DATASET]
        np.testing.assert_array_equal(
            overlap.pool_indices(DATASET, 42), np.arange(total))

    def test_pool_size_agrees_with_the_comparison_table(self):
        for dataset in ('cifar10', 'cifar100', 'mnist'):
            self.assertEqual(overlap.pool_indices(dataset, 42).size,
                             comparison.N_TRAIN[dataset])

    def test_pool_does_not_depend_on_the_seed(self):
        np.testing.assert_array_equal(
            overlap.pool_indices(DATASET, 42), overlap.pool_indices(DATASET, 43))

    def test_unknown_dataset_is_refused(self):
        with self.assertRaises(KeyError):
            overlap.pool_indices('not_a_dataset', 42)


class DiscoveryTest(unittest.TestCase):
    budget = 5000

    def _baseline_dir(self, scratch, method, architecture='resnet18', seed=42):
        return (Path(scratch) / 'data_selection_baselines' / method / DATASET /
                architecture / str(self.budget) / f'seed_{seed}')

    def test_selection_only_runs_are_found_though_discover_runs_misses_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = self._baseline_dir(tmp, 'herding')
            write_indices(directory, f'herding_{self.budget}_indices.pt', range(5))

            # No *_metrics.json: analysis.comparison cannot see this run at all.
            self.assertEqual(
                comparison.discover_runs(str(directory), 'herding', self.budget), [])
            found = overlap.discover_subsets(str(directory), 'herding', self.budget)
            self.assertEqual([k for k, _ in found], [None])

    def test_checkpoint_tags_are_separate_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_indices(directory, f'kl_faithful_{self.budget}_K5_cfgaaaa_indices.pt',
                          range(5))
            write_indices(directory, f'kl_faithful_{self.budget}_K10_cfgbbbb_indices.pt',
                          range(5))
            found = overlap.discover_subsets(str(directory), 'kl_faithful', self.budget)
            self.assertEqual(sorted(k for k, _ in found), [5, 10])

    def test_two_configs_in_one_seed_cell_fail_with_run_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            first, second = 'aaaaaaaaaaaa', 'bbbbbbbbbbbb'
            for run_id in (first, second):
                write_indices(
                    directory,
                    f'kl_faithful_{self.budget}_K10_cfg{run_id}_indices.pt', range(5))

            with self.assertRaises(RuntimeError) as caught:
                overlap.discover_subsets(str(directory), 'kl_faithful', self.budget)

            message = str(caught.exception)
            self.assertIn('configuration reruns', message)
            self.assertIn(first, message)
            self.assertIn(second, message)
            self.assertIn('K=10', message)

    def test_a_method_name_is_not_a_prefix_of_another(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_indices(directory,
                          f'kl_faithful_sgd_{self.budget}_K10_cfgaaaa_indices.pt',
                          range(5))
            self.assertEqual(
                overlap.discover_subsets(str(directory), 'kl_faithful', self.budget), [])
            self.assertEqual(
                len(overlap.discover_subsets(str(directory), 'kl_faithful_sgd',
                                             self.budget)), 1)

    def test_duplicate_indices_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_indices(Path(tmp), f'random_{self.budget}_indices.pt',
                                 [1, 2, 2, 3])
            with self.assertRaises(ValueError) as caught:
                overlap.load_indices(str(path))
            self.assertIn('duplicate', str(caught.exception))

    def test_collect_subsets_loads_every_architecture(self):
        with tempfile.TemporaryDirectory() as tmp:
            for architecture in ('resnet18', 'vgg16'):
                write_indices(self._baseline_dir(tmp, 'random', architecture),
                              f'random_{self.budget}_indices.pt', range(10))
            records = overlap.collect_subsets(
                tmp, DATASET, ['resnet18', 'vgg16'], [0.1], [42], methods=['random'])
        self.assertEqual(sorted(r['architecture'] for r in records),
                         ['resnet18', 'vgg16'])
        self.assertEqual(records[0]['indices'].size, 10)


class ContrastTest(unittest.TestCase):
    def test_architecture_contrast_pairs_the_same_method_and_seed(self):
        records = [
            subset(architecture='resnet18', indices=range(100)),
            subset(architecture='vgg16', indices=range(50, 150)),
            subset(architecture='resnet18', seed=43, indices=range(100)),
        ]
        rows = overlap.architecture_contrast(records)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['overlap'], 50)
        self.assertEqual(rows[0]['contrast'], overlap.ARCHITECTURE_CONTRAST)

    def test_architecture_contrast_pairs_across_a_checkpoint_count_mismatch(self):
        # The KL runs tag the checkpoints actually available, which can differ
        # between two architectures' trajectories.
        records = [subset(architecture='resnet18', k=10, indices=range(100)),
                   subset(architecture='vgg16', k=8, indices=range(100))]
        rows = overlap.architecture_contrast(records)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['k_mismatch'], 1)

    def test_method_contrast_stays_within_one_architecture_and_seed(self):
        records = [
            subset(method='kl_faithful', indices=range(100)),
            subset(method='herding', k=None, indices=range(100)),
            subset(method='herding', k=None, architecture='vgg16', indices=range(100)),
        ]
        rows = overlap.method_contrast(records)
        self.assertEqual(len(rows), 1)
        self.assertEqual({rows[0]['method_a'], rows[0]['method_b']},
                         {'kl_faithful', 'herding'})

    def test_seed_contrast_conditions_on_the_shared_part_of_the_two_pools(self):
        pool_a = overlap.pool_indices(DATASET, 42)
        pool_b = overlap.pool_indices(DATASET, 43)
        shared = np.intersect1d(pool_a, pool_b)
        # The same absolute images, addressed through each seed's own ordering.
        chosen = shared[:200]
        records = [
            subset(seed=42, indices=np.flatnonzero(np.isin(pool_a, chosen))),
            subset(seed=43, indices=np.flatnonzero(np.isin(pool_b, chosen))),
        ]
        rows = overlap.seed_contrast(records)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['overlap'], 200)
        self.assertAlmostEqual(rows[0]['kappa'], 1.0)
        self.assertEqual(rows[0]['n_pool'], shared.size)
        self.assertLess(rows[0]['n_pool'], POOL)

    def test_seed_contrast_scores_disjoint_selections_below_chance(self):
        pool_a = overlap.pool_indices(DATASET, 42)
        pool_b = overlap.pool_indices(DATASET, 43)
        shared = np.intersect1d(pool_a, pool_b)
        records = [
            subset(seed=42, indices=np.flatnonzero(np.isin(pool_a, shared[:200]))),
            subset(seed=43, indices=np.flatnonzero(np.isin(pool_b, shared[200:400]))),
        ]
        rows = overlap.seed_contrast(records)
        self.assertEqual(rows[0]['overlap'], 0)
        self.assertLess(rows[0]['kappa'], 0.0)

    def test_seed_floor_is_read_off_the_seed_contrast(self):
        rows = [
            dict(contrast=overlap.SEED_CONTRAST, fraction=0.1, architecture_a='resnet18',
                 method_a='kl_faithful', kappa=0.4),
            dict(contrast=overlap.SEED_CONTRAST, fraction=0.1, architecture_a='resnet18',
                 method_a='kl_faithful', kappa=0.6),
            dict(contrast=overlap.ARCHITECTURE_CONTRAST, fraction=0.1,
                 architecture_a='resnet18', method_a='kl_faithful', kappa=0.9),
        ]
        floor, _ = overlap.seed_floors(rows)[(0.1, 'resnet18', 'kl_faithful')]
        self.assertAlmostEqual(floor, 0.5)


class FloorRatioTest(unittest.TestCase):
    def test_a_solid_floor_normalises_the_architecture_kappa(self):
        self.assertAlmostEqual(overlap.floor_ratio(0.3, (0.6, 0.01)), 0.5)

    def test_a_floor_within_noise_of_chance_yields_no_ratio(self):
        # 0.005 ± 0.003 is not distinguishable from chance; dividing by it
        # would report an agreement ratio in the tens.
        self.assertTrue(np.isnan(overlap.floor_ratio(0.12, (0.005, 0.003))))

    def test_a_negative_floor_yields_no_ratio(self):
        self.assertTrue(np.isnan(overlap.floor_ratio(0.12, (-0.001, 0.0005))))

    def test_a_missing_floor_yields_no_ratio(self):
        self.assertTrue(np.isnan(overlap.floor_ratio(0.12, None)))


class ConsensusTest(unittest.TestCase):
    def test_poisson_binomial_sums_to_one_and_matches_the_binomial(self):
        distribution = overlap.poisson_binomial([0.1, 0.1, 0.1])
        self.assertAlmostEqual(distribution.sum(), 1.0)
        np.testing.assert_allclose(
            distribution, [0.729, 0.243, 0.027, 0.001], atol=1e-9)

    def test_poisson_binomial_handles_unequal_budgets(self):
        distribution = overlap.poisson_binomial([0.2, 0.5])
        np.testing.assert_allclose(distribution, [0.4, 0.5, 0.1], atol=1e-12)

    def test_full_agreement_puts_every_example_at_the_top_multiplicity(self):
        records = [subset(architecture=name, indices=range(100))
                   for name in ('resnet18', 'vgg16', 'inception')]
        rows = {row['selected_by']: row for row in overlap.consensus_rows(records)}
        self.assertEqual(rows[3]['count'], 100)
        self.assertEqual(rows[1]['count'], 0)
        self.assertEqual(rows[3]['union_size'], 100)
        self.assertGreater(rows[3]['ratio'], 1.0)

    def test_disjoint_selections_grow_the_union(self):
        records = [subset(architecture='resnet18', indices=range(100)),
                   subset(architecture='vgg16', indices=range(100, 200))]
        rows = {row['selected_by']: row for row in overlap.consensus_rows(records)}
        self.assertEqual(rows[2]['count'], 0)
        self.assertEqual(rows[1]['union_size'], 200)

    def test_a_single_architecture_has_nothing_to_agree_with(self):
        self.assertEqual(overlap.consensus_rows([subset()]), [])


class ClosestMethodTest(unittest.TestCase):
    @staticmethod
    def _pair(method_b, kappa, seed=42):
        return dict(contrast=overlap.METHOD_CONTRAST, dataset=DATASET, fraction=0.1,
                    architecture_a='resnet18', architecture_b='resnet18',
                    method_a='kl_faithful', method_b=method_b, seed_a=seed,
                    seed_b=seed, kappa=kappa, overlap_frac=kappa, class_tv=0.01)

    def test_methods_are_ranked_by_agreement_with_the_kl_reference(self):
        rows = [self._pair('herding', 0.2), self._pair('moso', 0.6),
                self._pair('random', 0.0)]
        ranked = overlap.closest_methods(rows, floors={})
        self.assertEqual([row['method'] for row in ranked],
                         ['moso', 'herding', 'random'])

    def test_the_reference_is_normalised_by_its_own_seed_floor(self):
        rows = [self._pair('moso', 0.3)]
        floors = {(0.1, 'resnet18', 'kl_faithful'): (0.6, 0.02)}
        ranked = overlap.closest_methods(rows, floors)
        self.assertAlmostEqual(ranked[0]['seed_floor'], 0.6)
        self.assertAlmostEqual(ranked[0]['kappa_over_floor'], 0.5)

    def test_every_kl_variant_is_its_own_reference(self):
        rows = [self._pair('moso', 0.3)]
        rows.append(dict(rows[0], method_a='kl_faithful_sgd'))
        ranked = overlap.closest_methods(rows, floors={})
        self.assertEqual(sorted(row['reference'] for row in ranked),
                         ['kl_faithful', 'kl_faithful_sgd'])

    def test_two_kl_variants_rank_against_each_other(self):
        rows = [dict(self._pair('kl_faithful_sgd', 0.8))]
        ranked = overlap.closest_methods(rows, floors={})
        self.assertEqual(
            {(row['reference'], row['method']) for row in ranked},
            {('kl_faithful', 'kl_faithful_sgd'), ('kl_faithful_sgd', 'kl_faithful')})

    def test_set_agreement_and_objective_gap_are_reported_separately(self):
        rows = [self._pair('moso', 0.05)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'd_exact_cifar10_resnet18_f0.1.csv'
            with open(path, 'w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    'dataset', 'architecture', 'fraction', 'method', 'seed',
                    'acc', 'd_alpha', 'd_rel'])
                writer.writeheader()
                writer.writerow(dict(dataset=DATASET, architecture='resnet18',
                                     fraction=0.1, method='kl_faithful', seed=42,
                                     acc=80.0, d_alpha=1.0, d_rel=0.50))
                writer.writerow(dict(dataset=DATASET, architecture='resnet18',
                                     fraction=0.1, method='moso', seed=42,
                                     acc=79.0, d_alpha=1.0, d_rel=0.52))
            scored = overlap.read_scored_tables([str(path)])
        ranked = overlap.closest_methods(rows, floors={}, scored=scored)
        self.assertAlmostEqual(ranked[0]['kappa_mean'], 0.05)
        # Barely any shared examples, nearly the same faithfulness.
        self.assertAlmostEqual(ranked[0]['d_rel_gap'], 0.02, places=6)
        self.assertAlmostEqual(ranked[0]['acc_gap'], 1.0, places=6)

    def test_a_table_without_d_rel_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'd_exact_wrong.csv'
            with open(path, 'w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle)
                writer.writerow(['dataset', 'method', 'seed', 'error'])
                writer.writerow([DATASET, 'moso', 42, 20.0])
            with self.assertRaises(ValueError) as caught:
                overlap.read_scored_tables([str(path)])
            self.assertIn('d_rel', str(caught.exception))


class MatrixTest(unittest.TestCase):
    def test_the_kappa_matrix_is_symmetric_with_kl_first(self):
        rows = [
            dict(contrast=overlap.METHOD_CONTRAST, fraction=0.1,
                 architecture_a='resnet18', method_a='herding', k_a='',
                 method_b='kl_faithful', k_b='10', kappa=0.3),
            dict(contrast=overlap.METHOD_CONTRAST, fraction=0.1,
                 architecture_a='resnet18', method_a='herding', k_a='',
                 method_b='moso', k_b='', kappa=0.1),
        ]
        labels, matrix = overlap.kappa_matrix(rows, 0.1, 'resnet18')
        self.assertEqual(labels[0], 'kl_faithful_K10')
        np.testing.assert_allclose(matrix, matrix.T, equal_nan=True)
        np.testing.assert_allclose(np.diag(matrix), 1.0)
        self.assertAlmostEqual(
            matrix[labels.index('herding'), labels.index('kl_faithful_K10')], 0.3)


if __name__ == '__main__':
    unittest.main()
