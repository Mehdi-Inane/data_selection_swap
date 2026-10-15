"""Transfer penalties and the source-by-target matrix layout."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_selection_swap.analysis import transfer_d
from data_selection_swap.analysis.comparison import RANDOM_METHOD

ARCHITECTURES = ('resnet18', 'vgg16', 'inception')


def row(source, target, d_rel, method='kl_faithful', k=10, seed=42, fraction=0.1):
    return dict(dataset='cifar10', fraction=fraction, budget=4500, method=method,
                k=k, seed=seed, source_architecture=source,
                target_architecture=target, d_alpha=d_rel, e_u_d=1.0, d_rel=d_rel,
                indices_path='x.pt', scoring_trajectory='t')


def square(matched=0.4, imported=0.55, seeds=(42,)):
    rows = []
    for seed in seeds:
        for source in ARCHITECTURES:
            for target in ARCHITECTURES:
                rows.append(row(source, target,
                                matched if source == target else imported, seed=seed))
    return rows


class TransferPenaltyTest(unittest.TestCase):
    def test_the_matched_diagonal_is_one_by_construction(self):
        rows = transfer_d.attach_transfer_penalty(square())
        diagonal = [r for r in rows
                    if r['source_architecture'] == r['target_architecture']]
        self.assertEqual(len(diagonal), len(ARCHITECTURES))
        for entry in diagonal:
            self.assertAlmostEqual(entry['transfer_penalty'], 1.0)

    def test_an_imported_subset_is_penalised_against_the_target_own_subset(self):
        rows = transfer_d.attach_transfer_penalty(square(matched=0.4, imported=0.55))
        off = [r for r in rows
               if r['source_architecture'] != r['target_architecture']]
        for entry in off:
            self.assertAlmostEqual(entry['transfer_penalty'], 0.55 / 0.4)
            self.assertAlmostEqual(entry['d_rel_matched'], 0.4)

    def test_an_interchangeable_subset_scores_one(self):
        rows = transfer_d.attach_transfer_penalty(square(matched=0.4, imported=0.4))
        self.assertTrue(all(np.isclose(r['transfer_penalty'], 1.0) for r in rows))

    def test_a_target_without_its_own_run_gets_no_penalty(self):
        rows = transfer_d.attach_transfer_penalty([row('resnet18', 'vgg16', 0.55)])
        self.assertTrue(np.isnan(rows[0]['transfer_penalty']))
        self.assertTrue(np.isnan(rows[0]['d_rel_matched']))

    def test_the_random_reference_is_not_mistaken_for_a_matched_run(self):
        # Its D_rel is 1.0 by definition; using it as the denominator would
        # silently turn every penalty into a plain D_rel.
        rows = square() + [dict(row('vgg16', 'vgg16', 1.0), method=RANDOM_METHOD, k='')]
        transfer_d.attach_transfer_penalty(rows)
        imported = next(r for r in rows if r['source_architecture'] == 'resnet18'
                        and r['target_architecture'] == 'vgg16')
        self.assertAlmostEqual(imported['d_rel_matched'], 0.4)

    def test_each_seed_is_penalised_against_its_own_diagonal(self):
        rows = square(seeds=(42, 43))
        for entry in rows:
            if entry['seed'] == 43:
                entry['d_rel'] *= 2.0          # a seed-wide offset, not a transfer cost
        transfer_d.attach_transfer_penalty(rows)
        penalties = {r['transfer_penalty'] for r in rows
                     if r['source_architecture'] != r['target_architecture']}
        self.assertEqual(len(penalties), 1)
        self.assertAlmostEqual(penalties.pop(), 0.55 / 0.4)


class MatrixTest(unittest.TestCase):
    def test_rows_are_the_selecting_architecture_and_columns_the_scoring_one(self):
        rows = transfer_d.attach_transfer_penalty(
            [row('resnet18', 'resnet18', 0.40), row('resnet18', 'vgg16', 0.60),
             row('vgg16', 'vgg16', 0.50), row('vgg16', 'resnet18', 0.70)])
        sources, targets, d_rel, penalty = transfer_d.transfer_matrix(
            rows, 0.1, 'kl_faithful')
        self.assertEqual(sources, ['resnet18', 'vgg16'])
        self.assertEqual(targets, ['resnet18', 'vgg16'])
        self.assertAlmostEqual(
            d_rel[sources.index('resnet18'), targets.index('vgg16')], 0.60)
        self.assertAlmostEqual(
            penalty[sources.index('resnet18'), targets.index('vgg16')], 0.60 / 0.50)

    def test_entries_average_over_seeds(self):
        rows = transfer_d.attach_transfer_penalty(
            [row('resnet18', 'vgg16', 0.40, seed=42),
             row('resnet18', 'vgg16', 0.60, seed=43)])
        _, _, d_rel, _ = transfer_d.transfer_matrix(rows, 0.1, 'kl_faithful')
        self.assertAlmostEqual(d_rel[0, 0], 0.50)

    def test_the_random_reference_has_no_matrix_of_its_own(self):
        rows = [dict(row('vgg16', 'vgg16', 1.0), method=RANDOM_METHOD, k='')]
        self.assertEqual(transfer_d.transfer_matrix(rows, 0.1, RANDOM_METHOD)[0], [])

    def test_another_fraction_is_a_separate_matrix(self):
        rows = transfer_d.attach_transfer_penalty(
            square() + [row('resnet18', 'vgg16', 0.9, fraction=0.2)])
        self.assertEqual(
            len(transfer_d.transfer_matrix(rows, 0.2, 'kl_faithful')[0]), 1)
        self.assertEqual(
            len(transfer_d.transfer_matrix(rows, 0.1, 'kl_faithful')[0]), 3)


class CheckpointDirTest(unittest.TestCase):
    def test_a_seed_template_is_filled_in(self):
        self.assertEqual(
            transfer_d.resolve_checkpoint_dir('/a/b/checkpoints/seed_{seed}', 42),
            '/a/b/checkpoints/seed_42')

    def test_a_seedless_launcher_path_descends_to_the_per_seed_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'seed_42').mkdir()
            self.assertEqual(transfer_d.resolve_checkpoint_dir(tmp, 42),
                             str(Path(tmp) / 'seed_42'))

    def test_a_directory_holding_only_the_manifest_is_used_as_is(self):
        from data_selection_swap import trajectory as ref

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ref.MANIFEST_NAME).write_text('{}', encoding='utf-8')
            self.assertEqual(transfer_d.resolve_checkpoint_dir(tmp, 42), tmp)

    def test_the_per_seed_child_beats_a_stale_parent_manifest(self):
        # A trajectory.json left directly in checkpoints/ by an older
        # (90%-split) run must not shadow the seed_<seed> trained on the full pool.
        from data_selection_swap import trajectory as ref

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ref.MANIFEST_NAME).write_text('{}', encoding='utf-8')
            (Path(tmp) / 'seed_42').mkdir()
            self.assertEqual(transfer_d.resolve_checkpoint_dir(tmp, 42),
                             str(Path(tmp) / 'seed_42'))


class WriteTest(unittest.TestCase):
    def test_every_declared_column_is_written(self):
        rows = transfer_d.attach_transfer_penalty(square())
        with tempfile.TemporaryDirectory() as tmp:
            path = transfer_d.write_csv(str(Path(tmp) / 'transfer.csv'), rows)
            header = Path(path).read_text(encoding='utf-8').splitlines()[0]
        self.assertEqual(header.split(','), transfer_d.FIELDS)


if __name__ == '__main__':
    unittest.main()
