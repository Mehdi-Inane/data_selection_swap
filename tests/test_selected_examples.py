"""Selected-example grids: K=1 filtering, sampling, and missing-method failure."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from data_selection_swap.analysis.checkpoint_ablation import full_method
from data_selection_swap.analysis.comparison import N_TRAIN
from data_selection_swap.analysis.selected_examples import (
    BASELINES, N_EXAMPLES, OURS_K, collect_rows, plot_selected_examples,
)
from data_selection_swap.analysis.selection_overlap import N_TOTAL, pool_indices

DATASET = 'cifar10'
ARCHITECTURE = 'resnet18'
SEED = 42
FRACTION = 0.1
BUDGET = int(FRACTION * N_TRAIN[DATASET])


def write_indices(directory, name, values):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    torch.save(torch.as_tensor(list(values), dtype=torch.long), path)
    return path


def plant_runs(scratch, skip=(), duplicate_k1=False):
    """One saved subset per figure row. K=1 and K=4 use disjoint index ranges."""
    method = full_method(ARCHITECTURE, 'sgd')
    kl_dir = (Path(scratch) / 'gradmatch_swap' / DATASET / ARCHITECTURE /
              str(BUDGET) / f'seed_{SEED}')
    write_indices(kl_dir, f'{method}_{BUDGET}_K1_cfgaaaa_indices.pt', range(0, 50))
    write_indices(kl_dir, f'{method}_{BUDGET}_K4_cfgbbbb_indices.pt', range(1000, 1050))
    if duplicate_k1:
        write_indices(kl_dir, f'{method}_{BUDGET}_K1_cfgcccc_indices.pt', range(2000, 2050))
    for offset, name in enumerate(BASELINES, start=1):
        if name in skip:
            continue
        directory = (Path(scratch) / 'data_selection_baselines' / name / DATASET /
                     ARCHITECTURE / str(BUDGET) / f'seed_{SEED}')
        start = offset * 100
        write_indices(directory, f'{name}_{BUDGET}_indices.pt', range(start, start + 50))
    return method


def fake_reader(absolute):
    return [(np.full((4, 4, 3), int(index) % 256, dtype=np.uint8), f'c{int(index) % 10}')
            for index in absolute]


class SelectedExamplesTest(unittest.TestCase):
    def test_k1_is_kept_and_other_checkpoint_counts_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            method = plant_runs(tmp)
            rows = collect_rows(tmp, DATASET, ARCHITECTURE, FRACTION, SEED,
                                N_EXAMPLES, example_seed=0)
        ours = rows[0]
        self.assertEqual(ours['method'], method)
        self.assertEqual(ours['k'], OURS_K)
        self.assertTrue(np.all(ours['pool_index'] < 50))
        self.assertTrue(np.all(ours['pool_index'] < 1000))

    def test_each_row_draws_six_in_range_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            plant_runs(tmp)
            rows = collect_rows(tmp, DATASET, ARCHITECTURE, FRACTION, SEED,
                                N_EXAMPLES, example_seed=0)
            pool = pool_indices(DATASET, SEED)
            self.assertEqual(len(rows), 1 + len(BASELINES))
            for row in rows:
                chosen = row['pool_index']
                self.assertEqual(chosen.size, N_EXAMPLES)
                self.assertEqual(len(set(int(v) for v in chosen)), N_EXAMPLES)
                saved = torch.load(row['indices_path'], weights_only=False)
                saved = set(int(v) for v in torch.as_tensor(saved).reshape(-1).tolist())
                self.assertTrue(set(int(v) for v in chosen).issubset(saved))
                np.testing.assert_array_equal(row['absolute'], pool[chosen])
                self.assertTrue(np.all(row['absolute'] >= 0))
                self.assertTrue(np.all(row['absolute'] < N_TOTAL[DATASET]))

    def test_a_missing_method_fails_before_any_figure_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            plant_runs(tmp, skip=('facility_location',))
            out = Path(tmp) / 'plots'
            called = []

            def reader(absolute):
                called.append(absolute)
                return fake_reader(absolute)

            with self.assertRaises(FileNotFoundError) as caught:
                plot_selected_examples(
                    tmp, DATASET, ARCHITECTURE, str(out), fraction=FRACTION,
                    seed=SEED, variant='sgd', reader=reader)
            self.assertIn('facility_location', str(caught.exception))
            self.assertEqual(called, [])
            self.assertFalse(out.exists())

    def test_two_k1_configs_fail_before_any_figure_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            plant_runs(tmp, duplicate_k1=True)
            out = Path(tmp) / 'plots'
            called = []

            def reader(absolute):
                called.append(True)
                return fake_reader(absolute)

            with self.assertRaises(RuntimeError) as caught:
                plot_selected_examples(
                    tmp, DATASET, ARCHITECTURE, str(out), fraction=FRACTION,
                    seed=SEED, variant='sgd', reader=reader)
            self.assertIn('configuration reruns', str(caught.exception))
            self.assertEqual(called, [])
            self.assertFalse(out.exists())

    def test_a_complete_grid_writes_png_and_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            plant_runs(tmp)
            out = Path(tmp) / 'plots'
            written = plot_selected_examples(
                tmp, DATASET, ARCHITECTURE, str(out), fraction=FRACTION,
                seed=SEED, variant='sgd', reader=fake_reader)
            self.assertEqual(
                [Path(path).suffix for path in written], ['.png', '.pdf'])
            for path in written:
                self.assertTrue(Path(path).is_file())
                self.assertGreater(Path(path).stat().st_size, 0)
