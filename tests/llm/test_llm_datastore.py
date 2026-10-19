"""Store round-trip, selection end-to-end, and diagnostic sanity tests."""

import json
import os
import tempfile
import unittest

import torch

from data_selection_swap.llm.config import SelectionConfig
from data_selection_swap.llm import datastore
from data_selection_swap.llm import scoring
from data_selection_swap.llm.swap import select


class StoreTest(unittest.TestCase):
    def test_save_load_roundtrip_preserves_bytes(self):
        store = datastore.make_synthetic_store(n=60, d=16, checkpoints=3, seed=2)
        with tempfile.TemporaryDirectory() as root:
            prov = store.save(root)
            loaded = datastore.FeatureStore.load(root)
        self.assertEqual(loaded.n, 60)
        self.assertEqual(loaded.weights, store.weights)
        self.assertEqual(loaded.provenance, prov)
        self.assertEqual(set(loaded.validation), {"tydiqa", "bbh"})
        for a, b in zip(loaded.train_features, store.train_features):
            self.assertTrue(torch.allclose(a, b.to(torch.float32).to(torch.float64)))

    def test_provenance_detects_tampering(self):
        store = datastore.make_synthetic_store(n=40, d=8, checkpoints=2, seed=5)
        with tempfile.TemporaryDirectory() as root:
            store.save(root)
            path = os.path.join(root, "features", "ckpt1.pt")
            torch.save(torch.randn(41, 8), path)          # corrupt: different bytes/size
            with self.assertRaises(ValueError):
                datastore.FeatureStore.load(root)

    def test_write_selected_jsonl(self):
        store = datastore.make_synthetic_store(n=50, d=8, checkpoints=2, seed=1)
        with tempfile.TemporaryDirectory() as root:
            out = os.path.join(root, "sub", "top_5.jsonl")
            n_written = datastore.write_selected_jsonl([3, 1, 4, 1, 5], store.pool, out)
            with open(out) as fh:
                lines = [json.loads(l) for l in fh]
        self.assertEqual(n_written, 5)
        self.assertEqual([l["id"] for l in lines], [3, 1, 4, 1, 5])


class SelectEndToEndTest(unittest.TestCase):
    def _store(self):
        return datastore.make_synthetic_store(n=120, d=24, checkpoints=4, seed=3)

    def test_targeted_selection_size_and_descend(self):
        store = self._store()
        feats = store.build_features()
        targets = store.subtask_targets("tydiqa")
        sel, hist, diag = select(feats, "targeted",
                                 SelectionConfig(setting="targeted", fraction=0.1,
                                                 rule="ranked", max_iters=100, seed=0),
                                 subtask_targets=targets)
        self.assertEqual(sel.numel(), 12)
        self.assertLessEqual(diag["value"], hist[0]["value"] + 1e-12)

    def test_faithful_selection_no_validation_needed(self):
        feats = self._store().build_features()
        sel, _, diag = select(feats, "faithful",
                              SelectionConfig(setting="faithful", fraction=0.1,
                                              rule="best", max_iters=50, seed=0))
        self.assertEqual(sel.numel(), 12)


class DiagnosticsTest(unittest.TestCase):
    def test_redundancy_and_overlap(self):
        store = datastore.make_synthetic_store(n=100, d=16, checkpoints=2, seed=4)
        feats = store.build_features()
        idx = torch.arange(20)
        red = scoring.mean_pairwise_cosine(feats, idx)
        self.assertGreaterEqual(red, -1.0001)
        self.assertLessEqual(red, 1.0001)
        self.assertAlmostEqual(scoring.jaccard(range(10), range(10)), 1.0)
        self.assertAlmostEqual(scoring.jaccard(range(10), range(10, 20)), 0.0)

    def test_spearman(self):
        self.assertAlmostEqual(scoring.spearman([1, 2, 3, 4], [1, 2, 3, 4]), 1.0)
        self.assertAlmostEqual(scoring.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)

    def test_length_stats(self):
        lengths = [1, 2, 3, 4, 5]
        stats = scoring.subset_length_stats(lengths, [0, 2, 4])
        self.assertAlmostEqual(stats["mean"], 3.0)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 5.0)


class IsolationTest(unittest.TestCase):
    def test_llm_package_pulls_in_no_cv_modules(self):
        import sys
        import data_selection_swap.llm as llm
        import importlib
        for name in llm.__all__:
            importlib.import_module(f"data_selection_swap.llm.{name}")
        # The LLM subpackage must not drag in torchvision or the CV experiment code.
        loaded = set(sys.modules)
        self.assertFalse(any(m == "torchvision" or m.startswith("torchvision.") for m in loaded))
        for forbidden in ("data_selection_swap.common", "data_selection_swap.objective",
                          "data_selection_swap.trajectory", "data_selection_swap.selection.kl"):
            self.assertNotIn(forbidden, loaded)


if __name__ == "__main__":
    unittest.main()
