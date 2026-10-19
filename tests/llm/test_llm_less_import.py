"""Ingest real LESS gradient-dump layouts and prove the score matches LESS.

Builds the on-disk files LESS ``collect_grads`` / ``matching.py`` produce
(``.../dim{D}/all_orig.pt``, ``.../dim{D}/grads-*.pt`` shards, and flattened
validation gradients), imports them via ``load_from_less``, and checks:

* pool concatenation order and per-checkpoint shapes,
* validation reshape into ``(n_subtasks, V, d)`` subtasks,
* the shard-merge fallback path,
* ``targeted_score`` equals a hand-written LESS ``matching.py`` computation
  (checkpoint-weighted, mean over val examples per subtask, max over subtasks),
* store save/load round-trips after import.
"""

import os
import tempfile
import unittest

import torch

from data_selection_swap.llm import datastore
from data_selection_swap.llm.targets import targeted_score


def _normalize_rows(x):
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


class LessImportTest(unittest.TestCase):
    def setUp(self):
        self.dim = 6
        self.ckpts = [105, 211]
        self.weights = [0.6, 0.4]
        self.train_files = ["dolly", "oasst1"]
        self.sizes = {"dolly": 5, "oasst1": 4}
        self.task = "tydiqa"
        self.ns = 3
        self.v = 2
        g = torch.Generator().manual_seed(123)
        # ground-truth tensors (pre-normalization, like LESS before all_orig)
        self.train = {
            c: {f: torch.randn(self.sizes[f], self.dim, generator=g, dtype=torch.float64)
                for f in self.train_files} for c in self.ckpts}
        self.val = {c: torch.randn(self.ns * self.v, self.dim, generator=g, dtype=torch.float64)
                    for c in self.ckpts}

    def _write_less_dir(self, root, shard_mode=False):
        for c in self.ckpts:
            for f in self.train_files:
                d = os.path.join(root, f"{f}-ckpt{c}-adam", f"dim{self.dim}")
                os.makedirs(d, exist_ok=True)
                block = self.train[c][f]
                if shard_mode:
                    # two grads shards that must be merged in index order
                    half = block.shape[0] // 2 or 1
                    torch.save(block[:half], os.path.join(d, f"grads-{half}.pt"))
                    torch.save(block[half:], os.path.join(d, f"grads-{block.shape[0]}.pt"))
                else:
                    torch.save(_normalize_rows(block), os.path.join(d, "all_orig.pt"))
            vd = os.path.join(root, f"{self.task}-ckpt{c}-sgd", f"dim{self.dim}")
            os.makedirs(vd, exist_ok=True)
            torch.save(_normalize_rows(self.val[c]), os.path.join(vd, "all_orig.pt"))

    def _load(self, root):
        return datastore.load_from_less(
            dim=self.dim, ckpts=self.ckpts, checkpoint_weights=self.weights,
            train_file_names=self.train_files,
            train_grad_template=os.path.join(root, "{name}-ckpt{ckpt}-adam", "dim{dim}"),
            tasks=[self.task],
            val_grad_template=os.path.join(root, "{task}-ckpt{ckpt}-sgd", "dim{dim}"),
            n_subtasks={self.task: self.ns})

    def test_shapes_and_concat_order(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_less_dir(root)
            store = self._load(root)
        n = sum(self.sizes.values())
        self.assertEqual(store.n, n)
        for k, c in enumerate(self.ckpts):
            expected = _normalize_rows(torch.cat([self.train[c][f] for f in self.train_files], 0))
            self.assertTrue(torch.allclose(store.train_features[k], expected, atol=1e-9))
        # weights renormalized to sum to 1 like LESS
        self.assertAlmostEqual(sum(store.weights), 1.0)

    def test_validation_reshaped_into_subtasks(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_less_dir(root)
            store = self._load(root)
        subs = store.subtask_targets(self.task)
        self.assertEqual(len(subs), self.ns)
        self.assertEqual(len(subs[0]), len(self.ckpts))
        self.assertEqual(tuple(subs[0][0].shape), (self.v, self.dim))

    def test_shard_merge_path(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_less_dir(root, shard_mode=True)
            store = self._load(root)
        c = self.ckpts[0]
        expected = _normalize_rows(torch.cat([self.train[c][f] for f in self.train_files], 0))
        self.assertTrue(torch.allclose(store.build_features().features[0], expected, atol=1e-9))

    def test_targeted_score_matches_less_matching(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_less_dir(root)
            store = self._load(root)
        feats = store.build_features()

        # reference: LESS matching.py (train/val already row-normalized in all_orig)
        w = [wv / sum(self.weights) for wv in self.weights]
        n = store.n
        per_subtask = torch.zeros(self.ns, n, dtype=torch.float64)
        for s in range(self.ns):
            score = torch.zeros(n, dtype=torch.float64)
            for k, c in enumerate(self.ckpts):
                train_norm = _normalize_rows(
                    torch.cat([self.train[c][f] for f in self.train_files], 0))
                val_all = _normalize_rows(self.val[c])
                val_sub = val_all.reshape(self.ns, self.v, self.dim)[s]      # [V, dim]
                inf = train_norm @ val_sub.t()                               # [n, V]
                score = score + w[k] * inf.mean(dim=1)
            per_subtask[s] = score
        reference = per_subtask.max(dim=0).values

        got = targeted_score(feats, store.subtask_targets(self.task), aggregation="max")
        self.assertTrue(torch.allclose(got, reference, atol=1e-8))

    def test_import_then_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_less_dir(root)
            store = self._load(root)
            out = os.path.join(root, "store")
            prov = store.save(out)
            loaded = datastore.FeatureStore.load(out)
        self.assertEqual(loaded.n, store.n)
        self.assertEqual(loaded.provenance, prov)
        self.assertEqual(set(loaded.validation), {self.task})
        self.assertEqual(len(loaded.subtask_targets(self.task)), self.ns)


if __name__ == "__main__":
    unittest.main()
