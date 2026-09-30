"""Focused tests for D(alpha) scoring, subset pairing, and correlation statistics."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.analyze.analyze_d_correlation import (
    load_rows,
    stratified_summary,
    summarize_blocks,
)
from scripts.experiments.d_correlation_common import (
    SCHEMA_VERSION,
    analysis_block_payload,
    atomic_write_json,
    generate_subset_indices,
    kendall_tau_b,
    sha256_indices,
    sha256_json,
    sha256_ordered_indices,
    spearman,
    validate_record,
)
from scripts.experiments.run_d_accuracy_correlation import (
    aggregate_gradient_mean,
    d_checkpoint_weights,
    enforce_batchnorm_policy,
    train_or_load_reference,
    torch_load,
    validate_learning_algorithm_alignment,
)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset, TensorDataset
except ImportError:  # Pure record/statistics tests still run on a login node.
    torch = nn = DataLoader = Subset = TensorDataset = None


def make_record(number: int, d_value: float, accuracy: float) -> dict:
    indices = [number, number + 10]
    subset_hash = sha256_indices(indices)
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "record_id": f"record-{number}",
        "dataset": "mnist",
        "architecture": "lenet",
        "split_seed": 42,
        "pool_hash": "pool-hash",
        "pool_size": 20,
        "subset": {
            "id": f"candidate-{number}",
            "sampler": "uniform",
            "seed": 100 + number,
            "fraction": 0.1,
            "budget": 2,
            "pool_indices": indices,
            "indices_hash": subset_hash,
        },
        "reference": {"fingerprint": "reference-hash"},
        "d_alpha": {
            "value": d_value,
            "subset_indices_hash": subset_hash,
            "score_id": "score-hash",
            "definition": "test-definition",
            "weighting": "uniform",
            "normalization": "sum",
            "scoring_transform": "test/clean deterministic transform",
            "model_mode": "eval_bn_free",
        },
        "retraining": {
            "subset_indices_hash": subset_hash,
            "optimizer": "sgd",
            "lr": 0.01,
            "epochs": 10,
            "batch_size": 32,
            "steps_per_reference_epoch": 4,
            "total_optimizer_steps": 40,
            "weight_decay": 0.0005,
            "scheduler": "CosineAnnealingLR",
            "augmentation": "training transform",
            "requested_seeds": [1, 2],
            "init_hash": "init-hash",
            "replicates": [
                {"seed": 1, "test_accuracy_percent": accuracy - 0.25,
                 "optimizer_steps": 40, "test_evaluations": 1},
                {"seed": 2, "test_accuracy_percent": accuracy + 0.25,
                 "optimizer_steps": 40, "test_evaluations": 1},
            ],
        },
    }
    record["analysis_block_id"] = sha256_json(analysis_block_payload(record))
    return record


class ObjectiveTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "PyTorch is unavailable")
    def test_aggregate_gradient_mean_matches_single_full_batch_backward(self):
        torch.manual_seed(3)
        dataset = TensorDataset(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]]),
            torch.tensor([0, 1, 0, 1]),
        )
        model = nn.Linear(2, 2)
        measured = aggregate_gradient_mean(
            torch, nn, DataLoader, dataset, model, "cpu", batch_size=2, num_workers=0,
        )
        model.zero_grad(set_to_none=True)
        inputs, targets = dataset.tensors
        nn.CrossEntropyLoss(reduction="mean")(model(inputs), targets).backward()
        expected = torch.cat([parameter.grad.reshape(-1) for parameter in model.parameters()]).double()
        self.assertTrue(torch.allclose(measured, expected, atol=1e-7, rtol=1e-6))

    def test_learning_rate_weight_uses_true_outgoing_lr(self):
        manifest = {
            "checkpoints": [
                {"epoch": 1, "outgoing_lr": 0.2, "incoming_lr": 999.0},
                {"epoch": 2, "outgoing_lr": 0.05, "incoming_lr": 999.0},
            ]
        }
        self.assertEqual(
            d_checkpoint_weights(manifest, "learning_rate", "sum"),
            [0.2, 0.05],
        )
        self.assertTrue(np.allclose(
            d_checkpoint_weights(manifest, "learning_rate", "mean"),
            [0.8, 0.2],
        ))

    def test_reference_and_subset_learning_algorithms_must_match(self):
        args = SimpleNamespace(
            reference_optimizer="sgd", reference_lr=0.01, reference_epochs=10,
            retrain_optimizer="adam", retrain_lr=0.001, retrain_epochs=10,
        )
        with self.assertRaisesRegex(ValueError, r"same \(optimizer, LR, epochs\)"):
            validate_learning_algorithm_alignment(args, {"num_epochs": 10})

    @unittest.skipIf(torch is None, "PyTorch is unavailable")
    def test_batchnorm_requires_explicit_surrogate_policy(self):
        model = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 2))
        with self.assertRaisesRegex(RuntimeError, "depend on the other samples"):
            enforce_batchnorm_policy(nn, model, "reject")
        self.assertEqual(enforce_batchnorm_policy(nn, model, "eval_frozen"), ["0"])

    @unittest.skipIf(torch is None, "PyTorch is unavailable")
    def test_reference_checkpoints_are_pre_update_states_with_outgoing_lr(self):
        class FakeCommon:
            @staticmethod
            def set_seed(seed):
                import random
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

            @staticmethod
            def build_model(cfg, num_classes, device, architecture=None):
                return nn.Linear(2, num_classes).to(device)

            @staticmethod
            def build_optimizer(params, kind, lr, weight_decay=0.0):
                return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)

        inputs = torch.arange(40, dtype=torch.float32).reshape(20, 2) / 20.0
        targets = torch.tensor([0, 1] * 10)
        full_train = TensorDataset(inputs, targets)
        trainset = Subset(full_train, list(range(18)))
        cfg = {
            "full_train": full_train,
            "num_classes": 2,
            "num_epochs": 2,
            "input_channels": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                reference_dir=directory,
                stage="all",
                architecture="lenet",
                batchnorm_policy="reject",
                reference_epochs=2,
                reference_lr=0.1,
                reference_optimizer="sgd",
                num_d_checkpoints=2,
                checkpoint_epochs=[1, 2],
                reference_seed=7,
                split_seed=7,
                weight_decay=0.0,
                batch_size=6,
                num_workers=0,
                reference_resume_every=1,
                dataset="mnist",
            )
            pool = list(range(18))
            pool_hash = sha256_ordered_indices(pool)
            manifest = train_or_load_reference(
                torch, nn, DataLoader, FakeCommon, cfg, trainset, pool,
                pool_hash, args, "cpu",
                {"git_commit": "test", "source_sha256": {"test": "source"}},
            )
            self.assertEqual(manifest["status"], "complete")
            first, second = manifest["checkpoints"]
            self.assertEqual(first["completed_epochs"], 0)
            self.assertEqual(second["completed_epochs"], 1)
            self.assertEqual((first["global_step"], second["global_step"]), (0, 3))
            self.assertEqual(manifest["config"]["total_optimizer_steps"], 6)
            self.assertEqual(first["state_convention"], "pre_epoch_before_outgoing_updates")
            self.assertAlmostEqual(first["outgoing_lr"], 0.1)
            self.assertAlmostEqual(second["outgoing_lr"], 0.05)
            init = torch_load(
                torch, Path(directory) / manifest["initialization"]["file"],
                map_location="cpu",
            )
            checkpoint_one = torch_load(
                torch, Path(directory) / first["file"], map_location="cpu",
            )["model"]
            checkpoint_two = torch_load(
                torch, Path(directory) / second["file"], map_location="cpu",
            )["model"]
            self.assertTrue(all(torch.equal(init[key], checkpoint_one[key]) for key in init))
            self.assertTrue(any(not torch.equal(init[key], checkpoint_two[key]) for key in init))


class SubsetGenerationTests(unittest.TestCase):
    def test_hashes_distinguish_pool_order_but_not_subset_order(self):
        self.assertEqual(sha256_indices([4, 1, 3]), sha256_indices([3, 4, 1]))
        self.assertNotEqual(
            sha256_ordered_indices([4, 1, 3]),
            sha256_ordered_indices([3, 4, 1]),
        )

    def test_stratified_sampler_is_exact_and_reproducible(self):
        labels = [0] * 8 + [1] * 4 + [2] * 2
        first = generate_subset_indices(labels, 7, "stratified", 91)
        second = generate_subset_indices(labels, 7, "stratified", 91)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 7)
        selected_labels = np.asarray(labels)[first]
        self.assertEqual(np.bincount(selected_labels, minlength=3).tolist(), [4, 2, 1])

    def test_loss_conditioned_sampler_stays_in_declared_candidate_pool(self):
        labels = [0] * 10
        losses = list(range(10))
        easy = generate_subset_indices(
            labels, 3, "easy", 7, losses=losses, loss_pool_multiplier=2.0,
        )
        hard = generate_subset_indices(
            labels, 3, "hard", 7, losses=losses, loss_pool_multiplier=2.0,
        )
        self.assertTrue(set(easy).issubset(set(range(6))))
        self.assertTrue(set(hard).issubset(set(range(4, 10))))


class PairingAndStatisticsTests(unittest.TestCase):
    def test_record_rejects_cross_subset_pairing(self):
        record = make_record(0, 2.0, 80.0)
        validate_record(record)
        record["d_alpha"]["subset_indices_hash"] = "wrong-subset"
        with self.assertRaisesRegex(ValueError, "not paired"):
            validate_record(record)

    def test_rank_statistics_have_expected_direction_with_ties(self):
        d = [1.0, 2.0, 2.0, 4.0]
        accuracy = [90.0, 80.0, 80.0, 60.0]
        self.assertAlmostEqual(spearman(d, accuracy), -1.0)
        self.assertAlmostEqual(kendall_tau_b(d, accuracy), -1.0)

    def test_analyzer_collapses_replicates_and_bootstraps_paired_subsets(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for number in range(6):
                path = Path(directory) / f"candidate_{number}" / "record.json"
                atomic_write_json(path, make_record(number, number + 1.0, 90.0 - 2.0 * number))
                paths.append(path)
            rows, excluded = load_rows(paths, skip_invalid=False)
            self.assertEqual(excluded, [])
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(row["accuracy_replicates"] == 2 for row in rows))
            groups = {rows[0]["analysis_block_id"]: rows}
            summaries, undersized = summarize_blocks(
                groups, n_bootstrap=300, confidence=0.95, seed=9, min_block_size=4,
            )
            self.assertEqual(undersized, [])
            self.assertAlmostEqual(summaries[0]["spearman"]["estimate"], -1.0)
            pooled = stratified_summary(
                [rows], n_bootstrap=300, confidence=0.95, seed=10,
            )
            self.assertAlmostEqual(pooled["estimate"], -1.0)
            self.assertGreater(pooled["probability_negative"], 0.95)


if __name__ == "__main__":
    unittest.main()
