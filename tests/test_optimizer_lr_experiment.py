"""Fast, dependency-free tests for the optimizer/LR experiment planner."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "experiments" / "optimizer_lr.py"
EXAMPLE = REPO_ROOT / "scripts" / "experiments" / "optimizer_lr.example.json"

SPEC = importlib.util.spec_from_file_location("optimizer_lr_experiment", SCRIPT)
optimizer_lr = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(optimizer_lr)


class OptimizerLrPlanTest(unittest.TestCase):
    def setUp(self):
        with EXAMPLE.open(encoding="utf-8") as handle:
            self.config = json.load(handle)

    def test_example_is_valid_and_has_expected_factorial_counts(self):
        optimizer_lr.validate_config(self.config)
        self.assertEqual(len(optimizer_lr.reference_tasks(self.config)), 20)
        self.assertEqual(len(optimizer_lr.selection_tasks(self.config)), 50)
        self.assertEqual(len(optimizer_lr.retrain_tasks(self.config)), 200)

    def test_ids_are_stable_unique_and_record_architecture(self):
        first = optimizer_lr.retrain_tasks(self.config)
        second = optimizer_lr.retrain_tasks(copy.deepcopy(self.config))
        self.assertEqual([task["id"] for task in first], [task["id"] for task in second])
        self.assertEqual(len({task["id"] for task in first}), len(first))
        self.assertTrue(all(task["architecture"] == "resnet18" for task in first))

    def test_score_flags_isolate_eta_weighting_and_adam_metric(self):
        self.assertEqual(optimizer_lr._score_flags("uniform"), ("uniform", "none"))
        self.assertEqual(optimizer_lr._score_flags("sgd"), ("sgd", "none"))
        self.assertEqual(optimizer_lr._score_flags("adam"), ("adam", "adam"))

    def test_adam_metric_on_sgd_reference_is_rejected(self):
        invalid = copy.deepcopy(self.config)
        invalid["selection"]["objectives_by_reference"]["sgd"].append("adam")
        with self.assertRaisesRegex(ValueError, "Adam D\\^2 metric"):
            optimizer_lr.validate_config(invalid)

    def test_completed_artifacts_are_immutable(self):
        task = optimizer_lr.reference_tasks(self.config)[0]
        with tempfile.TemporaryDirectory() as directory:
            final = Path(directory) / task["id"]
            final.mkdir()
            optimizer_lr._write_json(final / "_SUCCESS.json", {"task_id": task["id"]})
            self.assertTrue(optimizer_lr._completed(final, task))
            self.assertIsNone(optimizer_lr._begin(final, task))

    def test_reference_resume_flag_is_explicit(self):
        task = optimizer_lr.reference_tasks(self.config)[0]
        command = optimizer_lr.reference_command(
            task, Path("/data"), Path("/output"), download=False, resume=True
        )
        self.assertIn("--resume", command)


if __name__ == "__main__":
    unittest.main()
