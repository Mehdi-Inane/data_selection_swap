"""Dry-run tests for the LESS pipeline orchestrator (no execution)."""

import json
import os
import tempfile
import unittest

from data_selection_swap.llm import pipeline


def _config(**over):
    base = dict(vendor="/repo/vendor/LESS", repo_root="/repo",
                data_dir="/data/less", eval_data_dir="/data/eval",
                model="EleutherAI/pythia-410m", grads_dir="/work/GRADS",
                store_dir="/work/store", runs_dir="/work/runs",
                train_file_names=["a", "b"], tasks=["t1", "t2"],
                methods=["swap", "less", "random"], budgets=[0.05], seeds=[0],
                ckpts=[1, 2], checkpoint_weights=[0.5, 0.5], dim=8192, rule="ranked")
    base.update(over)
    return pipeline.PipelineConfig(**base)


class BuildStagesTest(unittest.TestCase):
    def setUp(self):
        self.cfg = _config()
        self.stages = pipeline.build_stages(self.cfg)
        self.by_name = {s.name: s for s in self.stages}

    def test_stage_counts(self):
        # 1 warmup + 2*2 train_grads + 2*2 val_grads + 1 store
        # + (2 tasks * 1 budget * 3 methods) select + same train + train*2 tasks eval
        self.assertEqual(len(self.stages), 1 + 4 + 4 + 1 + 6 + 6 + 12)

    def test_warmup_and_store_commands(self):
        w = self.by_name["warmup:seed0"]
        self.assertTrue(w.argv[1].endswith("warmup_lora_train.sh"))
        self.assertIn("/data/less", w.argv)
        self.assertIn("EleutherAI/pythia-410m", w.argv)
        self.assertEqual(w.cwd, "/repo/vendor/LESS")

        store = self.by_name["store:seed0"]
        self.assertIn("build_datastore.py", " ".join(store.argv))
        self.assertIn("--from-less", store.argv)
        self.assertEqual(store.cwd, "/repo")
        # store depends on every grad stage
        self.assertTrue(any("train_grads" in r for r in store.requires))
        self.assertTrue(any("val_grads" in r for r in store.requires))

    def test_selection_routes_to_the_right_script(self):
        swap = self.by_name["select:swap:t1:p0.05:seed0"]
        self.assertIn("select_swap.py", " ".join(swap.argv))
        self.assertIn("--setting", swap.argv)
        less = self.by_name["select:less:t1:p0.05:seed0"]
        self.assertIn("select_baselines.py", " ".join(less.argv))
        self.assertIn("--method", less.argv)
        self.assertIn("less", less.argv)

    def test_train_and_eval_chain(self):
        sel = "select:swap:t1:p0.05:seed0"
        train_names = [s.name for s in self.stages if sel in s.requires]
        self.assertTrue(any(n.startswith("train:") for n in train_names))
        train = self.by_name[next(n for n in train_names if n.startswith("train:"))]
        self.assertTrue(train.argv[1].endswith("lora_train.sh"))
        evals = [s for s in self.stages if train.name in s.requires]
        self.assertEqual(len(evals), 2)  # two eval tasks
        self.assertEqual(evals[0].env.get("DATA_DIR"), "/data/eval")
        self.assertIn("source eval_t1.sh", evals[0].argv[-1])

    def test_dry_run_executes_nothing(self):
        acted = pipeline.run_stages(self.stages, dry_run=True, echo=lambda *a, **k: None)
        self.assertEqual(len(acted), len(self.stages))

    def test_stage_filter(self):
        acted = pipeline.run_stages(self.stages, dry_run=True, only=["warmup"],
                                    echo=lambda *a, **k: None)
        self.assertTrue(all("warmup" in s.name for s in acted))
        self.assertEqual(len(acted), 1)

    def test_config_from_json(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cfg.json")
            with open(path, "w") as fh:
                json.dump({"vendor": "/v", "repo_root": "/r", "data_dir": "/d",
                           "eval_data_dir": "/e", "model": "m", "ckpts": [1],
                           "checkpoint_weights": [1.0]}, fh)
            cfg = pipeline.config_from_json(path)
        self.assertEqual(cfg.model, "m")
        self.assertEqual(cfg.ckpts, [1])
        self.assertEqual(cfg.tasks, ["mmlu", "tydiqa", "bbh"])  # defaults


if __name__ == "__main__":
    unittest.main()
