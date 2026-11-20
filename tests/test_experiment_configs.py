"""configs/*.yaml: every config resolves, and the paper configs run exactly
what the Slurm launchers run."""

import contextlib
import importlib.util
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_config", REPO / "scripts" / "experiments" / "run_config.py")
run_config = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_config)

CONFIGS = sorted((REPO / "configs").glob("*/*.yaml"))
HAS_FULL_ENVIRONMENT = all(importlib.util.find_spec(name) for name in ("torch", "torchvision"))


def _modern_bash():
    """A bash >= 4.4: older ones (macOS's 3.2) treat an empty "${array[@]}"
    under `set -u` as unbound, which stops the launchers before srun."""
    candidates = [os.environ.get("BASH_FOR_TESTS"), shutil.which("bash"),
                  "/opt/homebrew/bin/bash", "/usr/local/bin/bash"]
    for candidate in filter(None, candidates):
        try:
            version = subprocess.run(
                [candidate, "-c", 'echo "${BASH_VERSINFO[0]} ${BASH_VERSINFO[1]}"'],
                capture_output=True, text=True, check=True).stdout.split()
        except (OSError, subprocess.CalledProcessError):
            continue
        if tuple(map(int, version)) >= (4, 4):
            return candidate
    return None


BASH = _modern_bash()


def config(name):
    return run_config.load(REPO / "configs" / f"{name}.yaml")


def quiet_main(argv):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return run_config.main(argv)


class ConfigSchemaTest(unittest.TestCase):
    def test_every_config_resolves_and_names_are_unique(self):
        names = [run_config.load(path)["name"] for path in CONFIGS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(sorted(path.stem for path in CONFIGS), sorted(names))

    def test_stage_sizes(self):
        expected = {
            "paper/reference_trajectories": {"reference": 6},
            "paper/main_comparison": {"gradient_swap": 120, "baselines": 840},
            "paper/checkpoint_ablation": {"gradient_swap": 600},
            "paper/jl_ablation": {"gradient_swap": 120},
            "paper/warm_start": {"gradient_swap": 20, "score": 1},
            "paper/faithfulness_correlation": {"score": 6},
            "revision/exp_a_low_budget": {"cached_sketch": 360, "exact_algorithm1": 20,
                                          "exact_second_order": 80, "baselines": 420},
            "revision/cifar100_second_order": {"reference": 2, "single_swap": 8,
                                               "second_order": 16, "second_order_class_balanced": 8,
                                               "baselines": 16},
            "revision/cifar10_lt_second_order": {"reference": 2, "single_swap": 8,
                                                 "second_order": 16, "second_order_class_balanced": 8,
                                                 "baselines": 16},
            "revision/cifar100_finetune": {"reference": 1, "single_swap": 4,
                                           "second_order": 4,
                                           "second_order_class_balanced": 4, "baselines": 16},
            "revision/cifar100_linear_probe": {"probe": 15},
            "revision/cifar100_linear_probe_w0_w1": {"probe": 15},
        }
        for name, stages in expected.items():
            loaded = config(name)
            self.assertEqual({s: run_config.task_count(loaded, s) for s in loaded["stages"]},
                             stages, name)

    def test_flags(self):
        self.assertEqual(
            run_config.to_flags({"a": 1, "b": True, "c": False, "d": None, "e": [1, 2],
                                 "f": 1.0e-12, "g": "x"}),
            ["--a", "1", "--b", "--e", "1", "2", "--f", "1e-12", "--g", "x"])

    def test_invalid_configs_are_rejected(self):
        stage = {"script": "scripts/select/kl_full.py", "grid": {"seed": [1, 2]}}
        base = {"schema_version": 1, "name": "n", "description": "d", "stages": {"s": stage}}
        run_config.validate(base)
        broken = {
            "axis also an argument": {"args": {"seed": 3}},
            "unknown placeholder": {"args": {"x": "{nope}"}},
            "empty axis": {"grid": {"seed": []}},
            "missing choice": {"grid": {"m": ["a", "b"]}, "choices": {"m": {"a": {}}}},
            "missing script": {"script": None},
        }
        for label, change in broken.items():
            with self.subTest(label), self.assertRaises(run_config.ConfigError):
                run_config.validate({**base, "stages": {"s": {**stage, **change}}})
        with self.assertRaises(run_config.ConfigError):
            run_config.validate({**base, "status": "planned"})   # says nothing it requires

    def test_constants_are_filled_only_where_used(self):
        loaded = config("paper/warm_start")
        _, args = run_config.resolve(loaded, "score", 0, {"results": "R", "data_root": "D"})
        self.assertNotIn("init_ckpt", args)   # init_ckpt names {seed}; score has no seed

    def test_planned_configs_cannot_run(self):
        for path in CONFIGS:
            loaded = run_config.load(path)
            stage = next(iter(loaded["stages"]))
            code = quiet_main(["run-task", str(path), stage, "0", "--dry-run"])
            self.assertEqual(code, 2 if loaded.get("status") == "planned" else 0, path.name)

    def test_exp_a_variants(self):
        loaded = config("revision/exp_a_low_budget")
        runtime = {"results": "R", "data_root": "D"}
        _, alg1 = run_config.resolve(loaded, "cached_sketch", 0, runtime)
        _, exact = run_config.resolve(loaded, "exact_second_order", 0, runtime)
        _, balanced = run_config.resolve(loaded, "exact_second_order",
                                         run_config.task_count(loaded, "exact_second_order") - 1,
                                         runtime)
        self.assertEqual((alg1["swap_rule"], alg1["sketch_cache"], alg1["proj_dim"],
                          alg1["max_iters"]), ("ranked", True, 1024, 200))
        self.assertEqual((exact["swap_rule"], exact["proj_dim"], exact["max_iters"]),
                         ("second_order", 0, 1500))
        self.assertTrue(balanced["class_balanced"])
        self.assertNotIn("variant", alg1)

    @unittest.skipUnless(HAS_FULL_ENVIRONMENT, "needs the scripts' dependencies for --help")
    def test_every_argument_is_a_flag_of_its_script(self):
        for path in CONFIGS:
            loaded = run_config.load(path)
            if loaded.get("status", "ready") == "ready":
                self.assertEqual(run_config.check_flags(loaded), {}, path.name)


def _parse(argv, root):
    """(script relative to root, {flag: values}) of a python command line."""
    flags, key = {}, None
    for token in argv[2:]:
        if token.startswith("--"):
            key = token[2:]
            flags[key] = ()
        else:
            flags[key] += (token,)
    return os.path.relpath(argv[1], root), flags


# Script defaults the configs spell out and the launchers leave implicit.
EXPLICIT_DEFAULTS = {
    "lr": ("0.01",), "num_epochs": ("300",), "retrain_optimizer": ("sgd",),
    "swaps_per_sweep": ("1",), "candidates": ("1",), "refresh_every": ("0",),
    "weight_decay": ("0.0005",), "num_trials": ("8",), "surrogate_epochs": ("50",),
    "samples": ("10",),
}


@unittest.skipUnless(BASH, "needs bash >= 4.4 (set BASH_FOR_TESTS to point at one)")
class LauncherParityTest(unittest.TestCase):
    """
    Each paper config, against the launcher it replaces, run in a sandbox
    with stub srun/module/conda. Two differences are deliberate: every seed
    reads the seed-42 trajectory (Appendix G.1; the launchers default to
    seed-matched ones), so Gradient Swap passes --allow_manifest_mismatch.
    """

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.project = root / "project"
        (cls.project / "scripts" / "slurm").mkdir(parents=True)
        for helper in ("stage_dataset.sh", "result_layout.sh", "cluster_env.sh"):
            shutil.copy(REPO / "scripts" / "slurm" / helper,
                        cls.project / "scripts" / "slurm" / helper)
        cls.bin = root / "bin"
        cls.bin.mkdir()
        for name, body in (("module", "exit 0"), ("conda", "exit 0"),
                           ("srun", 'printf "%s\\0" "$@" > "$SRUN_LOG"'),
                           ("python", f'exec "{sys.executable}" "$@"')):
            (cls.bin / name).write_text(f"#!/bin/sh\n{body}\n")
            (cls.bin / name).chmod(0o755)
        cls.results, cls.data_root, cls.log = root / "results", root / "data", root / "srun"
        seeds = (42, 43, 44, 45, 46)
        for dataset in ("cifar10", "cifar100"):
            for architecture in ("resnet18", "vgg16", "inception"):
                for seed in seeds:
                    (cls.results / "data_selection_swap" / dataset / architecture
                     / "checkpoints" / f"seed_{seed}").mkdir(parents=True)
        for method in ("graphcut", "herding", "random", "el2n"):
            for seed in seeds:
                path = (cls.results / "data_selection_baselines" / method / "cifar10"
                        / "resnet18" / "5000" / f"seed_{seed}" / f"{method}_5000_indices.pt")
                path.parent.mkdir(parents=True)
                path.write_text("indices")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def srun_argv(self, script, task, env, args=()):
        """What the launcher hands to srun for this array task."""
        environment = {
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "HOME": os.environ.get("HOME", str(self.project)),
            "PROJECT_ROOT": str(self.project),
            "SCRATCH": str(self.results), "RESULTS_ROOT": str(self.results),
            "SLURM_TMPDIR": str(self.data_root), "SLURM_ARRAY_TASK_ID": str(task),
            "SRUN_LOG": str(self.log), **env,
        }
        result = subprocess.run([BASH, str(REPO / "scripts" / "slurm" / script), *args],
                                cwd=self.project, env=environment,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return [token.decode() for token in self.log.read_bytes().split(b"\0")[:-1]]

    def launcher(self, script, task, env):
        return _parse(self.srun_argv(script, task, env), self.project)

    def from_config(self, name, stage, task):
        runtime = {"results": str(self.results), "data_root": str(self.data_root)}
        script, args = run_config.resolve(config(name), stage, task, runtime)
        return _parse([sys.executable, str(REPO / script), *run_config.to_flags(args)], REPO)

    def assert_same_runs(self, name, stage, launcher, tasks, env=None):
        for task in tasks:
            with self.subTest(config=name, stage=stage, task=task):
                launcher_script, launcher_flags = self.launcher(launcher, task, env or {})
                config_script, config_flags = self.from_config(name, stage, task)
                self.assertEqual(config_script, launcher_script)
                for key, value in launcher_flags.items():
                    if key == "checkpoint_dir":
                        value = (re.sub(r"seed_\d+$", "seed_42", value[0]),)
                    self.assertEqual(config_flags.get(key), value, key)
                for key in set(config_flags) - set(launcher_flags):
                    expected = () if key == "allow_manifest_mismatch" else EXPLICIT_DEFAULTS[key]
                    self.assertEqual(config_flags[key], expected, key)

    def test_reference_trajectories(self):
        self.assert_same_runs("paper/reference_trajectories", "reference",
                              "run_full_data.sh", range(6), {"SEEDS_OVERRIDE": "42"})

    def test_main_comparison(self):
        self.assert_same_runs("paper/main_comparison", "gradient_swap",
                              "run_kl_full_selection.sh", (0, 1, 7, 23, 59, 60, 98, 119))
        # One task per method (methods change every 20 tasks), then the far end.
        self.assert_same_runs("paper/main_comparison", "baselines", "run_baselines.sh",
                              (*range(0, 140, 20), 451, 839))

    def test_checkpoint_ablation(self):
        self.assert_same_runs("paper/checkpoint_ablation", "gradient_swap",
                              "run_kl_full_selection_checkpoints.sh",
                              (0, 20, 40, 60, 80, 333, 599))

    def test_jl_ablation(self):
        self.assert_same_runs("paper/jl_ablation", "gradient_swap",
                              "run_kl_full_jl_ablation.sh", (0, 20, 40, 60, 80, 100, 119),
                              {"DATASETS": "cifar10", "ARCHITECTURES": "resnet18"})

    def test_warm_start(self):
        self.assert_same_runs("paper/warm_start", "gradient_swap",
                              "run_kl_full_from_baselines.sh", range(20))
        self.assert_same_runs("paper/warm_start", "score", "run_score_warmstart.sh", (0,))

    def test_generic_launcher_stages_the_dataset_and_runs_the_task(self):
        config_path = "configs/paper/main_comparison.yaml"
        argv = self.srun_argv("run_config.sh", 7, {"PROJECT_ROOT": str(REPO)},
                              args=(config_path, "baselines"))
        self.assertEqual(argv[1:], [str(REPO / "scripts" / "experiments" / "run_config.py"),
                                    "run-task", str(REPO / config_path), "baselines", "7",
                                    "--data-root", str(self.data_root)])
        self.assertTrue((self.data_root / "cifar10_data").is_dir())   # task 7 is CIFAR-10


if __name__ == "__main__":
    unittest.main()
