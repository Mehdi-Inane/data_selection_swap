#!/usr/bin/env python3
"""Controlled SGD/Adam and learning-rate experiment.

This driver deliberately separates the experiment into three stages:

1. ``reference`` trains a seed-matched full-data trajectory.
2. ``selection`` selects a subset once and freezes its indices.
3. ``retrain`` evaluates those exact indices under an optimizer/LR grid.

The separation matters.  Re-running selection for every retraining setting
would mix selection noise into the optimizer comparison, while the historical
KL entry point also uses filenames that do not encode the retraining optimizer
or learning rate.  Here every artifact directory is keyed by a canonical
configuration hash, completed directories are immutable, and every downstream
stage records hashes of its inputs.

The configuration can independently vary:

* reference optimizer and reference learning rate;
* score objective (uniform, SGD eta weighting, or Adam eta/D^2 weighting);
* retraining optimizer and retraining learning rate.

The default example compares uniform versus learning-rate-weighted scores on
the same trajectory.  On Adam trajectories it additionally compares the SGD
eta-only score with the full Adam-preconditioned score, isolating the effect of
the preconditioner.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
SCHEMA_VERSION = 1
STAGES = ("reference", "selection", "retrain")
SCORE_OBJECTIVES = ("uniform", "sgd", "adam")
KNOWN_DATASETS = ("cifar10", "cifar100", "mnist", "imagenet")
KNOWN_ARCHITECTURES = ("resnet18", "resnet34", "lenet", "convnet")


def _source_fingerprint() -> str:
    """Fingerprint maintained experiment code so stale results are not reused."""
    digest = hashlib.sha256()
    paths = sorted((PROJECT_ROOT / "src").rglob("*.py"))
    paths += [
        PROJECT_ROOT / "scripts" / "experiments" / name
        for name in ("optimizer_lr.py", "compare_swap_rules.py")
    ]
    paths = [path for path in paths if path.is_file()]
    for path in paths:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


SOURCE_FINGERPRINT = _source_fingerprint()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _short_hash(value: Any, length: int = 16) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:length]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_artifacts(root: Path) -> dict[str, str]:
    manifest_path = root / "artifacts.sha256.json"
    expected = _read_json(manifest_path)
    mismatches = []
    for relative, wanted in expected.items():
        path = root / relative
        if not path.is_file():
            mismatches.append(f"{relative}: missing")
            continue
        actual = _sha256(path)
        if actual != wanted:
            mismatches.append(f"{relative}: sha256={actual}, expected={wanted}")
    if mismatches:
        raise RuntimeError(
            f"Artifact verification failed below {root}: " + "; ".join(mismatches)
        )
    return expected


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _require_keys(mapping: dict[str, Any], keys: Iterable[str], context: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{context} is missing required key(s): {', '.join(missing)}")


def _positive_numbers(values: Iterable[Any], context: str) -> None:
    values = list(values)
    if not values:
        raise ValueError(f"{context} must not be empty")
    if any(not isinstance(value, (int, float)) or value <= 0 for value in values):
        raise ValueError(f"{context} must contain only positive numbers: {values!r}")


def _no_duplicates(values: Iterable[Any], context: str) -> None:
    values = list(values)
    if len(values) != len(set(values)):
        raise ValueError(f"{context} contains duplicates: {values!r}")


def validate_config(config: dict[str, Any]) -> None:
    """Validate the experiment specification without importing torch."""
    _require_keys(
        config,
        ["schema_version", "experiment_name", "datasets", "architectures", "fractions", "seeds",
         "reference", "selection", "retraining"],
        "config",
    )
    if config["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version={config['schema_version']!r}; expected {SCHEMA_VERSION}"
        )
    if not config["datasets"] or any(not isinstance(x, str) or not x for x in config["datasets"]):
        raise ValueError("datasets must be a non-empty list of names")
    if (not config["architectures"] or
            any(not isinstance(x, str) or not x for x in config["architectures"])):
        raise ValueError("architectures must be a non-empty list of names")
    unknown_datasets = sorted(set(config["datasets"]) - set(KNOWN_DATASETS))
    unknown_architectures = sorted(set(config["architectures"]) - set(KNOWN_ARCHITECTURES))
    if unknown_datasets:
        raise ValueError(f"Unknown dataset(s): {unknown_datasets}; choose from {KNOWN_DATASETS}")
    if unknown_architectures:
        raise ValueError(
            f"Unknown architecture(s): {unknown_architectures}; choose from {KNOWN_ARCHITECTURES}"
        )
    for values, context in [
        (config["datasets"], "datasets"), (config["architectures"], "architectures"),
        (config["fractions"], "fractions"), (config["seeds"], "seeds"),
    ]:
        _no_duplicates(values, context)
    fractions = config["fractions"]
    if not fractions or any(not isinstance(x, (int, float)) or not 0 < x < 1 for x in fractions):
        raise ValueError("fractions must contain numbers strictly between zero and one")
    seeds = config["seeds"]
    if not seeds or any(not isinstance(x, int) or x < 0 for x in seeds):
        raise ValueError("seeds must contain non-negative integers")

    reference = config["reference"]
    _require_keys(
        reference,
        ["optimizers", "batch_size", "num_workers", "num_checkpoints", "spacing",
         "weight_decay", "beta1", "beta2", "adam_eps", "num_epochs"],
        "reference",
    )
    if not reference["optimizers"]:
        raise ValueError("reference.optimizers must not be empty")
    for optimizer, settings in reference["optimizers"].items():
        if optimizer not in ("sgd", "adam"):
            raise ValueError(f"reference optimizer must be sgd or adam, got {optimizer!r}")
        _require_keys(settings, ["learning_rates"], f"reference.optimizers.{optimizer}")
        _positive_numbers(settings["learning_rates"],
                          f"reference.optimizers.{optimizer}.learning_rates")
        _no_duplicates(settings["learning_rates"],
                       f"reference.optimizers.{optimizer}.learning_rates")
    if reference["spacing"] not in ("linear", "log"):
        raise ValueError("reference.spacing must be 'linear' or 'log'")
    if reference["num_epochs"] is not None and reference["num_epochs"] <= 0:
        raise ValueError("reference.num_epochs must be null or positive")
    _positive_numbers([reference["batch_size"], reference["num_checkpoints"]], "reference sizes")
    if (reference["num_workers"] < 0 or reference["weight_decay"] < 0 or
            reference["adam_eps"] <= 0 or not 0 <= reference["beta1"] < 1 or
            not 0 <= reference["beta2"] < 1):
        raise ValueError("reference workers/optimizer hyperparameters are outside valid ranges")

    selection = config["selection"]
    _require_keys(
        selection,
        ["objectives_by_reference", "rule", "num_checkpoints", "checkpoint_selection",
         "checkpoint_weighting", "scoring_transform", "include_bias", "proj_dim",
         "max_iters", "max_pairs", "batch_pairs", "batch_size", "num_workers",
         "adam_eps"],
        "selection",
    )
    extra_objective_maps = sorted(
        set(selection["objectives_by_reference"]) - set(reference["optimizers"])
    )
    if extra_objective_maps:
        raise ValueError(
            "selection.objectives_by_reference contains unused reference optimizer(s): "
            f"{extra_objective_maps}"
        )
    for reference_optimizer in reference["optimizers"]:
        if reference_optimizer not in selection["objectives_by_reference"]:
            raise ValueError(
                "selection.objectives_by_reference has no entry for reference optimizer "
                f"{reference_optimizer!r}"
            )
        objectives = selection["objectives_by_reference"][reference_optimizer]
        if not objectives:
            raise ValueError(f"No selection objectives for {reference_optimizer!r}")
        unknown = sorted(set(objectives) - set(SCORE_OBJECTIVES))
        if unknown:
            raise ValueError(f"Unknown score objective(s): {unknown}")
        _no_duplicates(objectives,
                       f"selection.objectives_by_reference.{reference_optimizer}")
        if reference_optimizer == "sgd" and "adam" in objectives:
            raise ValueError(
                "The Adam D^2 metric needs Adam second moments and cannot score an SGD reference"
            )
    if selection["rule"] not in ("ranked", "best"):
        raise ValueError("selection.rule must be 'ranked' (Algorithm 3) or 'best' (Algorithm 2)")
    if selection["checkpoint_selection"] not in ("spread", "prefix"):
        raise ValueError("selection.checkpoint_selection must be spread or prefix")
    if selection["checkpoint_weighting"] not in ("point", "window"):
        raise ValueError("selection.checkpoint_weighting must be point or window")
    if selection["scoring_transform"] not in ("test", "train"):
        raise ValueError("selection.scoring_transform must be test or train")
    if (selection["proj_dim"] < 0 or selection["max_iters"] <= 0 or
            selection["max_pairs"] < 0 or selection["batch_pairs"] <= 0 or
            selection["num_checkpoints"] <= 0 or selection["num_workers"] < 0):
        raise ValueError(
            "selection requires proj_dim/max_pairs/num_workers >= 0 and "
            "max_iters/batch_pairs/num_checkpoints > 0"
        )
    if selection["adam_eps"] <= 0:
        raise ValueError("selection.adam_eps must be positive")
    if selection["batch_size"] != reference["batch_size"]:
        raise ValueError(
            "selection.batch_size must equal reference.batch_size: checkpoint step "
            "metadata and trajectory provenance are batch-size specific"
        )
    if "adam" in {
        objective
        for values in selection["objectives_by_reference"].values()
        for objective in values
    } and selection["proj_dim"] != 0:
        raise ValueError("Adam's coordinate metric requires selection.proj_dim=0")
    if "adam" in {
        objective
        for values in selection["objectives_by_reference"].values()
        for objective in values
    } and selection["adam_eps"] != reference["adam_eps"]:
        raise ValueError(
            "selection.adam_eps must match reference.adam_eps for the Adam preconditioner"
        )

    retraining = config["retraining"]
    _require_keys(
        retraining,
        ["optimizers", "batch_size", "num_workers", "num_epochs", "eval_every",
         "scheduler", "weight_decay", "beta1", "beta2", "adam_eps"],
        "retraining",
    )
    if retraining["scheduler"] != "cosine":
        raise ValueError("Only the explicitly controlled cosine scheduler is supported")
    if not retraining["optimizers"]:
        raise ValueError("retraining.optimizers must not be empty")
    for optimizer, settings in retraining["optimizers"].items():
        if optimizer not in ("sgd", "adam"):
            raise ValueError(f"retraining optimizer must be sgd or adam, got {optimizer!r}")
        _require_keys(settings, ["learning_rates"], f"retraining.optimizers.{optimizer}")
        _positive_numbers(settings["learning_rates"],
                          f"retraining.optimizers.{optimizer}.learning_rates")
        _no_duplicates(settings["learning_rates"],
                       f"retraining.optimizers.{optimizer}.learning_rates")
    if retraining["num_epochs"] is not None and retraining["num_epochs"] <= 0:
        raise ValueError("retraining.num_epochs must be null or positive")
    _positive_numbers(
        [retraining["batch_size"], retraining["eval_every"]], "retraining sizes"
    )
    if (retraining["num_workers"] < 0 or retraining["weight_decay"] < 0 or
            retraining["adam_eps"] <= 0 or not 0 <= retraining["beta1"] < 1 or
            not 0 <= retraining["beta2"] < 1):
        raise ValueError("retraining workers/optimizer hyperparameters are outside valid ranges")


def load_config(path: Path) -> dict[str, Any]:
    config = _read_json(path)
    validate_config(config)
    return config


def _task_id(stage: str, identity: dict[str, Any]) -> str:
    return f"{stage[:3]}-{_short_hash({
        'schema': SCHEMA_VERSION,
        'source_fingerprint': SOURCE_FINGERPRINT,
        'stage': stage,
        **identity,
    })}"


def reference_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    settings = config["reference"]
    common = {key: value for key, value in settings.items() if key != "optimizers"}
    for dataset in config["datasets"]:
        for architecture in config["architectures"]:
            for seed in config["seeds"]:
                for optimizer, optimizer_settings in settings["optimizers"].items():
                    for learning_rate in optimizer_settings["learning_rates"]:
                        identity = {
                            "experiment": config["experiment_name"],
                            "dataset": dataset,
                            "architecture": architecture,
                            "seed": seed,
                            "reference_optimizer": optimizer,
                            "reference_lr": float(learning_rate),
                            "reference": common,
                        }
                        tasks.append({"stage": "reference",
                                      "id": _task_id("reference", identity), **identity})
    return tasks


def selection_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    selection = config["selection"]
    selection_settings = {
        key: value for key, value in selection.items() if key != "objectives_by_reference"
    }
    for reference in reference_tasks(config):
        objectives = selection["objectives_by_reference"][reference["reference_optimizer"]]
        for fraction in config["fractions"]:
            for objective in objectives:
                identity = {
                    "experiment": config["experiment_name"],
                    "dataset": reference["dataset"],
                    "architecture": reference["architecture"],
                    "fraction": float(fraction),
                    "seed": reference["seed"],
                    "reference_id": reference["id"],
                    "reference_optimizer": reference["reference_optimizer"],
                    "reference_lr": reference["reference_lr"],
                    "score_objective": objective,
                    "selection": selection_settings,
                }
                tasks.append({"stage": "selection", "id": _task_id("selection", identity),
                              **identity})
    return tasks


def retrain_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    settings = config["retraining"]
    common = {key: value for key, value in settings.items() if key != "optimizers"}
    for selection in selection_tasks(config):
        for optimizer, optimizer_settings in settings["optimizers"].items():
            for learning_rate in optimizer_settings["learning_rates"]:
                identity = {
                    "experiment": config["experiment_name"],
                    "dataset": selection["dataset"],
                    "architecture": selection["architecture"],
                    "fraction": selection["fraction"],
                    "seed": selection["seed"],
                    "reference_id": selection["reference_id"],
                    "reference_optimizer": selection["reference_optimizer"],
                    "reference_lr": selection["reference_lr"],
                    "selection_id": selection["id"],
                    "score_objective": selection["score_objective"],
                    "retrain_optimizer": optimizer,
                    "retrain_lr": float(learning_rate),
                    "retraining": common,
                }
                tasks.append({"stage": "retrain", "id": _task_id("retrain", identity),
                              **identity})
    return tasks


def tasks_for(config: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    if stage == "reference":
        return reference_tasks(config)
    if stage == "selection":
        return selection_tasks(config)
    if stage == "retrain":
        return retrain_tasks(config)
    raise ValueError(f"Unknown stage {stage!r}")


def resolve_output_root(config_path: Path, config: dict[str, Any], override: str | None) -> Path:
    raw = override or config.get("output_root")
    if raw is None:
        scratch = os.environ.get("SCRATCH")
        raw = str(Path(scratch) / "data_selection_swap" / "optimizer_lr") if scratch else str(
            PROJECT_ROOT / "results" / "optimizer_lr"
        )
    expanded = Path(os.path.expandvars(os.path.expanduser(str(raw))))
    if not expanded.is_absolute():
        expanded = (config_path.parent / expanded).resolve()
    return expanded


def task_dir(output_root: Path, task: dict[str, Any]) -> Path:
    plural = {"reference": "references", "selection": "selections", "retrain": "retrains"}
    return output_root / plural[task["stage"]] / task["id"]


def _find_task(config: dict[str, Any], stage: str, task_id: int) -> dict[str, Any]:
    tasks = tasks_for(config, stage)
    if not 0 <= task_id < len(tasks):
        raise IndexError(f"task-id {task_id} outside 0..{len(tasks) - 1} for stage {stage}")
    return tasks[task_id]


def _score_flags(objective: str) -> tuple[str, str]:
    if objective == "uniform":
        return "uniform", "none"
    if objective == "sgd":
        return "sgd", "none"
    if objective == "adam":
        return "adam", "adam"
    raise ValueError(f"Unknown score objective: {objective}")


def reference_command(task: dict[str, Any], data_dir: Path, save_dir: Path,
                      download: bool, resume: bool = False) -> list[str]:
    cfg = task["reference"]
    command = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "train" / "reference.py"),
        "--dataset", task["dataset"], "--architecture", task["architecture"],
        "--data_dir", str(data_dir),
        "--save_dir", str(save_dir), "--seed", str(task["seed"]),
        "--optimizer", task["reference_optimizer"], "--lr", str(task["reference_lr"]),
        "--weight_decay", str(cfg["weight_decay"]), "--beta1", str(cfg["beta1"]),
        "--beta2", str(cfg["beta2"]), "--adam_eps", str(cfg["adam_eps"]),
        "--batch_size", str(cfg["batch_size"]), "--num_workers", str(cfg["num_workers"]),
        "--num_checkpoints", str(cfg["num_checkpoints"]), "--spacing", cfg["spacing"],
    ]
    if cfg["num_epochs"] is not None:
        command.extend(["--num_epochs", str(cfg["num_epochs"])])
    if download:
        command.append("--download")
    if resume:
        command.append("--resume")
    return command


def selection_command(task: dict[str, Any], reference_dir: Path, data_dir: Path,
                      download: bool) -> list[str]:
    cfg = task["selection"]
    objective, metric = _score_flags(task["score_objective"])
    command = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "select" / "kl.py"),
        "--dataset", task["dataset"], "--architecture", task["architecture"],
        "--data_dir", str(data_dir),
        "--checkpoint_dir", str(reference_dir), "--fraction", str(task["fraction"]),
        "--seed", str(task["seed"]), "--optimizer", objective, "--metric", metric,
        "--rule", cfg["rule"], "--num_checkpoints", str(cfg["num_checkpoints"]),
        "--checkpoint_selection", cfg["checkpoint_selection"],
        "--checkpoint_weighting", cfg["checkpoint_weighting"],
        "--scoring_transform", cfg["scoring_transform"],
        "--proj_dim", str(cfg["proj_dim"]), "--max_iters", str(cfg["max_iters"]),
        "--adam_eps", str(cfg["adam_eps"]),
        "--max_pairs", str(cfg["max_pairs"]), "--batch_pairs", str(cfg["batch_pairs"]),
        "--batch_size", str(cfg["batch_size"]), "--num_workers", str(cfg["num_workers"]),
        "--selection_only",
    ]
    command.append("--include_bias" if cfg["include_bias"] else "--no_include_bias")
    if download:
        command.append("--download")
    return command


def _git_provenance() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(PROJECT_ROOT), *args], text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain")
    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "source_fingerprint": SOURCE_FINGERPRINT,
        "python": sys.version,
        "platform": platform.platform(),
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    old_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC_ROOT) + (os.pathsep + old_path if old_path else "")
    env["PYTHONHASHSEED"] = "0"
    return env


def _run_command(command: list[str], env: dict[str, str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def _completed(final_dir: Path, task: dict[str, Any]) -> bool:
    marker = final_dir / "_SUCCESS.json"
    if not marker.exists():
        return False
    value = _read_json(marker)
    if value.get("task_id") != task["id"]:
        raise RuntimeError(f"Completion marker in {final_dir} belongs to another task")
    return True


@contextmanager
def _task_lock(final_dir: Path):
    """Prevent duplicate array/requeue instances from running one task concurrently."""
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = final_dir.parent / f".{final_dir.name}.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is already running task {final_dir.name}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "host": platform.node()}))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _begin(final_dir: Path, task: dict[str, Any]) -> tuple[Path, Path] | None:
    if _completed(final_dir, task):
        print(f"Already complete; leaving immutable artifact unchanged: {final_dir}")
        return None
    if final_dir.exists():
        raise RuntimeError(
            f"Refusing to overwrite incomplete artifact directory {final_dir}. "
            "Move it aside after inspecting it, then retry."
        )
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{task['id']}-", dir=final_dir.parent))
    payload = work / "payload"
    payload.mkdir()
    return work, payload


def _publish(work: Path, payload: Path, final_dir: Path, task: dict[str, Any]) -> None:
    _write_json(payload / "task.json", task)
    _write_json(payload / "_SUCCESS.json", {
        "task_id": task["id"],
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    })
    os.replace(payload, final_dir)
    shutil.rmtree(work)
    print(f"Published immutable artifact: {final_dir}")


def _ensure_cuda(require_cuda: bool, allow_cpu: bool) -> None:
    if not require_cuda or allow_cpu:
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This configuration requires CUDA, but no GPU is visible. Pass --allow-cpu only "
            "for a deliberate smoke test; do not mix CPU and GPU timings."
        )


def run_reference(task: dict[str, Any], output_root: Path, data_dir: Path,
                  download: bool, allow_cpu: bool, require_cuda: bool) -> None:
    _ensure_cuda(require_cuda, allow_cpu)
    final_dir = task_dir(output_root, task)
    if _completed(final_dir, task):
        print(f"Already complete; leaving immutable artifact unchanged: {final_dir}")
        return
    final_dir.mkdir(parents=True, exist_ok=True)
    state_path = final_dir / "training_state.pth"
    manifest_path = final_dir / "trajectory.json"
    resume = state_path.exists()
    if manifest_path.exists() and not resume:
        raise RuntimeError(
            f"Partial reference {final_dir} has a manifest but no resumable training state. "
            "Move it aside after inspection; it will not be overwritten."
        )
    command = reference_command(task, data_dir, final_dir, download, resume=resume)
    try:
        _run_command(command, _environment())
        manifest = _read_json(manifest_path)
        expected_epochs = task["reference"]["num_epochs"]
        expected_epochs = int(expected_epochs or manifest["num_epochs"])
        checks = {
            "optimizer": task["reference_optimizer"],
            "seed": task["seed"],
            "dataset": task["dataset"],
            "architecture": task["architecture"],
            "lr": task["reference_lr"],
        }
        for key, expected in checks.items():
            if manifest.get(key) != expected:
                raise RuntimeError(
                    f"Reference manifest mismatch for {key}: {manifest.get(key)!r} != {expected!r}"
                )
        initial_hash = manifest.get("initial_state_sha256")
        if (not isinstance(initial_hash, str) or len(initial_hash) != 64 or
                any(character not in "0123456789abcdef" for character in initial_hash)):
            raise RuntimeError("Reference manifest has no valid initial_state_sha256")
        if manifest.get("epochs_completed") != expected_epochs:
            raise RuntimeError(
                f"Reference stopped at epoch {manifest.get('epochs_completed')}; "
                f"expected {expected_epochs}"
            )
        checkpoint_files = sorted(final_dir.glob("checkpoint_*.pth"))
        if len(checkpoint_files) != len(manifest.get("checkpoints", [])):
            raise RuntimeError("Reference checkpoint files disagree with trajectory.json")
        artifacts = {path.relative_to(final_dir).as_posix(): _sha256(path)
                     for path in [manifest_path, *checkpoint_files]}
        _write_json(final_dir / "artifacts.sha256.json", artifacts)
        _write_json(final_dir / "task.json", task)
        _write_json(final_dir / "provenance.json", {
            **_git_provenance(), "command": command,
            "resumed": resume,
            "data_dir": str(data_dir.resolve()), "artifacts": artifacts,
        })
        # The marker is written last. Downstream jobs never consume a partial
        # trajectory, while an interrupted reference can resume in place.
        _write_json(final_dir / "_SUCCESS.json", {
            "task_id": task["id"],
            "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        })
        print(f"Published resumable reference artifact: {final_dir}")
    except Exception:
        print(f"Reference failed; preserved resumable directory: {final_dir}", file=sys.stderr)
        raise


def _single_match(root: Path, pattern: str, required: bool = True) -> Path | None:
    matches = sorted(root.rglob(pattern))
    if not matches and not required:
        return None
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {pattern!r} below {root}, found {matches}")
    return matches[0]


def run_selection(task: dict[str, Any], config: dict[str, Any], output_root: Path,
                  data_dir: Path, download: bool, allow_cpu: bool,
                  require_cuda: bool) -> None:
    _ensure_cuda(require_cuda, allow_cpu)
    references = {item["id"]: item for item in reference_tasks(config)}
    reference = references[task["reference_id"]]
    reference_dir = task_dir(output_root, reference)
    if not _completed(reference_dir, reference):
        raise RuntimeError(f"Reference dependency is incomplete: {reference_dir}")
    reference_hashes = _verify_artifacts(reference_dir)

    final_dir = task_dir(output_root, task)
    started = _begin(final_dir, task)
    if started is None:
        return
    work, payload = started
    scratch = work / "isolated_scratch"
    command = selection_command(task, reference_dir, data_dir, download)
    env = _environment()
    env["SCRATCH"] = str(scratch)
    try:
        _run_command(command, env)
        generated = scratch / "gradmatch_swap"
        copied: dict[str, str] = {}
        for pattern, destination, required in [
            ("*_indices.pt", "indices.pt", True),
            ("*_objective.json", "objective.json", True),
            ("*_timing.json", "timing.json", True),
            ("*_training.log", "selection.log", False),
        ]:
            source = _single_match(generated, pattern, required=required)
            if source is not None:
                shutil.copy2(source, payload / destination)
                copied[destination] = _sha256(payload / destination)

        _write_json(payload / "artifacts.sha256.json", copied)
        _write_json(payload / "provenance.json", {
            **_git_provenance(), "command": command,
            "data_dir": str(data_dir.resolve()),
            "reference_dir": str(reference_dir.resolve()),
            "reference_artifact_manifest_sha256": _sha256(
                reference_dir / "artifacts.sha256.json"
            ),
            "reference_artifacts": reference_hashes,
            "artifacts": copied,
        })
        _publish(work, payload, final_dir, task)
    except Exception:
        print(f"Selection failed; preserved diagnostic work directory: {work}", file=sys.stderr)
        raise


def _evaluate(model: Any, loader: Any, device: str) -> float:
    import torch

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            correct += model(inputs).argmax(1).eq(targets).sum().item()
            total += targets.size(0)
    if total == 0:
        raise RuntimeError("Test loader is empty")
    return 100.0 * correct / total


def run_retrain(task: dict[str, Any], config: dict[str, Any], output_root: Path,
                data_dir: Path, download: bool, allow_cpu: bool,
                require_cuda: bool) -> None:
    _ensure_cuda(require_cuda, allow_cpu)
    selections = {item["id"]: item for item in selection_tasks(config)}
    selection = selections[task["selection_id"]]
    selection_dir = task_dir(output_root, selection)
    if not _completed(selection_dir, selection):
        raise RuntimeError(f"Selection dependency is incomplete: {selection_dir}")
    _verify_artifacts(selection_dir)

    final_dir = task_dir(output_root, task)
    if _completed(final_dir, task):
        print(f"Already complete; leaving immutable artifact unchanged: {final_dir}")
        return
    final_dir.mkdir(parents=True, exist_ok=True)
    state_path = final_dir / "training_state.pth"
    resume = state_path.exists()
    try:
        sys.path.insert(0, str(SRC_ROOT))
        import torch
        import torch.nn as nn
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from torch.utils.data import DataLoader, Subset, random_split
        from data_selection_swap import common

        settings = task["retraining"]
        common.set_seed(task["seed"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = common.get_dataset_config(
            task["dataset"], str(data_dir), download=download,
            architecture=task["architecture"],
        )
        n_val = int(0.1 * len(cfg["full_train"]))
        n_train = len(cfg["full_train"]) - n_val
        trainset, _ = random_split(
            cfg["full_train"], [n_train, n_val],
            generator=torch.Generator().manual_seed(task["seed"]),
        )
        indices_path = selection_dir / "indices.pt"
        selected_indices = torch.load(indices_path, map_location="cpu").long().reshape(-1)
        expected_budget = int(task["fraction"] * n_train)
        if selected_indices.numel() != expected_budget:
            raise RuntimeError(
                f"Selected-index count {selected_indices.numel()} != expected {expected_budget}"
            )
        if (selected_indices < 0).any() or (selected_indices >= n_train).any():
            raise RuntimeError("Selected indices are outside the deterministic training split")
        if selected_indices.unique().numel() != selected_indices.numel():
            raise RuntimeError("Selected indices contain duplicates")

        train_generator = torch.Generator().manual_seed(task["seed"] + 10_000)
        trainloader = DataLoader(
            Subset(trainset, selected_indices.tolist()), batch_size=settings["batch_size"],
            shuffle=True, generator=train_generator, pin_memory=True,
            num_workers=settings["num_workers"],
        )
        testloader = DataLoader(
            cfg["testset"], batch_size=settings["batch_size"], shuffle=False,
            pin_memory=True, num_workers=settings["num_workers"],
        )

        # Seed immediately before construction: all retraining cells for a seed
        # start from byte-identical weights, independent of their selection stage.
        common.set_seed(task["seed"])
        model = common.build_model(
            cfg, cfg["num_classes"], device, architecture=task["architecture"]
        )
        initial_state_sha256 = common.state_dict_fingerprint(model.state_dict())
        criterion = nn.CrossEntropyLoss()
        optimizer = common.build_optimizer(
            model.parameters(), task["retrain_optimizer"], task["retrain_lr"],
            weight_decay=settings["weight_decay"],
            betas=(settings["beta1"], settings["beta2"]), eps=settings["adam_eps"],
        )
        num_epochs = int(settings["num_epochs"] or cfg["num_epochs"])
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

        history: list[dict[str, Any]] = []
        epoch_seconds: list[float] = []
        start_epoch = 1
        prior_total_seconds = 0.0
        if resume:
            try:
                state = torch.load(state_path, map_location=device, weights_only=False)
            except TypeError:  # torch < 2.0 has no weights_only argument
                state = torch.load(state_path, map_location=device)
            if state.get("task_id") != task["id"]:
                raise RuntimeError(
                    f"Retraining state belongs to {state.get('task_id')}, not {task['id']}"
                )
            if state.get("initial_state_sha256") != initial_state_sha256:
                raise RuntimeError(
                    "Initial model fingerprint changed since the interrupted run; "
                    "refusing a non-reproducible resume."
                )
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start_epoch = int(state["epoch"]) + 1
            history = list(state["history"])
            epoch_seconds = list(state["epoch_seconds"])
            prior_total_seconds = float(state.get("total_seconds", sum(epoch_seconds)))
            train_generator.set_state(state["train_generator_state"].cpu())
            torch.set_rng_state(state["torch_rng_state"].cpu())
            if device == "cuda" and state.get("cuda_rng_state_all") is not None:
                torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
            if state.get("python_rng_state") is not None:
                random.setstate(state["python_rng_state"])
            print(f"Resuming retraining task {task['id']} at epoch {start_epoch}", flush=True)

        segment_started = time.perf_counter()
        total_seconds = prior_total_seconds
        for epoch in range(start_epoch, num_epochs + 1):
            if device == "cuda":
                torch.cuda.synchronize()
            epoch_started = time.perf_counter()
            epoch_lr = float(optimizer.param_groups[0]["lr"])
            model.train()
            running_loss = 0.0
            examples = 0
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(inputs), targets)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item()) * targets.size(0)
                examples += targets.size(0)
            scheduler.step()
            if device == "cuda":
                torch.cuda.synchronize()
            epoch_seconds.append(time.perf_counter() - epoch_started)

            if epoch % settings["eval_every"] == 0 and epoch != num_epochs:
                print(
                    f"epoch={epoch}/{num_epochs} optimizer={task['retrain_optimizer']} "
                    f"lr={task['retrain_lr']:.6g} loss={running_loss / examples:.5f}",
                    flush=True,
                )

            # The official test set is touched once, at the declared terminal
            # epoch. This prevents a periodic test trace from becoming an
            # undeclared model-selection channel.
            if epoch == num_epochs:
                accuracy = _evaluate(model, testloader, device)
                row = {
                    "epoch": epoch, "learning_rate": epoch_lr,
                    "train_loss": running_loss / examples,
                    "test_accuracy": accuracy,
                }
                history.append(row)
                print(
                    f"epoch={epoch}/{num_epochs} optimizer={task['retrain_optimizer']} "
                    f"lr={task['retrain_lr']:.6g} loss={row['train_loss']:.5f} "
                    f"test_accuracy={accuracy:.3f}", flush=True,
                )

            total_seconds = prior_total_seconds + (time.perf_counter() - segment_started)
            training_state = {
                "task_id": task["id"], "epoch": epoch,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "history": history,
                "epoch_seconds": epoch_seconds, "total_seconds": total_seconds,
                "initial_state_sha256": initial_state_sha256,
                "train_generator_state": train_generator.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": (
                    torch.cuda.get_rng_state_all() if device == "cuda" else None
                ),
                "python_rng_state": random.getstate(),
            }
            state_tmp = state_path.with_name(f"{state_path.name}.tmp.{os.getpid()}")
            torch.save(training_state, state_tmp)
            os.replace(state_tmp, state_path)

        if not history or history[-1]["epoch"] != num_epochs:
            raise RuntimeError("Final-epoch test accuracy was not evaluated")
        model_path = final_dir / "model.pth"
        torch.save(model.state_dict(), model_path)
        metrics = {
            "primary_metric": "final_test_accuracy",
            "final_test_accuracy": history[-1]["test_accuracy"],
            "best_observed_test_accuracy": history[-1]["test_accuracy"],
            "test_evaluations": 1,
            "history": history,
            "num_epochs": num_epochs,
            "optimizer": task["retrain_optimizer"],
            "initial_learning_rate": task["retrain_lr"],
            "scheduler": settings["scheduler"],
            "initial_state_sha256": initial_state_sha256,
        }
        timing = {
            "total_seconds": total_seconds,
            "epoch_seconds": epoch_seconds,
            "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
            "includes": "subset retraining and one final test evaluation; excludes selection",
        }
        _write_json(final_dir / "metrics.json", metrics)
        _write_json(final_dir / "timing.json", timing)
        copied_indices = final_dir / "indices.input.pt"
        shutil.copy2(indices_path, copied_indices)
        artifacts = {
            "indices.input.pt": _sha256(copied_indices),
            "model.pth": _sha256(model_path),
            "metrics.json": _sha256(final_dir / "metrics.json"),
            "timing.json": _sha256(final_dir / "timing.json"),
        }
        _write_json(final_dir / "artifacts.sha256.json", artifacts)
        _write_json(final_dir / "task.json", task)
        _write_json(final_dir / "provenance.json", {
            **_git_provenance(),
            "resumed": resume,
            "data_dir": str(data_dir.resolve()),
            "selection_dir": str(selection_dir.resolve()),
            "selection_artifact_manifest_sha256": _sha256(
                selection_dir / "artifacts.sha256.json"
            ),
            "selected_indices_sha256": artifacts["indices.input.pt"],
            "initial_state_sha256": initial_state_sha256,
            "artifacts": artifacts,
        })
        _write_json(final_dir / "_SUCCESS.json", {
            "task_id": task["id"],
            "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        })
        # The final model is the durable state once the success marker exists;
        # discard the much larger optimizer checkpoint to keep large sweeps tractable.
        state_path.unlink(missing_ok=True)
        print(f"Published resumable retraining artifact: {final_dir}")
    except Exception:
        print(f"Retraining failed; preserved resumable directory: {final_dir}", file=sys.stderr)
        raise


def _selection_objectives(selection_dir: Path) -> tuple[float | None, float | None]:
    path = selection_dir / "objective.json"
    if not path.exists():
        return None, None
    history = _read_json(path)
    values = [float(row["objective"]) for row in history if "objective" in row]
    return (values[0], values[-1]) if values else (None, None)


def summarize(config: dict[str, Any], output_root: Path) -> tuple[Path, Path]:
    selections = {task["id"]: task for task in selection_tasks(config)}
    references = {task["id"]: task for task in reference_tasks(config)}
    rows: list[dict[str, Any]] = []
    for task in retrain_tasks(config):
        directory = task_dir(output_root, task)
        row = {
            "task_id": task["id"], "status": "missing", "dataset": task["dataset"],
            "architecture": task["architecture"],
            "fraction": task["fraction"], "seed": task["seed"],
            "reference_optimizer": task["reference_optimizer"],
            "reference_lr": task["reference_lr"],
            "reference_initial_state_sha256": None,
            "score_objective": task["score_objective"],
            "retrain_optimizer": task["retrain_optimizer"],
            "retrain_lr": task["retrain_lr"],
            "final_test_accuracy": None, "best_observed_test_accuracy": None,
            "selection_objective_initial": None, "selection_objective_final": None,
            "selection_D_final": None, "selection_certificate": None,
            "selection_swaps": None, "selection_stop_reason": None,
            "selection_pipeline_seconds": None, "retrain_seconds": None,
        }
        if _completed(directory, task):
            metrics = _read_json(directory / "metrics.json")
            timing = _read_json(directory / "timing.json")
            selection = selections[task["selection_id"]]
            reference = references[task["reference_id"]]
            reference_manifest = _read_json(
                task_dir(output_root, reference) / "trajectory.json"
            )
            selection_dir = task_dir(output_root, selection)
            initial, final = _selection_objectives(selection_dir)
            selection_history = _read_json(selection_dir / "objective.json")
            selection_final = selection_history[-1] if selection_history else {}
            selection_timing = _read_json(selection_dir / "timing.json")
            row.update({
                "status": "complete",
                "final_test_accuracy": metrics["final_test_accuracy"],
                "best_observed_test_accuracy": metrics["best_observed_test_accuracy"],
                "reference_initial_state_sha256": reference_manifest.get(
                    "initial_state_sha256"
                ),
                "selection_objective_initial": initial,
                "selection_objective_final": final,
                "selection_D_final": selection_final.get(
                    "D", selection_timing.get("D")
                ),
                "selection_certificate": selection_timing.get(
                    "certificate", selection_final.get("certificate")
                ),
                "selection_swaps": selection_timing.get(
                    "swaps", selection_final.get("swaps")
                ),
                "selection_stop_reason": selection_timing.get(
                    "reason", selection_final.get("reason")
                ),
                "selection_pipeline_seconds": selection_timing.get("selection_seconds"),
                "retrain_seconds": timing["total_seconds"],
            })
        rows.append(row)

    summary_dir = output_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    raw_path = summary_dir / "optimizer_lr_runs.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    group_fields = [
        "dataset", "architecture", "fraction", "reference_optimizer", "reference_lr",
        "score_objective", "retrain_optimizer", "retrain_lr",
    ]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row["status"] == "complete":
            key = tuple(row[field] for field in group_fields)
            groups.setdefault(key, []).append(row)
    aggregated = []
    for key, members in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        record = {**dict(zip(group_fields, key)), "n_seeds": len(members)}
        for metric in ("final_test_accuracy", "selection_D_final",
                       "selection_certificate", "selection_pipeline_seconds",
                       "retrain_seconds"):
            values = [float(member[metric]) for member in members
                      if member[metric] is not None]
            record[f"mean_{metric}"] = statistics.fmean(values) if values else None
            record[f"sample_std_{metric}"] = (
                statistics.stdev(values) if len(values) > 1 else None
            )
        accuracies = [float(member["final_test_accuracy"]) for member in members]
        record["min_final_test_accuracy"] = min(accuracies)
        record["max_final_test_accuracy"] = max(accuracies)
        record["selection_stop_reasons"] = sorted({
            str(member["selection_stop_reason"]) for member in members
        })
        aggregated.append(record)
    aggregate_path = summary_dir / "optimizer_lr_aggregate.json"
    _write_json(aggregate_path, {
        "primary_metric": "final_test_accuracy",
        "complete_runs": sum(row["status"] == "complete" for row in rows),
        "planned_runs": len(rows), "groups": aggregated,
    })
    return raw_path, aggregate_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(subparser: argparse.ArgumentParser, with_output: bool = True) -> None:
        subparser.add_argument("--config", required=True, type=Path)
        if with_output:
            subparser.add_argument("--output-root", default=None)

    validate = subparsers.add_parser("validate", help="validate a JSON configuration")
    common(validate, with_output=False)

    count = subparsers.add_parser("count", help="print the number of tasks in one stage")
    common(count, with_output=False)
    count.add_argument("--stage", required=True, choices=STAGES)

    listing = subparsers.add_parser("list", help="print the deterministic task plan as JSONL")
    common(listing)
    listing.add_argument("--stage", choices=STAGES, default=None)

    field = subparsers.add_parser("task-field", help="print one top-level field for a task")
    common(field, with_output=False)
    field.add_argument("--stage", required=True, choices=STAGES)
    field.add_argument("--task-id", required=True, type=int)
    field.add_argument("--field", required=True)

    run = subparsers.add_parser("run-task", help="run or dry-run one array task")
    common(run)
    run.add_argument("--stage", required=True, choices=STAGES)
    run.add_argument("--task-id", required=True, type=int)
    run.add_argument("--data-dir", required=True, type=Path)
    run.add_argument("--download", action="store_true")
    run.add_argument("--allow-cpu", action="store_true",
                     help="Explicitly allow CPU execution for small smoke tests")
    run.add_argument("--dry-run", action="store_true")

    summary = subparsers.add_parser("summarize", help="write raw and seed-aggregate summaries")
    common(summary)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    config = load_config(config_path)

    if args.command == "validate":
        counts = {stage: len(tasks_for(config, stage)) for stage in STAGES}
        print(json.dumps({"valid": True, "counts": counts}, sort_keys=True))
        return 0
    if args.command == "count":
        print(len(tasks_for(config, args.stage)))
        return 0
    if args.command == "task-field":
        task = _find_task(config, args.stage, args.task_id)
        if args.field not in task:
            raise KeyError(f"Task has no top-level field {args.field!r}")
        value = task[args.field]
        print(value if isinstance(value, (str, int, float)) else _canonical_json(value))
        return 0

    output_root = resolve_output_root(config_path, config, args.output_root)
    if args.command == "list":
        stages = (args.stage,) if args.stage else STAGES
        for stage in stages:
            for index, task in enumerate(tasks_for(config, stage)):
                print(_canonical_json({"array_index": index, "artifact_dir":
                                      str(task_dir(output_root, task)), **task}))
        return 0
    if args.command == "summarize":
        raw, aggregate = summarize(config, output_root)
        print(f"Wrote {raw}\nWrote {aggregate}")
        return 0
    if args.command == "run-task":
        task = _find_task(config, args.stage, args.task_id)
        if args.dry_run:
            preview: dict[str, Any] = {
                "task": task, "artifact_dir": str(task_dir(output_root, task)),
                "data_dir": str(args.data_dir),
            }
            if args.stage == "reference":
                preview["command"] = reference_command(
                    task, args.data_dir, task_dir(output_root, task), args.download
                )
            elif args.stage == "selection":
                refs = {item["id"]: item for item in reference_tasks(config)}
                preview["command"] = selection_command(
                    task, task_dir(output_root, refs[task["reference_id"]]),
                    args.data_dir, args.download,
                )
            else:
                preview["operation"] = "retrain frozen indices with the recorded optimizer/LR"
            print(json.dumps(preview, indent=2, sort_keys=True))
            return 0

        require_cuda = bool(config.get("require_cuda", True))
        with _task_lock(task_dir(output_root, task)):
            if args.stage == "reference":
                run_reference(task, output_root, args.data_dir, args.download,
                              args.allow_cpu, require_cuda)
            elif args.stage == "selection":
                run_selection(task, config, output_root, args.data_dir, args.download,
                              args.allow_cpu, require_cuda)
            else:
                run_retrain(task, config, output_root, args.data_dir, args.download,
                            args.allow_cpu, require_cuda)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
