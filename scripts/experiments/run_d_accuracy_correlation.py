#!/usr/bin/env python3
"""Run a leakage-free D(alpha) versus final-test-accuracy experiment.

For each dataset/network/split, this workflow trains (or validates and reuses)
one full-pool reference trajectory.  It then creates several subsets without
using validation or test labels, computes the all-parameter manuscript
gradient-discrepancy D(alpha) exactly on the declared recorded checkpoints, and
retrains the *same* subset from one shared initialization.  (Using K recorded
states is an explicit quadrature approximation to an every-update trajectory.)
The official test set is evaluated exactly once, after the final retraining
epoch. The reference, D score, and subset retraining all use the same clean,
deterministic training view so augmentation cannot silently change the loss
whose gradient is being measured.

The output unit is one ``record.json`` per subset.  Each record contains the
indices and their hash, D(alpha), every final-accuracy replicate, the trajectory
fingerprint, and a hash of all nuisance conditions.  This makes an accidental
join between D from one subset and accuracy from another detectable.

Examples
--------
Train/reuse a reference and evaluate a candidate panel::

    python scripts/experiments/run_d_accuracy_correlation.py \
      --stage all --dataset cifar10 --architecture resnet18 \
      --data-dir /datasets/cifar --output-dir /scratch/d_correlation \
      --fractions 0.1 0.2 0.4 --samplers uniform stratified easy hard \
      --subsets-per-sampler 3 --retrain-replicates 3 \
      --batchnorm-policy eval_frozen

Run the two expensive stages separately (the arguments affecting the reference
must be identical)::

    python scripts/experiments/run_d_accuracy_correlation.py ... --stage reference
    python scripts/experiments/run_d_accuracy_correlation.py ... --stage evaluate

``easy`` and ``hard`` mean random samples from low/high reference-loss pools;
they use only the clean training pool at the last recorded pre-epoch reference
checkpoint.  They
are deliberately included to create a useful quality range for a correlation
study and are not proposed selection algorithms.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import platform
import random
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.experiments.d_correlation_common import (  # noqa: E402
    SCHEMA_VERSION,
    analysis_block_payload,
    atomic_write_json,
    canonical_indices,
    generate_subset_indices,
    load_json,
    sha256_file,
    sha256_indices,
    sha256_json,
    sha256_ordered_indices,
    validate_record,
)


DATASETS = ("cifar10", "cifar100", "mnist")
ARCHITECTURES = ("resnet18", "resnet34", "lenet", "convnet")
SAMPLERS = ("uniform", "stratified", "easy", "hard")
D_DEFINITION = (
    "sum_{k in recorded checkpoints} w_k || mean_{i:alpha_i=1} grad ell(theta_k,z_i) "
    "- mean_{i in pool} grad ell(theta_k,z_i) ||_2^2"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_runtime():
    """Delay heavy imports so --help and pure unit tests work on login nodes."""

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Subset
        from data_selection_swap import common
    except ImportError as error:
        raise RuntimeError(
            "The experiment runtime needs torch, torchvision, and the repository's "
            "normal dependencies. Install requirements.txt and use `pip install -e .`."
        ) from error
    return torch, nn, DataLoader, Subset, common


def run_git(args: list[str]) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_provenance() -> dict[str, Any]:
    tracked = [
        Path(__file__).resolve(),
        PROJECT_ROOT / "scripts/experiments/d_correlation_common.py",
        PROJECT_ROOT / "src/data_selection_swap/common.py",
    ]
    status = run_git(["status", "--porcelain=v1"])
    return {
        "git_commit": run_git(["rev-parse", "HEAD"]),
        "git_dirty": bool(status),
        "git_status_sha256": sha256_json(status or ""),
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in tracked
        },
    }


def runtime_provenance(torch, device: str) -> dict[str, Any]:
    result = {
        "created_at_utc": utc_now(),
        "command": sys.argv,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": getattr(torch, "__version__", "unknown"),
        "device": device,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        result["cuda_device"] = torch.cuda.get_device_name(torch.device(device))
        result["cuda_version"] = getattr(torch.version, "cuda", None)
    return result


def atomic_torch_save(torch, value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def torch_load(torch, path: Path, *, map_location: str | Any) -> Any:
    # Explicit weights_only=False is required by newer PyTorch for optimizer/RNG
    # resume records, but older supported releases do not know the keyword.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def cpu_state_dict(model) -> dict[str, Any]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def choose_device(torch, requested: str) -> str:
    if requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"requested {requested!r}, but CUDA is unavailable")
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_epochs(requested: int, cfg: dict[str, Any]) -> int:
    epochs = int(requested) if requested else int(cfg["num_epochs"])
    if epochs < 1:
        raise ValueError("epoch count must be positive")
    return epochs


def resolve_lr(requested: float | None, optimizer: str) -> float:
    if requested is not None:
        if requested <= 0.0:
            raise ValueError("learning rate must be positive")
        return float(requested)
    return 0.001 if optimizer in {"adam", "adamw"} else 0.01


def validate_learning_algorithm_alignment(args, cfg: dict[str, Any]) -> None:
    reference = (
        args.reference_optimizer,
        resolve_lr(args.reference_lr, args.reference_optimizer),
        resolve_epochs(args.reference_epochs, cfg),
    )
    retraining = (
        args.retrain_optimizer,
        resolve_lr(args.retrain_lr, args.retrain_optimizer),
        resolve_epochs(args.retrain_epochs, cfg),
    )
    if reference != retraining:
        raise ValueError(
            "The D(alpha) correlation experiment requires the full-pool reference and "
            "subset fits to use the same (optimizer, LR, epochs). Got reference="
            f"{reference}, subset={retraining}. Run a separate matched cell for each "
            "optimizer/LR; use optimizer_lr.py for deliberate cross-protocol ablations."
        )


def checkpoint_epochs(total_epochs: int, count: int, explicit: list[int] | None) -> list[int]:
    if explicit:
        epochs = sorted(set(int(value) for value in explicit))
        if epochs[0] < 1 or epochs[-1] > total_epochs:
            raise ValueError(f"--checkpoint-epochs must lie in [1, {total_epochs}]")
        if epochs[-1] != total_epochs:
            raise ValueError(
                "--checkpoint-epochs must include the state at the start of the final training epoch"
            )
        return epochs
    if count < 1:
        raise ValueError("--num-d-checkpoints must be positive")
    values = np.linspace(1, total_epochs, min(count, total_epochs))
    epochs = sorted(set(int(round(value)) for value in values))
    if epochs[-1] != total_epochs:
        epochs.append(total_epochs)
    return epochs


def pool_indices_and_hash(trainset) -> tuple[list[int], str]:
    if not hasattr(trainset, "indices"):
        raise TypeError("shared loader did not return a Subset with stable pool indices")
    values = canonical_indices(int(i) for i in trainset.indices)
    # Preserve the actual pool order separately: pool-relative subset positions
    # refer to trainset.indices in this order, not in sorted order.
    ordered = [int(i) for i in trainset.indices]
    if len(values) != len(ordered):
        raise ValueError("selection pool contains duplicate global indices")
    return ordered, sha256_ordered_indices(ordered)


def build_data(common, args, *, scoring_transform: str = "test"):
    cfg = common.get_dataset_config(
        args.dataset, args.data_dir, download=args.download,
        architecture=args.architecture,
    )
    trainset, eval_loader, testloader, n_train = common.make_loaders(
        cfg, args.split_seed, args.batch_size, args.num_workers,
        scoring_transform=scoring_transform,
    )
    ordered_pool_indices, pool_hash = pool_indices_and_hash(trainset)
    if len(ordered_pool_indices) != n_train:
        raise ValueError("shared loader's n_train disagrees with trainset length")
    return cfg, trainset, eval_loader, testloader, ordered_pool_indices, pool_hash


def reference_config(args, cfg, pool_hash: str, init_hash: str, epochs: int,
                     learning_rate: float, checkpoints: list[int], code: dict[str, Any]) -> dict[str, Any]:
    pool_size = int(len(cfg["full_train"]) - int(0.1 * len(cfg["full_train"])))
    steps_per_epoch = int(math.ceil(pool_size / args.batch_size))
    return {
        "schema_version": 1,
        "dataset": args.dataset,
        "architecture": args.architecture,
        "split_seed": args.split_seed,
        "reference_seed": args.reference_seed,
        "pool_hash": pool_hash,
        "pool_size": pool_size,
        "num_classes": int(cfg["num_classes"]),
        "input_channels": int(cfg.get("input_channels", 3)),
        "optimizer": args.reference_optimizer,
        "lr": learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": epochs,
        "batch_size": args.batch_size,
        "steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": epochs * steps_per_epoch,
        "scheduler": "CosineAnnealingLR",
        "checkpoint_state_convention": (
            "pre_epoch model state before the epoch's outgoing optimizer updates; "
            "outgoing_lr is the LR used by those updates; the terminal post-training "
            "state is not included"
        ),
        "checkpoint_epochs": checkpoints,
        "init_hash": init_hash,
        "train_transform": "clean deterministic test transform (shared with D and retraining)",
        "source": code,
    }


def capture_rng_state(torch, loader_generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader_generator": loader_generator.get_state(),
    }


def restore_rng_state(torch, state: dict[str, Any], loader_generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    loader_generator.set_state(state["loader_generator"])


def validate_reference_manifest(manifest: dict[str, Any], expected_config_id: str,
                                reference_dir: Path, *, require_complete: bool) -> None:
    if manifest.get("config_id") != expected_config_id:
        raise RuntimeError(
            "reference directory contains a different immutable configuration. "
            f"Expected {expected_config_id}, found {manifest.get('config_id')}. "
            "Use a new --reference-dir; existing evidence is never silently overwritten."
        )
    if require_complete and manifest.get("status") != "complete":
        raise RuntimeError(
            f"reference trajectory is {manifest.get('status')!r}, not complete: {reference_dir}"
        )
    if manifest.get("status") == "complete":
        for item in manifest.get("checkpoints", []):
            path = reference_dir / item["file"]
            if not path.is_file():
                raise RuntimeError(f"reference checkpoint is missing: {path}")
            actual = sha256_file(path)
            if actual != item["sha256"]:
                raise RuntimeError(f"reference checkpoint hash mismatch: {path}")


def batchnorm_layer_names(nn, model) -> list[str]:
    batchnorm_base = nn.modules.batchnorm._BatchNorm
    return [name for name, module in model.named_modules() if isinstance(module, batchnorm_base)]


def enforce_batchnorm_policy(nn, model, policy: str) -> list[str]:
    names = batchnorm_layer_names(nn, model)
    if names and policy == "reject":
        preview = ", ".join(names[:4]) + (" ..." if len(names) > 4 else "")
        raise RuntimeError(
            "This architecture contains BatchNorm ("
            f"{preview}). Train-mode BatchNorm gradients depend on the other samples in "
            "the batch, so the manuscript's per-example D(alpha) decomposition is not "
            "defined without an extra convention. Re-run with "
            "--batchnorm-policy eval_frozen to score the explicit frozen-running-statistics "
            "surrogate, or use the BN-free LeNet control."
        )
    if names:
        print(
            "WARNING: BatchNorm detected. D(alpha) will be exact for the model in eval "
            "mode with frozen running statistics, NOT for train-mode gradients. This "
            "surrogate is recorded in every result and analyzed in a separate block.",
            file=sys.stderr,
        )
    return names


def prepare_initialization(torch, common, cfg, args, device: str,
                           reference_dir: Path) -> tuple[Path, str, Any]:
    init_path = reference_dir / "shared_initialization.pt"
    common.set_seed(args.reference_seed)
    model = common.build_model(
        cfg, cfg["num_classes"], device, architecture=args.architecture,
    )
    if init_path.exists():
        state = torch_load(torch, init_path, map_location="cpu")
        model.load_state_dict(state)
    else:
        atomic_torch_save(torch, cpu_state_dict(model), init_path)
    return init_path, sha256_file(init_path), model


def train_or_load_reference(torch, nn, DataLoader, common, cfg, trainset,
                            ordered_pool_indices: list[int], pool_hash: str,
                            args, device: str, code: dict[str, Any]) -> dict[str, Any]:
    reference_dir = Path(args.reference_dir).expanduser().resolve()
    reference_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = reference_dir / "reference_manifest.json"
    init_candidate_path = reference_dir / "shared_initialization.pt"
    if args.stage == "evaluate":
        if not manifest_path.is_file():
            raise RuntimeError(
                f"--stage evaluate requires a completed reference manifest: {manifest_path}"
            )
        if not init_candidate_path.is_file():
            raise RuntimeError(
                f"--stage evaluate requires the shared initialization: {init_candidate_path}"
            )
    pool_path = reference_dir / "pool_indices.json"
    if pool_path.exists():
        stored_pool = load_json(pool_path)
        if stored_pool.get("pool_hash") != pool_hash or stored_pool.get("global_indices") != ordered_pool_indices:
            raise RuntimeError(
                f"selection-pool mapping in {pool_path} disagrees with seed/data configuration"
            )
    else:
        atomic_write_json(pool_path, {
            "split_seed": args.split_seed,
            "pool_hash": pool_hash,
            "global_indices": ordered_pool_indices,
        })

    init_path, init_hash, model = prepare_initialization(
        torch, common, cfg, args, device, reference_dir,
    )
    bn_names = enforce_batchnorm_policy(nn, model, args.batchnorm_policy)
    epochs = resolve_epochs(args.reference_epochs, cfg)
    learning_rate = resolve_lr(args.reference_lr, args.reference_optimizer)
    desired_checkpoints = checkpoint_epochs(
        epochs, args.num_d_checkpoints, args.checkpoint_epochs,
    )
    source_for_config = {
        "git_commit": code.get("git_commit"),
        "source_sha256": code["source_sha256"],
    }
    config = reference_config(
        args, cfg, pool_hash, init_hash, epochs, learning_rate,
        desired_checkpoints, source_for_config,
    )
    config["has_batchnorm"] = bool(bn_names)
    config_id = sha256_json(config)
    latest_path = reference_dir / "reference_latest.pt"

    if manifest_path.exists():
        manifest = load_json(manifest_path)
        validate_reference_manifest(
            manifest, config_id, reference_dir,
            require_complete=args.stage == "evaluate",
        )
        if manifest.get("status") == "complete":
            return manifest
    else:
        manifest = {
            "schema_version": 1,
            "status": "running",
            "created_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "config_id": config_id,
            "config": config,
            "initialization": {"file": init_path.name, "sha256": init_hash},
            "pool_indices": {"file": pool_path.name, "pool_hash": pool_hash},
            "checkpoints": [],
            "progress": {"completed_epoch": 0, "train_loss": []},
        }
        atomic_write_json(manifest_path, manifest)

    optimizer = common.build_optimizer(
        model.parameters(), args.reference_optimizer, learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loader_generator = torch.Generator().manual_seed(args.reference_seed + 1701)
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        pin_memory=device.startswith("cuda"), num_workers=args.num_workers,
        generator=loader_generator,
    )
    if len(trainloader) != int(config["steps_per_epoch"]):
        raise RuntimeError(
            f"reference loader has {len(trainloader)} steps, manifest declares "
            f"{config['steps_per_epoch']}"
        )
    start_epoch = 1
    if latest_path.is_file():
        latest = torch_load(torch, latest_path, map_location=device)
        if latest.get("config_id") != config_id:
            raise RuntimeError("reference resume state has the wrong config_id")
        saved_epoch = int(latest["completed_epoch"])
        # The lightweight JSON progress may be newer than the periodic heavy
        # resume state. Truncate it to the model/optimizer state we can restore.
        manifest["progress"]["completed_epoch"] = saved_epoch
        for history_key in ("train_loss", "epoch_seconds"):
            if history_key in manifest["progress"]:
                manifest["progress"][history_key] = manifest["progress"][history_key][:saved_epoch]
        model.load_state_dict(latest["model"])
        optimizer.load_state_dict(latest["optimizer"])
        scheduler.load_state_dict(latest["scheduler"])
        restore_rng_state(torch, latest["rng"], loader_generator)
        start_epoch = saved_epoch + 1
        recorded_epochs = {int(item["epoch"]) for item in manifest.get("checkpoints", [])}
        for checkpoint_epoch in desired_checkpoints:
            if checkpoint_epoch > saved_epoch or checkpoint_epoch in recorded_epochs:
                continue
            checkpoint_path = reference_dir / f"checkpoint_epoch_{checkpoint_epoch:04d}.pt"
            if not checkpoint_path.is_file():
                raise RuntimeError(
                    "resume state passed a requested checkpoint, but its artifact is missing: "
                    f"{checkpoint_path}"
                )
            payload = torch_load(torch, checkpoint_path, map_location="cpu")
            if payload.get("config_id") != config_id or int(payload.get("epoch", -1)) != checkpoint_epoch:
                raise RuntimeError(f"orphan checkpoint has incompatible metadata: {checkpoint_path}")
            manifest["checkpoints"].append({
                "file": checkpoint_path.name,
                "epoch": checkpoint_epoch,
                "outgoing_lr": float(payload["outgoing_lr"]),
                "completed_epochs": int(payload["completed_epochs"]),
                "global_step": int(payload["global_step"]),
                "state_convention": payload["state_convention"],
                "sha256": sha256_file(checkpoint_path),
            })
        manifest["checkpoints"].sort(key=lambda item: int(item["epoch"]))
        print(f"Resuming reference at epoch {start_epoch}/{epochs}")
    elif manifest["progress"]["completed_epoch"]:
        if not latest_path.is_file():
            raise RuntimeError(
                f"reference manifest is partial but resume state is missing: {latest_path}"
            )
    else:
        common.set_seed(args.reference_seed)
        model.load_state_dict(torch_load(torch, init_path, map_location=device))

    criterion = nn.CrossEntropyLoss()
    checkpoint_by_epoch = {
        int(item["epoch"]): item for item in manifest.get("checkpoints", [])
    }
    for epoch in range(start_epoch, epochs + 1):
        lr_used = float(optimizer.param_groups[0]["lr"])
        if epoch in desired_checkpoints and epoch not in checkpoint_by_epoch:
            # Save theta immediately before the outgoing update block whose LR
            # weights its discrepancy. This avoids tagging a post-update state
            # with the LR that produced it and excludes the terminal theta_T.
            checkpoint_path = reference_dir / f"checkpoint_epoch_{epoch:04d}.pt"
            atomic_torch_save(torch, {
                "model": cpu_state_dict(model),
                "epoch": epoch,
                "completed_epochs": epoch - 1,
                "global_step": (epoch - 1) * int(config["steps_per_epoch"]),
                "outgoing_lr": lr_used,
                "state_convention": "pre_epoch_before_outgoing_updates",
                "config_id": config_id,
            }, checkpoint_path)
            checkpoint_record = {
                "file": checkpoint_path.name,
                "epoch": epoch,
                "completed_epochs": epoch - 1,
                "global_step": (epoch - 1) * int(config["steps_per_epoch"]),
                "outgoing_lr": lr_used,
                "state_convention": "pre_epoch_before_outgoing_updates",
                "sha256": sha256_file(checkpoint_path),
            }
            manifest["checkpoints"].append(checkpoint_record)
            manifest["checkpoints"].sort(key=lambda item: int(item["epoch"]))
            checkpoint_by_epoch[epoch] = checkpoint_record

        model.train()
        loss_sum = 0.0
        sample_count = 0
        started = time.perf_counter()
        for inputs, targets in trainloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * int(targets.numel())
            sample_count += int(targets.numel())
        scheduler.step()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        manifest["progress"]["completed_epoch"] = epoch
        manifest["progress"]["train_loss"].append(loss_sum / sample_count)
        manifest["progress"].setdefault("epoch_seconds", []).append(elapsed)

        should_resume_save = (
            epoch % args.reference_resume_every == 0
            or epoch in desired_checkpoints
            or epoch == epochs
        )
        if should_resume_save:
            atomic_torch_save(torch, {
                "config_id": config_id,
                "completed_epoch": epoch,
                "model": cpu_state_dict(model),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": capture_rng_state(torch, loader_generator),
            }, latest_path)
        manifest["updated_at_utc"] = utc_now()
        atomic_write_json(manifest_path, manifest)
        print(
            f"Reference epoch {epoch:03d}/{epochs}: "
            f"loss={loss_sum / sample_count:.5f}, lr={lr_used:.6g}, {elapsed:.1f}s"
        )

    actual_epochs = [int(item["epoch"]) for item in manifest["checkpoints"]]
    if actual_epochs != desired_checkpoints:
        raise RuntimeError(
            f"reference checkpoints {actual_epochs} do not match requested {desired_checkpoints}"
        )
    fingerprint_payload = {
        "config_id": config_id,
        "initialization_sha256": init_hash,
        "pool_hash": pool_hash,
        "checkpoints": [
            {key: item[key] for key in (
                "epoch", "completed_epochs", "global_step", "outgoing_lr",
                "state_convention", "sha256"
            )}
            for item in manifest["checkpoints"]
        ],
    }
    manifest["fingerprint"] = sha256_json(fingerprint_payload)
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = utc_now()
    manifest["updated_at_utc"] = utc_now()
    atomic_write_json(manifest_path, manifest)
    validate_reference_manifest(manifest, config_id, reference_dir, require_complete=True)
    return manifest


def checkpoint_state(torch, path: Path, device: str) -> dict[str, Any]:
    payload = torch_load(torch, path, map_location=device)
    if not isinstance(payload, dict) or "model" not in payload:
        raise RuntimeError(f"not a D-correlation reference checkpoint: {path}")
    return payload


def collect_pool_metadata(torch, nn, model, eval_loader, device: str,
                          *, with_losses: bool) -> tuple[list[int], list[float] | None]:
    labels: list[int] = []
    losses: list[float] = []
    criterion = nn.CrossEntropyLoss(reduction="none")
    model.eval()
    with torch.no_grad():
        for inputs, targets in eval_loader:
            labels.extend(int(value) for value in targets.tolist())
            if with_losses:
                inputs = inputs.to(device, non_blocking=True)
                targets_device = targets.to(device, non_blocking=True)
                batch_losses = criterion(model(inputs), targets_device)
                losses.extend(float(value) for value in batch_losses.detach().cpu().tolist())
    return labels, losses if with_losses else None


def sanitized_id(raw: str) -> str:
    clean = "".join(character if character.isalnum() or character in "-_" else "_" for character in raw)
    clean = clean.strip("_")
    if not clean:
        raise ValueError(f"subset id {raw!r} has no safe filename characters")
    return clean[:120]


def load_external_candidates(torch, path: Path, n_pool: int,
                             ordered_pool_indices: list[int]) -> list[dict[str, Any]]:
    payload = load_json(path)
    entries = payload.get("subsets") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError("external subset JSON must be a list or {'subsets': [...]} object")
    global_to_pool = {global_index: position for position, global_index in enumerate(ordered_pool_indices)}
    candidates = []
    for number, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"external subset entry {number} is not an object")
        if "indices_file" in entry:
            indices_path = Path(entry["indices_file"])
            if not indices_path.is_absolute():
                indices_path = path.parent / indices_path
            if indices_path.suffix.lower() in {".pt", ".pth"}:
                index_payload = torch_load(torch, indices_path, map_location="cpu")
            else:
                index_payload = load_json(indices_path)
            indices = index_payload.get("indices", index_payload) if isinstance(index_payload, dict) else index_payload
            if hasattr(indices, "tolist"):
                indices = indices.tolist()
        else:
            indices = entry.get("indices", entry.get("pool_indices"))
        if indices is None:
            raise ValueError(f"external subset entry {number} has no indices")
        index_space = entry.get("index_space", "pool")
        if index_space == "global":
            try:
                indices = [global_to_pool[int(value)] for value in indices]
            except KeyError as error:
                raise ValueError(f"global index {error.args[0]} is outside the selection pool") from error
        elif index_space != "pool":
            raise ValueError("external index_space must be 'pool' or 'global'")
        indices = canonical_indices(indices)
        if not indices or indices[-1] >= n_pool or indices[0] < 0:
            raise ValueError(f"external subset {number} contains an out-of-range pool index")
        budget = len(indices)
        fraction = float(entry.get("fraction", budget / n_pool))
        if not math.isclose(fraction * n_pool, budget, rel_tol=0.0, abs_tol=1e-7):
            raise ValueError(
                f"external subset {number}: int(fraction*n_pool) != number of indices"
            )
        candidates.append({
            "id": sanitized_id(str(entry.get("id", f"external_{number:03d}"))),
            "sampler": str(entry.get("sampler", "external")),
            "seed": int(entry.get("seed", -1)),
            "fraction": fraction,
            "budget": budget,
            "pool_indices": indices,
            "indices_hash": sha256_indices(indices),
            "source": str(path),
        })
    return candidates


def make_candidates(torch, args, labels: list[int], losses: list[float] | None,
                    ordered_pool_indices: list[int]) -> list[dict[str, Any]]:
    n_pool = len(labels)
    candidates: list[dict[str, Any]] = []
    seen_hashes: dict[str, str] = {}
    for fraction_number, fraction in enumerate(args.fractions):
        budget = int(float(fraction) * n_pool)
        if not 1 <= budget < n_pool:
            raise ValueError(
                f"fraction {fraction} produces budget {budget}; require 1 <= budget < {n_pool}"
            )
        for sampler_number, sampler in enumerate(args.samplers):
            for repeat in range(args.subsets_per_sampler):
                base_seed = (
                    args.subset_seed_base
                    + 100_003 * fraction_number
                    + 10_007 * sampler_number
                    + repeat
                )
                # Extremely rare random collisions, and deterministic easy/hard
                # collisions in tiny pools, are retried rather than double-counted.
                for attempt in range(100):
                    seed = base_seed + 104_729 * attempt
                    indices = generate_subset_indices(
                        labels, budget, sampler, seed, losses=losses,
                        loss_pool_multiplier=args.loss_pool_multiplier,
                    )
                    index_hash = sha256_indices(indices)
                    if index_hash not in seen_hashes:
                        break
                else:
                    raise RuntimeError(
                        f"could not generate a unique {sampler} subset after 100 attempts; "
                        "reduce --subsets-per-sampler or enlarge --loss-pool-multiplier"
                    )
                candidate_id = sanitized_id(
                    f"f{fraction:.6g}_{sampler}_{repeat:03d}_s{seed}"
                )
                candidate = {
                    "id": candidate_id,
                    "sampler": sampler,
                    "seed": seed,
                    "fraction": float(fraction),
                    "budget": budget,
                    "pool_indices": indices,
                    "indices_hash": index_hash,
                    "source": "generated from selection-pool information only",
                }
                candidates.append(candidate)
                seen_hashes[index_hash] = candidate_id

    if args.external_subsets:
        for candidate in load_external_candidates(
            torch, Path(args.external_subsets).expanduser().resolve(), n_pool,
            ordered_pool_indices,
        ):
            previous = seen_hashes.get(candidate["indices_hash"])
            if previous is not None:
                raise ValueError(
                    f"external subset {candidate['id']} duplicates {previous}; duplicate subsets "
                    "must not be counted twice in a correlation"
                )
            candidates.append(candidate)
            seen_hashes[candidate["indices_hash"]] = candidate["id"]

    if not candidates:
        raise ValueError("candidate panel is empty; specify --samplers or --external-subsets")
    ids = [candidate["id"] for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate ids must be unique")
    return candidates


def d_checkpoint_weights(manifest: dict[str, Any], weighting: str,
                         normalization: str) -> list[float]:
    checkpoints = sorted(manifest["checkpoints"], key=lambda item: int(item["epoch"]))
    if weighting == "uniform":
        weights = np.ones(len(checkpoints), dtype=np.float64)
    elif weighting == "learning_rate":
        weights = np.asarray(
            [float(item["outgoing_lr"]) for item in checkpoints], dtype=np.float64
        )
    else:
        raise ValueError(f"unknown D weighting: {weighting}")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
        raise ValueError("D checkpoint weights must be finite, non-negative, and not all zero")
    if normalization == "mean":
        weights = weights / weights.sum()
    elif normalization != "sum":
        raise ValueError(f"unknown D normalization: {normalization}")
    return [float(value) for value in weights]


def flat_trainable_parameters(model) -> list[Any]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def aggregate_gradient_mean(torch, nn, DataLoader, dataset, model, device: str,
                            batch_size: int, num_workers: int):
    params = flat_trainable_parameters(model)
    total_dim = sum(parameter.numel() for parameter in params)
    accumulator = torch.zeros(total_dim, dtype=torch.float64, device=device)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        pin_memory=device.startswith("cuda"), num_workers=num_workers,
    )
    model.eval()
    count = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)
        criterion(model(inputs), targets).backward()
        pieces = [
            (parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)).reshape(-1)
            for parameter in params
        ]
        accumulator.add_(torch.cat(pieces).double())
        count += int(targets.numel())
    model.zero_grad(set_to_none=True)
    if count != len(dataset):
        raise RuntimeError(f"gradient pass saw {count} examples, expected {len(dataset)}")
    return accumulator / float(count)


def score_candidates(torch, nn, DataLoader, Subset, common, cfg, trainset,
                     manifest: dict[str, Any], candidates: list[dict[str, Any]],
                     args, device: str, experiment_dir: Path) -> dict[str, dict[str, Any]]:
    clean_pool = common.scoring_subset(cfg, trainset, "test")
    model = common.build_model(
        cfg, cfg["num_classes"], device, architecture=args.architecture,
    )
    bn_names = enforce_batchnorm_policy(nn, model, args.batchnorm_policy)
    checkpoints = sorted(manifest["checkpoints"], key=lambda item: int(item["epoch"]))
    weights = d_checkpoint_weights(manifest, args.d_weighting, args.d_normalization)
    score_config = {
        "reference_fingerprint": manifest["fingerprint"],
        "definition": D_DEFINITION,
        "weighting": args.d_weighting,
        "normalization": args.d_normalization,
        "weights": weights,
        "scoring_transform": "test/clean deterministic transform",
        "model_mode": "eval_frozen_batchnorm" if bn_names else "eval_bn_free",
        "batchnorm_layers": bn_names,
        "gradient_parameters": "all trainable model parameters",
        "loss": "unweighted cross entropy; weight decay excluded",
        "trajectory_sampling": (
            "exact aggregate gradients at the declared pre-epoch checkpoints; "
            "K-checkpoint quadrature approximation rather than every optimizer update"
        ),
        "batch_size": args.d_batch_size,
    }
    score_id = sha256_json(score_config)
    score_job_id = sha256_json({
        "score_id": score_id,
        "candidate_hashes": {
            candidate["id"]: candidate["indices_hash"] for candidate in candidates
        },
    })
    partial_path = experiment_dir / "d_scores_progress.json"
    if partial_path.exists():
        progress = load_json(partial_path)
        if progress.get("score_job_id") != score_job_id:
            raise RuntimeError(
                f"existing D-score progress has a different configuration: {partial_path}"
            )
    else:
        progress = {
            "schema_version": 1,
            "status": "running",
            "score_id": score_id,
            "score_job_id": score_job_id,
            "config": score_config,
            "completed_checkpoint_sha256": [],
            "totals": {candidate["id"]: 0.0 for candidate in candidates},
            "contributions": {candidate["id"]: [] for candidate in candidates},
        }
        atomic_write_json(partial_path, progress)

    reference_dir = Path(args.reference_dir).expanduser().resolve()
    complete_hashes = set(progress["completed_checkpoint_sha256"])
    for checkpoint_number, (item, weight) in enumerate(zip(checkpoints, weights)):
        if item["sha256"] in complete_hashes:
            continue
        path = reference_dir / item["file"]
        payload = checkpoint_state(torch, path, device)
        model.load_state_dict(payload["model"])
        full_mean = aggregate_gradient_mean(
            torch, nn, DataLoader, clean_pool, model, device,
            args.d_batch_size, args.num_workers,
        )
        for candidate_number, candidate in enumerate(candidates):
            subset = Subset(clean_pool, candidate["pool_indices"])
            subset_mean = aggregate_gradient_mean(
                torch, nn, DataLoader, subset, model, device,
                args.d_batch_size, args.num_workers,
            )
            difference = subset_mean - full_mean
            raw = float(torch.dot(difference, difference).item())
            weighted = float(weight * raw)
            progress["totals"][candidate["id"]] += weighted
            progress["contributions"][candidate["id"]].append({
                "checkpoint_sha256": item["sha256"],
                "epoch": int(item["epoch"]),
                "global_step": int(item["global_step"]),
                "outgoing_lr": float(item["outgoing_lr"]),
                "state_convention": item["state_convention"],
                "weight": weight,
                "raw_squared_gradient_discrepancy": raw,
                "weighted_contribution": weighted,
            })
            print(
                f"D checkpoint {checkpoint_number + 1}/{len(checkpoints)}, "
                f"subset {candidate_number + 1}/{len(candidates)}: "
                f"{candidate['id']} contribution={weighted:.6g}"
            )
        progress["completed_checkpoint_sha256"].append(item["sha256"])
        progress["updated_at_utc"] = utc_now()
        atomic_write_json(partial_path, progress)

    if progress["completed_checkpoint_sha256"] != [item["sha256"] for item in checkpoints]:
        raise RuntimeError("D-score progress checkpoint order is inconsistent")
    progress["status"] = "complete"
    progress["completed_at_utc"] = utc_now()
    atomic_write_json(partial_path, progress)
    return {
        candidate["id"]: {
            "value": float(progress["totals"][candidate["id"]]),
            "contributions": progress["contributions"][candidate["id"]],
            "score_id": score_id,
            "score_job_id": score_job_id,
            "config": score_config,
        }
        for candidate in candidates
    }


def resolved_retrain_seeds(args) -> list[int]:
    if args.retrain_seeds:
        seeds = [int(seed) for seed in args.retrain_seeds]
        if len(seeds) != len(set(seeds)):
            raise ValueError("--retrain-seeds must be unique")
        return seeds
    if args.retrain_replicates < 1:
        raise ValueError("--retrain-replicates must be positive")
    return [args.retrain_seed_base + args.split_seed * 1000 + i for i in range(args.retrain_replicates)]


def experiment_config(args, manifest: dict[str, Any], cfg: dict[str, Any],
                      pool_hash: str, code: dict[str, Any]) -> dict[str, Any]:
    retrain_epochs = resolve_epochs(args.retrain_epochs, cfg)
    retrain_lr = resolve_lr(args.retrain_lr, args.retrain_optimizer)
    external_hash = None
    if args.external_subsets:
        external_hash = sha256_file(Path(args.external_subsets).expanduser().resolve())
    return {
        "schema_version": 1,
        "dataset": args.dataset,
        "architecture": args.architecture,
        "split_seed": args.split_seed,
        "pool_hash": pool_hash,
        "reference_fingerprint": manifest["fingerprint"],
        "candidate_panel": {
            "fractions": [float(value) for value in args.fractions],
            "samplers": list(args.samplers),
            "subsets_per_sampler": args.subsets_per_sampler,
            "subset_seed_base": args.subset_seed_base,
            "loss_pool_multiplier": args.loss_pool_multiplier,
            "external_subsets_sha256": external_hash,
        },
        "d_alpha": {
            "definition": D_DEFINITION,
            "weighting": args.d_weighting,
            "normalization": args.d_normalization,
            "scoring_transform": "test/clean deterministic transform",
            "batchnorm_policy": args.batchnorm_policy,
            "batch_size": args.d_batch_size,
        },
        "retraining": {
            "optimizer": args.retrain_optimizer,
            "lr": retrain_lr,
            "weight_decay": args.weight_decay,
            "epochs": retrain_epochs,
            "batch_size": args.batch_size,
            "steps_per_reference_epoch": int(manifest["config"]["steps_per_epoch"]),
            "total_optimizer_steps": int(manifest["config"]["total_optimizer_steps"]),
            "scheduler": "CosineAnnealingLR",
            "training_view": "clean deterministic test transform (same examples/loss view as D)",
            "requested_seeds": resolved_retrain_seeds(args),
            "shared_initialization_sha256": manifest["initialization"]["sha256"],
            "test_evaluations_per_fit": 1,
            "learning_algorithm_matched_to_reference": True,
        },
        "source": {
            "git_commit": code.get("git_commit"),
            "source_sha256": code["source_sha256"],
        },
    }


def candidate_manifest_path(output_root: Path, args, experiment_id: str) -> tuple[Path, Path]:
    experiment_dir = (
        output_root / args.dataset / args.architecture
        / f"split_{args.split_seed}" / f"experiment_{experiment_id[:16]}"
    )
    return experiment_dir, experiment_dir / "candidate_manifest.json"


def validate_candidate(candidate: dict[str, Any], n_pool: int) -> None:
    indices = canonical_indices(candidate["pool_indices"])
    if not indices or indices[0] < 0 or indices[-1] >= n_pool:
        raise ValueError(f"candidate {candidate.get('id')} has out-of-range indices")
    if len(indices) != int(candidate["budget"]):
        raise ValueError(f"candidate {candidate.get('id')} budget mismatch")
    if sha256_indices(indices) != candidate["indices_hash"]:
        raise ValueError(f"candidate {candidate.get('id')} hash mismatch")


def prepare_candidates(torch, nn, common, cfg, trainset, eval_loader,
                       ordered_pool_indices: list[int], manifest: dict[str, Any],
                       args, device: str, experiment_config_value: dict[str, Any],
                       experiment_dir: Path, candidate_path: Path) -> list[dict[str, Any]]:
    experiment_id = sha256_json(experiment_config_value)
    if candidate_path.exists():
        saved = load_json(candidate_path)
        if saved.get("experiment_id") != experiment_id:
            raise RuntimeError(f"candidate manifest configuration mismatch: {candidate_path}")
        candidates = saved["candidates"]
        for candidate in candidates:
            validate_candidate(candidate, len(trainset))
        return candidates

    final_item = max(manifest["checkpoints"], key=lambda item: int(item["epoch"]))
    final_payload = checkpoint_state(
        torch, Path(args.reference_dir).expanduser().resolve() / final_item["file"], device,
    )
    model = common.build_model(
        cfg, cfg["num_classes"], device, architecture=args.architecture,
    )
    model.load_state_dict(final_payload["model"])
    wants_losses = bool({"easy", "hard"}.intersection(args.samplers))
    labels, losses = collect_pool_metadata(
        torch, nn, model, eval_loader, device, with_losses=wants_losses,
    )
    if len(labels) != len(trainset):
        raise RuntimeError("clean scoring loader order/length does not match selection pool")
    candidates = make_candidates(torch, args, labels, losses, ordered_pool_indices)
    for candidate in candidates:
        validate_candidate(candidate, len(trainset))
    experiment_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(candidate_path, {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "experiment_config": experiment_config_value,
        "created_at_utc": utc_now(),
        "pool_hash": experiment_config_value["pool_hash"],
        "candidate_count": len(candidates),
        "candidates": candidates,
    })
    return candidates


def final_test_accuracy(torch, model, testloader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            predictions = model(inputs).argmax(dim=1)
            correct += int(predictions.eq(targets).sum().item())
            total += int(targets.numel())
    if total == 0:
        raise RuntimeError("test loader is empty")
    return 100.0 * correct / total


def train_subset_replicate(torch, nn, DataLoader, Subset, common, cfg, trainset,
                           testloader, candidate: dict[str, Any], seed: int,
                           args, device: str, init_path: Path,
                           model_output_path: Path | None,
                           steps_per_reference_epoch: int) -> dict[str, Any]:
    common.set_seed(seed)
    model = common.build_model(
        cfg, cfg["num_classes"], device, architecture=args.architecture,
    )
    model.load_state_dict(torch_load(torch, init_path, map_location=device))
    optimizer = common.build_optimizer(
        model.parameters(), args.retrain_optimizer,
        resolve_lr(args.retrain_lr, args.retrain_optimizer),
        weight_decay=args.weight_decay,
    )
    epochs = resolve_epochs(args.retrain_epochs, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed + 4001)
    loader = DataLoader(
        Subset(trainset, candidate["pool_indices"]),
        batch_size=args.batch_size, shuffle=True,
        pin_memory=device.startswith("cuda"), num_workers=args.num_workers,
        generator=generator,
    )
    history = []
    steps_per_reference_epoch = int(steps_per_reference_epoch)
    if steps_per_reference_epoch < 1:
        raise ValueError("steps_per_reference_epoch must be positive")
    iterator = iter(loader)
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        completed_steps = 0
        while completed_steps < steps_per_reference_epoch:
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                inputs, targets = next(iterator)
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(targets.numel())
            count += int(targets.numel())
            completed_steps += 1
        scheduler.step()
        history.append(total_loss / count)
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            print(
                f"Retrain {candidate['id']} seed={seed}: epoch {epoch}/{epochs}, "
                f"train_loss={history[-1]:.5f}, steps={completed_steps}"
            )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    # This is intentionally the first and only access to the official test set
    # for this fit. No checkpoint is chosen using test performance.
    accuracy = final_test_accuracy(torch, model, testloader, device)
    result = {
        "seed": seed,
        "test_accuracy_percent": accuracy,
        "test_evaluation_epoch": epochs,
        "test_evaluations": 1,
        "training_seconds": training_seconds,
        "optimizer_steps": epochs * steps_per_reference_epoch,
        "steps_per_reference_epoch": steps_per_reference_epoch,
        "train_loss_by_epoch": history,
    }
    if model_output_path is not None:
        atomic_torch_save(torch, cpu_state_dict(model), model_output_path)
        result["model_file"] = model_output_path.name
        result["model_sha256"] = sha256_file(model_output_path)
    return result


def initial_record(candidate: dict[str, Any], score: dict[str, Any], args,
                   manifest: dict[str, Any], experiment_id: str,
                   experiment_config_value: dict[str, Any], code: dict[str, Any],
                   runtime: dict[str, Any]) -> dict[str, Any]:
    retrain_cfg = experiment_config_value["retraining"]
    record_id = sha256_json({
        "experiment_id": experiment_id,
        "subset_indices_hash": candidate["indices_hash"],
    })
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "record_id": record_id,
        "record_config_id": sha256_json({
            "experiment_id": experiment_id,
            "candidate": candidate,
            "d_score_id": score["score_id"],
        }),
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "dataset": args.dataset,
        "architecture": args.architecture,
        "split_seed": args.split_seed,
        "pool_hash": experiment_config_value["pool_hash"],
        "pool_size": int(manifest["config"]["pool_size"]),
        "subset": copy.deepcopy(candidate),
        "reference": {
            "directory": str(Path(args.reference_dir).expanduser().resolve()),
            "config_id": manifest["config_id"],
            "fingerprint": manifest["fingerprint"],
            "checkpoint_sha256": [item["sha256"] for item in manifest["checkpoints"]],
        },
        "d_alpha": {
            "value": score["value"],
            "subset_indices_hash": candidate["indices_hash"],
            "score_id": score["score_id"],
            "score_job_id": score["score_job_id"],
            "definition": score["config"]["definition"],
            "weighting": score["config"]["weighting"],
            "normalization": score["config"]["normalization"],
            "checkpoint_weights": score["config"]["weights"],
            "scoring_transform": score["config"]["scoring_transform"],
            "model_mode": score["config"]["model_mode"],
            "batchnorm_layers": score["config"]["batchnorm_layers"],
            "interpretation": (
                "exact aggregate-gradient discrepancy at the recorded checkpoints for the "
                "deterministic eval-mode model; with BatchNorm this is a frozen-running-"
                "statistics surrogate, not the batch-dependent train-mode gradient"
            ),
            "gradient_parameters": score["config"]["gradient_parameters"],
            "loss": score["config"]["loss"],
            "trajectory_sampling": score["config"]["trajectory_sampling"],
            "per_checkpoint": score["contributions"],
        },
        "retraining": {
            "subset_indices_hash": candidate["indices_hash"],
            "optimizer": retrain_cfg["optimizer"],
            "lr": retrain_cfg["lr"],
            "weight_decay": retrain_cfg["weight_decay"],
            "epochs": retrain_cfg["epochs"],
            "batch_size": retrain_cfg["batch_size"],
            "steps_per_reference_epoch": retrain_cfg["steps_per_reference_epoch"],
            "total_optimizer_steps": retrain_cfg["total_optimizer_steps"],
            "scheduler": retrain_cfg["scheduler"],
            "augmentation": retrain_cfg["training_view"],
            "init_hash": retrain_cfg["shared_initialization_sha256"],
            "requested_seeds": retrain_cfg["requested_seeds"],
            "test_protocol": "official test set evaluated once after final epoch; no best-epoch selection",
            "replicates": [],
        },
        "provenance": {"source": code, "runtime": runtime},
    }
    record["analysis_block_id"] = sha256_json(analysis_block_payload(record))
    return record


def run_candidate_retraining(torch, nn, DataLoader, Subset, common, cfg, trainset,
                             testloader, candidates: list[dict[str, Any]],
                             scores: dict[str, dict[str, Any]], manifest: dict[str, Any],
                             args, device: str, experiment_dir: Path,
                             experiment_id: str, experiment_config_value: dict[str, Any],
                             code: dict[str, Any], runtime: dict[str, Any]) -> None:
    records_dir = experiment_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    init_path = Path(args.reference_dir).expanduser().resolve() / manifest["initialization"]["file"]
    if sha256_file(init_path) != manifest["initialization"]["sha256"]:
        raise RuntimeError(f"shared initialization hash mismatch: {init_path}")

    requested_seeds = resolved_retrain_seeds(args)
    for number, candidate in enumerate(candidates):
        candidate_dir = records_dir / f"{sanitized_id(candidate['id'])}_{candidate['indices_hash'][:10]}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        record_path = candidate_dir / "record.json"
        expected = initial_record(
            candidate, scores[candidate["id"]], args, manifest, experiment_id,
            experiment_config_value, code, runtime,
        )
        if record_path.exists():
            record = load_json(record_path)
            if record.get("record_config_id") != expected["record_config_id"]:
                raise RuntimeError(f"existing subset record has a different configuration: {record_path}")
            if record.get("status") == "complete":
                completed = {int(item["seed"]) for item in record["retraining"]["replicates"]}
                if completed == set(requested_seeds):
                    print(f"Skipping completed subset {number + 1}/{len(candidates)}: {candidate['id']}")
                    continue
                raise RuntimeError(f"completed record has the wrong replicate seeds: {record_path}")
        else:
            record = expected
            atomic_write_json(record_path, record)

        completed_seeds = {int(item["seed"]) for item in record["retraining"]["replicates"]}
        for seed in requested_seeds:
            if seed in completed_seeds:
                continue
            model_path = candidate_dir / f"model_seed_{seed}.pt" if args.save_models else None
            replicate = train_subset_replicate(
                torch, nn, DataLoader, Subset, common, cfg, trainset, testloader,
                candidate, seed, args, device, init_path, model_path,
                experiment_config_value["retraining"]["steps_per_reference_epoch"],
            )
            record["retraining"]["replicates"].append(replicate)
            record["retraining"]["replicates"].sort(key=lambda item: int(item["seed"]))
            record["updated_at_utc"] = utc_now()
            atomic_write_json(record_path, record)
        actual_seeds = {int(item["seed"]) for item in record["retraining"]["replicates"]}
        if actual_seeds != set(requested_seeds):
            raise RuntimeError(f"retraining replicate set is incomplete for {candidate['id']}")
        record["status"] = "complete"
        record["completed_at_utc"] = utc_now()
        record["updated_at_utc"] = utc_now()
        atomic_write_json(record_path, record)
        accuracies = [item["test_accuracy_percent"] for item in record["retraining"]["replicates"]]
        print(
            f"Completed subset {number + 1}/{len(candidates)}: {candidate['id']}, "
            f"D={record['d_alpha']['value']:.6g}, "
            f"accuracy={np.mean(accuracies):.3f}% +/- {np.std(accuracies):.3f}"
        )


def finalize_experiment(experiment_dir: Path, experiment_id: str,
                        experiment_config_value: dict[str, Any],
                        candidates: list[dict[str, Any]]) -> None:
    record_paths = sorted((experiment_dir / "records").glob("*/record.json"))
    records_by_hash = {}
    for path in record_paths:
        record = load_json(path)
        validate_record(record)
        subset_hash = record["subset"]["indices_hash"]
        if subset_hash in records_by_hash:
            raise RuntimeError(f"duplicate completed record for subset hash {subset_hash}")
        records_by_hash[subset_hash] = path
    expected_hashes = {candidate["indices_hash"] for candidate in candidates}
    if set(records_by_hash) != expected_hashes:
        missing = expected_hashes - set(records_by_hash)
        extra = set(records_by_hash) - expected_hashes
        raise RuntimeError(f"record set mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    summary = {
        "schema_version": 1,
        "status": "complete",
        "experiment_id": experiment_id,
        "experiment_config": experiment_config_value,
        "completed_at_utc": utc_now(),
        "candidate_count": len(candidates),
        "records": [
            {
                "file": str(path.relative_to(experiment_dir)),
                "sha256": sha256_file(path),
                "subset_indices_hash": subset_hash,
            }
            for subset_hash, path in sorted(records_by_hash.items())
        ],
    }
    atomic_write_json(experiment_dir / "experiment_manifest.json", summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--stage", choices=("reference", "evaluate", "all"), default="all")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument(
        "--architecture", choices=ARCHITECTURES, default=None,
        help="Default: lenet for MNIST, resnet18 otherwise",
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--reference-dir", default=None,
        help="Shared trajectory directory. Default: OUTPUT/references/DATASET/ARCH/split_SEED/refseed_SEED",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--reference-seed", type=int, default=None)

    reference = parser.add_argument_group("reference trajectory")
    reference.add_argument("--reference-optimizer", choices=("sgd", "adam", "adamw"), default="sgd")
    reference.add_argument("--reference-lr", type=float, default=None)
    reference.add_argument("--reference-epochs", type=int, default=0, help="0 uses the dataset default")
    reference.add_argument("--num-d-checkpoints", type=int, default=5)
    reference.add_argument("--checkpoint-epochs", nargs="+", type=int, default=None)
    reference.add_argument("--reference-resume-every", type=int, default=5)
    reference.add_argument("--weight-decay", type=float, default=5e-4)

    panel = parser.add_argument_group("candidate subset panel")
    panel.add_argument("--fractions", nargs="+", type=float, default=[0.1, 0.2, 0.4])
    panel.add_argument("--samplers", nargs="*", choices=SAMPLERS,
                       default=["uniform", "stratified", "easy", "hard"])
    panel.add_argument("--subsets-per-sampler", type=int, default=1)
    panel.add_argument("--subset-seed-base", type=int, default=730_001)
    panel.add_argument("--loss-pool-multiplier", type=float, default=2.0)
    panel.add_argument(
        "--external-subsets", default=None,
        help="Optional JSON candidate list with pool- or dataset-global indices",
    )

    scoring = parser.add_argument_group("D(alpha) scoring")
    scoring.add_argument("--d-weighting", choices=("uniform", "learning_rate"), default="uniform")
    scoring.add_argument(
        "--d-normalization", choices=("sum", "mean"), default="sum",
        help="sum matches the manuscript's stacked-feature D; mean only changes scale for fixed checkpoints",
    )
    scoring.add_argument("--d-batch-size", type=int, default=256)
    scoring.add_argument(
        "--batchnorm-policy", choices=("reject", "eval_frozen"), default="reject",
        help="Train-mode BN is batch-dependent. 'reject' is safest; 'eval_frozen' explicitly scores the frozen-BN surrogate.",
    )

    retraining = parser.add_argument_group("subset retraining")
    retraining.add_argument("--retrain-optimizer", choices=("sgd", "adam", "adamw"), default="sgd")
    retraining.add_argument("--retrain-lr", type=float, default=None)
    retraining.add_argument("--retrain-epochs", type=int, default=0, help="0 uses the dataset default")
    retraining.add_argument("--retrain-replicates", type=int, default=1)
    retraining.add_argument("--retrain-seed-base", type=int, default=910_000)
    retraining.add_argument("--retrain-seeds", nargs="+", type=int, default=None)
    retraining.add_argument("--save-models", action="store_true")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--batch-size", type=int, default=128)
    runtime.add_argument("--num-workers", type=int, default=4)
    runtime.add_argument("--device", default="auto", help="auto, cpu, cuda, or e.g. cuda:1")
    args = parser.parse_args()

    if args.architecture is None:
        args.architecture = "lenet" if args.dataset == "mnist" else "resnet18"
    if args.reference_seed is None:
        args.reference_seed = args.split_seed
    if not args.fractions or any(not 0.0 < value < 1.0 for value in args.fractions):
        parser.error("every --fractions value must lie strictly between 0 and 1")
    if len(set(args.fractions)) != len(args.fractions):
        parser.error("--fractions values must be unique")
    if len(set(args.samplers)) != len(args.samplers):
        parser.error("--samplers values must be unique")
    if args.subsets_per_sampler < 1 and args.samplers:
        parser.error("--subsets-per-sampler must be positive")
    if args.loss_pool_multiplier < 1.0:
        parser.error("--loss-pool-multiplier must be at least 1")
    if args.reference_resume_every < 1:
        parser.error("--reference-resume-every must be positive")
    if args.batch_size < 1 or args.d_batch_size < 1:
        parser.error("batch sizes must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.weight_decay < 0.0:
        parser.error("--weight-decay cannot be negative")
    if args.stage != "reference" and not args.samplers and not args.external_subsets:
        parser.error("evaluation needs --samplers and/or --external-subsets")
    return args


def main() -> None:
    args = parse_args()
    torch, nn, DataLoader, Subset, common = load_runtime()
    device = choose_device(torch, args.device)
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.reference_dir is None:
        args.reference_dir = str(
            output_root / "references" / args.dataset / args.architecture
            / f"split_{args.split_seed}" / f"refseed_{args.reference_seed}"
        )
    code = source_provenance()
    runtime = runtime_provenance(torch, device)
    cfg, trainset, eval_loader, testloader, ordered_pool_indices, pool_hash = build_data(
        common, args, scoring_transform="test",
    )
    validate_learning_algorithm_alignment(args, cfg)
    clean_trainset = common.scoring_subset(cfg, trainset, "test")
    manifest = train_or_load_reference(
        torch, nn, DataLoader, common, cfg, clean_trainset,
        ordered_pool_indices, pool_hash, args, device, code,
    )
    print(
        f"Reference ready: fingerprint={manifest['fingerprint'][:16]}, "
        f"checkpoints={len(manifest['checkpoints'])}, directory={args.reference_dir}"
    )
    if args.stage == "reference":
        return

    config = experiment_config(args, manifest, cfg, pool_hash, code)
    experiment_id = sha256_json(config)
    experiment_dir, candidate_path = candidate_manifest_path(output_root, args, experiment_id)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    candidates = prepare_candidates(
        torch, nn, common, cfg, trainset, eval_loader, ordered_pool_indices,
        manifest, args, device, config, experiment_dir, candidate_path,
    )
    print(f"Candidate panel: {len(candidates)} unique subsets in {experiment_dir}")
    scores = score_candidates(
        torch, nn, DataLoader, Subset, common, cfg, trainset, manifest,
        candidates, args, device, experiment_dir,
    )
    run_candidate_retraining(
        torch, nn, DataLoader, Subset, common, cfg, clean_trainset, testloader,
        candidates, scores, manifest, args, device, experiment_dir,
        experiment_id, config, code, runtime,
    )
    finalize_experiment(experiment_dir, experiment_id, config, candidates)
    print(
        f"Experiment complete: {len(candidates)} paired D/accuracy records. "
        f"Analyze with scripts/analyze/analyze_d_correlation.py --input {output_root} "
        f"--output-dir {output_root / 'analysis'}"
    )


if __name__ == "__main__":
    main()
