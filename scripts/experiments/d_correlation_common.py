"""Pure utilities shared by the D(alpha) correlation runner and analyzer.

This module deliberately has no PyTorch, torchvision, scipy, or matplotlib
imports.  Keeping record validation and the statistical core lightweight makes
it possible to audit and unit-test the pairing logic on login nodes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1


def canonical_json(value: Any) -> str:
    """Return a stable JSON representation suitable for hashing."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: os.PathLike[str] | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_indices(indices: Iterable[int]) -> list[int]:
    """Validate a subset and return sorted, unique integer indices."""

    values = [int(i) for i in indices]
    if len(values) != len(set(values)):
        raise ValueError("subset indices must be unique")
    return sorted(values)


def sha256_indices(indices: Iterable[int]) -> str:
    values = canonical_indices(indices)
    array = np.asarray(values, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(b"d-alpha-indices-v1\0")
    digest.update(np.asarray([array.size], dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def sha256_ordered_indices(indices: Iterable[int]) -> str:
    """Hash an ordered index mapping (unlike subset hashes, order matters)."""

    values = [int(i) for i in indices]
    if len(values) != len(set(values)):
        raise ValueError("ordered indices must be unique")
    array = np.asarray(values, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(b"d-alpha-ordered-indices-v1\0")
    digest.update(np.asarray([array.size], dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_write_json(path: os.PathLike[str] | str, value: Any) -> None:
    """Atomically replace ``path`` with an indented, deterministic JSON file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_json(path: os.PathLike[str] | str) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def largest_remainder_quotas(labels: Sequence[int], budget: int) -> dict[int, int]:
    """Allocate an exact proportional class budget by largest remainder."""

    y = np.asarray(labels, dtype=np.int64)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional sequence")
    if not 1 <= int(budget) <= y.size:
        raise ValueError(f"budget must be in [1, {y.size}], got {budget}")
    classes, counts = np.unique(y, return_counts=True)
    raw = counts.astype(np.float64) * (float(budget) / float(y.size))
    quota = np.floor(raw).astype(np.int64)
    remaining = int(budget - quota.sum())
    # Stable class-id tie breaking makes this reproducible across NumPy versions.
    order = np.lexsort((classes, -(raw - quota)))
    quota[order[:remaining]] += 1
    return {int(c): int(q) for c, q in zip(classes, quota)}


def generate_subset_indices(
    labels: Sequence[int],
    budget: int,
    sampler: str,
    seed: int,
    *,
    losses: Sequence[float] | None = None,
    loss_pool_multiplier: float = 2.0,
) -> list[int]:
    """Generate pool-relative indices without looking at validation/test data.

    ``easy`` and ``hard`` sample uniformly from a low- or high-reference-loss
    candidate pool rather than deterministically taking the same top-k set on
    every repeat.  The reference loss must have been measured on the clean
    selection pool at the final reference checkpoint.
    """

    y = np.asarray(labels, dtype=np.int64)
    n = int(y.size)
    budget = int(budget)
    if y.ndim != 1 or n == 0:
        raise ValueError("labels must be a non-empty one-dimensional sequence")
    if not 1 <= budget < n:
        raise ValueError(f"budget must be in [1, {n - 1}], got {budget}")
    rng = np.random.default_rng(int(seed))

    if sampler == "uniform":
        chosen = rng.choice(n, size=budget, replace=False)
    elif sampler == "stratified":
        pieces = []
        for class_id, quota in largest_remainder_quotas(y, budget).items():
            if quota:
                class_positions = np.flatnonzero(y == class_id)
                pieces.append(rng.choice(class_positions, size=quota, replace=False))
        chosen = np.concatenate(pieces)
        rng.shuffle(chosen)
    elif sampler in {"easy", "hard"}:
        if losses is None:
            raise ValueError(f"sampler={sampler!r} requires reference losses")
        loss = np.asarray(losses, dtype=np.float64)
        if loss.shape != (n,) or not np.all(np.isfinite(loss)):
            raise ValueError(f"losses must contain {n} finite values")
        if loss_pool_multiplier < 1.0:
            raise ValueError("loss_pool_multiplier must be at least 1")
        candidate_size = min(n, max(budget, int(math.ceil(loss_pool_multiplier * budget))))
        # mergesort makes ties deterministic before the seeded sample.
        order = np.argsort(loss, kind="mergesort")
        candidates = order[:candidate_size] if sampler == "easy" else order[-candidate_size:]
        chosen = rng.choice(candidates, size=budget, replace=False)
    else:
        raise ValueError(
            f"unknown sampler {sampler!r}; choose uniform, stratified, easy, or hard"
        )

    return canonical_indices(chosen.tolist())


def average_ranks(values: Sequence[float]) -> np.ndarray:
    """One-based average ranks with exact tie handling."""

    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if not np.all(np.isfinite(x)):
        raise ValueError("values must be finite")
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    start = 0
    while start < x.size:
        stop = start + 1
        while stop < x.size and x[order[stop]] == x[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size < 2:
        return float("nan")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    return float(np.dot(a, b) / denom) if denom > 0.0 else float("nan")


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson(average_ranks(x), average_ranks(y))


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall's tau-b, including ties in either coordinate."""

    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size < 2:
        return float("nan")
    concordant = discordant = ties_x = ties_y = 0
    for i in range(a.size - 1):
        dx = np.sign(a[i + 1 :] - a[i])
        dy = np.sign(b[i + 1 :] - b[i])
        concordant += int(np.sum((dx * dy) > 0))
        discordant += int(np.sum((dx * dy) < 0))
        ties_x += int(np.sum((dx == 0) & (dy != 0)))
        ties_y += int(np.sum((dy == 0) & (dx != 0)))
    numerator = concordant - discordant
    denom = math.sqrt(
        (concordant + discordant + ties_x)
        * (concordant + discordant + ties_y)
    )
    return float(numerator / denom) if denom > 0.0 else float("nan")


def paired_bootstrap(
    x: Sequence[float],
    y: Sequence[float],
    statistic,
    *,
    n_bootstrap: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Bootstrap paired rows; x and y are always resampled together."""

    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("x and y must be equally sized one-dimensional arrays")
    estimate = float(statistic(a, b))
    rng = np.random.default_rng(int(seed))
    draws = []
    for _ in range(int(n_bootstrap)):
        take = rng.integers(0, a.size, size=a.size)
        value = float(statistic(a[take], b[take]))
        if math.isfinite(value):
            draws.append(value)
    if not draws:
        return {
            "estimate": estimate,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "probability_negative": float("nan"),
            "valid_bootstrap": 0,
        }
    alpha = 0.5 * (1.0 - confidence)
    values = np.asarray(draws)
    return {
        "estimate": estimate,
        "ci_low": float(np.quantile(values, alpha)),
        "ci_high": float(np.quantile(values, 1.0 - alpha)),
        "probability_negative": float(np.mean(values < 0.0)),
        "valid_bootstrap": int(values.size),
    }


def analysis_block_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    """Nuisance conditions that must be fixed within a correlation block."""

    reference = record["reference"]
    d_alpha = record["d_alpha"]
    retraining = record["retraining"]
    subset = record["subset"]
    return {
        "dataset": record["dataset"],
        "architecture": record["architecture"],
        "split_seed": int(record["split_seed"]),
        "pool_hash": record["pool_hash"],
        "fraction": float(subset["fraction"]),
        "budget": int(subset["budget"]),
        "reference_fingerprint": reference["fingerprint"],
        "d_score_id": d_alpha["score_id"],
        "d_definition": d_alpha["definition"],
        "d_weighting": d_alpha["weighting"],
        "d_normalization": d_alpha["normalization"],
        "scoring_transform": d_alpha["scoring_transform"],
        "model_mode": d_alpha["model_mode"],
        "retrain_optimizer": retraining["optimizer"],
        "retrain_lr": float(retraining["lr"]),
        "retrain_epochs": int(retraining["epochs"]),
        "retrain_batch_size": int(retraining["batch_size"]),
        "retrain_steps_per_reference_epoch": int(retraining["steps_per_reference_epoch"]),
        "retrain_total_optimizer_steps": int(retraining["total_optimizer_steps"]),
        "retrain_weight_decay": float(retraining["weight_decay"]),
        "retrain_scheduler": retraining["scheduler"],
        "retrain_augmentation": retraining["augmentation"],
        "retrain_seeds": [int(seed) for seed in retraining["requested_seeds"]],
        "retrain_init_hash": retraining["init_hash"],
    }


def validate_record(record: Mapping[str, Any]) -> None:
    """Reject incomplete, unpaired, or internally inconsistent records."""

    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version={record.get('schema_version')!r}")
    if record.get("status") != "complete":
        raise ValueError(f"record status is {record.get('status')!r}, not 'complete'")
    subset = record.get("subset", {})
    indices = canonical_indices(subset.get("pool_indices", []))
    pool_size = int(record.get("pool_size", -1))
    if pool_size < 2:
        raise ValueError("record has no valid pool_size")
    if not indices or indices[0] < 0 or indices[-1] >= pool_size:
        raise ValueError("subset contains an out-of-range pool index")
    if len(indices) != int(subset.get("budget", -1)):
        raise ValueError("subset budget does not match the number of indices")
    fraction_product = float(subset.get("fraction", -1.0)) * pool_size
    if int(fraction_product) != len(indices) and not math.isclose(
        fraction_product, len(indices), rel_tol=0.0, abs_tol=1e-7
    ):
        raise ValueError("subset fraction, pool_size, and budget are inconsistent")
    if sha256_indices(indices) != subset.get("indices_hash"):
        raise ValueError("subset indices_hash does not match pool_indices")
    d_alpha = record.get("d_alpha", {})
    if d_alpha.get("subset_indices_hash") != subset.get("indices_hash"):
        raise ValueError("D(alpha) was not paired with this subset hash")
    value = float(d_alpha.get("value", float("nan")))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("D(alpha) must be finite and non-negative")
    contributions = d_alpha.get("per_checkpoint")
    if contributions is not None:
        contribution_sum = sum(float(item["weighted_contribution"]) for item in contributions)
        if not math.isclose(value, contribution_sum, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("D(alpha) does not equal its checkpoint contributions")
    retraining = record.get("retraining", {})
    if retraining.get("subset_indices_hash") != subset.get("indices_hash"):
        raise ValueError("retraining was not paired with this subset hash")
    replicates = retraining.get("replicates", [])
    if not replicates:
        raise ValueError("record has no completed retraining replicate")
    seeds = set()
    for replicate in replicates:
        seed = int(replicate["seed"])
        if seed in seeds:
            raise ValueError(f"duplicate retraining seed {seed}")
        seeds.add(seed)
        accuracy = float(replicate["test_accuracy_percent"])
        if not 0.0 <= accuracy <= 100.0:
            raise ValueError(f"invalid test accuracy {accuracy}")
        if int(replicate.get("optimizer_steps", -1)) != int(retraining["total_optimizer_steps"]):
            raise ValueError("retraining replicate used the wrong number of optimizer steps")
        if int(replicate.get("test_evaluations", -1)) != 1:
            raise ValueError("each retraining replicate must evaluate the test set exactly once")
    requested_seeds = {int(seed) for seed in retraining.get("requested_seeds", [])}
    if seeds != requested_seeds:
        raise ValueError("completed retraining seeds do not match requested_seeds")
    expected = sha256_json(analysis_block_payload(record))
    if record.get("analysis_block_id") != expected:
        raise ValueError("analysis_block_id does not match the recorded protocol")


def flatten_record(record: Mapping[str, Any], source: str = "") -> dict[str, Any]:
    validate_record(record)
    replicates = record["retraining"]["replicates"]
    accuracies = np.asarray(
        [float(item["test_accuracy_percent"]) for item in replicates], dtype=np.float64
    )
    return {
        "source": source,
        "record_id": record["record_id"],
        "analysis_block_id": record["analysis_block_id"],
        "dataset": record["dataset"],
        "architecture": record["architecture"],
        "split_seed": int(record["split_seed"]),
        "fraction": float(record["subset"]["fraction"]),
        "budget": int(record["subset"]["budget"]),
        "sampler": record["subset"]["sampler"],
        "subset_seed": int(record["subset"]["seed"]),
        "subset_id": record["subset"]["id"],
        "subset_hash": record["subset"]["indices_hash"],
        "d_alpha": float(record["d_alpha"]["value"]),
        "accuracy_mean": float(accuracies.mean()),
        "accuracy_sd": float(accuracies.std(ddof=1)) if accuracies.size > 1 else 0.0,
        "accuracy_replicates": int(accuracies.size),
        "accuracy_values": accuracies.tolist(),
        "reference_fingerprint": record["reference"]["fingerprint"],
    }
