"""On-disk gradient-feature store bridging LESS and the selector.

LESS's Step 2 dumps projected Adam-LoRA gradient features per checkpoint and its
Step 3 dumps mean validation gradients per target subtask. This module defines a
small, explicit on-disk layout so the selector never depends on LESS's internal
file conventions or imports LESS:

    <root>/
      manifest.json                 # n, weights, checkpoint names, provenance hash
      features/<ckpt>.pt            # [n, d] float32/64, one file per checkpoint
      val/<task>/<subtask>/<ckpt>.pt  # [d] mean validation gradient
      pool.jsonl                    # (optional) one JSON record per pool example

``FeatureStore.load`` / ``make_synthetic`` let the whole pipeline run in tests
with no GPU and no dataset; ``load_less_dumps`` (Phase 0) will point the same
objects at real LESS outputs. The provenance hash records the exact feature
bytes so a selection can never silently mix checkpoints or pool orders.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from .features import TrajectoryFeatures

MANIFEST_NAME = "manifest.json"

#: Subtask counts used by LESS ``matching.py`` to reshape flattened validation
#: gradients (``authoritative for the released data layout``).
N_SUBTASKS = {"mmlu": 57, "bbh": 27, "tydiqa": 9}


def _provenance(root: str, weights: Sequence[float], normalize: str) -> str:
    """Hash every stored feature/gradient ``.pt`` byte plus weights/normalize.

    Walks ``features/`` and ``val/`` so tampering with any checkpoint, task or
    subtask file is detected on load, independent of file ordering.
    """
    h = hashlib.sha256()
    h.update(normalize.encode())
    for w in weights:
        h.update(repr(float(w)).encode())
        h.update(b"\x00")
    tracked = []
    for sub in ("features", "val"):
        base = os.path.join(root, sub)
        for dirpath, _, files in os.walk(base):
            for f in files:
                if f.endswith(".pt"):
                    tracked.append(os.path.join(dirpath, f))
    for path in sorted(tracked):
        st = os.stat(path)
        rel = os.path.relpath(path, root)
        h.update(f"{rel}\t{st.st_size}\t{st.st_mtime_ns}".encode())
        h.update(b"\n")
    return h.hexdigest()


@dataclass
class FeatureStore:
    """Trajectory feature store plus (optional) validation gradients and pool."""

    n: int
    weights: List[float]
    checkpoint_names: List[str]
    train_features: List[torch.Tensor]
    validation: Dict[str, List[List[torch.Tensor]]] = field(default_factory=dict)
    normalize: str = "l2"
    provenance: str = ""
    pool: Optional[List[dict]] = None

    def build_features(self) -> TrajectoryFeatures:
        return TrajectoryFeatures(self.train_features, weights=self.weights,
                                  normalize=self.normalize)

    def subtask_targets(self, task: str) -> List[List[torch.Tensor]]:
        if task not in self.validation:
            raise KeyError(f"no validation gradients for task {task!r}")
        return self.validation[task]

    # -- persistence ------------------------------------------------------
    def save(self, root: str) -> str:
        os.makedirs(os.path.join(root, "features"), exist_ok=True)
        files = []
        for name, feat in zip(self.checkpoint_names, self.train_features):
            path = os.path.join(root, "features", f"{name}.pt")
            torch.save(feat.to(torch.float32), path)
            files.append(path)
        for task, subtasks in self.validation.items():
            for s, sub in enumerate(subtasks):
                subdir = os.path.join(root, "val", task, f"subtask{s}")
                os.makedirs(subdir, exist_ok=True)
                for name, vec in zip(self.checkpoint_names, sub):
                    torch.save(vec.to(torch.float32), os.path.join(subdir, f"{name}.pt"))
        if self.pool is not None:
            with open(os.path.join(root, "pool.jsonl"), "w") as fh:
                for rec in self.pool:
                    fh.write(json.dumps(rec) + "\n")
        manifest = {
            "n": self.n,
            "weights": self.weights,
            "checkpoint_names": self.checkpoint_names,
            "normalize": self.normalize,
            "validation_tasks": sorted(self.validation),
            "provenance": self.provenance or _provenance(root, self.weights, self.normalize),
        }
        with open(os.path.join(root, MANIFEST_NAME), "w") as fh:
            json.dump(manifest, fh, indent=2)
        return manifest["provenance"]

    @classmethod
    def load(cls, root: str, device: str = "cpu") -> "FeatureStore":
        with open(os.path.join(root, MANIFEST_NAME)) as fh:
            manifest = json.load(fh)
        names = manifest["checkpoint_names"]
        feats = [torch.load(os.path.join(root, "features", f"{name}.pt"),
                            map_location=device).to(torch.float64) for name in names]
        validation: Dict[str, List[List[torch.Tensor]]] = {}
        for task in manifest.get("validation_tasks", []):
            task_dir = os.path.join(root, "val", task)
            subdirs = sorted(os.listdir(task_dir),
                             key=lambda s: int(s.replace("subtask", "")))
            validation[task] = [
                [torch.load(os.path.join(task_dir, s, f"{name}.pt"),
                            map_location=device).to(torch.float64) for name in names]
                for s in subdirs
            ]
        pool = None
        pool_path = os.path.join(root, "pool.jsonl")
        if os.path.exists(pool_path):
            with open(pool_path) as fh:
                pool = [json.loads(line) for line in fh if line.strip()]
        store = cls(n=manifest["n"], weights=manifest["weights"], checkpoint_names=names,
                    train_features=feats, validation=validation,
                    normalize=manifest["normalize"], provenance=manifest["provenance"], pool=pool)
        stored = manifest["provenance"]
        recomputed = _provenance(root, manifest["weights"], manifest["normalize"])
        if stored and stored != recomputed:
            raise ValueError(
                f"feature-store provenance mismatch ({stored[:8]}... != {recomputed[:8]}...): "
                "the feature bytes changed since this store was written")
        return store


def write_selected_jsonl(indices: Sequence[int], pool: Sequence[dict], out_path: str) -> int:
    """Write the selected subset as LESS-style JSONL (one pool record per line).

    Returns the number of records written. Records keep their original pool
    fields so the file drops straight into LESS's ``lora_train.sh``.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    written = 0
    with open(out_path, "w") as fh:
        for idx in indices:
            fh.write(json.dumps(pool[int(idx)]) + "\n")
            written += 1
    return written


def make_synthetic_store(n=200, d=32, checkpoints=4, tasks=("tydiqa", "bbh"),
                         subtasks=2, normalize="l2", seed=0, pool=True) -> FeatureStore:
    """A deterministic in-memory store for tests and smoke runs (no GPU/data)."""
    g = torch.Generator().manual_seed(seed)
    names = [f"ckpt{k + 1}" for k in range(checkpoints)]
    feats = [torch.randn(n, d, generator=g) for _ in names]
    weights = [1.0 / checkpoints] * checkpoints
    validation = {
        task: [[torch.rand(d, generator=g) + 0.2 for _ in names] for _ in range(subtasks)]
        for task in tasks
    }
    records = [{"id": i, "text": f"example {i}"} for i in range(n)] if pool else None
    return FeatureStore(n=n, weights=weights, checkpoint_names=names, train_features=feats,
                        validation=validation, normalize=normalize, pool=records)


def _load_dim_dir(dirpath: str) -> torch.Tensor:
    """Load one LESS ``.../dim{D}`` directory as a float64 ``[N, d]`` tensor.

    Prefers ``all_orig.pt`` (already row-normalized by LESS), then
    ``all_unormalized.pt``, then merges the raw ``grads-{count}.pt`` shards in
    index order — the same three states LESS leaves on disk.
    """
    all_orig = os.path.join(dirpath, "all_orig.pt")
    if os.path.exists(all_orig):
        return torch.load(all_orig, map_location="cpu").to(torch.float64)
    all_un = os.path.join(dirpath, "all_unormalized.pt")
    if os.path.exists(all_un):
        return torch.load(all_un, map_location="cpu").to(torch.float64)
    shards = [f for f in os.listdir(dirpath)
              if f.startswith("grads-") and f.endswith(".pt")]
    if not shards:
        raise FileNotFoundError(
            f"no all_orig.pt / all_unormalized.pt / grads-*.pt under {dirpath}")
    shards.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
    parts = [torch.load(os.path.join(dirpath, s), map_location="cpu") for s in shards]
    return torch.cat(parts, dim=0).to(torch.float64)


def load_from_less(
    dim: int,
    ckpts: Sequence[int],
    checkpoint_weights: Sequence[float],
    train_file_names: Sequence[str],
    train_grad_template: str,
    tasks: Sequence[str],
    val_grad_template: str,
    n_subtasks: Optional[Dict[str, int]] = None,
    pool_files: Optional[Dict[str, str]] = None,
) -> "FeatureStore":
    """Ingest LESS Step-2/Step-3 gradient dumps into a :class:`FeatureStore`.

    Templates mirror LESS ``matching.sh`` path conventions (formatted per
    checkpoint):
      ``train_grad_template`` with ``{name}``/``{ckpt}``/``{dim}``, e.g.
        ``"{grad_root}/{name}-ckpt{ckpt}-adam/dim{dim}"``
      ``val_grad_template`` with ``{task}``/``{ckpt}``/``{dim}``, e.g.
        ``"{grad_root}/{task}-ckpt{ckpt}-sgd/dim{dim}"``

    The pool is the concatenation of ``train_file_names`` in the given order
    (per-file row counts are preserved so selected indices map back to records).
    Validation gradients ``[N_valid, d]`` are reshaped to
    ``(n_subtasks[task], V, d)`` and stored per subtask, so the selector computes
    LESS's mean-over-val-examples / max-over-subtasks score exactly.
    """
    if len(ckpts) != len(checkpoint_weights):
        raise ValueError("ckpts and checkpoint_weights must match in length")
    total = float(sum(checkpoint_weights))
    if total <= 0:
        raise ValueError("checkpoint_weights must sum to a positive value")
    weights = [float(w) / total for w in checkpoint_weights]  # LESS renormalizes too

    names = [f"ckpt{c}" for c in ckpts]
    train_features: List[torch.Tensor] = []
    per_file_counts: List[List[int]] = []
    for k, c in enumerate(ckpts):
        blocks, counts = [], []
        for name in train_file_names:
            block = _load_dim_dir(train_grad_template.format(name=name, ckpt=c, dim=dim))
            blocks.append(block)
            counts.append(block.shape[0])
        per_file_counts.append(counts)
        train_features.append(torch.cat(blocks, dim=0))

    n = train_features[0].shape[0]
    width = train_features[0].shape[1]
    for f in train_features:
        if f.shape != (n, width):
            raise ValueError("every checkpoint must produce the same pool size and width")

    validation: Dict[str, List[List[torch.Tensor]]] = {}
    n_subtasks = n_subtasks or {}
    for task in tasks:
        ns = n_subtasks.get(task, N_SUBTASKS.get(task))
        if ns is None:
            raise ValueError(f"unknown subtask count for task {task!r}; pass n_subtasks")
        by_ckpt = []
        for c in ckpts:
            mat = _load_dim_dir(val_grad_template.format(task=task, ckpt=c, dim=dim))
            if mat.shape[0] % ns != 0:
                raise ValueError(
                    f"{task} validation rows {mat.shape[0]} not divisible by "
                    f"n_subtasks={ns}")
            v = mat.shape[0] // ns
            by_ckpt.append(mat.reshape(ns, v, width))
        # validation[task][subtask][ckpt] -> [V, width]
        validation[task] = [[by_ckpt[k][s] for k in range(len(ckpts))] for s in range(ns)]

    pool = None
    if pool_files:
        pool = []
        for name in train_file_names:
            with open(pool_files[name]) as fh:
                pool.extend(json.loads(line) for line in fh if line.strip())

    return FeatureStore(n=n, weights=weights, checkpoint_names=names,
                        train_features=train_features, validation=validation,
                        normalize="l2", pool=pool)
