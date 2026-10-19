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


def _provenance(files: Sequence[str], weights: Sequence[float], normalize: str) -> str:
    h = hashlib.sha256()
    h.update(normalize.encode())
    for w in weights:
        h.update(repr(float(w)).encode())
        h.update(b"\x00")
    for path in sorted(files):
        st = os.stat(path)
        h.update(f"{os.path.basename(path)}\t{st.st_size}\t{st.st_mtime_ns}".encode())
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
            "provenance": self.provenance or _provenance(files, self.weights, self.normalize),
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
        recomputed = _provenance(
            [os.path.join(root, "features", f"{name}.pt") for name in names],
            manifest["weights"], manifest["normalize"])
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
