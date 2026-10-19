#!/usr/bin/env python3
"""LESS and random baselines on a gradient-feature store (same output schema).

``less`` reproduces LESS's independent top-k by its influence score (cosine to
the validation gradient, ``max`` over subtasks); ``random`` samples uniformly.
Both write the same ``selected_indices.pt`` / ``selected.jsonl`` layout as
``select_swap.py`` so downstream training/evaluation treats them identically.
"""

import argparse
import json
import os

import torch

from data_selection_swap.llm import datastore
from data_selection_swap.llm.config import SelectionConfig
from data_selection_swap.llm.objective import select_topk
from data_selection_swap.llm.targets import build_score


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--method", choices=["random", "less"], required=True)
    p.add_argument("--task", default=None, help="validation task (required for --method less)")
    p.add_argument("--fraction", type=float, default=0.05)
    p.add_argument("--aggregation", choices=["max", "mean"], default="max")
    p.add_argument("--normalize", choices=["l2", "none"], default="l2")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.method == "less" and not args.task:
        p.error("--task is required for --method less")

    store = datastore.FeatureStore.load(args.store)
    features = store.build_features()
    m = SelectionConfig(fraction=args.fraction).subset_size(features.n)

    if args.method == "random":
        sel = torch.sort(torch.randperm(features.n, generator=torch.Generator().manual_seed(args.seed))[:m]).values
    else:
        c = build_score(features, "targeted",
                        subtask_targets=store.subtask_targets(args.task),
                        aggregation=args.aggregation)
        sel = select_topk(c, m)

    os.makedirs(args.out, exist_ok=True)
    torch.save(sel, os.path.join(args.out, "selected_indices.pt"))
    if store.pool is not None:
        datastore.write_selected_jsonl(sel.tolist(), store.pool,
                                       os.path.join(args.out, "selected.jsonl"))
    report = {"method": args.method, "task": args.task, "fraction": args.fraction,
              "normalize": args.normalize, "seed": args.seed,
              "provenance": store.provenance, "subset_size": int(sel.numel())}
    with open(os.path.join(args.out, "diagnostics.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
