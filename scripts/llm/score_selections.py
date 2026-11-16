#!/usr/bin/env python3
"""Score existing selections (objective f, mean influence, redundancy) on one store.

Reads only: for every ``select:<method>:<task>:p<fraction>:seed<seed>`` directory
under ``--runs`` it loads ``selected_indices.pt`` and evaluates the shared set
objective f (objective.py, minimized), the mean LESS influence score, and the
final-checkpoint mean pairwise cosine — putting swap, LESS top-k and random on
one footing, including the baselines whose thin diagnostics recorded no metrics.
"""

import argparse
import json
import os

import torch

from data_selection_swap.llm import datastore, scoring
from data_selection_swap.llm.objective import SwapObjective
from data_selection_swap.llm.targets import build_score


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", required=True)
    p.add_argument("--runs", required=True, help="runs dir containing select:* dirs")
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--methods", nargs="+", default=["swap", "less", "random"])
    p.add_argument("--fraction", type=float, default=0.05)
    p.add_argument("--aggregation", choices=["max", "mean"], default="max")
    p.add_argument("--out", default=None, help="default: <runs>/score_selections.json")
    args = p.parse_args()

    store = datastore.FeatureStore.load(args.store)
    features = store.build_features()

    results = []
    for task in args.tasks:
        c = build_score(features, "targeted", subtask_targets=store.subtask_targets(task),
                        aggregation=args.aggregation)
        obj = SwapObjective(features, c)
        for method in args.methods:
            d = os.path.join(args.runs, f"select:{method}:{task}:p{args.fraction}:seed0")
            sel_path = os.path.join(d, "selected_indices.pt")
            if not os.path.exists(sel_path):
                continue
            idx = torch.load(sel_path).to(torch.long)
            row = {"method": method, "task": task, "provenance": store.provenance,
                   "subset_size": int(idx.numel()),
                   "objective_f": obj.value(idx),
                   "mean_influence": float(c.index_select(0, idx).mean()),
                   "redundancy_final_ckpt": scoring.mean_pairwise_cosine(features, idx)}
            results.append(row)
            print(json.dumps(row), flush=True)

    out = args.out or os.path.join(args.runs, "score_selections.json")
    with open(out, "w") as fh:
        json.dump({"store": args.store, "provenance": store.provenance,
                   "results": results}, fh, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
