#!/usr/bin/env python3
"""Select a subset from a gradient-feature store with the swap selector.

Works on a real LESS-derived store or a synthetic smoke store (see
``build_datastore.py``). For the ``targeted`` setting it also reports, side by
side, LESS's independent top-k on the same influence score — the direct
swap-vs-LESS comparison (objective value, overlap, redundancy) that the study is
about. Writes everything under ``--out``; touches nothing else.
"""

import argparse
import json
import os

import torch

from data_selection_swap.llm import datastore, scoring
from data_selection_swap.llm.config import SelectionConfig
from data_selection_swap.llm.objective import SwapObjective, select_topk
from data_selection_swap.llm.swap import select
from data_selection_swap.llm.targets import build_score


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", required=True, help="feature-store root")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--setting", choices=["targeted", "faithful"], default="targeted")
    p.add_argument("--task", default=None, help="validation task (targeted setting)")
    p.add_argument("--fraction", type=float, default=0.05)
    p.add_argument("--normalize", choices=["l2", "none"], default="l2")
    p.add_argument("--aggregation", choices=["max", "mean"], default="max")
    p.add_argument("--rule", choices=["ranked", "best"], default="ranked")
    p.add_argument("--max-iters", type=int, default=2000)
    p.add_argument("--block-size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.setting == "targeted" and not args.task:
        p.error("--task is required for the targeted setting")

    store = datastore.FeatureStore.load(args.store)
    features = store.build_features()
    targets = store.subtask_targets(args.task) if args.task else None

    config = SelectionConfig(setting=args.setting, fraction=args.fraction,
                             normalize=args.normalize, aggregation=args.aggregation,
                             rule=args.rule, max_iters=args.max_iters,
                             block_size=args.block_size, seed=args.seed)
    sel, history, diag = select(features, args.setting, config, subtask_targets=targets)

    os.makedirs(args.out, exist_ok=True)
    torch.save(sel, os.path.join(args.out, "selected_indices.pt"))
    if store.pool is not None:
        datastore.write_selected_jsonl(sel.tolist(), store.pool,
                                       os.path.join(args.out, "selected.jsonl"))

    report = {"setting": args.setting, "task": args.task, "fraction": args.fraction,
              "normalize": args.normalize, "aggregation": args.aggregation,
              "rule": args.rule, "seed": args.seed, "provenance": store.provenance,
              "subset_size": int(sel.numel()), "swap": diag}

    # Direct comparison against LESS's independent top-k on the same score.
    c = build_score(features, args.setting, subtask_targets=targets,
                    aggregation=args.aggregation)
    obj = SwapObjective(features, c)
    less_topk = select_topk(c, config.subset_size(features.n))
    report["less_topk"] = {
        "value": obj.value(less_topk),
        "jaccard_vs_swap": scoring.jaccard(less_topk.tolist(), sel.tolist()),
        "redundancy": scoring.mean_pairwise_cosine(features, less_topk),
    }
    report["swap"]["redundancy"] = scoring.mean_pairwise_cosine(features, sel)

    with open(os.path.join(args.out, "diagnostics.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
