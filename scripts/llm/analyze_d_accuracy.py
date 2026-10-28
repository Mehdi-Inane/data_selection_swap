#!/usr/bin/env python3
"""Spearman correlation between a selection score and downstream accuracy.

Port of the CV project's D(α)-vs-accuracy study. Reads run rows (score = the
selection objective value by default; accuracy from each run's ``accuracy.json``)
and reports the rank correlation overall and per method. A score that predicts
accuracy well is evidence the selection ranking is doing the real work.
"""

import argparse
import json

from data_selection_swap.llm import correlation, plotting


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="run tree with diagnostics.json / accuracy.json")
    p.add_argument("--score-key", default="value")
    p.add_argument("--acc-key", default="accuracy")
    p.add_argument("--group", nargs="*", default=["method"])
    p.add_argument("--task", default=None, help="restrict to one task")
    args = p.parse_args()

    rows = plotting.read_run_rows(args.root)
    if args.task:
        rows = [r for r in rows if str(r.get("task")) == args.task]
    result = correlation.score_vs_accuracy(
        rows, score_key=args.score_key, acc_key=args.acc_key, group_keys=tuple(args.group))

    # tuple keys are not JSON-serializable; render them for display
    printable = {("all" if k == "all" else "|".join(k)): v for k, v in result.items()}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
