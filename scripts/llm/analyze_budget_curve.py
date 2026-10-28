#!/usr/bin/env python3
"""Aggregate selection runs into a budget-efficiency curve (plot + optional JSON).

Reads every ``diagnostics.json`` (and optional ``accuracy.json``) under ``ROOT``,
groups runs by task and method, and plots the metric (default downstream
``accuracy``) against the selection fraction. Use ``--value-key value`` to plot
the selection objective instead of accuracy.
"""

import argparse
import json
import sys

from data_selection_swap.llm import plotting


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="run tree with diagnostics.json / accuracy.json")
    p.add_argument("--value-key", default="accuracy")
    p.add_argument("--out", default="budget_curve.png")
    p.add_argument("--json", default=None, help="optional: dump merged rows to JSON")
    p.add_argument("--stdout", action="store_true", help="print the curve summary as JSON")
    args = p.parse_args()

    rows = plotting.read_run_rows(args.root)
    curve = plotting.budget_curve(rows, value_key=args.value_key)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=2)
    if args.stdout:
        json.dump(curve, sys.stdout, indent=2)
        print()

    path = plotting.save_budget_curve(rows, args.out, value_key=args.value_key)
    if path:
        print(f"wrote {path} ({sum(len(v) for v in curve.values())} series across "
              f"{len(curve)} tasks)")
    else:
        print("matplotlib unavailable: wrote curve summary only "
              f"(use --json/--stdout); {len(rows)} runs across {len(curve)} tasks")


if __name__ == "__main__":
    main()
