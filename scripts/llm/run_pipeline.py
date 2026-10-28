#!/usr/bin/env python3
"""Orchestrate the LESS + selection pipeline from a JSON config.

Dry-runs by default (prints the exact command sequence); pass ``--run`` to
execute. ``--stage`` filters by substring (e.g. ``--stage warmup --stage store``)
so you can resume or run stages independently on a cluster. The command plan is
built by ``data_selection_swap.llm.pipeline`` (see ``docs/llm_less_protocol.md``).
"""

import argparse

from data_selection_swap.llm import pipeline


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="JSON PipelineConfig")
    p.add_argument("--run", action="store_true", help="execute stages (default: dry-run)")
    p.add_argument("--stage", action="append", default=None,
                   help="substring filter on stage names (repeatable)")
    p.add_argument("--list", action="store_true", help="just list stage names")
    args = p.parse_args()

    cfg = pipeline.config_from_json(args.config)
    stages = pipeline.build_stages(cfg)

    if args.list:
        for st in stages:
            print(st.name)
        return
    acted = pipeline.run_stages(stages, dry_run=not args.run, only=args.stage)
    mode = "ran" if args.run else "dry-run"
    print(f"\n{mode}: {len(acted)}/{len(stages)} stages")


if __name__ == "__main__":
    main()
