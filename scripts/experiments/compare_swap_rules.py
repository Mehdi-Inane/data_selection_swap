#!/usr/bin/env python3
"""Paired comparison of manuscript Algorithm 2 and Algorithm 3.

The implementation reuses the immutable reference/selection/retraining stages
from ``optimizer_lr.py``, but exposes only one controlled optimizer/LR setting
and varies exactly one factor: ``ranked`` (Algorithm 3) versus ``best``
(Algorithm 2).  Algorithm 2 is required to use ``max_pairs=0`` so it remains a
full best-swap search rather than a truncated approximation.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import optimizer_lr as core


RULES = ("ranked", "best")


def validate_config(config: dict[str, Any]) -> None:
    core._require_keys(
        config,
        ["schema_version", "experiment_name", "datasets", "architectures", "fractions",
         "seeds", "reference", "selection", "retraining"],
        "swap-rule config",
    )
    if config["schema_version"] != core.SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {core.SCHEMA_VERSION}")
    reference = config["reference"]
    core._require_keys(reference, ["optimizer", "learning_rate"], "reference")
    retraining = config["retraining"]
    core._require_keys(retraining, ["optimizer", "learning_rate"], "retraining")
    selection = config["selection"]
    core._require_keys(selection, ["rules", "score_objective", "max_pairs"], "selection")
    if tuple(selection["rules"]) != RULES:
        raise ValueError("selection.rules must be exactly ['ranked', 'best']")
    if selection["max_pairs"] != 0:
        raise ValueError(
            "Algorithm 2 must use max_pairs=0; a cap is not the full best-swap algorithm"
        )
    for _, variant in variant_configs(config):
        core.validate_config(variant)


def _convert(config: dict[str, Any], rule: str) -> dict[str, Any]:
    reference = config["reference"]
    selection = config["selection"]
    retraining = config["retraining"]
    reference_optimizer = reference["optimizer"]
    retrain_optimizer = retraining["optimizer"]
    return {
        "schema_version": core.SCHEMA_VERSION,
        "experiment_name": config["experiment_name"],
        "output_root": config.get("output_root"),
        "require_cuda": config.get("require_cuda", True),
        "datasets": config["datasets"],
        "architectures": config["architectures"],
        "fractions": config["fractions"],
        "seeds": config["seeds"],
        "reference": {
            **{key: value for key, value in reference.items()
               if key not in ("optimizer", "learning_rate")},
            "optimizers": {
                reference_optimizer: {"learning_rates": [reference["learning_rate"]]}
            },
        },
        "selection": {
            **{key: value for key, value in selection.items()
               if key not in ("rules", "score_objective")},
            "rule": rule,
            "objectives_by_reference": {
                reference_optimizer: [selection["score_objective"]]
            },
        },
        "retraining": {
            **{key: value for key, value in retraining.items()
               if key not in ("optimizer", "learning_rate")},
            "optimizers": {
                retrain_optimizer: {"learning_rates": [retraining["learning_rate"]]}
            },
        },
    }


def variant_configs(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [(rule, _convert(config, rule)) for rule in RULES]


def reference_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    # Selection rule is absent from reference identity, so both converted
    # configurations intentionally produce the same IDs.
    return core.reference_tasks(_convert(config, RULES[0]))


def selection_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = []
    for rule, variant in variant_configs(config):
        for task in core.selection_tasks(variant):
            tasks.append({**task, "swap_rule": rule})
    return tasks


def retrain_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = []
    for rule, variant in variant_configs(config):
        for task in core.retrain_tasks(variant):
            tasks.append({**task, "swap_rule": rule})
    return tasks


def tasks_for(config: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    if stage == "reference":
        return reference_tasks(config)
    if stage == "selection":
        return selection_tasks(config)
    if stage == "retrain":
        return retrain_tasks(config)
    raise ValueError(stage)


def _find_task(config: dict[str, Any], stage: str, index: int) -> dict[str, Any]:
    tasks = tasks_for(config, stage)
    if not 0 <= index < len(tasks):
        raise IndexError(f"task-id {index} outside 0..{len(tasks) - 1} for {stage}")
    return tasks[index]


def _variant_for(config: dict[str, Any], rule: str) -> dict[str, Any]:
    return _convert(config, rule)


def _load(path: Path) -> dict[str, Any]:
    config = core._read_json(path)
    validate_config(config)
    return config


def _selection_diagnostics(directory: Path) -> dict[str, Any]:
    timing = core._read_json(directory / "timing.json")
    history = core._read_json(directory / "objective.json")
    final = history[-1] if history else {}
    phases = timing.get("phases_seconds", {})
    return {
        "D_alpha": final.get("D", timing.get("D")),
        "objective": final.get("objective", timing.get("objective")),
        "swaps": timing.get("swaps", final.get("swaps")),
        "pair_searches": timing.get("pair_searches", final.get("pair_searches")),
        "certificate": timing.get("certificate", final.get("certificate")),
        "certificate_valid": timing.get("certificate_valid"),
        "globally_optimal": timing.get("globally_optimal"),
        "one_swap_local_optimum": timing.get("one_swap_local_optimum"),
        "algorithm_2_exact": timing.get("algorithm_2_exact"),
        "pairs_screened": timing.get("pairs_screened"),
        "stop_reason": timing.get("reason", final.get("reason")),
        "scoring_seconds": phases.get("scoring"),
        "swap_search_seconds": phases.get("selection"),
        "selection_pipeline_seconds": timing.get("selection_seconds"),
    }


def _difference(left: Any, right: Any) -> float | None:
    return None if left is None or right is None else float(left) - float(right)


def summarize(config: dict[str, Any], output_root: Path) -> tuple[Path, Path, Path]:
    selections = {task["id"]: task for task in selection_tasks(config)}
    rows: list[dict[str, Any]] = []
    for task in retrain_tasks(config):
        run_dir = core.task_dir(output_root, task)
        selection = selections[task["selection_id"]]
        selection_dir = core.task_dir(output_root, selection)
        row = {
            "task_id": task["id"], "status": "missing", "dataset": task["dataset"],
            "architecture": task["architecture"], "fraction": task["fraction"],
            "seed": task["seed"], "swap_rule": task["swap_rule"],
            "reference_id": task["reference_id"],
            "reference_optimizer": task["reference_optimizer"],
            "reference_lr": task["reference_lr"],
            "score_objective": task["score_objective"],
            "retrain_optimizer": task["retrain_optimizer"],
            "retrain_lr": task["retrain_lr"],
            "final_test_accuracy": None, "initial_state_sha256": None,
            "D_alpha": None, "objective": None, "swaps": None,
            "pair_searches": None, "certificate": None, "certificate_valid": None,
            "globally_optimal": None, "one_swap_local_optimum": None,
            "algorithm_2_exact": None, "pairs_screened": None, "stop_reason": None,
            "scoring_seconds": None, "swap_search_seconds": None,
            "selection_pipeline_seconds": None,
        }
        if core._completed(run_dir, task) and core._completed(selection_dir, selection):
            metrics = core._read_json(run_dir / "metrics.json")
            row.update(_selection_diagnostics(selection_dir))
            row.update({
                "status": "complete",
                "final_test_accuracy": metrics["final_test_accuracy"],
                "initial_state_sha256": metrics["initial_state_sha256"],
            })
        rows.append(row)

    summary_dir = output_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    raw_path = summary_dir / "swap_rule_runs.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    group_fields = [
        "dataset", "architecture", "fraction", "reference_optimizer", "reference_lr",
        "score_objective", "retrain_optimizer", "retrain_lr", "swap_rule",
    ]
    metrics_to_aggregate = [
        "final_test_accuracy", "D_alpha", "swaps", "certificate",
        "swap_search_seconds", "selection_pipeline_seconds",
    ]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row["status"] == "complete":
            groups.setdefault(tuple(row[field] for field in group_fields), []).append(row)
    aggregates = []
    for key, members in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        record = {**dict(zip(group_fields, key)), "n_seeds": len(members)}
        for metric in metrics_to_aggregate:
            values = [float(member[metric]) for member in members if member[metric] is not None]
            record[f"mean_{metric}"] = statistics.fmean(values) if values else None
            record[f"sample_std_{metric}"] = (
                statistics.stdev(values) if len(values) > 1 else None
            )
        aggregates.append(record)
    aggregate_path = summary_dir / "swap_rule_aggregate.json"
    core._write_json(aggregate_path, {
        "primary_metric": "final_test_accuracy",
        "complete_runs": sum(row["status"] == "complete" for row in rows),
        "planned_runs": len(rows), "groups": aggregates,
    })

    pair_fields = [
        "dataset", "architecture", "fraction", "seed", "reference_optimizer",
        "reference_lr", "score_objective", "retrain_optimizer", "retrain_lr",
    ]
    by_pair: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row["status"] == "complete":
            by_pair.setdefault(tuple(row[field] for field in pair_fields), {})[
                row["swap_rule"]
            ] = row
    pairs = []
    for key, rules in sorted(by_pair.items(), key=lambda item: tuple(map(str, item[0]))):
        if set(rules) != set(RULES):
            continue
        ranked, best = rules["ranked"], rules["best"]
        pairs.append({
            **dict(zip(pair_fields, key)),
            "same_reference": ranked["reference_id"] == best["reference_id"],
            "same_initialization": (
                ranked["initial_state_sha256"] == best["initial_state_sha256"]
            ),
            "best_minus_ranked_accuracy": _difference(
                best["final_test_accuracy"], ranked["final_test_accuracy"]
            ),
            "best_minus_ranked_D_alpha": _difference(
                best["D_alpha"], ranked["D_alpha"]
            ),
            "best_over_ranked_swap_time": (
                best["swap_search_seconds"] / ranked["swap_search_seconds"]
                if ranked["swap_search_seconds"] else None
            ),
            "best_algorithm_2_exact": best["algorithm_2_exact"],
            "ranked_certificate": ranked["certificate"],
            "best_certificate": best["certificate"],
        })
    paired_group_fields = [field for field in pair_fields if field != "seed"]
    paired_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for pair in pairs:
        paired_groups.setdefault(
            tuple(pair[field] for field in paired_group_fields), []
        ).append(pair)
    paired_aggregates = []
    for key, members in sorted(
            paired_groups.items(), key=lambda item: tuple(map(str, item[0]))):
        record = {**dict(zip(paired_group_fields, key)), "n_pairs": len(members)}
        for metric in ("best_minus_ranked_accuracy", "best_minus_ranked_D_alpha",
                       "best_over_ranked_swap_time"):
            values = [float(member[metric]) for member in members
                      if member[metric] is not None]
            record[f"mean_{metric}"] = statistics.fmean(values) if values else None
            record[f"sample_std_{metric}"] = (
                statistics.stdev(values) if len(values) > 1 else None
            )
        record["all_same_reference"] = all(member["same_reference"] for member in members)
        record["all_same_initialization"] = all(
            member["same_initialization"] for member in members
        )
        record["all_best_searches_exact"] = all(
            member["best_algorithm_2_exact"] is True for member in members
        )
        paired_aggregates.append(record)
    pairs_path = summary_dir / "swap_rule_paired.json"
    core._write_json(pairs_path, {
        "pairs": pairs, "n_complete_pairs": len(pairs),
        "paired_aggregates": paired_aggregates,
    })
    return raw_path, aggregate_path, pairs_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--config", required=True, type=Path)

    count = subparsers.add_parser("count")
    count.add_argument("--config", required=True, type=Path)
    count.add_argument("--stage", required=True, choices=core.STAGES)

    listing = subparsers.add_parser("list")
    listing.add_argument("--config", required=True, type=Path)
    listing.add_argument("--stage", choices=core.STAGES, default=None)
    listing.add_argument("--output-root", default=None)

    field = subparsers.add_parser("task-field")
    field.add_argument("--config", required=True, type=Path)
    field.add_argument("--stage", required=True, choices=core.STAGES)
    field.add_argument("--task-id", required=True, type=int)
    field.add_argument("--field", required=True)

    run = subparsers.add_parser("run-task")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--stage", required=True, choices=core.STAGES)
    run.add_argument("--task-id", required=True, type=int)
    run.add_argument("--data-dir", required=True, type=Path)
    run.add_argument("--output-root", default=None)
    run.add_argument("--download", action="store_true")
    run.add_argument("--allow-cpu", action="store_true")
    run.add_argument("--dry-run", action="store_true")

    summary = subparsers.add_parser("summarize")
    summary.add_argument("--config", required=True, type=Path)
    summary.add_argument("--output-root", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    config = _load(config_path)
    if args.command == "validate":
        print(json.dumps({
            "valid": True,
            "counts": {stage: len(tasks_for(config, stage)) for stage in core.STAGES},
        }, sort_keys=True))
        return 0
    if args.command == "count":
        print(len(tasks_for(config, args.stage)))
        return 0
    if args.command == "task-field":
        task = _find_task(config, args.stage, args.task_id)
        value = task[args.field]
        print(value if isinstance(value, (str, int, float)) else core._canonical_json(value))
        return 0

    output_root = core.resolve_output_root(config_path, config, args.output_root)
    if args.command == "list":
        stages = (args.stage,) if args.stage else core.STAGES
        for stage in stages:
            for index, task in enumerate(tasks_for(config, stage)):
                print(core._canonical_json({
                    "array_index": index,
                    "artifact_dir": str(core.task_dir(output_root, task)), **task,
                }))
        return 0
    if args.command == "summarize":
        for path in summarize(config, output_root):
            print(f"Wrote {path}")
        return 0
    if args.command == "run-task":
        task = _find_task(config, args.stage, args.task_id)
        rule = task.get("swap_rule", RULES[0])
        variant = _variant_for(config, rule)
        if args.dry_run:
            preview: dict[str, Any] = {
                "task": task, "artifact_dir": str(core.task_dir(output_root, task)),
                "data_dir": str(args.data_dir),
            }
            if args.stage == "reference":
                preview["command"] = core.reference_command(
                    task, args.data_dir, core.task_dir(output_root, task), args.download
                )
            elif args.stage == "selection":
                references = {item["id"]: item for item in core.reference_tasks(variant)}
                preview["command"] = core.selection_command(
                    task, core.task_dir(output_root, references[task["reference_id"]]),
                    args.data_dir, args.download,
                )
            else:
                preview["operation"] = "paired retraining from frozen indices"
            print(json.dumps(preview, indent=2, sort_keys=True))
            return 0

        require_cuda = bool(config.get("require_cuda", True))
        with core._task_lock(core.task_dir(output_root, task)):
            if args.stage == "reference":
                core.run_reference(task, output_root, args.data_dir, args.download,
                                   args.allow_cpu, require_cuda)
            elif args.stage == "selection":
                core.run_selection(task, variant, output_root, args.data_dir, args.download,
                                   args.allow_cpu, require_cuda)
            else:
                core.run_retrain(task, variant, output_root, args.data_dir, args.download,
                                 args.allow_cpu, require_cuda)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
