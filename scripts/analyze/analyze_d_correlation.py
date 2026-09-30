#!/usr/bin/env python3
"""Analyze paired D(alpha) / final-test-accuracy experiment records.

The subset, rather than an individual retraining replicate, is the statistical
unit.  Accuracies are averaged within subset before the point estimate is
computed; the bootstrap resamples paired subsets and also resamples retraining
replicates within each selected subset.  Correlations are never computed by
pooling raw values across datasets, architectures, budgets, or trajectories.

Outputs
-------
paired_observations.csv
    One row per unique subset, with retraining replicates collapsed.
correlations_by_block.csv
    Spearman, log-D Pearson, and Kendall tau-b within every fixed protocol.
correlation_summary.json
    Machine-readable provenance, exclusions, estimates, and bootstrap CIs.
correlation_table.md
    Compact paper-friendly table.
d_accuracy_scatter.png
    Raw paired observations, faceted by dataset and architecture.
d_accuracy_within_block_ranks.png
    Scale-free, within-block percentile ranks used for the pooled estimate.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.d_correlation_common import (  # noqa: E402
    atomic_write_json,
    average_ranks,
    flatten_record,
    kendall_tau_b,
    load_json,
    pearson,
    spearman,
)


def discover_records(inputs: list[str]) -> list[Path]:
    found: set[Path] = set()
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if path.is_file():
            found.add(path)
        elif path.is_dir():
            found.update(p.resolve() for p in path.rglob("record.json"))
        else:
            raise FileNotFoundError(f"input does not exist: {path}")
    return sorted(found)


def load_rows(paths: list[Path], skip_invalid: bool) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        try:
            row = flatten_record(load_json(path), source=str(path))
            key = (row["analysis_block_id"], row["subset_hash"])
            previous = seen.get(key)
            if previous is not None:
                same = (
                    previous["d_alpha"] == row["d_alpha"]
                    and previous["accuracy_values"] == row["accuracy_values"]
                )
                if same:
                    excluded.append({"path": str(path), "reason": "exact duplicate record"})
                    continue
                raise ValueError(
                    "duplicate subset hash in the same analysis block has conflicting results; "
                    f"first seen at {previous['source']}"
                )
            seen[key] = row
            rows.append(row)
        except Exception as error:
            if not skip_invalid:
                raise ValueError(f"invalid record {path}: {error}") from error
            excluded.append({"path": str(path), "reason": str(error)})
    rows.sort(
        key=lambda r: (
            r["dataset"], r["architecture"], r["split_seed"], r["fraction"],
            r["sampler"], r["subset_seed"], r["subset_hash"],
        )
    )
    return rows, excluded


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def safe_log_d(values: np.ndarray) -> np.ndarray:
    positive = values[values > 0.0]
    floor = float(positive.min()) * 1e-6 if positive.size else np.finfo(np.float64).tiny
    return np.log10(np.maximum(values, floor))


def draw_accuracy_mean(row: dict[str, Any], rng: np.random.Generator) -> float:
    values = np.asarray(row["accuracy_values"], dtype=np.float64)
    if values.size == 1:
        return float(values[0])
    # Parametric assumptions are unnecessary: bootstrap the replicate mean.
    take = rng.integers(0, values.size, size=values.size)
    return float(values[take].mean())


def bootstrap_block(
    rows: list[dict[str, Any]],
    statistic,
    *,
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    d = np.asarray([r["d_alpha"] for r in rows], dtype=np.float64)
    accuracy = np.asarray([r["accuracy_mean"] for r in rows], dtype=np.float64)
    estimate = float(statistic(d, accuracy))
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(n_bootstrap):
        take = rng.integers(0, len(rows), size=len(rows))
        sampled = [rows[int(i)] for i in take]
        sampled_d = np.asarray([r["d_alpha"] for r in sampled], dtype=np.float64)
        sampled_a = np.asarray([draw_accuracy_mean(r, rng) for r in sampled], dtype=np.float64)
        value = float(statistic(sampled_d, sampled_a))
        if math.isfinite(value):
            draws.append(value)
    if not draws:
        return {
            "estimate": finite_or_none(estimate),
            "ci_low": None,
            "ci_high": None,
            "probability_negative": None,
            "valid_bootstrap": 0,
        }
    alpha = 0.5 * (1.0 - confidence)
    samples = np.asarray(draws)
    return {
        "estimate": finite_or_none(estimate),
        "ci_low": float(np.quantile(samples, alpha)),
        "ci_high": float(np.quantile(samples, 1.0 - alpha)),
        "probability_negative": float(np.mean(samples < 0.0)),
        "valid_bootstrap": int(samples.size),
    }


def block_label(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    return {
        "analysis_block_id": first["analysis_block_id"],
        "dataset": first["dataset"],
        "architecture": first["architecture"],
        "split_seed": first["split_seed"],
        "fraction": first["fraction"],
        "budget": first["budget"],
        "reference_fingerprint": first["reference_fingerprint"],
        "n_subsets": len(rows),
        "n_accuracy_fits": int(sum(r["accuracy_replicates"] for r in rows)),
    }


def summarize_blocks(
    groups: dict[str, list[dict[str, Any]]],
    *,
    n_bootstrap: int,
    confidence: float,
    seed: int,
    min_block_size: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    summaries: list[dict[str, Any]] = []
    undersized: list[str] = []
    for number, (block_id, rows) in enumerate(sorted(groups.items())):
        entry = block_label(rows)
        if len(rows) < min_block_size:
            undersized.append(block_id)
            entry["status"] = "insufficient_subsets"
            entry["minimum_required"] = min_block_size
            summaries.append(entry)
            continue
        entry["status"] = "included"
        block_seed = seed + 1009 * (number + 1)
        entry["spearman"] = bootstrap_block(
            rows, spearman, n_bootstrap=n_bootstrap,
            confidence=confidence, seed=block_seed,
        )
        entry["pearson_log10_d"] = bootstrap_block(
            rows,
            lambda d, a: pearson(safe_log_d(np.asarray(d)), a),
            n_bootstrap=n_bootstrap,
            confidence=confidence, seed=block_seed + 1,
        )
        entry["kendall_tau_b"] = {
            "estimate": finite_or_none(
                kendall_tau_b(
                    [r["d_alpha"] for r in rows],
                    [r["accuracy_mean"] for r in rows],
                )
            )
        }
        summaries.append(entry)
    return summaries, undersized


def within_block_rank_arrays(groups: list[list[dict[str, Any]]]) -> tuple[np.ndarray, np.ndarray]:
    x_parts, y_parts = [], []
    for rows in groups:
        d = np.asarray([r["d_alpha"] for r in rows], dtype=np.float64)
        a = np.asarray([r["accuracy_mean"] for r in rows], dtype=np.float64)
        # Centre each block so no dataset/budget/architecture mean can drive the result.
        x = average_ranks(d) / (len(rows) + 1.0)
        y = average_ranks(a) / (len(rows) + 1.0)
        x_parts.append(x - x.mean())
        y_parts.append(y - y.mean())
    return np.concatenate(x_parts), np.concatenate(y_parts)


def stratified_summary(
    groups: list[list[dict[str, Any]]],
    *,
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    observed_x, observed_y = within_block_rank_arrays(groups)
    estimate = pearson(observed_x, observed_y)
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(n_bootstrap):
        # Resample whole blocks, then paired subsets within each chosen block.
        block_take = rng.integers(0, len(groups), size=len(groups))
        sampled_groups = []
        for index in block_take:
            source = groups[int(index)]
            row_take = rng.integers(0, len(source), size=len(source))
            sampled = []
            for row_index in row_take:
                original = source[int(row_index)]
                clone = dict(original)
                clone["accuracy_mean"] = draw_accuracy_mean(original, rng)
                sampled.append(clone)
            sampled_groups.append(sampled)
        x, y = within_block_rank_arrays(sampled_groups)
        value = pearson(x, y)
        if math.isfinite(value):
            draws.append(value)
    if not draws:
        return {
            "estimate": finite_or_none(estimate), "ci_low": None, "ci_high": None,
            "probability_negative": None, "valid_bootstrap": 0,
            "n_blocks": len(groups), "n_subsets": int(observed_x.size),
        }
    alpha = 0.5 * (1.0 - confidence)
    samples = np.asarray(draws)
    return {
        "estimate": finite_or_none(estimate),
        "ci_low": float(np.quantile(samples, alpha)),
        "ci_high": float(np.quantile(samples, 1.0 - alpha)),
        "probability_negative": float(np.mean(samples < 0.0)),
        "valid_bootstrap": int(samples.size),
        "n_blocks": len(groups),
        "n_subsets": int(observed_x.size),
        "definition": "Pearson association of D and accuracy percentile ranks centred within each block",
    }


def write_observations_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "record_id", "analysis_block_id", "dataset", "architecture", "split_seed",
        "fraction", "budget", "sampler", "subset_seed", "subset_id", "subset_hash",
        "d_alpha", "accuracy_mean", "accuracy_sd", "accuracy_replicates",
        "reference_fingerprint", "source",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def write_block_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fields = [
        "analysis_block_id", "dataset", "architecture", "split_seed", "fraction",
        "budget", "n_subsets", "n_accuracy_fits", "status", "spearman",
        "spearman_ci_low", "spearman_ci_high", "probability_spearman_negative",
        "pearson_log10_d", "pearson_ci_low", "pearson_ci_high", "kendall_tau_b",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summaries:
            s = item.get("spearman", {})
            p = item.get("pearson_log10_d", {})
            k = item.get("kendall_tau_b", {})
            writer.writerow({
                "analysis_block_id": item["analysis_block_id"],
                "dataset": item["dataset"], "architecture": item["architecture"],
                "split_seed": item["split_seed"], "fraction": item["fraction"],
                "budget": item["budget"], "n_subsets": item["n_subsets"],
                "n_accuracy_fits": item["n_accuracy_fits"], "status": item["status"],
                "spearman": s.get("estimate"), "spearman_ci_low": s.get("ci_low"),
                "spearman_ci_high": s.get("ci_high"),
                "probability_spearman_negative": s.get("probability_negative"),
                "pearson_log10_d": p.get("estimate"), "pearson_ci_low": p.get("ci_low"),
                "pearson_ci_high": p.get("ci_high"),
                "kendall_tau_b": k.get("estimate"),
            })


def format_ci(stat: dict[str, Any]) -> str:
    if stat.get("estimate") is None:
        return "NA"
    if stat.get("ci_low") is None:
        return f"{stat['estimate']:.3f}"
    return f"{stat['estimate']:.3f} [{stat['ci_low']:.3f}, {stat['ci_high']:.3f}]"


def write_markdown(path: Path, summaries: list[dict[str, Any]], overall: dict[str, Any]) -> None:
    lines = [
        "# D(alpha) versus final test accuracy",
        "",
        "Lower D(alpha) is hypothesized to predict higher accuracy, so a negative correlation is supportive.",
        "Each row fixes dataset, architecture, split, budget, trajectory, D definition, and retraining protocol.",
        "Accuracy replicates are averaged per subset; brackets are configured paired-bootstrap confidence intervals.",
        "",
        "| Dataset | Network | Split | Fraction | Subsets | Spearman | Pearson(log10 D) | Kendall tau-b |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        if item["status"] != "included":
            spearman_text = f"insufficient (need {item['minimum_required']})"
            pearson_text = kendall_text = "NA"
        else:
            spearman_text = format_ci(item["spearman"])
            pearson_text = format_ci(item["pearson_log10_d"])
            value = item["kendall_tau_b"]["estimate"]
            kendall_text = "NA" if value is None else f"{value:.3f}"
        lines.append(
            f"| {item['dataset']} | {item['architecture']} | {item['split_seed']} | "
            f"{item['fraction']:.3g} | {item['n_subsets']} | {spearman_text} | "
            f"{pearson_text} | {kendall_text} |"
        )
    lines.extend([
        "",
        "## Stratified pooled association",
        "",
        f"Within-block rank association: **{format_ci(overall)}** over "
        f"{overall['n_subsets']} subsets in {overall['n_blocks']} blocks.",
        "This is not a raw pooled correlation: ranks are computed and centred inside each block first.",
        "The bootstrap directional probability is descriptive, not a classical p-value.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def make_plots(output_dir: Path, rows: list[dict[str, Any]], included_groups: list[list[dict[str, Any]]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required for plots; rerun with --no-plots") from error

    panels = sorted({(r["dataset"], r["architecture"]) for r in rows})
    columns = min(3, max(1, len(panels)))
    nrows = int(math.ceil(len(panels) / columns))
    fig, axes = plt.subplots(nrows, columns, figsize=(5.2 * columns, 4.1 * nrows), squeeze=False)
    fractions = sorted({r["fraction"] for r in rows})
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.88, max(1, len(fractions))))
    color_for = dict(zip(fractions, colors))
    for axis, panel in zip(axes.flat, panels):
        selected = [r for r in rows if (r["dataset"], r["architecture"]) == panel]
        for fraction in fractions:
            points = [r for r in selected if r["fraction"] == fraction]
            if not points:
                continue
            axis.errorbar(
                [r["d_alpha"] for r in points],
                [r["accuracy_mean"] for r in points],
                yerr=[r["accuracy_sd"] for r in points],
                fmt="o", ms=4.5, alpha=0.75, capsize=2,
                color=color_for[fraction], label=f"fraction={fraction:g}",
            )
        if all(r["d_alpha"] > 0.0 for r in selected):
            axis.set_xscale("log")
        axis.set_title(f"{panel[0]} / {panel[1]}")
        axis.set_xlabel("D(alpha), all-parameter recorded-trajectory score")
        axis.set_ylabel("Final test accuracy (%)")
        axis.grid(alpha=0.2)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(loc="best", fontsize=8, frameon=False)
    for axis in axes.flat[len(panels) :]:
        axis.set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "d_accuracy_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.4, 5.2))
    for rows_in_block in included_groups:
        d = np.asarray([r["d_alpha"] for r in rows_in_block])
        a = np.asarray([r["accuracy_mean"] for r in rows_in_block])
        x = average_ranks(d) / (len(rows_in_block) + 1.0)
        y = average_ranks(a) / (len(rows_in_block) + 1.0)
        axis.scatter(x, y, s=24, alpha=0.5)
    x_all, y_all = within_block_rank_arrays(included_groups)
    # Re-add 0.5 to centred ranks for an intuitive percentile plot.
    if np.dot(x_all, x_all) > 0:
        slope = float(np.dot(x_all, y_all) / np.dot(x_all, x_all))
        line_x = np.linspace(-0.45, 0.45, 100)
        axis.plot(line_x + 0.5, slope * line_x + 0.5, color="black", lw=1.5)
    axis.set_xlabel("Within-block percentile rank of D(alpha) (lower is better)")
    axis.set_ylabel("Within-block percentile rank of final accuracy")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "d_accuracy_within_block_ranks.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", nargs="+", required=True, help="Record file(s) or directories searched recursively")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--min-block-size", type=int, default=4)
    parser.add_argument("--skip-invalid", action="store_true", help="Exclude invalid records instead of failing closed")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.n_bootstrap < 100:
        parser.error("--n-bootstrap must be at least 100")
    if not 0.0 < args.confidence < 1.0:
        parser.error("--confidence must lie strictly between 0 and 1")
    if args.min_block_size < 3:
        parser.error("--min-block-size must be at least 3")
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = discover_records(args.input)
    if not paths:
        raise SystemExit("no record.json files found")
    rows, excluded = load_rows(paths, args.skip_invalid)
    if not rows:
        raise SystemExit("no valid completed records found")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["analysis_block_id"]].append(row)
    summaries, undersized = summarize_blocks(
        groups, n_bootstrap=args.n_bootstrap, confidence=args.confidence,
        seed=args.seed, min_block_size=args.min_block_size,
    )
    included_groups = [groups[item["analysis_block_id"]] for item in summaries if item["status"] == "included"]
    if not included_groups:
        raise SystemExit(
            f"no block has at least {args.min_block_size} unique paired subsets; "
            "generate more subset candidates"
        )
    overall = stratified_summary(
        included_groups, n_bootstrap=args.n_bootstrap,
        confidence=args.confidence, seed=args.seed + 7919,
    )

    write_observations_csv(output_dir / "paired_observations.csv", rows)
    write_block_csv(output_dir / "correlations_by_block.csv", summaries)
    write_markdown(output_dir / "correlation_table.md", summaries, overall)
    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_direction": "negative: lower D(alpha) should accompany higher final test accuracy",
        "statistical_unit": "unique subset; retraining replicates are averaged before point estimation",
        "bootstrap": {
            "method": "paired subset resampling, with replicate-mean resampling within subset; pooled result also resamples blocks",
            "draws": args.n_bootstrap,
            "confidence": args.confidence,
            "seed": args.seed,
            "directional_probability_is_p_value": False,
        },
        "inputs_discovered": len(paths),
        "valid_unique_subsets": len(rows),
        "excluded": excluded,
        "undersized_blocks": undersized,
        "blocks": summaries,
        "stratified_pooled_rank_association": overall,
    }
    atomic_write_json(output_dir / "correlation_summary.json", summary)
    if not args.no_plots:
        make_plots(output_dir, rows, included_groups)
    print(
        f"Analyzed {len(rows)} paired subsets in {len(groups)} blocks. "
        f"Stratified rank association={format_ci(overall)}. Outputs: {output_dir}"
    )


if __name__ == "__main__":
    main()
