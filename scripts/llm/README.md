# `scripts/llm` — LLM instruction-tuning selection entry points

Thin CLIs over the isolated `data_selection_swap.llm` subpackage. They run on a
**gradient-feature store** (a directory with `manifest.json`, `features/<ckpt>.pt`,
`val/<task>/<subtask>/<ckpt>.pt`, optional `pool.jsonl`). Nothing here imports the
CV experiment code, and nothing here writes outside `--out`.

## Environment

Use the **separate** `llm_select` env (`requirements-llm.txt`). The selector needs
only `torch`; LESS/eval dependencies are for Phase 0 only.

## Quick smoke test (no GPU, no dataset)

```bash
python scripts/llm/build_datastore.py --out /tmp/store --n 5000 --d 512 --checkpoints 4
python scripts/llm/select_swap.py     --store /tmp/store --out /tmp/runs/swap_tydiqa \
    --setting targeted --task tydiqa --fraction 0.05 --rule ranked
python scripts/llm/select_swap.py     --store /tmp/store --out /tmp/runs/faithful \
    --setting faithful --fraction 0.05 --rule best
python scripts/llm/select_baselines.py --store /tmp/store --out /tmp/runs/less_tydiqa \
    --method less --task tydiqa --fraction 0.05
python scripts/llm/select_baselines.py --store /tmp/store --out /tmp/runs/random \
    --method random --fraction 0.05
python scripts/llm/analyze_selection.py /tmp/runs --csv /tmp/summary.csv
```

`select_swap.py --setting targeted` prints a `diagnostics.json` comparing the swap
subset to LESS's independent top-k on the *same* influence score (`jaccard_vs_swap`,
`swap_redundancy` vs `less_redundancy`) — the core comparison, available immediately.

## Scripts

| Script | Purpose |
|---|---|
| `build_datastore.py` | Make a synthetic store (smoke), or `--from-less` to ingest real LESS gradient dumps. |
| `select_swap.py` | Faithful (A) or targeted (B) selection; writes indices/JSONL + diagnostics vs LESS. |
| `select_baselines.py` | LESS top-k and random baselines, same output schema. |
| `run_pipeline.py` | Orchestrate the full LESS + selection pipeline from a JSON config (dry-run default). |
| `analyze_selection.py` | Flatten many `diagnostics.json` into one table/CSV. |
| `analyze_budget_curve.py` | Budget-efficiency curves (accuracy or objective value) vs selection fraction. |
| `analyze_d_accuracy.py` | Spearman correlation of a selection score against downstream accuracy. |

## Reproducing / running the pipeline (Phase 0/2, GPU)

```bash
python scripts/llm/run_pipeline.py --config scripts/llm/pipeline.example.json --list   # show stages
python scripts/llm/run_pipeline.py --config scripts/llm/pipeline.example.json          # dry-run
python scripts/llm/run_pipeline.py --config scripts/llm/pipeline.example.json --run    # execute
# after eval writes runs/.../accuracy.json:
python scripts/llm/analyze_budget_curve.py runs --value-key accuracy --out bc.png
python scripts/llm/analyze_d_accuracy.py runs --score-key value --group method
```

See `docs/llm_less_protocol.md` for the `PipelineConfig` JSON schema and stage list.

Output layout per run dir: `selected_indices.pt`, `selected.jsonl` (if the store has a
pool), `diagnostics.json`. Selected JSONL is LESS-compatible for the training step.
