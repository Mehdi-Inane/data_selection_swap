# LLM fine-tuning selection protocol (vs. LESS)

Protocol for extending the faithful/trajectory selector to LESS-style instruction
tuning. The plan lives in
`~/.opencode/plan/2026-10-19-less-gradient-swap-implementation-plan.md`; this is the
in-repo runbook. **Everything here is additive and isolated** from the CV experiments.

## Guardrails (do not violate)

- LLM code lives only in `src/data_selection_swap/llm/`, `scripts/llm/`, `tests/llm/`.
- Do **not** edit `common.py`, `paths.py`, `objective.py`, `trajectory.py`,
  `selection/*`, `scripts/{select,train,analyze,plot,experiments}/*`, or the core
  `requirements.txt`. The `llm` package must import only `numpy`/`torch` (enforced by
  `tests/llm/test_llm_datastore.py::IsolationTest`).
- LESS stays an external checkout in `vendor/LESS/` (gitignored). Adapters read its
  **outputs** (files) into the `FeatureStore` layout; we never import or fork LESS.
- Keep the CV suite green: the LLM work must not change any existing CV test result.

## Method (one objective, two settings)

Fixed-size subset objective on the LESS trajectory feature store:

```
f(I) = (1/m^2) Σ_{i,j∈I} K_ij − (2/m) Σ_{i∈I} c_i,   K_ij = Σ_k w_k ⟨Φ_i, Φ_j⟩_{M_k}
```

- **Setting A (`--setting faithful`)** — `c` from the pool-mean gradient; `f` equals the
  manuscript discrepancy `D(I)` up to a constant (proved in `tests/llm`). Validation-free.
- **Setting B (`--setting targeted`)** — `c` is LESS's cosine influence to the validation
  gradient (`max` over subtasks). **LESS = `top-k` on `c`**; the swap adds the `K` redundancy
  term. This is the apples-to-apples comparison.
- `--normalize l2` (default) = LESS's cosine (the sequence-length correction); `--normalize
  none` is the raw ablation. `--rule best` is exact (Algorithm 2); `--rule ranked` is cheap
  (Algorithm 3). Decisions are float64.

## Phases

### Phase 0 — reproduction gate (needs GPU + `less_data`)  [importer DONE; runs pending]
1. `conda create -n llm_select ... && pip install -r requirements-llm.txt && pip install -e .`
2. `git clone https://github.com/princeton-nlp/LESS vendor/LESS`; download
   `princeton-nlp/less_data`; adapt Pythia formatting in the external checkout only.
3. Run LESS's own pipeline (external) to produce its gradient dumps — warmup LoRA (5%, 4
   epochs) → `get_train_lora_grads.sh` (train, dim 8192, `dim8192/all_orig.pt`) →
   `get_eval_lora_grads.sh` (per-task validation gradients) → into e.g. `GRADS/`.
4. **Ingest (implemented):** `build_datastore.py --from-less` reads those dumps into a
   `FeatureStore` (concatenates the train files into one pool, reshapes flattened
   validation gradients into subtasks). `tests/llm/test_llm_less_import.py` proves the
   imported `targeted_score` equals LESS `matching.py` bit-for-bit, so our `less` baseline
   (`select_baselines.py --method less`) *is* LESS's selection.
   ```bash
   python scripts/llm/build_datastore.py --out llm_store --from-less --dim 8192 \
     --ckpts 105 211 317 420 --checkpoint-weights 1.6877e-05 1.2859e-05 7.7030e-06 2.5616e-06 \
     --train-file-names flan_v2 cot dolly oasst1 \
     --train-grad-template 'GRADS/{name}-ckpt{ckpt}-adam/dim{dim}' \
     --val-grad-template      'GRADS/{task}-ckpt{ckpt}-sgd/dim{dim}' \
     --tasks mmlu tydiqa bbh \
     --pool-files flan_v2=data/train/flan_v2.jsonl cot=data/train/cot.jsonl \
                  dolly=data/train/dolly.jsonl oasst1=data/train/oasst1.jsonl
   ```
5. Reproduce **Random** and **LESS** (TyDiQA, BBH @ 5%, 3 seeds): select with
   `select_baselines.py` → hand the `selected.jsonl` to LESS `lora_train.sh` + open-instruct
   eval. **Gate:** within ±1–2 pts of the paper's Pythia rows; freeze into
   `tests/llm/fixtures/less_targets.json`.

### Phase 1 — port correctness  [DONE, no GPU]
`tests/llm/*` pass on the real package: identity to `D(I)`, exact swap deltas,
brute-force optimality, LESS top-k parity, store round-trip + provenance, CV isolation.

### Phase 2 — Setting B headline
For each model (Pythia-410M → 1.4B/2.8B → Mistral-7B), one store per (model, seed) reused by
all methods/budgets `{1,2,5,10,25}%`: run `select_swap.py` (targeted) + `select_baselines.py`
(less/random) + vendored BM25/DSIR/RDS/IFD/INFLUEN2 → train → eval → `analyze_selection.py`.
Report accuracy-vs-budget + AUC. Win condition: swap ≥ LESS, largest gap at 1–5%.

### Phase 3 — Setting A generality, Phase 4 — analysis, Phase 5 — scale
See the plan. `scoring.py` provides redundancy / Jaccard / Spearman diagnostics used there.

## Cost note

LESS releases **no prebuilt gradient store** — warmup + per-example gradient extraction is the
dominant cost, so build one store per (model, seed) and reuse it across every method, budget,
task, and both settings.
