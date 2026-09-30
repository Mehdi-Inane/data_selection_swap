# Controlled experiment drivers

## SGD, Adam, and learning-rate experiment

`optimizer_lr.py` implements a paired factorial experiment without coupling
subset selection to final retraining:

1. **Reference:** train one seed-matched full-data trajectory for each
   `(dataset, architecture, seed, reference optimizer, reference LR)` cell.
2. **Selection:** select once from each reference trajectory with one of:
   `uniform` (equal checkpoint weights), `sgd` (`eta_k` weights), or `adam`
   (`eta_k` weights and the Adam `D_k^2` metric).
3. **Retraining:** reuse the frozen indices under every requested
   `(retraining optimizer, retraining LR)` cell.

This separates three questions that should not be collapsed into one label:

- Does the optimizer used to create the reference trajectory matter?
- Does including its learning-rate schedule in `D(alpha)` matter?
- Does the optimizer or learning rate used to train the selected subset matter?

Compare `uniform` and `sgd` on the **same** reference task to isolate checkpoint
learning-rate weighting. On an Adam reference, compare `sgd` and `adam` to
isolate the coordinatewise Adam preconditioner. Varying the reference base LR
changes the learned trajectory; because checkpoint weights are normalized,
uniformly rescaling a cosine schedule does not by itself change the relative
`eta_k` weights.

Every task has a configuration-derived ID. Completed artifact directories are
immutable, partial directories are never treated as results, and downstream
stages record SHA-256 hashes of their inputs. All retraining cells for a seed
construct byte-identical initial weights and use an explicit data-loader seed.
Interrupted reference and retraining tasks checkpoint their optimizer,
scheduler, random-generator, and epoch state and resume only when the task ID
and initialization fingerprint match; selection publishes atomically.
The primary outcome is final-epoch test accuracy, and the official test set is
evaluated exactly once per fit. `eval_every` controls train-loss progress
logging only; it is not a test-set evaluation interval.

### Inspect the plan locally

```bash
python scripts/experiments/optimizer_lr.py validate \
  --config scripts/experiments/optimizer_lr.example.json

python scripts/experiments/optimizer_lr.py list \
  --config scripts/experiments/optimizer_lr.example.json \
  --stage selection

python scripts/experiments/optimizer_lr.py run-task \
  --config scripts/experiments/optimizer_lr.example.json \
  --stage selection --task-id 0 \
  --data-dir /path/to/data --output-root /path/to/results --dry-run
```

The supplied example deliberately uses one subset fraction but is still a
substantial full factorial: 20 reference tasks, 50 selection tasks, and 200
retraining tasks. Reduce seeds/LR values for a smoke test before launching the
full plan. Add datasets or architectures to the corresponding top-level JSON
lists to cross them explicitly.

### Submit with dependencies on Slurm

```bash
scripts/slurm/submit_optimizer_lr.sh \
  scripts/experiments/optimizer_lr.example.json \
  "$SCRATCH/data_selection_swap/optimizer_lr_v1"
```

The submitter creates three arrays with `afterok` dependencies. Existing
successful task artifacts are skipped safely, so resubmitting a plan is
idempotent. Use `OPT_LR_MAX_PARALLEL` to change the array throttle and
`OPT_LR_DATA_DIR` to bypass node-local dataset staging.

After all arrays complete:

```bash
python scripts/experiments/optimizer_lr.py summarize \
  --config scripts/experiments/optimizer_lr.example.json \
  --output-root "$SCRATCH/data_selection_swap/optimizer_lr_v1"
```

This writes a row-level CSV and a JSON aggregation with the mean and **sample**
standard deviation across seeds. Missing runs remain visible in the row-level
file rather than being silently dropped.

## Algorithm 2 versus Algorithm 3

`compare_swap_rules.py` is a separate paired experiment for the manuscript's
two swap rules. Its configuration permits only one reference optimizer/LR,
one scoring objective, and one retraining optimizer/LR. It then generates two
selection cells from the same trajectory:

- `ranked`: Algorithm 3, which tests the top gradient-ranked exchange;
- `best`: Algorithm 2, which searches for the best exact exchange.

The validator requires `max_pairs=0`. A truncated pair search is useful as a
runtime approximation, but it is not the full Algorithm 2 requested by this
experiment and therefore cannot be mislabeled here. Both cells use the same
dataset split, seed, budget, checkpoints, clean scoring view, initial subset,
and deterministic retraining initialization.

Inspect and submit the example with:

```bash
python scripts/experiments/compare_swap_rules.py validate \
  --config scripts/experiments/swap_rules.example.json

scripts/slurm/submit_swap_rules.sh \
  scripts/experiments/swap_rules.example.json \
  "$SCRATCH/data_selection_swap/swap_rules_v1"
```

After completion, run:

```bash
python scripts/experiments/compare_swap_rules.py summarize \
  --config scripts/experiments/swap_rules.example.json \
  --output-root "$SCRATCH/data_selection_swap/swap_rules_v1"
```

The outputs include row-level and seed-aggregate test accuracy, `D(alpha)`,
number of swaps and pair searches, optimality certificate, exact-search flag,
scoring time, and swap-search time. A paired JSON file reports Algorithm 2
minus Algorithm 3 differences and verifies that each pair used the same
reference and byte-identical retraining initialization.
