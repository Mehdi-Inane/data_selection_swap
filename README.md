# Towards Faithful Data Selection: Statistical Limits and Scalable Algorithms

This repository reproduces the experiments in
[Towards Faithful Data Selection: Statistical Limits and Scalable Algorithms](docs/data_selection_manuscript.pdf) by Ahmed-Mehdi Inane*, Mahdi Haghifam*, Gintare Karolina Dziugaite, and Ioannis Mitliagkas.

Gradient Swap selects a fixed-size training subset whose gradients stay close
to the full-data gradients along a saved training trajectory. The subset is
then evaluated by retraining a fresh model.

The published comparison uses CIFAR-10 and CIFAR-100 with ResNet-18, VGG-16,
and Inception. The selector in those runs is full-network Gradient Swap
(`scripts/select/kl_full.py`). Random, Herding, Moderate Coreset, EL2N, MoSo,
GraphCut, and Facility Location are the static baselines.

## Install

Python 3.10 or newer is recommended (3.9 is the minimum). From the repository
root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

A CUDA build of PyTorch is required for the full experiments. CPU is enough
for the unit tests and for inspecting saved results.

GraphCut and Facility Location need `apricot-select`, which is in
`requirements.txt`. GradMatch is not part of the manuscript and needs a
separate [CORDS](https://github.com/decile-team/cords) install. Files under
`scripts/legacy/` are old prototypes and are not used below.

## Data and results

CIFAR-10 and CIFAR-100 are downloaded by passing `--download`. Point
`--data_dir` at the directory you want them stored in:

```text
data/cifar10/
data/cifar100/
```

Every run writes under a results root. The root is `RESULTS_ROOT` if that
variable is set, otherwise `SCRATCH`, otherwise `./results` in the current
working directory.

```text
results/data_selection_swap/<dataset>/<architecture>/checkpoints/seed_42/
results/gradmatch_swap/<dataset>/<architecture>/<budget>/seed_<seed>/
results/data_selection_baselines/<method>/<dataset>/<architecture>/<budget>/seed_<seed>/
```

`<budget>` is `int(fraction * 45000)` on CIFAR, because selection uses the
90% training pool. Pass `--scratch /path/to/results` to analysis scripts when
the results root is not the one above.

On a cluster, export `RESULTS_ROOT` before `sbatch`. Set `CONDA_MODULE` and
`CONDA_ENV` only if the job should load a module and activate a conda
environment. Otherwise the job uses the Python already on `PATH`. Optional
dataset archives are `CIFAR10_ARCHIVE`, `CIFAR100_ARCHIVE`, and
`IMAGENET_RAW_DIR` (ImageNet is not used in the manuscript).

## Experiment configs

Each experiment is one YAML file: `configs/paper/` holds the manuscript's,
`configs/revision/` the implementation guide's Exp A–C. A config lists, for
each stage, the script, a grid, and the arguments. Task i of a stage is the
i-th grid point (last axis fastest), and every argument becomes a flag of
the script, so the file shows every setting the run receives. What a config
cannot set is fixed in code: SGD with Nesterov momentum 0.9, weight decay
5e-4, the cosine schedule, augmentation, and 300 retraining epochs.

| Config | Manuscript | Stages (tasks) |
|---|---|---|
| `paper/reference_trajectories.yaml` | step 1, Appendix G.1 | reference (6) |
| `paper/main_comparison.yaml` | Figure 2 | gradient_swap (120), baselines (840) |
| `paper/checkpoint_ablation.yaml` | Table 1, Appendix J | gradient_swap (600) |
| `paper/faithfulness_correlation.yaml` | Figure 1, Appendix H | score (6) |
| `paper/warm_start.yaml` | Appendix I | gradient_swap (20), score (1) |
| `paper/jl_ablation.yaml` | Appendix K | gradient_swap (120) |
| `revision/exp_a_low_budget.yaml` | guide Exp A | cached_sketch (360), exact_algorithm1 (20), exact_second_order (80), baselines (420) |
| `revision/cifar100_second_order.yaml` | CIFAR-100: Random, GraphCut, Algorithm 1, second-order k = 8, 16, class-balanced k = 16 | reference (2), single_swap (8), second_order (16), second_order_class_balanced (8), baselines (16) |
| `revision/cifar10_lt_second_order.yaml` | CIFAR-10-LT: the same methods | reference (2), single_swap (8), second_order (16), second_order_class_balanced (8), baselines (16) |
| `revision/exp_b_contamination.yaml` | guide Exp B (planned) | reference, gradient_swap, baselines |
| `revision/exp_c1_coarse_cifar100.yaml` | guide Exp C.1 (planned) | reference, gradient_swap, baselines |
| `revision/exp_c2_long_tail.yaml` | guide Exp C.2 (planned) | reference, gradient_swap, baselines, small_validation |
| `revision/exp_c3_scalability.yaml` | guide Exp C.3 (planned) | reference, gradient_swap, baselines |

```bash
python scripts/experiments/run_config.py validate configs/*/*.yaml   # --check-flags: also against each script's --help
python scripts/experiments/run_config.py list configs/paper/main_comparison.yaml gradient_swap
python scripts/experiments/run_config.py run-task configs/paper/main_comparison.yaml gradient_swap 0
scripts/slurm/submit_config.sh configs/paper/main_comparison.yaml gradient_swap   # one Slurm array
scripts/slurm/submit_config.sh configs/paper/main_comparison.yaml                 # one array per stage
```

`cifar10_lt` and `cifar100_lt` are the standard long-tailed CIFAR pools:
class c keeps n_max · 50^(−c/(C−1)) training images (13,996 and 12,608),
one fixed draw (`common.LONG_TAIL_RATIO`, `LONG_TAIL_SEED`), with the
balanced test set. Every script accepts them as `--dataset`; each needs its
own reference trajectory.

The paper configs run what the launchers in steps 1–6 run, flag for flag
(`tests/test_experiment_configs.py` checks this), with one deliberate
difference: every seed reads the seed-42 trajectory, as Appendix G.1
describes, where the launchers default to each seed's own. A `planned`
config describes an experiment whose code does not exist yet; it validates
and lists, but does not run.

## Protocol used in the paper

Section 5 and Appendix G of the manuscript:

| | |
|---|---|
| Data | CIFAR-10 and CIFAR-100, 90/10 train/validation split |
| Models | ResNet-18, VGG-16, Inception |
| Training | SGD, Nesterov momentum 0.9, learning rate 0.01, weight decay `5e-4`, cosine schedule, 300 epochs, batch size 128, cross-entropy |
| Budgets | 10%, 20%, 40%, and 50% of the training pool |
| Seeds | 42, 43, 44, 45, 46 |
| Scoring view | clean test images (`--scoring_transform test`) |
| Gradient Swap | full network, at most 200 swaps, SGD checkpoint weights `w_k = eta_k` |

Appendix G.1 trains **one** reference trajectory, on seed 42, and reuses it
for every selection seed. The other four seeds therefore score a pool that is
not identical to the reference pool. Reproduce that by pointing
`--checkpoint_dir` at `checkpoints/seed_42` and passing
`--allow_manifest_mismatch`. Leaving the flag off keeps the stricter check:
the trajectory seed and pool must match the selection seed.

The same optimizer and schedule are used to train the reference model and to
retrain on a selected subset. These grids take hundreds of GPU-days. The
smoke test below checks that a single command runs.

## Smoke test

One epoch, one seed, selection only (no 300-epoch retrain):

```bash
python scripts/train/full_data.py \
  --dataset cifar10 \
  --architecture resnet18 \
  --data_dir data/cifar10 \
  --save_dir results/data_selection_swap/cifar10/resnet18/checkpoints/seed_42 \
  --num_epochs 1 \
  --seed 42 \
  --download

python scripts/select/kl_full.py \
  --dataset cifar10 \
  --architecture resnet18 \
  --data_dir data/cifar10 \
  --checkpoint_dir results/data_selection_swap/cifar10/resnet18/checkpoints/seed_42 \
  --fraction 0.1 \
  --num_checkpoints 1 \
  --optimizer sgd \
  --metric none \
  --max_iters 2 \
  --seed 42 \
  --selection_only
```

The published runs use 300 epochs and `--max_iters 200`. Drop
`--num_epochs` and `--selection_only` for that.

## 1. Reference trajectory

Train the seed-42 full-data model and store a checkpoint at the end of every
epoch, with the learning rate of that epoch. One job per dataset and
architecture (6 jobs):

```bash
SEEDS_OVERRIDE=42 sbatch --array=0-5 scripts/slurm/run_full_data.sh
```

Locally, repeat the smoke-test training command without `--num_epochs`, once
per dataset and architecture. The save directory must stay
`.../checkpoints/seed_42`. Later stages read that directory.

## 2. Main comparison (Figure 2)

Full-network Gradient Swap at **K = 1** (the epoch-1 checkpoint), plus the
seven baselines, at every fraction and seed.

One Gradient Swap cell:

```bash
python scripts/select/kl_full.py \
  --dataset cifar10 \
  --architecture resnet18 \
  --data_dir data/cifar10 \
  --checkpoint_dir results/data_selection_swap/cifar10/resnet18/checkpoints/seed_42 \
  --fraction 0.1 \
  --num_checkpoints 1 \
  --checkpoint_selection spread \
  --checkpoint_weighting point \
  --optimizer sgd \
  --metric none \
  --max_iters 200 \
  --seed 43 \
  --allow_manifest_mismatch \
  --download
```

Seed 42 does not need `--allow_manifest_mismatch`. Seeds 43–46 do.

The full grid is 2 datasets × 3 architectures × 4 fractions × 5 seeds:

```bash
sbatch scripts/slurm/run_kl_full_selection.sh
sbatch scripts/slurm/run_baselines.sh
```

`run_kl_full_selection.sh` defaults to `NUM_CHECKPOINTS=1`. Baselines are
Random, Herding, Moderate, EL2N, MoSo, GraphCut, and Facility Location.
Accuracy curves are:

```bash
python scripts/plot/result_plots.py \
  --dataset cifar10 \
  --architecture resnet18 \
  --scratch results
```

Repeat `--dataset` and `--architecture` for the other five panels. Figures
land in `results/data_selection_baselines/plots/`.

## 3. Checkpoint count (Table 1 and Appendix J)

Same training setup, with K in {1, 4, 10, 15, 20}. K = 1 is the epoch-1
checkpoint. Larger K takes checkpoints spaced across the 300 epochs.
`run_kl_full_selection_checkpoints.sh` is this experiment (600 tasks).
`run_checkpoint_ablation.sh` is an older last-layer sweep and is not Table 1
or Appendix J.

```bash
sbatch scripts/slurm/run_kl_full_selection_checkpoints.sh

python scripts/plot/checkpoint_ablation.py \
  --dataset cifar10 \
  --architectures resnet18 vgg16 inception \
  --checkpoints 1 4 10 15 20 \
  --scratch results
```

Run the plot once with `--dataset cifar100` for Appendix J's second table.
Table 1 is the CIFAR-10 ResNet-18 rows.

## 4. Faithfulness correlation (Figure 1 and Appendix H)

After the subsets in Section 2 exist, score every saved subset on **all 300**
reference checkpoints. One point is one method and seed at the 10% budget.
The reference is again `checkpoints/seed_42`.

```bash
python scripts/analyze/score_subsets.py \
  --dataset cifar10 \
  --architecture resnet18 \
  --data_dir data/cifar10 \
  --checkpoint_dir results/data_selection_swap/cifar10/resnet18/checkpoints/seed_42 \
  --fractions 0.1 \
  --seeds 42 43 44 45 46 \
  --num_checkpoints 0 \
  --weighting sgd \
  --metric none \
  --checkpoint_weighting point \
  --scoring_transform test \
  --allow_manifest_mismatch \
  --scratch results \
  --download
```

`--num_checkpoints 0` means every stored checkpoint. The table is
`results/data_selection_baselines/d_exact_cifar10_resnet18_f0.1.csv`.

```bash
sbatch scripts/slurm/run_score_subsets.sh

python scripts/plot/d_accuracy_scatter.py \
  --input results/data_selection_baselines/d_exact_cifar10_resnet18_f0.1.csv \
  --dataset cifar10 \
  --architecture resnet18 \
  --scratch results
```

The Slurm array is one task per dataset and architecture (6 tasks). Repeat
the plot command for each CSV. Figure 1 in the main text is ResNet-18 on
CIFAR-10 and VGG-16 on CIFAR-100. Appendix H is all six panels.

## 5. Warm start (Appendix I)

Start Gradient Swap from a baseline subset instead of a random subset.
The swap uses K = 10 checkpoints. The reported objective uses all 300
checkpoints. The grid is CIFAR-10, ResNet-18, 10% of the pool, started from
GraphCut, Herding, Random, and EL2N, seeds 42–46.

Run Section 2 for that cell first, then:

```bash
sbatch scripts/slurm/run_kl_full_from_baselines.sh
sbatch scripts/slurm/run_score_warmstart.sh
```

The second job writes the before/after table under
`results/data_selection_baselines/baseline_warmstart/`.

One local warm start, from EL2N, seed 42:

```bash
python scripts/select/kl_full.py \
  --dataset cifar10 \
  --architecture resnet18 \
  --data_dir data/cifar10 \
  --checkpoint_dir results/data_selection_swap/cifar10/resnet18/checkpoints/seed_42 \
  --fraction 0.1 \
  --num_checkpoints 10 \
  --optimizer sgd \
  --metric none \
  --max_iters 200 \
  --seed 42 \
  --init_method el2n \
  --init_indices results/data_selection_baselines/el2n/cifar10/resnet18/4500/seed_42/el2n_4500_indices.pt \
  --download
```

## 6. Johnson–Lindenstrauss embeddings (Appendix K)

Count-sketch the full-network gradient to width 1024, 2048, or 4096, and
compare with the exact aggregate (`p = 0`, which is the K = 1 and K = 4
runs from Section 3). The manuscript table is CIFAR-10, ResNet-18, fractions
10–50%, seeds 42–46, K in {1, 4}.

```bash
DATASETS=cifar10 ARCHITECTURES=resnet18 \
  sbatch --array=0-119 scripts/slurm/run_kl_full_jl_ablation.sh

python scripts/plot/jl_ablation.py \
  --dataset cifar10 \
  --architecture resnet18 \
  --checkpoints 1 4 \
  --proj_dims 0 1024 2048 4096 \
  --scratch results
```

The default array `0-719` also runs CIFAR-100 and the other two
architectures. Those extra cells are not in Appendix K.

## 7. Selected examples (Appendix L)

Six training images from each method's 10% subset, seed 42, for every
dataset and architecture. Indices are mapped back through the seed-42 split
and shown without crop, flip, or normalization.

```bash
sbatch scripts/slurm/run_selected_examples.sh

python scripts/plot/selected_examples.py \
  --dataset cifar10 \
  --architecture resnet18 \
  --fraction 0.1 \
  --seed 42 \
  --data_dir data/cifar10 \
  --scratch results \
  --download
```

## 8. Second-order k×k Gradient Swap (not in the manuscript)

Algorithm 1 ranks exchanges by the linear score `s_r` alone. High-norm
examples dominate that ranking, their own curvature `C_rr/m²` makes the
top-ranked exchange increase `G_T`, and Algorithm 1 stops at that first
rejection. `--swap_rule second_order` shortlists by each example's exact
single-point change instead, `d_i = -s_i + C_ii/m²` for the `k` drops and
`a_j = s_j + C_jj/m²` for the `k` adds, and applies the best of the
shortlisted exchanges if its exact change `d_i + a_j - 2C_ij/m²` is below
`-accept_tol`, so `G_T` strictly decreases. The rule is
`selection/second_order.py`; both engines are in `selection/kl_full.py`.

| | Mode A: cached sketch | Mode B: exact |
|---|---|---|
| Selected by | `--proj_dim p` with p > 0 | `--proj_dim 0` |
| Built once | sketched, centred per-example gradients `[n, K·p]` in float32 (205 MB at n = 50,000, p = 1024, K = 1) | per-example curvature `C_rr`, `n` scalars |
| Per sweep | one matvec and a `k×k` matmul, no network pass | one JVP sweep over the pool, plus per-example gradients of the `2k` shortlisted images |

Options: `--shortlist_k` (default 32); `--class_balanced` with
`--class_shortlist_k` (default 16), which starts from a draw stratified at
`per_class_budget` quotas and only exchanges examples of the same class;
`--sketch_cache`, which runs Algorithm 1 on the Mode A cache. The method
name gains `_so<k>`, `_cb<k_c>` or `_cache`. An Algorithm 1 run that leaves
these options at their defaults keeps its configuration fingerprint.

`kl_full` also saves `*_init_indices.pt`, the starting subset. Every
accepted entry of `*_objective.json` records `dropped` and `added`, so
`subsets.replay_swaps` rebuilds the subset after any number of swaps. Its
timing file adds `initial_G_T`, `final_G_T`, `relative_G_T_drop_pct`,
`jaccard_with_init`, `sweeps_run` and `selection_forward_calls` (0 for
Mode A). Every selector's timing file now carries `ref_epochs_used`,
`ref_train_sec`, `extra_probe_epochs`, `extra_train_sec`,
`scoring_or_sketch_sec`, `swap_selection_sec`, `total_selection_sec` and the
peak CUDA memory of each phase (`peak_scoring_vram_mb`, ...).

Exp A, the comparison at the new low budgets, is
`configs/revision/exp_a_low_budget.yaml` (see Experiment configs):

```bash
CONFIG=configs/revision/exp_a_low_budget.yaml
scripts/slurm/submit_config.sh $CONFIG cached_sketch        # every variant, p = 1024
scripts/slurm/submit_config.sh $CONFIG exact_algorithm1     # ResNet-18, R^d
scripts/slurm/submit_config.sh $CONFIG exact_second_order   # ResNet-18, R^d
scripts/slurm/submit_config.sh $CONFIG baselines            # the 2% and 5% budgets
```

Pin one GPU type for all of them (add `--gres=gpu:<type>:1`) when the
timings and memory are to be compared. The checks are in
`tests/test_second_order_swap.py` and need only torch:

```bash
python -m unittest tests.test_second_order_swap tests.test_swap_algorithms
```

## What each run writes

```text
*_indices.pt      selected pool-relative indices
*_init_indices.pt starting subset of a Gradient Swap run
*_timing.json     scoring and selection time
*_objective.json  Gradient Swap objective history
*_metrics.json    test accuracy while retraining the subset
*_model.pth       retrained model
```

`scripts/analyze/aggregate_comparison.py --dataset cifar10 --scratch results`
prints mean and standard deviation of accuracy and time across seeds.

## Experiments that are not in the manuscript

`scripts/experiments/` has three extra drivers: an optimizer and learning-rate
factorial, a comparison of the two swap rules, and a controlled correlation
between `D(alpha)` and accuracy on networks without BatchNorm. They are
documented in [scripts/experiments/README.md](scripts/experiments/README.md)
and [scripts/experiments/README_D_CORRELATION.md](scripts/experiments/README_D_CORRELATION.md).
The last-layer selector `scripts/select/kl.py` is also not the Gradient Swap
used in Section 5.

## Reproducibility notes

- Selection, retraining and scoring all use the whole training set as the pool (no
  train/val split). Use the same `--seed` for the reference trajectory, selection,
  and retraining.
- New trajectory manifests store and consumers verify a hash of the ordered
  pool indices; a seed label alone is not accepted as proof of pool identity. A
  trajectory trained before the split was removed (a 45,000-image pool) is rejected,
  so retrain it with `scripts/slurm/run_full_data.sh`. Scoring takes the trajectory's
  training batch size from `--reference_batch_size` (default 128), separate from the
  scoring `--batch_size`.
- Use the same `--scoring_transform` for every method. `test` gives a stable,
  unaugmented scoring view; `train` reproduces the earlier runs.
- Prefer `--checkpoint_selection spread`; `prefix` intentionally reproduces the
  early-checkpoint experiments.
- Keep `--checkpoint_weighting point` for the exact `eta_k` formulas above. Use
  `window` only when treating a sparse checkpoint as representative of a range of
  optimizer steps.
- The files in `scripts/legacy/` are retained for provenance and are not part of
  the maintained workflow.
- Exact Algorithm 2 has quadratic worst-case pair count per iteration and can
  be substantially slower than Algorithm 3. Use a small smoke configuration
  before allocating the full grid; any positive `max_pairs` is reported as a
  truncated, non-certifying search.
