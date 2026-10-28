# Faithful data selection

This repository reproduces the experiments in
[Towards Faithful Data Selection](docs/data_selection_manuscript.pdf).
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

## What each run writes

```text
*_indices.pt      selected pool-relative indices
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
