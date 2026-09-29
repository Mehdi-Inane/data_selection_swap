# Faithful data selection

This repository implements the trajectory-based data-selection framework in
[the manuscript](docs/data_selection_manuscript.pdf). It selects a fixed-size
training subset whose gradients reproduce the full-data gradients along a saved
training trajectory, then evaluates the subset by retraining a fresh model.

The maintained code supports CIFAR-100 and ImageNet, a factored last-layer method,
a matrix-free full-network method, and Random, Herding, Moderate Coreset, EL2N,
MoSo, GraphCut, Facility Location, and GradMatch baselines. All maintained methods
share the same split, model construction, retraining loop, timing convention, and
artifact layout described in
[the experimental protocol](docs/experimental_protocol.md).

## Optimizer-aware scoring: SGD and Adam

**The SGD and Adam scores requested for this project are implemented directly.**
At checkpoint `k`, define the difference between the selected-subset and full-data
mean gradients as

```text
Delta g_k(I) = mean_{i in I} grad l(theta_k, z_i)
               - mean_{i in S} grad l(theta_k, z_i).
```

The available objectives are:

| Mode | Objective | Equivalent gradient-feature scaling |
|---|---|---|
| `uniform` | `sum_k ||Delta g_k(I)||^2` | `1` |
| `sgd` | `sum_k eta_k ||Delta g_k(I)||^2` | `sqrt(eta_k)` |
| `adam` | `sum_k eta_k ||Delta g_k(I) / (sqrt(vhat_k) + eps)||^2` | `sqrt(eta_k) / (sqrt(vhat_k) + eps)` per coordinate |
| `adam_update` | `sum_k eta_k^2 ||Delta g_k(I) / (sqrt(vhat_k) + eps)||^2` | update-mismatch ablation, not the manuscript KL score |

`scripts/train/full_data.py` keeps its checkpoints as ordinary model state dicts
and writes `trajectory.json`, including the learning rate `eta_k`, step, and epoch
for every saved SGD checkpoint. `scripts/train/reference.py --optimizer adam`
additionally saves Adam's second-moment accumulator and optimizer step. The scorer
bias-corrects it to obtain `vhat_k` before constructing the coordinatewise metric.

The implementation does not need to materialize the scaled per-example gradients.
It applies the mathematically equivalent checkpoint weight `w_k = eta_k` and Adam
metric

```text
M_k[j] = 1 / (sqrt(vhat_k[j]) + eps)^2
```

inside the same matrix-free kernel-vector product and swap algorithm. Positive
checkpoint weights are normalized to sum to one; this changes the reported scale
but not the selected subset or any swap decision.

Use these flags to choose the score:

```text
--optimizer uniform                         unweighted reference
--optimizer sgd  --metric none              SGD score D_eta
--optimizer adam --metric adam              Adam score D_Adam
--checkpoint_weighting point                use eta_k exactly (default)
--checkpoint_weighting window               optional sum of eta over each checkpoint window
```

For Adam, `--metric auto` is equivalent to `--metric adam`. Adam's coordinatewise
metric is incompatible with the independent factor projections used by
`--proj_dim`, so Adam runs require `--proj_dim 0`.

## Repository layout

```text
src/data_selection_swap/
  common.py                 datasets, models, retraining, timing, and artifacts
  trajectory.py             eta_k weights, Adam vhat metric, checkpoint resolution
  objective.py              exact full-network D(alpha) evaluation
  selection/
    kl.py                    factored last-layer KL-Faithful selector
    kl_full.py               matrix-free all-parameter selector
    *.py                     maintained baseline implementations
  training/                 full-data and optimizer-aware trajectory training
  analysis/                 discovery, aggregation, scoring, and plotting logic
scripts/
  train/                    training entry points
  select/                   selection entry points
  analyze/                  aggregation and objective-scoring entry points
  plot/                     plotting entry points
  slurm/                    cluster launchers and dataset staging
  legacy/                   superseded prototypes with hard-coded assumptions
docs/                       manuscript and experimental protocol
notebooks/                  exploratory notebooks
```

The reusable implementation lives under `src/`; files under `scripts/` are thin,
human-facing entry points.

## Installation

Python 3.10 or newer is recommended. Create or activate an environment, then run:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

The code imports the CORDS `ResNet18` and GradMatch implementation. Use the same
CORDS installation as the experiment environment (the cluster launchers activate
the `gradmatch` Conda environment). The editable install is important: it makes
every script import the package from `src/data_selection_swap` consistently.

Dataset roots should have these layouts:

```text
/path/to/cifar100/                torchvision CIFAR-100 root
/path/to/imagenet/train/<class>/  ImageNet training images
/path/to/imagenet/val/<class>/    ImageNet validation images
```

Pass `--download` for CIFAR-100 if it is not already present. ImageNet must be
staged manually or through `scripts/slurm/stage_dataset.sh`.

## Run an SGD experiment

### 1. Train the shared full-data trajectory

```bash
python scripts/train/full_data.py \
  --dataset cifar100 \
  --data_dir /path/to/cifar100 \
  --save_dir /path/to/sgd_checkpoints \
  --lr 0.01 \
  --batch_size 128 \
  --seed 42
```

This writes model checkpoints, `trajectory.json` with each checkpoint's `eta_k`,
`train_time.json`, and `train.log`.

### 2. Select with the learning-rate-weighted SGD score

```bash
python scripts/select/kl.py \
  --dataset cifar100 \
  --data_dir /path/to/cifar100 \
  --checkpoint_dir /path/to/sgd_checkpoints \
  --fraction 0.1 \
  --num_checkpoints 10 \
  --checkpoint_selection spread \
  --checkpoint_weighting point \
  --optimizer sgd \
  --metric none \
  --rule ranked \
  --seed 42
```

Use `--rule best` for exact best-swap search in the factored last-layer kernel.
It is certifying but can be much more expensive. Use `--selection_only` to save
indices and timing without the final retraining stage.

## Run an Adam experiment

### 1. Train an optimizer-aware Adam reference trajectory

```bash
python scripts/train/reference.py \
  --dataset cifar100 \
  --data_dir /path/to/cifar100 \
  --save_dir /path/to/adam_checkpoints \
  --optimizer adam \
  --lr 0.001 \
  --num_checkpoints 10 \
  --spacing linear \
  --seed 42
```

Each checkpoint record contains the model weights, step, learning rate, Adam
hyperparameters, and second-moment state needed to reconstruct `vhat_k`.

### 2. Select with the Adam-preconditioned score

```bash
python scripts/select/kl.py \
  --dataset cifar100 \
  --data_dir /path/to/cifar100 \
  --checkpoint_dir /path/to/adam_checkpoints \
  --fraction 0.1 \
  --num_checkpoints 10 \
  --checkpoint_weighting point \
  --optimizer adam \
  --metric adam \
  --adam_eps 1e-8 \
  --proj_dim 0 \
  --retrain_optimizer adam \
  --seed 42
```

To isolate the effect of coordinate scaling from checkpoint weighting, useful
ablations are:

```bash
# Equal checkpoint weights, Adam coordinate metric
python scripts/select/kl.py ... --optimizer uniform --metric adam

# SGD eta_k weights, no coordinate metric
python scripts/select/kl.py ... --optimizer sgd --metric none

# Full Adam score
python scripts/select/kl.py ... --optimizer adam --metric adam
```

The first and third commands need Adam reference checkpoints because an SGD run
does not contain second moments.

## Full-network selector

Use the same trajectory and optimizer flags with the matrix-free all-parameter
implementation:

```bash
python scripts/select/kl_full.py \
  --dataset cifar100 \
  --data_dir /path/to/cifar100 \
  --checkpoint_dir /path/to/adam_checkpoints \
  --fraction 0.1 \
  --optimizer adam \
  --metric adam \
  --checkpoint_weighting point \
  --max_iters 200 \
  --seed 42
```

This is substantially more expensive than the last-layer factorization. Its
`--max_iters` counts full gradient-ranking sweeps, not cheap scalar iterations.

## Run baselines

Every baseline uses the shared protocol and accepts the common dataset,
checkpoint, fraction, seed, and `--selection_only` arguments. Examples:

```bash
python scripts/select/random_selection.py \
  --dataset cifar100 --data_dir /path/to/cifar100 \
  --checkpoint_dir /path/to/sgd_checkpoints --fraction 0.1 --seed 42

python scripts/select/herding.py \
  --dataset cifar100 --data_dir /path/to/cifar100 \
  --checkpoint_dir /path/to/sgd_checkpoints --fraction 0.1 --kernel features

python scripts/select/submodular.py \
  --dataset cifar100 --data_dir /path/to/cifar100 \
  --checkpoint_dir /path/to/sgd_checkpoints --fraction 0.1 \
  --function graphcut --kernel features
```

Other entry points are `moderate.py`, `el2n.py`, `moso.py`, and `gradmatch.py`
under `scripts/select/`.

## Aggregate, score, and plot

The scripts use `$SCRATCH` by default. Pass `--scratch` when results live
elsewhere.

```bash
# Accuracy and timing summary
python scripts/analyze/aggregate_comparison.py \
  --dataset cifar100 \
  --scratch /path/to/scratch

# Re-score every saved subset on one common full-network objective
python scripts/analyze/score_subsets.py \
  --dataset cifar100 \
  --data_dir /path/to/cifar100 \
  --checkpoint_dir /path/to/sgd_checkpoints \
  --weighting sgd \
  --checkpoint_weighting point \
  --scratch /path/to/scratch

# Checkpoint-count ablation plot
python scripts/plot/checkpoint_ablation.py \
  --dataset cifar100 \
  --scratch /path/to/scratch
```

The aggregator discovers optimizer variants separately (for example,
`kl_faithful`, `kl_faithful_sgd`, and `kl_faithful_adam`), so their accuracy
and timing rows can be compared directly. Non-default metric, checkpoint-window,
and momentum ablations receive additional filename tags and do not overwrite the
main runs.

Selection runs save these artifacts:

```text
*_indices.pt      selected pool-relative indices
*_timing.json     extra training, scoring, and selection wall-clock
*_objective.json  KL-Faithful objective history
*_metrics.json    test accuracy during subset retraining
*_model.pth       final retrained model
```

## Slurm

The launchers resolve the repository root from their own location and set
`PYTHONPATH` automatically, so they can be submitted from any directory:

```bash
sbatch scripts/slurm/run_full_data.sh
sbatch scripts/slurm/run_kl_selection.sh
sbatch scripts/slurm/run_baselines.sh
sbatch scripts/slurm/run_checkpoint_ablation.sh
sbatch scripts/slurm/run_gradmatch_selection.sh
```

Environment variables such as `DATASET_OVERRIDE`, `CHECKPOINT_DIR`,
`NUM_CHECKPOINTS`, `CHECKPOINT_SELECTION`, `CHECKPOINT_WEIGHTING`, `OPTIMIZER`,
`METRIC`, and `RULE` override the main KL launcher settings. Adjust the dataset
archives and cluster-specific paths in `scripts/slurm/stage_dataset.sh`.

For example:

```bash
# SGD score on the ordinary full-data trajectory
OPTIMIZER=sgd METRIC=none CHECKPOINT_WEIGHTING=point \
  sbatch scripts/slurm/run_kl_selection.sh

# Adam score on an optimizer-aware trajectory
CHECKPOINT_DIR=/path/to/adam_checkpoints OPTIMIZER=adam METRIC=adam \
  CHECKPOINT_WEIGHTING=point sbatch scripts/slurm/run_kl_selection.sh
```

## Reproducibility notes

- Use the same `--seed` for the 90/10 split, selection, and retraining.
- Use the same `--scoring_transform` for every method. `test` gives a stable,
  unaugmented scoring view; `train` reproduces the earlier runs.
- Prefer `--checkpoint_selection spread`; `prefix` intentionally reproduces the
  early-checkpoint experiments.
- Keep `--checkpoint_weighting point` for the exact `eta_k` formulas above. Use
  `window` only when treating a sparse checkpoint as representative of a range of
  optimizer steps.
- The files in `scripts/legacy/` are retained for provenance and are not part of
  the maintained workflow.
