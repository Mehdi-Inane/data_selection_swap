# Data Selection Baselines: Multi-Dataset / Multi-Architecture Update

This README documents the changes made to support the new empirical setting:

- **Datasets**: `cifar10`, `cifar100`
- **Architectures**: `resnet18`, `vgg16`, `inception`
- **Total combinations**: 6

It also documents the KL update to handle checkpoints that now contain optimizer state, and the new `--full_gradient` mode.

---

## 1) What was changed

### `selection_common.py`

#### Dataset support
- Added `cifar10` support in `get_dataset_config`.
- Kept `cifar100` and `imagenet` support.
- Added robust root resolution (`resolve_data_root`) so staged datasets are found from common layouts (`$SLURM_TMPDIR`, `<dataset>_data`, etc.).

#### Architecture support
- Replaced the previous single-model setup with architecture-aware model creation:
  - `resnet18`
  - `vgg16` (`vgg16_bn` with CIFAR-adapted classifier)
  - `inception` (`googlenet` with CIFAR-adapted stem)
- Added shared `ARCHS = ["resnet18", "vgg16", "inception"]`.

#### Feature extraction hook generalization
- Added `get_classifier_layer(model)` to locate the final linear classifier across architectures.
- Updated feature extraction to hook this final classifier input (penultimate representation), instead of assuming `model.linear`.

#### Checkpoint compatibility
- Added `unwrap_checkpoint_state` and `load_model_checkpoint`.
- All baseline loaders now correctly handle both:
  - raw `state_dict` checkpoints
  - full training checkpoints with keys like `model`, `optimizer`, `scheduler`, etc.

#### Shared parser + output layout
- `base_argparser` now includes:
  - `--dataset {cifar10,cifar100,imagenet}`
  - `--arch {resnet18,vgg16,inception}`
- `get_save_paths(...)` now stores outputs under architecture-specific directories:
  - `$SCRATCH/data_selection_baselines/<method>/<dataset>/<arch>/<budget>/seed_<seed>`
  - This avoids collisions between different architectures on the same dataset.

---

### Baseline training scripts updated

Updated scripts:
- `train_random_selection.py`
- `train_herding_selection.py`
- `train_moderate_coreset_selection.py`
- `train_moso_selection.py`
- `train_el2n_selection.py`
- `train_submodular_selection.py`

Changes applied consistently:
- Use `args.arch` when building models.
- Save under architecture-specific output directories.
- Logging now includes architecture.
- Checkpoint loading paths now use the shared robust loader for full checkpoints.

---

### `train_kl_selection.py`

#### Dataset/architecture updates
- Added `--dataset {cifar10,cifar100,imagenet}`.
- Added `--arch {resnet18,vgg16,inception}`.
- Switched dataset/model construction to shared `selection_common` logic for consistency with baselines.
- KL outputs now include architecture in path:
  - `$SCRATCH/gradmatch_swap/<dataset>/<arch>/<budget>/seed_<seed>`

#### Checkpoint format update (optimizer state now present)
- Checkpoint loading now unwraps full checkpoints and loads only model weights when needed.
- This prevents failures when trajectory checkpoints include optimizer/scheduler/RNG states.

#### New option: `--full_gradient`
- Added `--full_gradient` to compute exact full-network per-sample gradients instead of last-layer Kronecker factors.
- Added `--full_grad_chunk` to control per-sample-gradient micro-batch size.
- Implemented a **streaming kernel-vector operator** for memory safety:
  - computes exact `K @ v` in two passes per checkpoint,
  - avoids materializing full per-sample gradient matrices in RAM.
- The implementation remains learning-rate weighted (`eta_k`) consistently.
- This mode is intentionally slow but memory-aware and intended for CIFAR-scale experimentation.
- `--full_gradient` is blocked for ImageNet in this implementation.

#### Learning-rate weighting
- Existing sqrt/eta logic for the default factorized path is preserved.
- In full-gradient mode, equivalent eta weighting is applied at kernel accumulation level.

---

### SLURM launch scripts updated for the 6 combinations

#### `run_baselines.sh`
- Array expanded to cover:
  - 6 combos × 7 methods × 4 fractions × 5 seeds = 840 tasks (`0-839`).
- Added internal mapping from `SLURM_ARRAY_TASK_ID` to:
  - dataset, architecture, method, fraction, seed.
- Checkpoint path now architecture-aware:
  - `$SCRATCH/data_selection_swap/<dataset>/<arch>/checkpoints`
- Passes `--arch` into every baseline run.
- Optional overrides retained/added:
  - `DATASET_OVERRIDE`, `ARCH_OVERRIDE`, `METHODS`.

#### `alg3.sh` (KL launcher)
- Array expanded to:
  - 6 combos × 4 fractions × 5 seeds = 120 tasks (`0-119`).
- Added dataset+architecture mapping and architecture-aware checkpoint path.
- Passes `--arch` to `train_kl_selection.py`.

#### `checkpoint_ablation.sh`
- Array expanded to:
  - 6 combos × 4 fractions × 5 seeds × 5 checkpoint-count settings = 600 tasks (`0-599`).
- Added dataset+architecture mapping and architecture-aware checkpoint path.
- Passes `--arch` to `train_kl_selection.py`.

---

## 2) How to run

## Full trajectory generation (already updated before this task)
Use:
- `run_full_data.sh`

It already maps 6 array tasks to all dataset/architecture combinations and writes checkpoints to:
- `$SCRATCH/data_selection_swap/<dataset>/<arch>/checkpoints`

## Baselines
Use:
- `run_baselines.sh`

By default it runs all 6 combinations × configured methods/fractions/seeds.

## KL-Faithful
Use:
- `alg3.sh`

Optional exact full-gradient mode (memory-aware, slow):
- add `--full_gradient --full_grad_chunk <small_int>` to `train_kl_selection.py` invocation.

## KL checkpoint ablation
Use:
- `checkpoint_ablation.sh`

---

## 3) Validation done

- Python compile checks passed for all updated selection scripts.
- Shell syntax checks passed for:
  - `run_baselines.sh`
  - `alg3.sh`
  - `checkpoint_ablation.sh`

