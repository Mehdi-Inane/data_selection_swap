# Experimental protocol: trajectory selection and controlled comparisons

Two quantities are compared: **final test accuracy after retraining on the
subset** and **time to produce the subset**. The configuration-hashed drivers
under `scripts/experiments/` are the source of truth for controlled claims.
The older all-baseline launcher is useful for exploratory comparisons, but its
method-specific training work means it is not, by itself, a pure
selection-rule-only experiment.

## 1. Held fixed for every method

| Component | Setting (source: `data_selection_swap.selection.kl`) |
|---|---|
| Data | Canonical comparison grid: CIFAR-10 / CIFAR-100, 90/10 train/val `random_split(seed)`; selection pool = the 90% split |
| Budgets | `fraction ∈ {0.1, 0.2, 0.4, 0.5}` of `n_train`, exact size `int(fraction · n_train)` |
| Seeds | 42–46; the seed fixes the split, the method's randomness and the retrain init |
| Architecture | Canonical grid: torchvision ResNet-18, VGG-16-BN, or GoogLeNet/Inception. ResNet-34, LeNet, and ConvNet remain available for mh-only controls. |
| Retraining | Optimizer matched to the score (SGD for uniform/SGD, Adam for Adam), cosine schedule, 300 / 350 epochs, bs 128, unweighted CE |
| Metric | Final-epoch top-1 test accuracy. Controlled drivers evaluate the official test set once per fit. |
| Scoring view | `--scoring_transform` — same value for every method (see §4) |
| Hardware | Same GPU type for every timed job, 4 workers, one job per GPU |

Every reference is seed matched to the split it represents. Its manifest stores
the dataset, architecture, seed, schedule, initial-state fingerprint, and hash
of the ordered pool indices; consumers reject any mismatch. Controlled
retraining uses frozen indices, a byte-identical initialization for paired
cells, and a dedicated shuffle generator. Use `--retrain_optimizer` explicitly
when holding the retraining optimizer fixed across optimizer-aware scoring
ablations.

Baseline outputs use
`data_selection_baselines/<method>/<dataset>/<architecture>/<budget>/seed_<seed>`;
KL outputs use
`gradmatch_swap/<dataset>/<architecture>/<budget>/seed_<seed>`. Architecture
is part of both the manifest and directory identity, so incompatible
checkpoints or results cannot silently share a comparison cell.

## 2. What each method is given (its canonical protocol)

The shared resource is a **per-seed** full-data trajectory from
`scripts/train/full_data.py` (checkpoints at epochs 1–10, then every 10 epochs).
Sparse checkpoint selections are spaced in optimizer-step time rather than
saved-file position. Each method uses as much of the trajectory as its protocol
requires and trains any additional probe/surrogate models itself.

| Method | Signal | From | Rule | Class-balanced |
|---|---|---|---|---|
| KL-Faithful | last-layer grads (L, Φ), all ckpts | full trajectory | single-swap descent | no |
| Random | – | – | uniform | no |
| Herding | Φ | last ckpt | Welling herding, per class | yes |
| Moderate | Φ, distance to class-median prototype | last ckpt | median band | no (repo) |
| EL2N | ‖softmax − onehot‖, mean over K=10 | 10 fresh probes × 20 epochs | top-k | no (repo) |
| MoSo | exact leave-one-out gradient score, full-net grads | 8 surrogates × 50 epochs on disjoint parts, 10 sampled epochs | top-k per class | yes (repo) |
| GraphCut | Euclidean sim. on Φ, λ=1 | last ckpt | apricot lazy greedy, per class | yes |
| Facility Location | Euclidean sim. on Φ | last ckpt | apricot lazy greedy, per class | yes |

No method, including KL-Faithful, is tuned on the test set; baselines use the
defaults of their papers / reference code.

## 3. Trajectory-kernel baseline sensitivity (legacy)

KL-Faithful can use trajectory last-layer gradients, which is more information
than most baselines use. The following legacy variants give other rules a
trajectory-gradient kernel:

| Variant | Flag |
|---|---|
| `herding_grad_trajectory`, `graphcut_grad_trajectory`, `facility_location_grad_trajectory` | `--kernel grad_trajectory`: per-class unweighted trajectory-gradient kernel |
| `*_grad` | `--kernel grad`: same kernel, last checkpoint only |
| `el2n_trajectory` | `--source trajectory`: trajectory checkpoint at epoch 20 instead of new probes |
| `moso_trajectory[_lastlayer]` | `--source trajectory`: 10 evenly spaced trajectory checkpoints with their cosine η_t; optionally KL's last-layer gradient space |

Run with `METHODS="..." sbatch --array=... scripts/slurm/run_baselines.sh`.
These variants currently sum every checkpoint with unit weight. They are not
identical-information controls for a KL run that uses a subset of checkpoints,
learning-rate weights, Adam metrics, or a different bias/view convention. Do
not describe them as selection-rule-only comparisons. Use
`compare_swap_rules.py` for the exact Algorithm 2/3 controlled comparison.

## 4. Known confound: augmentation during scoring

The factored KL selector can score the training set through the *training*
transform (random crop and flip), so features differ from one checkpoint to
the next and from one run to the next. Moderate-DS and data_diet score clean images; MoSo
scores augmented ones. For the main table, use one value of
`--scoring_transform` for **all** methods, including KL-Faithful; `test` is
the cleaner choice. The default, `test`, is stable across checkpoints; `train`
reproduces the earlier augmented-view KL-Faithful runs.

## 5. Optimizer-aware objective

The main optimizer-aware comparison uses `--checkpoint_weighting point`:

- `--optimizer uniform --metric none`: equal checkpoint weights;
- `--optimizer sgd --metric none`: `w_k = eta_k`;
- `--optimizer adam --metric adam`: `w_k = eta_k` and
  `M_k[j] = (sqrt(vhat_k[j]) + eps)^-2`.

The SGD full-data trainer records `eta_k` in `trajectory.json`. Adam additionally
requires `scripts/train/reference.py --optimizer adam`, which saves the raw
second moment and optimizer step needed for bias correction. Use
`--checkpoint_weighting window` only as an explicit integrated-schedule
ablation. Result discovery reports the variants as separate methods, so they
can be compared without overwriting one another.

## 6. Timing

Every script writes `<method>_<budget>_timing.json` using a CUDA-synchronised
wall-clock that starts at the first checkpoint load. The timing is split into phases:

- `extra_training`: training the method needs beyond the shared trajectory
  (EL2N probes, MoSo surrogates);
- `scoring`: checkpoint loads plus forward/backward passes over the pool;
- `selection`: the solver (swap descent, greedy, sort).

Dataset staging, model construction and retraining are excluded; they are
identical across methods.

`reference_epochs_used` records how far into the shared trajectory a method
had to go: KL-Faithful and the embedding methods need all of it (300), and
`el2n_trajectory` needs 20. The full-data trainer writes per-epoch times to
`train_time.json`, and `scripts/analyze/aggregate_comparison.py` reports:

- **selection**: the method's own cost, assuming the trajectory already exists
  (e.g. it was trained anyway);
- **total**: selection plus the cost of the trajectory epochs it consumed, i.e. the
  end-to-end cost from nothing.

Report both. The first reflects the amortised setting KL-Faithful targets; the
second is the honest cost when no full-data model exists yet.

## 7. Running

Launchers use the Python on `PATH`. Set `CONDA_MODULE` and `CONDA_ENV` when a
cluster module and conda environment should be activated. Results go to
`RESULTS_ROOT`, then `SCRATCH`, then `./results`. Use Python 3.9+ and
PyTorch 2.0+ so full-network Gradient Swap can import `torch.func`.

The manuscript commands, including the shared seed-42 trajectory, are in the
repository README. Optimizer/learning-rate, swap-rule, and D(alpha)
correlation drivers under `scripts/experiments/` are extra controlled
experiments and are not reported in the manuscript.

To measure selection time without the retraining cost, add `--selection_only`.
For exact commands, task counts, immutable artifact schemas, and analysis, see
`scripts/experiments/README.md` and `README_D_CORRELATION.md` in that directory.

## 8. Interpretation boundary

The manuscript's KL theorem assumes full-batch Langevin dynamics. The
maintained reference trainers use ordinary minibatch SGD/Adam (momentum,
weight decay, and no injected Langevin noise), so `D(alpha)` is an empirical
recorded-trajectory gradient-discrepancy score. Algorithm 2 and Algorithm 3 are
implemented faithfully for that quadratic score; this does not by itself turn
the experiment into a verification of the full Langevin KL theorem.

With BatchNorm, independently scored examples use the model in evaluation mode
with frozen running statistics. This is a reproducible surrogate, not the
batch-dependent train-mode gradient field. The D/accuracy experiment rejects
BatchNorm by default and treats explicitly enabled frozen-BN runs as separate
sensitivity blocks.
