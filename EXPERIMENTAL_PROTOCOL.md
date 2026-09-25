# Experimental protocol: KL-Faithful vs. data-selection baselines

Two quantities are compared: **test accuracy after retraining on the subset**
and **time to produce the subset**. Everything that is not the selection rule
is held fixed; every selection rule is run the way its authors run it.

## 1. Held fixed for every method

| Component | Setting (source: `train_kl_selection.py`) |
|---|---|
| Data | CIFAR-100 / ImageNet, 90/10 train/val `random_split(seed)`; selection pool = the 90% split |
| Budgets | `fraction ∈ {0.1, 0.2, 0.4, 0.5}` of `n_train`, exact size `int(fraction · n_train)` |
| Seeds | 42–46; the seed fixes the split, the method's randomness and the retrain init |
| Architecture | cords ResNet-18 (CIFAR stem on CIFAR-100), for selection **and** retraining |
| Retraining | SGD 0.01, momentum 0.9, nesterov, wd 5e-4, cosine, 300 / 350 epochs, bs 128, unweighted CE |
| Metric | Final-epoch top-1 test accuracy (no best-epoch picking on the test set) |
| Scoring view | `--scoring_transform` — same value for every method (see §4) |
| Hardware | Same GPU type (`rtx8000`), 4 workers, one job per GPU |

The retrain loop is `selection_common.retrain_on_subset`, a copy of steps 8–9
of `train_kl_selection.py`. Differences in accuracy therefore come only from
`selected_indices`.

## 2. What each method is given (its canonical protocol)

The shared resource is the full-data trajectory from `train_full_data.py`
(checkpoints at epochs 1–10, then every 10 epochs). Each method uses as much of
it as its paper uses, and trains anything extra it needs itself.

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

## 3. Same-information ablation

KL-Faithful uses the *whole* trajectory's last-layer gradients, which is more
information than most baselines use. To separate "better signal" from
"better selection rule", each baseline can be given the same signal:

| Variant | Flag |
|---|---|
| `herding_grad_trajectory`, `graphcut_grad_trajectory`, `facility_location_grad_trajectory` | `--kernel grad_trajectory`: per-class K = Σ_t (L_t L_tᵀ) ⊙ (Φ_t Φ_tᵀ), exactly KL-Faithful's kernel |
| `*_grad` | `--kernel grad`: same kernel, last checkpoint only |
| `el2n_trajectory` | `--source trajectory`: trajectory checkpoint at epoch 20 instead of new probes |
| `moso_trajectory[_lastlayer]` | `--source trajectory`: 10 evenly spaced trajectory checkpoints with their cosine η_t; optionally KL's last-layer gradient space |

Run with `METHODS="..." sbatch --array=... run_baselines.sh`.

## 4. Known confound: augmentation during scoring

`train_kl_selection.py` scores the training set through the *training*
transform (random crop and flip), so features differ from one checkpoint to
the next and from one run to the next. Moderate-DS and data_diet score clean images; MoSo
scores augmented ones. For the main table, use one value of
`--scoring_transform` for **all** methods, including KL-Faithful; `test` is
the cleaner choice. The default, `train`, reproduces the existing KL-Faithful
runs.

## 5. Timing

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
`el2n_trajectory` needs 20. `train_full_data.py` writes per-epoch times to
`train_time.json`, and `aggregate_selection_comparison.py` reports:

- **selection**: the method's own cost, assuming the trajectory already exists
  (e.g. it was trained anyway);
- **total**: selection plus the cost of the trajectory epochs it consumed, i.e. the
  end-to-end cost from nothing.

Report both. The first reflects the amortised setting KL-Faithful targets; the
second is the honest cost when no full-data model exists yet.

## 6. Running

```bash
sbatch run_full_data.sh                          # once per dataset: trajectory + train_time.json
sbatch alg3.sh                                   # KL-Faithful
sbatch run_baselines.sh                          # 7 canonical baselines × 4 fractions × 5 seeds
python aggregate_selection_comparison.py --dataset cifar100
```

To measure selection time without the retraining cost, add `--selection_only`.
