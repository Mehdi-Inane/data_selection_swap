# D(alpha) / subset-quality correlation experiment

This workflow tests the manuscript claim that a subset with lower
`D(alpha)` should produce a better final model. It supports CIFAR-10,
CIFAR-100, and MNIST, and the repository's ResNet-18, ResNet-34, LeNet, and
ConvNet model factory.

The runner computes the all-parameter manuscript quantity exactly at the
declared recorded checkpoints

```text
D(alpha) = sum_k w_k || mean_{i in I} grad ell(theta_k, z_i)
                         - mean_{i in S} grad ell(theta_k, z_i) ||_2^2.
```

`--d-weighting uniform --d-normalization sum` is the direct stacked-feature
form. `learning_rate` is a declared ablation. Each reference checkpoint is the
pre-epoch state immediately before its outgoing optimizer-update block, and it
stores the LR used by those updates. The terminal post-training state is not
included. This prevents a state from being mislabeled with the LR that produced
it and keeps the state/learning-rate pairing used by LR-weighted D honest.
The K-checkpoint score is still a quadrature approximation to evaluating every
optimizer update on the full reference trajectory; K and every checkpoint hash
are part of the recorded scoring configuration.

The reference trajectory, D scoring passes, and subset retraining all use the
same clean deterministic image view. This is intentional: dynamic crop/flip
augmentation would make the loss scored by D differ from the loss used to train
the two models. This correlation experiment favors a well-defined aligned loss;
it must not be mixed with the repository's augmented-training result blocks.
The runner likewise requires the reference and subset fits to have identical
optimizer, learning rate, epoch count, batch size, weight decay, and cosine
schedule. Run separate matched cells to study SGD and Adam; the dedicated
`optimizer_lr.py` workflow handles deliberately crossed optimizer/LR ablations.
Because a subset has fewer batches in one data pass, its loader is cycled until
it has taken the same number of optimizer updates as the full-pool reference in
each reference-equivalent epoch. Thus both endpoints are measured at the same
`T`; the exact step count is stored in the manifest and every replicate.

## BatchNorm qualification

Train-mode BatchNorm makes a sample's gradient depend on the rest of its
mini-batch, so it does not admit the per-example decomposition above. The
runner therefore defaults to `--batchnorm-policy reject`.

- Use LeNet as the BN-free primary control.
- To run a ResNet or the current ConvNet, pass
  `--batchnorm-policy eval_frozen`. The resulting D is exact for the
  deterministic model with frozen running statistics, not for train-mode
  BatchNorm. This qualification is written into every record and creates a
  separate analysis block.

## Run

```bash
python scripts/experiments/run_d_accuracy_correlation.py \
  --stage all \
  --dataset cifar10 \
  --architecture lenet \
  --data-dir /path/to/datasets/cifar10 \
  --output-dir /path/to/results/d_correlation \
  --fractions 0.1 0.2 0.4 \
  --samplers uniform stratified easy hard \
  --subsets-per-sampler 3 \
  --retrain-replicates 3 \
  --download
```

The reference and evaluation phases can be scheduled separately. All
reference-affecting arguments must remain identical; an incompatible reuse
fails rather than overwriting evidence.

```bash
python scripts/experiments/run_d_accuracy_correlation.py ... --stage reference
python scripts/experiments/run_d_accuracy_correlation.py ... --stage evaluate
```

The Slurm launcher provides a six-cell network grid (BN-free LeNet and the
explicit frozen-BN ResNet-18 sensitivity on each dataset) over five split
seeds:

```bash
# Default array is 6 configurations x 5 seeds = 30 tasks.
STAGE=reference sbatch scripts/slurm/run_d_accuracy_correlation.sh

# Submit only after the reference array succeeds.
STAGE=evaluate sbatch scripts/slurm/run_d_accuracy_correlation.sh
```

Set `DOWNLOAD=1` only where compute nodes are allowed to download torchvision
datasets. Otherwise pre-stage them under `$DATA_ROOT_BASE/{cifar10,cifar100,mnist}`.
Publication runs should generally use at least eight unique subsets per fixed
block and three retraining replicates; the launcher defaults are smaller to
avoid accidentally launching an unexpectedly large allocation.

`easy` and `hard` are random samples from low- and high-reference-loss pools.
They use no validation/test information and are included to widen the subset
quality range, not as proposed selection algorithms. Existing selector outputs
can be supplied with `--external-subsets` as JSON:

```json
{
  "subsets": [
    {
      "id": "algorithm3_seed42",
      "sampler": "algorithm3",
      "seed": 42,
      "index_space": "pool",
      "fraction": 0.1,
      "indices_file": "algorithm3_seed42_indices.json"
    }
  ]
}
```

The number of indices must equal `fraction * pool_size`. Dataset-global
indices are also accepted with `"index_space": "global"`.

## Analyze

```bash
python scripts/analyze/analyze_d_correlation.py \
  --input /path/to/results/d_correlation \
  --output-dir /path/to/results/d_correlation/analysis \
  --n-bootstrap 5000
```

The subset is the unit of analysis. Retraining replicates are averaged within
subset, then paired subsets are bootstrapped; replicate uncertainty is also
resampled. Correlations are computed inside blocks that fix dataset, network,
split, budget, trajectory, D definition, and retraining protocol. The pooled
summary uses centred within-block ranks, never raw D or accuracy pooled across
incommensurate settings. A negative correlation supports the claim.

Every completed `record.json` embeds the exact pool-relative indices and hash,
the same hash beside D and retraining, checkpoint hashes, the shared-init hash,
and the full protocol. Invalid or conflicting records fail closed by default.
