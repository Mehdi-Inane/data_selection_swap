"""
train_herding_selection.py
Herding subset selection (Welling, ICML 2009: "Herding Dynamical Weights
to Learn", https://dl.acm.org/doi/abs/10.1145/1553374.1553517).

Algorithm (per class c, on penultimate features φ)
────────────────────────────────────────────────────
    μ_c   = mean_{i in class c} φ_i                     (class mean)
    w_0   = μ_c
    for k = 1 .. quota_c:
        i_k   = argmax_{i not yet selected} <w_{k-1}, φ_i>
        w_k   = w_{k-1} + μ_c - φ_{i_k}

Since w_{k-1} = k μ_c - Σ_{s<k} φ_{i_s}, the score is written with the
kernel K = Φ Φ^T only:

    <w_{k-1}, φ_i> = k · mean_j K_ij  -  Σ_{s<k} K_{i, i_s}

so --kernel features reproduces the feature-space rule exactly, and
--kernel grad / grad_trajectory run the same herding dynamics in the
last-layer-gradient RKHS (kernel herding, Chen, Welling & Smola 2010).

Herding greedily picks the sample that pulls the running average of
selected features closest to the true class mean — the same "moment
matching" rule used to build exemplar sets in iCaRL. It is a purely
combinatorial, *deterministic given φ* rule (no training happens during
selection), so — matching the public convention for this method — we
read φ from a single converged reference network rather than a full
trajectory.
"""

import json

import torch

import selection_common as common


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm: per-class kernel herding
# ─────────────────────────────────────────────────────────────────────────────
def herding_select(kernels, class_idx, quota, device: str):
    """Return the global indices selected by herding."""
    selected_indices = []
    for c, idx in enumerate(class_idx):
        q = min(int(quota[c].item()), idx.numel())
        if q == 0:
            continue

        K         = kernels[c].to(device).double()     # [n_c, n_c]
        mean_sim  = K.mean(dim=1)                       # <μ_c, φ_i>
        sel_sim   = torch.zeros_like(mean_sim)          # Σ_s K_{i, i_s}
        available = torch.ones_like(mean_sim, dtype=torch.bool)

        for k in range(1, q + 1):
            scores = (k * mean_sim - sel_sim).masked_fill(~available, -float('inf'))
            local_best = torch.argmax(scores).item()
            available[local_best] = False
            sel_sim += K[:, local_best]
            selected_indices.append(idx[local_best].item())

    return torch.tensor(selected_indices, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = common.base_argparser("Herding subset selection")
    p.add_argument('--embedding_checkpoint', default=None, type=str,
                    help='Checkpoint to embed with. Default: last checkpoint '
                         'in --checkpoint_dir (the converged reference model).')
    p.add_argument('--kernel', default='features',
                   choices=['features', 'grad', 'grad_trajectory'],
                   help="'features' is canonical herding; the gradient kernels "
                        "give herding the same information KL-Faithful uses.")
    args = p.parse_args()

    common.set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── 1. Dataset ───────────────────────────────────────────────────────────
    cfg = common.get_dataset_config(args.dataset, args.data_dir, download=args.download)
    NUM_CLASSES = cfg['num_classes']

    trainset, eval_loader, testloader, n_train = common.make_loaders(
        cfg, args.seed, args.batch_size, args.num_workers, args.scoring_transform)
    budget = int(args.fraction * n_train)

    method = 'herding' if args.kernel == 'features' else f'herding_{args.kernel}'
    paths = common.get_save_paths(method, args.dataset, budget, args.seed)
    logger = common.get_logger(__name__, paths['log'])
    logger.info(f"Dataset={args.dataset}  budget={budget}  seed={args.seed}  "
                f"kernel={args.kernel}  device={device}")

    timer = common.SelectionTimer(device)
    ref_model = common.build_model(cfg, NUM_CLASSES, device)

    # ── 2. Reference embedding → per-class kernels ───────────────────────────
    with timer.phase('scoring'):
        kernels, class_idx, ckpts = common.build_class_kernels(
            args.kernel, ref_model, eval_loader, args.checkpoint_dir,
            args.embedding_checkpoint, NUM_CLASSES, device)
    logger.info(f"Built per-class '{args.kernel}' kernels from {len(ckpts)} checkpoint(s).")

    # ── 3. Herding selection ─────────────────────────────────────────────────
    logger.info(f"Running herding selection (budget={budget}/{n_train}) ...")
    targets = torch.empty(n_train, dtype=torch.long)
    for c, idx in enumerate(class_idx):
        targets[idx] = c
    with timer.phase('selection'):
        quota = common.per_class_budget(targets, NUM_CLASSES, budget)
        selected_indices = herding_select(kernels, class_idx, quota, device)
    logger.info(f"Selected {len(selected_indices)} samples across {NUM_CLASSES} classes.")

    common.save_timing(paths['timing'], timer, logger, method=method,
                       reference_epochs_used=common.reference_epochs_used(ckpts))
    with open(paths['scores'], 'w') as fh:
        json.dump({"method": method, "kernel": args.kernel, "checkpoints": ckpts,
                    "n_selected": len(selected_indices)}, fh, indent=4)

    # ── 4. Retrain on selected subset (identical protocol to every baseline) ─
    common.finish(cfg, trainset, selected_indices, device, args, logger, paths)


if __name__ == "__main__":
    main()
