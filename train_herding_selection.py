"""
train_herding_selection.py
Herding subset selection (Welling, 2009; Chen, Welling & Smola, ICML 2010:
"Super-Samples from Kernel Herding", https://dl.acm.org/doi/abs/10.1145/1553374.1553517).

Algorithm (per class c, on penultimate features φ)
────────────────────────────────────────────────────
    μ_c   = mean_{i in class c} φ_i                     (class mean)
    w_0   = μ_c
    for k = 1 .. quota_c:
        i_k   = argmax_{i not yet selected} <w_{k-1}, φ_i>
        w_k   = w_{k-1} + μ_c - φ_{i_k}

Herding greedily picks the sample that pulls the running average of
selected features closest to the true class mean — the same "moment
matching" rule used to build exemplar sets in iCaRL. It is a purely
combinatorial, *deterministic given φ* rule (no training happens during
selection), so — matching the public convention for this method — we
read φ from a single converged reference network rather than a full
trajectory.

This script mirrors the section structure of train_kl_selection.py so
the two are easy to diff; only the "Algorithm" section differs, plus a
Step 7 selection instead of Algorithm 1's iterative swap descent.
"""

import json
import logging

import torch

import selection_common as common


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm: per-class herding
# ─────────────────────────────────────────────────────────────────────────────
def herding_select(phi: torch.Tensor, targets: torch.Tensor, num_classes: int,
                    budget: int, device: str):
    """Return the global indices selected by herding, and per-class order."""
    quota = common.per_class_budget(targets, num_classes, budget)
    phi = phi.to(device)

    selected_indices = []
    for c in range(num_classes):
        class_idx = torch.where(targets == c)[0].to(device)
        q = int(quota[c].item())
        if q == 0 or class_idx.numel() == 0:
            continue

        phi_c = phi[class_idx]                       # [n_c, F]
        mu_c  = phi_c.mean(dim=0)                     # [F]
        w     = mu_c.clone()

        available = torch.ones(class_idx.numel(), dtype=torch.bool, device=device)
        for _ in range(min(q, class_idx.numel())):
            scores = phi_c @ w                         # [n_c]
            scores = scores.masked_fill(~available, -float('inf'))
            local_best = torch.argmax(scores).item()
            available[local_best] = False
            selected_indices.append(class_idx[local_best].item())
            w = w + mu_c - phi_c[local_best]

    return torch.tensor(selected_indices, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = common.base_argparser("Herding subset selection")
    p.add_argument('--embedding_checkpoint', default=None, type=str,
                    help='Checkpoint to embed with. Default: last checkpoint '
                         'in --checkpoint_dir (the converged reference model).')
    args = p.parse_args()

    common.set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── 1. Dataset ───────────────────────────────────────────────────────────
    cfg = common.get_dataset_config(args.dataset, args.data_dir, download=args.download)
    NUM_CLASSES = cfg['num_classes']

    trainset, eval_loader, testloader, n_train = common.make_loaders(
        cfg, args.seed, args.batch_size, args.num_workers)
    budget = int(args.fraction * n_train)

    paths = common.get_save_paths('herding', args.dataset, budget, args.seed)
    logger = common.get_logger(__name__, paths['log'])
    logger.info(f"Dataset={args.dataset}  budget={budget}  seed={args.seed}  device={device}")

    # ── 2. Reference embedding ───────────────────────────────────────────────
    ckpt = common.resolve_embedding_checkpoint(args.checkpoint_dir, args.embedding_checkpoint)
    logger.info(f"Embedding with reference checkpoint: {ckpt}")
    ref_model = common.build_model(cfg, NUM_CLASSES, device)
    ref_model.load_state_dict(torch.load(ckpt, map_location=device))

    _, phi, targets = common.extract_probs_features_targets(ref_model, eval_loader, device)

    # ── 3. Herding selection ─────────────────────────────────────────────────
    logger.info(f"Running herding selection (budget={budget}/{n_train}) ...")
    selected_indices = herding_select(phi, targets, NUM_CLASSES, budget, device)
    logger.info(f"Selected {len(selected_indices)} samples across {NUM_CLASSES} classes.")

    torch.save(selected_indices, paths['indices'])
    with open(paths['scores'], 'w') as fh:
        json.dump({"method": "herding", "embedding_checkpoint": ckpt,
                    "n_selected": len(selected_indices)}, fh, indent=4)

    # ── 4. Retrain on selected subset (identical protocol to every baseline) ─
    common.retrain_on_subset(cfg, trainset, selected_indices, device, args, logger, paths)


if __name__ == "__main__":
    main()