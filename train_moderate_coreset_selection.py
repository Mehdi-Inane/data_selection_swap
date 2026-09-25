"""
train_moderate_coreset_selection.py
Moderate Coreset (Xia et al., ICLR 2023: "Moderate Coreset: A Universal
Method of Data Selection for Real-world Data-efficient Deep Learning",
https://openreview.net/forum?id=7D5EECbOaf9). Port of selection.py in
https://github.com/tmllab/2023_ICLR_Moderate-DS.

Algorithm (on penultimate features φ of a converged model)
────────────────────────────────────────────────────────────
    p_c   = coordinate-wise median of {φ_i : y_i = c}      (class prototype)
    d_i   = || φ_i - p_{y_i} ||_2                          (score)
    keep the `budget` samples whose d_i are closest to the median of d,
    i.e. the central band of argsort(d) over the WHOLE training set
    (the reference implementation is not class-balanced).

The reference repo trains a base model on the full data and embeds with
its best checkpoint; here that is the last checkpoint of the shared
trajectory (no extra training).
"""

import json

import torch

import selection_common as common


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm: distance-to-median-prototype, keep the median band
# ─────────────────────────────────────────────────────────────────────────────
def moderate_distances(phi: torch.Tensor, targets: torch.Tensor, num_classes: int):
    prototypes = torch.zeros(num_classes, phi.shape[1], dtype=phi.dtype)
    for c in range(num_classes):
        members = phi[targets == c]
        if members.numel() > 0:
            prototypes[c] = members.median(dim=0).values
    return (phi - prototypes[targets]).norm(dim=1)


def moderate_select(distance: torch.Tensor, budget: int) -> torch.Tensor:
    n = distance.numel()
    sorted_idx = torch.argsort(distance)
    start = round(n * (0.5 - budget / (2.0 * n)))
    start = min(max(start, 0), n - budget)
    return sorted_idx[start:start + budget]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = common.base_argparser("Moderate Coreset subset selection")
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
        cfg, args.seed, args.batch_size, args.num_workers, args.scoring_transform)
    budget = int(args.fraction * n_train)

    paths = common.get_save_paths('moderate', args.dataset, budget, args.seed)
    logger = common.get_logger(__name__, paths['log'])
    logger.info(f"Dataset={args.dataset}  budget={budget}  seed={args.seed}  device={device}")

    timer = common.SelectionTimer(device)

    # ── 2. Reference embedding ───────────────────────────────────────────────
    ckpt = common.resolve_embedding_checkpoint(args.checkpoint_dir, args.embedding_checkpoint)
    logger.info(f"Embedding with reference checkpoint: {ckpt}")
    ref_model = common.build_model(cfg, NUM_CLASSES, device)
    with timer.phase('scoring'):
        ref_model.load_state_dict(torch.load(ckpt, map_location=device))
        _, phi, targets = common.extract_probs_features_targets(ref_model, eval_loader, device)

    # ── 3. Moderate selection ────────────────────────────────────────────────
    logger.info(f"Running moderate coreset selection (budget={budget}/{n_train}) ...")
    with timer.phase('selection'):
        distance = moderate_distances(phi, targets, NUM_CLASSES)
        selected_indices = moderate_select(distance, budget)
    logger.info(f"Selected {len(selected_indices)} samples; distance band "
                f"[{distance[selected_indices].min():.3f}, {distance[selected_indices].max():.3f}] "
                f"(median {distance.median():.3f}).")

    common.save_timing(paths['timing'], timer, logger, method='moderate',
                       reference_epochs_used=common.reference_epochs_used([ckpt]))
    with open(paths['scores'], 'w') as fh:
        json.dump({"method": "moderate", "embedding_checkpoint": ckpt,
                    "n_selected": len(selected_indices)}, fh, indent=4)
    torch.save(distance, paths['scores'].replace('.json', '.pt'))

    # ── 4. Retrain on selected subset (identical protocol to every baseline) ─
    common.finish(cfg, trainset, selected_indices, device, args, logger, paths)


if __name__ == "__main__":
    main()
