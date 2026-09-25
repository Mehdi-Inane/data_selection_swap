"""
train_el2n_selection.py
EL2N scores (Paul, Ganguli & Dziugaite, NeurIPS 2021: "Deep Learning on a
Data Diet: Finding Important Examples Early in Training",
https://arxiv.org/abs/2107.07075). Port of the l2_error score +
keep_max_scores / offset subsets in https://github.com/mansheej/data_diet.

Algorithm
─────────
    for k = 1 .. K:  train a fresh network (independent init) for
                     `score_epoch` epochs on the full training set
    EL2N(x, y) = mean_k || softmax(f_k(x)) - onehot(y) ||_2
    keep the `budget` highest-scoring samples (global, not per class),
    optionally after skipping the `offset` hardest ones.

--source probes      (paper protocol, default) K=10 probe runs at epoch 20;
                     their training time is charged to EL2N.
--source trajectory  K=1, the shared trajectory's checkpoint at
                     `score_epoch` (no extra training) — the same-information
                     variant.
Probes use exactly the retrain protocol (SGD, lr, cosine over the full
num_epochs), so a probe at epoch 20 matches the shared trajectory at
epoch 20 up to initialisation.
"""

import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import selection_common as common


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm: EL2N score and keep-max selection
# ─────────────────────────────────────────────────────────────────────────────
def el2n_scores(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return common.logit_grads(probs, targets).norm(dim=1)


def el2n_select(scores: torch.Tensor, budget: int, offset: int = 0) -> torch.Tensor:
    order = torch.argsort(scores, descending=True)
    offset = min(offset, scores.numel() - budget)
    return order[offset:offset + budget]


def train_probe(cfg, trainset, epochs, seed, device, args):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = common.build_model(cfg, cfg['num_classes'], device)
    loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                        pin_memory=True, num_workers=args.num_workers,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['num_epochs'])
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        common.train_one_epoch(model, loader, optimizer, criterion, device)
        scheduler.step()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = common.base_argparser("EL2N subset selection")
    p.add_argument('--source', default='probes', choices=['probes', 'trajectory'])
    p.add_argument('--score_epoch', default=20, type=int,
                   help='Epoch at which EL2N is computed (paper: 20).')
    p.add_argument('--num_probes', default=10, type=int,
                   help='K independent probe runs averaged (paper: 10). '
                        'Ignored for --source trajectory.')
    p.add_argument('--offset', default=0, type=int,
                   help='Skip the `offset` highest-scoring samples (data_diet '
                        'offset subsets, for high pruning rates).')
    args = p.parse_args()

    common.set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── 1. Dataset ───────────────────────────────────────────────────────────
    cfg = common.get_dataset_config(args.dataset, args.data_dir, download=args.download)
    NUM_CLASSES = cfg['num_classes']

    trainset, eval_loader, testloader, n_train = common.make_loaders(
        cfg, args.seed, args.batch_size, args.num_workers, args.scoring_transform)
    budget = int(args.fraction * n_train)

    method = 'el2n' if args.source == 'probes' else 'el2n_trajectory'
    if args.offset:
        method += f'_offset{args.offset}'
    paths = common.get_save_paths(method, args.dataset, budget, args.seed)
    logger = common.get_logger(__name__, paths['log'])
    logger.info(f"Dataset={args.dataset}  budget={budget}  seed={args.seed}  "
                f"source={args.source}  score_epoch={args.score_epoch}  device={device}")

    timer = common.SelectionTimer(device)

    # ── 2. EL2N scores ───────────────────────────────────────────────────────
    scores = torch.zeros(n_train)
    if args.source == 'trajectory':
        ckpt = os.path.join(args.checkpoint_dir, f"checkpoint_{args.score_epoch:03d}.pth")
        if not os.path.isfile(ckpt):
            raise ValueError(f"No trajectory checkpoint at epoch {args.score_epoch}: {ckpt}")
        ref_model = common.build_model(cfg, NUM_CLASSES, device)
        with timer.phase('scoring'):
            ref_model.load_state_dict(torch.load(ckpt, map_location=device))
            probs, _, targets = common.extract_probs_features_targets(ref_model, eval_loader, device)
            scores = el2n_scores(probs, targets)
        reference_epochs = args.score_epoch
        num_probes = 1
    else:
        for k in range(args.num_probes):
            probe_seed = args.seed * 1000 + k
            logger.info(f"Probe {k + 1}/{args.num_probes}: training {args.score_epoch} epochs "
                        f"(seed {probe_seed}) ...")
            with timer.phase('extra_training'):
                probe = train_probe(cfg, trainset, args.score_epoch, probe_seed, device, args)
            with timer.phase('scoring'):
                probs, _, targets = common.extract_probs_features_targets(probe, eval_loader, device)
                scores += el2n_scores(probs, targets) / args.num_probes
            del probe
        reference_epochs = 0
        num_probes = args.num_probes

    # ── 3. Keep-max selection ────────────────────────────────────────────────
    with timer.phase('selection'):
        selected_indices = el2n_select(scores, budget, args.offset)
    logger.info(f"Selected {len(selected_indices)} samples; EL2N range "
                f"[{scores[selected_indices].min():.4f}, {scores[selected_indices].max():.4f}].")

    common.save_timing(paths['timing'], timer, logger, method=method,
                       reference_epochs_used=reference_epochs,
                       extra_training_epochs=0 if args.source == 'trajectory'
                       else args.num_probes * args.score_epoch)
    with open(paths['scores'], 'w') as fh:
        json.dump({"method": method, "source": args.source, "score_epoch": args.score_epoch,
                    "num_probes": num_probes, "offset": args.offset,
                    "n_selected": len(selected_indices)}, fh, indent=4)
    torch.save(scores, paths['scores'].replace('.json', '.pt'))

    # ── 4. Retrain on selected subset (identical protocol to every baseline) ─
    common.set_seed(args.seed)
    common.finish(cfg, trainset, selected_indices, device, args, logger, paths)


if __name__ == "__main__":
    main()
