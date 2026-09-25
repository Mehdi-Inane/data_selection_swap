"""
train_moso_selection.py
MoSo — Moving-one-Sample-out (Tan et al., NeurIPS 2023: "Data Pruning via
Moving-one-Sample-out", https://arxiv.org/abs/2310.14664). Port of
surrogate_training.py + scoring.py (MoSo_scoring_exact) + retraining.py
(nopt2) in https://github.com/hrtan/MoSo.

Score (first-order MoSo, Eq. 4, "exact" leave-one-out mean in the repo)
─────────────────────────────────────────────────────────────────────────
For a sample z in a set of size N, with g = ∇ℓ(z; w_t) and
ḡ = (1/N) Σ_j ∇ℓ(z_j; w_t):

    s_t(z) = [ (2N-3) ||ḡ||²  -  ||g||²  +  (2N-4) <ḡ - g/N, g> ] / (N-1)²
    MoSo(z) = Σ_{t in sampled checkpoints}  η_t · s_t(z)

Selection keeps the highest-MoSo samples, class-balanced (top-k per class).

--source surrogates  (paper protocol, default) split the training set into
                     I disjoint parts, train one surrogate per part for 50
                     epochs with the repo's surrogate hyper-parameters, and
                     score each part with its own surrogate at 10 randomly
                     sampled epochs. Surrogate training time is charged to MoSo.
--source trajectory  I=1, N=n_train, score at `samples` evenly spaced
                     checkpoints of the shared trajectory with the cosine LR
                     that produced them (no extra training) — the
                     same-information variant.

--grad_space full        per-sample gradients of every parameter (repo).
--grad_space last_layer  g_i = vec(l_i φ_i^T), the exact Kronecker factors
                         KL-Faithful uses; ḡ·g_i = l_i^T Ḡ φ_i and
                         ||g_i||² = ||l_i||² ||φ_i||², so no G is formed.
"""

import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.func import functional_call, grad, vmap
from torch.utils.data import DataLoader, Subset

import selection_common as common


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm: gradient statistics → exact MoSo score
# ─────────────────────────────────────────────────────────────────────────────
def moso_exact(dot, sq, gbar_sq, N):
    """dot_i = <ḡ, g_i>, sq_i = ||g_i||², gbar_sq = ||ḡ||² (float64 tensors)."""
    cross = dot - sq / N
    return ((2 * N - 3) * gbar_sq - sq + (2 * N - 4) * cross) / (N - 1) ** 2


def grad_stats_last_layer(model, loader, device, chunk=8192):
    probs, phi, targets = common.extract_probs_features_targets(model, loader, device)
    L = common.logit_grads(probs, targets)
    N = L.shape[0]

    Gbar = torch.zeros(L.shape[1], phi.shape[1], dtype=torch.float64, device=device)
    for s in range(0, N, chunk):
        Gbar += L[s:s + chunk].to(device).double().T @ phi[s:s + chunk].to(device).double()
    Gbar /= N

    dot, sq = [], []
    for s in range(0, N, chunk):
        L_c   = L[s:s + chunk].to(device).double()
        phi_c = phi[s:s + chunk].to(device).double()
        dot.append(((L_c @ Gbar) * phi_c).sum(dim=1).cpu())
        sq.append((L_c.pow(2).sum(dim=1) * phi_c.pow(2).sum(dim=1)).cpu())
    return torch.cat(dot), torch.cat(sq), Gbar.pow(2).sum().item(), N


def grad_stats_full(model, loader, device, chunk):
    model.eval()
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    # Pass 1: ḡ. In eval mode (BN running stats) the gradient of the summed
    # loss is exactly the sum of per-sample gradients.
    gbar = {n: torch.zeros_like(p) for n, p in named}
    N = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(inputs), targets, reduction='sum').backward()
        for n, p in named:
            gbar[n] += p.grad
        N += targets.numel()
    model.zero_grad(set_to_none=True)
    gbar = {n: (g / N).flatten() for n, g in gbar.items()}
    gbar_sq = sum(g.double().pow(2).sum() for g in gbar.values()).item()

    # Pass 2: per-sample <ḡ, g_i> and ||g_i||² without storing g_i.
    params  = {n: p.detach() for n, p in named}
    buffers = {n: b.detach() for n, b in model.named_buffers()}

    def loss_fn(p, x, y):
        out = functional_call(model, (p, buffers), (x.unsqueeze(0),))
        return F.cross_entropy(out, y.unsqueeze(0))

    per_sample_grad = vmap(grad(loss_fn), in_dims=(None, 0, 0))

    dot, sq = [], []
    for inputs, targets in loader:
        for x, y in zip(inputs.split(chunk), targets.split(chunk)):
            g = per_sample_grad(params, x.to(device), y.to(device))
            d = torch.zeros(y.numel(), dtype=torch.float64, device=device)
            s = torch.zeros_like(d)
            for n in params:
                g_n = g[n].flatten(1)
                d += (g_n @ gbar[n]).double()
                s += g_n.double().pow(2).sum(dim=1)
            dot.append(d.cpu())
            sq.append(s.cpu())
            del g
    return torch.cat(dot), torch.cat(sq), gbar_sq, N


def moso_step_scores(model, loader, lr, device, args):
    if args.grad_space == 'last_layer':
        dot, sq, gbar_sq, N = grad_stats_last_layer(model, loader, device)
    else:
        dot, sq, gbar_sq, N = grad_stats_full(model, loader, device, args.grad_chunk)
    return lr * moso_exact(dot, sq, gbar_sq, N)


def evenly_spaced_checkpoints(ckpt_paths, samples, max_epoch=None):
    by_epoch = {common.checkpoint_epoch(p): p for p in ckpt_paths}
    epochs = sorted(e for e in by_epoch if max_epoch is None or e <= max_epoch)
    horizon = epochs[-1]
    chosen = []
    for target in np.linspace(horizon / samples, horizon, samples):
        remaining = [e for e in epochs if e not in chosen]
        if not remaining:
            break
        chosen.append(min(remaining, key=lambda e: abs(e - target)))
    return [(e, by_epoch[e]) for e in sorted(chosen)]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = common.base_argparser("MoSo subset selection")
    p.add_argument('--source', default='surrogates', choices=['surrogates', 'trajectory'])
    p.add_argument('--grad_space', default='full', choices=['full', 'last_layer'])
    p.add_argument('--samples', default=10, type=int,
                   help='Checkpoints sampled per surrogate / from the trajectory (repo: 10).')
    p.add_argument('--grad_chunk', default=32, type=int,
                   help='Per-sample-gradient micro-batch for --grad_space full.')
    # surrogate protocol (repo README / surrogate_training.py defaults)
    p.add_argument('--num_trials', default=8, type=int, help='I: number of disjoint parts.')
    p.add_argument('--surrogate_epochs', default=50, type=int)
    p.add_argument('--surrogate_lr', default=0.1, type=float)
    p.add_argument('--surrogate_wd', default=2e-4, type=float)
    p.add_argument('--surrogate_bs', default=256, type=int)
    # trajectory variant
    p.add_argument('--trajectory_lr', default=0.01, type=float,
                   help='Base LR train_full_data.py used (to recover η_t).')
    p.add_argument('--max_epoch', default=None, type=int,
                   help='Only use trajectory checkpoints up to this epoch.')
    args = p.parse_args()

    common.set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── 1. Dataset ───────────────────────────────────────────────────────────
    cfg = common.get_dataset_config(args.dataset, args.data_dir, download=args.download)
    NUM_CLASSES = cfg['num_classes']

    trainset, eval_loader, testloader, n_train = common.make_loaders(
        cfg, args.seed, args.batch_size, args.num_workers, args.scoring_transform)
    budget = int(args.fraction * n_train)
    scoring_set = eval_loader.dataset

    method = 'moso' if args.source == 'surrogates' else 'moso_trajectory'
    if args.grad_space == 'last_layer':
        method += '_lastlayer'
    paths = common.get_save_paths(method, args.dataset, budget, args.seed)
    logger = common.get_logger(__name__, paths['log'])
    logger.info(f"Dataset={args.dataset}  budget={budget}  seed={args.seed}  "
                f"source={args.source}  grad_space={args.grad_space}  device={device}")

    timer = common.SelectionTimer(device)
    scores = torch.zeros(n_train, dtype=torch.float64)
    targets = torch.tensor([trainset.dataset.targets[i] for i in trainset.indices])

    # ── 2. MoSo scores ───────────────────────────────────────────────────────
    if args.source == 'trajectory':
        chosen = evenly_spaced_checkpoints(common.list_checkpoints(args.checkpoint_dir),
                                           args.samples, args.max_epoch)
        logger.info(f"Scoring at trajectory epochs {[e for e, _ in chosen]}")
        model = common.build_model(cfg, NUM_CLASSES, device)
        for epoch, ckpt in chosen:
            lr = common.cosine_lr(args.trajectory_lr, epoch, cfg['num_epochs'])
            with timer.phase('scoring'):
                model.load_state_dict(torch.load(ckpt, map_location=device))
                scores += moso_step_scores(model, eval_loader, lr, device, args)
        reference_epochs = max(e for e, _ in chosen)
        extra_epochs = 0
    else:
        perm  = torch.randperm(n_train, generator=torch.Generator().manual_seed(args.seed))
        parts = perm.chunk(args.num_trials)
        rng   = random.Random(args.seed)
        for trial, part in enumerate(parts):
            part = part.sort().values
            score_epochs = set(rng.sample(range(args.surrogate_epochs),
                                          min(args.samples, args.surrogate_epochs)))
            logger.info(f"Surrogate {trial + 1}/{args.num_trials}: |S_i|={part.numel()}, "
                        f"scoring at epochs {sorted(score_epochs)}")

            torch.manual_seed(args.seed * 1000 + trial)
            model = common.build_model(cfg, NUM_CLASSES, device)
            train_loader = DataLoader(Subset(trainset, part.tolist()), batch_size=args.surrogate_bs,
                                      shuffle=True, pin_memory=True, num_workers=args.num_workers)
            score_loader = DataLoader(Subset(scoring_set, part.tolist()), batch_size=args.batch_size,
                                      shuffle=False, pin_memory=True, num_workers=args.num_workers)
            optimizer = optim.SGD(model.parameters(), lr=args.surrogate_lr, momentum=0.9,
                                  weight_decay=args.surrogate_wd, nesterov=True)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.surrogate_epochs, eta_min=1e-4)
            criterion = nn.CrossEntropyLoss()

            part_scores = torch.zeros(part.numel(), dtype=torch.float64)
            for epoch in range(args.surrogate_epochs):
                with timer.phase('extra_training'):
                    common.train_one_epoch(model, train_loader, optimizer, criterion, device)
                if epoch in score_epochs:
                    lr = scheduler.get_last_lr()[0]
                    with timer.phase('scoring'):
                        part_scores += moso_step_scores(model, score_loader, lr, device, args)
                scheduler.step()
            scores[part] = part_scores
            del model
        reference_epochs = 0
        extra_epochs = args.surrogate_epochs  # I surrogates on N/I samples each

    # ── 3. Class-balanced top-k (repo: nopt2) ────────────────────────────────
    with timer.phase('selection'):
        selected_indices = common.per_class_topk(scores, targets, NUM_CLASSES, budget)
    logger.info(f"Selected {len(selected_indices)} samples across {NUM_CLASSES} classes.")

    common.save_timing(paths['timing'], timer, logger, method=method,
                       reference_epochs_used=reference_epochs,
                       extra_training_epochs=extra_epochs)
    with open(paths['scores'], 'w') as fh:
        json.dump({"method": method, "source": args.source, "grad_space": args.grad_space,
                    "samples": args.samples, "num_trials": args.num_trials,
                    "n_selected": len(selected_indices)}, fh, indent=4)
    torch.save(scores, paths['scores'].replace('.json', '.pt'))

    # ── 4. Retrain on selected subset (identical protocol to every baseline) ─
    common.set_seed(args.seed)
    common.finish(cfg, trainset, selected_indices, device, args, logger, paths)


if __name__ == "__main__":
    main()
