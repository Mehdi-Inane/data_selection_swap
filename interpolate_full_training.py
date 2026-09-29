"""
interpolate_full_runs.py

1) Trains --num_runs (default 3) FULL-DATA models, ResNet18 on the whole training split
   (all of CIFAR-100 train except the 10% validation hold-out), using the recipe of
   train_kl_selection.py's retraining stage (SGD lr 0.01 / momentum 0.9 / nesterov / wd 5e-4,
   cosine annealing over --num_epochs=300, batch size 128).

   Run k uses seed = --seed + k  (k = 0, 1, 2, ...).  Only the LAST epoch is saved:
       {out_dir}/full_seed{seed+k}_epoch{num_epochs}.pth
   Existing finished checkpoints are reused (pass --retrain_full to force retraining).

2) Evaluates CIFAR-100 *test* error/loss of
       theta(a,b) = (1-a-b) theta_run0 + a theta_run1 + b theta_run2     (a,b >= 0, a+b <= 1)
   and plots the result (heatmap + contour, as in Ganesh et al. ICML'23 Fig. 5).
   (0,0)=run0, (1,0)=run1, (0,1)=run2.  Each pair of runs is one edge of the triangle.

Seed semantics
  By default the seed controls the train/val split AND init AND data order (so each run
  also sees a different 90% training split).  If you want the runs to differ ONLY in init /
  data order, pass --split_seed S to keep the split fixed across runs.

Example
  python interpolate_full_runs.py --data_dir $SLURM_TMPDIR/cifar100_data \
      --seed 42 --num_runs 3 --out_dir $SCRATCH/connectivity/full_runs_seed42
"""
import argparse
import itertools
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from cords.utils.models import ResNet18

MEAN, STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

# identical to get_dataset_config('cifar100') in train_kl_selection.py
TF_TRAIN = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])
TF_TEST = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_model(device):
    m = ResNet18(num_classes=100)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m.to(device)


def load_sd(path, device):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):  # unwrap common checkpoint formats
        for k in ("model_state_dict", "state_dict", "model"):
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break
    obj = {k.replace("module.", "", 1): v for k, v in obj.items()}
    return {k: v.to(device) for k, v in obj.items()}


def cache(dataset, device, batch, workers=4):
    xs, ys = [], []
    for x, y in DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers):
        xs.append(x); ys.append(y)
    x, y = torch.cat(xs).to(device), torch.cat(ys).to(device)
    return [(x[i:i + batch], y[i:i + batch]) for i in range(0, len(x), batch)]


@torch.no_grad()
def evaluate(model, batches):
    model.eval()
    tot, wrong, n = 0.0, 0, 0
    for x, y in batches:
        out = model(x)
        tot += F.cross_entropy(out, y, reduction="sum").item()
        wrong += out.argmax(1).ne(y).sum().item()
        n += y.numel()
    return tot / n, 100.0 * wrong / n


def train_full(args, seed, device, ckpt_path, test_cache):
    """
    Full-data model (whole training split).  `seed` controls init + data order
    (and the train/val split unless --split_seed is given).
    Saves ONLY the final-epoch weights to ckpt_path.
    """
    set_seed(seed)
    split_seed = args.split_seed if args.split_seed is not None else seed

    full_train = datasets.CIFAR100(root=args.data_dir, train=True, download=args.download,
                                   transform=TF_TRAIN)
    n_val = int(0.1 * len(full_train))
    n_train = len(full_train) - n_val
    trainset, _ = random_split(full_train, [n_train, n_val],
                               generator=torch.Generator().manual_seed(split_seed))
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=args.num_workers)

    model = make_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    tag = f"full seed={seed}"
    print(f"[{tag}] training on {n_train} samples (val hold-out {n_val}), split_seed={split_seed}, "
          f"epochs={args.num_epochs}", flush=True)
    accuracy_history = {}
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        running_loss = 0.0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        if epoch % 10 == 0:
            print(f"[{tag}] epoch {epoch:03d}/{args.num_epochs} | "
                  f"train loss {running_loss / len(trainloader):.4f}", flush=True)
        if epoch % 50 == 0 or epoch == args.num_epochs:
            _, err = evaluate(model, test_cache)  # leaves model in eval(); next epoch re-enters train()
            accuracy_history[epoch] = 100.0 - err
            print(f"[{tag}] epoch {epoch:03d}/{args.num_epochs} | test acc {100.0 - err:.2f}%", flush=True)

    tmp = ckpt_path + ".tmp"  # write-then-rename so a crash never leaves a half-written "final" file
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, ckpt_path)
    with open(os.path.splitext(ckpt_path)[0] + "_metrics.json", "w") as fh:
        json.dump(accuracy_history, fh, indent=4)
    print(f"[{tag}] saved final (epoch {args.num_epochs}) weights -> {ckpt_path}", flush=True)


class Evaluator:
    def __init__(self, device, test_cache, bn_cache):
        self.model, self.test, self.bn = make_model(device), test_cache, bn_cache
        self.bns = [m for m in self.model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]

    @torch.no_grad()
    def __call__(self, sd):
        self.model.load_state_dict(sd)
        if self.bn:  # re-estimate BN running stats (no gradient steps, no weight updates)
            for m in self.bns:
                m.reset_running_stats(); m.momentum = None
            self.model.train()
            for x, _ in self.bn:
                self.model(x)
            for m in self.bns:
                m.momentum = 0.1
        return evaluate(self.model, self.test)


def mix(sds, w):
    out = {}
    for k, v in sds[0].items():
        out[k] = sum(wi * sd[k] for wi, sd in zip(w, sds) if wi > 0) if v.is_floating_point() else v.clone()
    return out


def scan(ev, sds, N):
    loss = np.full((N + 1, N + 1), np.nan)
    err = np.full((N + 1, N + 1), np.nan)
    for i in range(N + 1):
        for j in range(N + 1 - i):
            a, b = i / N, j / N
            loss[i, j], err[i, j] = ev(mix(sds, (max(0.0, 1 - a - b), a, b)))
        print(f"  row {i + 1}/{N + 1} done", flush=True)
    return loss, err


def edges(G, N, labels):
    """1-D paths along the triangle edges; G[i,j], i = weight on model 1, j = weight on model 2."""
    l0, l1, l2 = labels
    return {f"{l0}→{l1}": [G[i, 0] for i in range(N + 1)],
            f"{l0}→{l2}": [G[0, j] for j in range(N + 1)],
            f"{l1}→{l2}": [G[N - j, j] for j in range(N + 1)]}


def barrier(path):
    return max(path) - 0.5 * (path[0] + path[-1])


def plot(res, key, label, path, vmax=None):
    N = res["N"]
    l0, l1, l2 = res["labels"]
    G = res[key]
    pts = [(i / N, j / N) for i in range(N + 1) for j in range(N + 1 - i)]
    xs, ys = zip(*pts)
    tri = mtri.Triangulation(xs, ys)
    z = np.array([G[int(round(x * N)), int(round(y * N))] for x, y in pts])
    vmin = z.min()
    vmax = vmax or z.max()
    z = np.clip(z, vmin, vmax)
    levels = np.linspace(vmin, vmax, 16)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3), squeeze=False)
    for c in range(2):
        ax = axes[0, c]
        if c == 0:
            im = ax.tripcolor(tri, z, shading="gouraud", vmin=vmin, vmax=vmax, cmap="viridis")
        else:
            im = ax.tricontourf(tri, z, levels=levels, vmin=vmin, vmax=vmax, cmap="viridis")
            ax.tricontour(tri, z, levels=levels, colors="k", linewidths=0.4)
        corners = [((0, 0), l0, G[0, 0]), ((1, 0), l1, G[N, 0]), ((0, 1), l2, G[0, N])]
        for (cx, cy), lab, v in corners:
            ax.scatter([cx], [cy], c="w", edgecolors="k", s=45, zorder=5)
            ax.annotate(f"{lab}\n{v:.2f}", (cx, cy), textcoords="offset points", xytext=(6, 6), fontsize=8,
                        zorder=6, bbox=dict(boxstyle="round,pad=0.15", fc="w", alpha=0.7, lw=0))
        ax.set_xlim(-0.12, 1.15); ax.set_ylim(-0.12, 1.15); ax.set_aspect("equal")
        ax.set_xlabel(f"a  (weight on {l1})"); ax.set_ylabel(f"b  (weight on {l2})")
        if c == 0:
            ax.set_title(f"{l0} / {l1} / {l2}", loc="left", fontsize=10)
    fig.colorbar(im, ax=axes.ravel().tolist(), label=label, shrink=0.8)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--download", action="store_true")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=42,
                   help="seed of the FIRST run; run k uses seed + k")
    p.add_argument("--num_runs", type=int, default=3,
                   help="number of full-data runs (the interpolation triangle needs exactly 3)")
    p.add_argument("--split_seed", type=int, default=None,
                   help="if set, use this seed for the train/val split of ALL runs "
                        "(default: split seed = run seed)")
    # full-data training (same defaults as train_kl_selection.py / cifar100 config)
    p.add_argument("--num_epochs", type=int, default=300, help="full-data training epochs (cosine T_max)")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--retrain_full", action="store_true",
                   help="retrain even if the final checkpoint already exists in out_dir")
    p.add_argument("--grid_res", type=int, default=20, help="N: (N+1)(N+2)/2 evaluations")
    p.add_argument("--no_bn_recal", action="store_true",
                   help="interpolate BN running stats linearly instead of re-estimating them")
    p.add_argument("--bn_samples", type=int, default=6400, help="unaugmented train images for BN re-estimation")
    p.add_argument("--vmax", type=float, default=None, help="clip colour scale")
    args = p.parse_args()

    if args.num_runs != 3:
        raise ValueError("The interpolation triangle is defined for exactly 3 runs (--num_runs 3).")

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seeds = [args.seed + k for k in range(args.num_runs)]
    ckpts = {s: os.path.join(args.out_dir, f"full_seed{s}_epoch{args.num_epochs}.pth") for s in seeds}
    for s in seeds:
        print(f"seed {s}: {ckpts[s]}")

    kw = dict(root=args.data_dir, download=args.download, transform=TF_TEST)
    test_cache = cache(datasets.CIFAR100(train=False, **kw), device, 1000)
    bn_cache = None
    if not args.no_bn_recal:
        tr = datasets.CIFAR100(train=True, **kw)
        idx = np.random.RandomState(0).choice(len(tr), args.bn_samples, replace=False).tolist()
        bn_cache = cache(Subset(tr, idx), device, 128)

    # ── train the full-data runs sequentially: seed, seed+1, seed+2 ──────────────────
    for s in seeds:
        if os.path.exists(ckpts[s]) and not args.retrain_full:
            print(f"[full seed={s}] found finished epoch-{args.num_epochs} checkpoint; skipping "
                  f"(pass --retrain_full to redo)")
        else:
            train_full(args, s, device, ckpts[s], test_cache)

    # ── interpolation among the three runs ───────────────────────────────────────────
    sds = [load_sd(ckpts[s], device) for s in seeds]
    labels = [f"seed {s}" for s in seeds]
    ev = Evaluator(device, test_cache, bn_cache)
    pkeys = [k for k, _ in ev.model.named_parameters()]
    l2 = lambda a, b: math.sqrt(sum(((a[k] - b[k]) ** 2).sum().item() for k in pkeys))

    print(f"[interp] scanning grid N={args.grid_res}")
    loss, err = scan(ev, sds, args.grid_res)
    res = dict(labels=labels, N=args.grid_res, loss=loss, err=err)

    summary = dict(
        seeds=seeds,
        split_seed=args.split_seed,
        l2={f"{labels[i]}-{labels[j]}": l2(sds[i], sds[j]) for i, j in itertools.combinations(range(3), 2)},
        **{f"{k}_{m}_barrier": barrier(path) for m, G in (("err", err), ("loss", loss))
           for k, path in edges(G, args.grid_res, labels).items()},
        **{f"{k}_err_path": path for k, path in edges(err, args.grid_res, labels).items()},
        **{f"{k}_loss_path": path for k, path in edges(loss, args.grid_res, labels).items()},
        corner_test_err=[float(err[0, 0]), float(err[args.grid_res, 0]), float(err[0, args.grid_res])],
        max_err_in_triangle=float(np.nanmax(err)), max_loss_in_triangle=float(np.nanmax(loss)),
    )
    with open(os.path.join(args.out_dir, "interp_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    np.savez(os.path.join(args.out_dir, "grid_full_runs.npz"), loss=loss, err=err, seeds=np.array(seeds))

    plot(res, "err", "test error (%)", os.path.join(args.out_dir, "interp_test_error.png"), args.vmax)
    plot(res, "loss", "test loss", os.path.join(args.out_dir, "interp_test_loss.png"), args.vmax)

    print("\n" + "=" * 80)
    print("L2 distances:", {k: round(v, 1) for k, v in summary["l2"].items()})
    print("test-err barrier (pp):")
    for k in edges(err, args.grid_res, labels):
        print(f"  {k:<24s}{summary[f'{k}_err_barrier']:8.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()