"""
interpolate_checkpoints.py

1) Trains the FULL-DATA model:  ResNet18 from a fixed-seed init (--seed) on the whole training
   split (all of CIFAR-100 train except the 10% validation hold-out), using exactly the recipe of
   train_kl_selection.py's retraining stage (same seeded random_split, train augmentations,
   SGD lr 0.01 / momentum 0.9 / nesterov / wd 5e-4, cosine annealing over --num_epochs=300,
   batch size 128).  Only the LAST epoch is saved:  {out_dir}/full_seed{seed}_epoch300.pth
   (if that file already exists it is reused, so re-running the plotting stage does not retrain;
   pass --retrain_full to force it).

2) NO retraining of ours / X.  Evaluates CIFAR-100 *test* error/loss of
       theta(a,b) = (1-a-b) theta_full + a theta_ours + b theta_X     (a,b >= 0, a+b <= 1)
   using the already-trained final checkpoints of ours and X, and plots the result
   (heatmap + contour, as in Ganesh et al. ICML'23 Fig. 5).  (0,0)=full, (1,0)=ours, (0,1)=X.

Default paths follow your directory layout:
  ours : {SCRATCH}/gradmatch_swap/cifar100/{budget}/seed_{seed}/kl_faithful_{budget}_K{K}_model.pth
  X    : /home/mila/a/ahmedm/scratch/data_selection_baselines/{name}/cifar100/{budget}/seed_{seed}/{name}_{budget}_model.pth
  full : {out_dir}/full_seed{seed}_epoch{num_epochs}.pth      (trained by this script)

--seed is used for BOTH the train/val split (so the full model trains on exactly the same
training split that ours/X were selected from and retrained on) and the init / data order.

Example
  python interpolate_checkpoints.py --data_dir $SLURM_TMPDIR/cifar100_data \
      --baselines gradmatch random kcenter alg3 --budget 13500 --seed 42 --K 10 \
      --out_dir $SCRATCH/connectivity/cifar100_seed42
"""
import argparse
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
SCRATCH_DEFAULT = "/home/mila/a/ahmedm/scratch"

# identical to get_dataset_config('cifar100') in train_kl_selection.py
TF_TRAIN = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])
TF_TEST = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def set_seed(seed=42):  # identical to train_kl_selection.py
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


def train_full(args, device, ckpt_path, test_cache):
    """
    Full-data model, same logic as train_kl_selection.py (steps 3 + 8), except the "subset" is the
    whole training split.  Saves ONLY the final-epoch weights to ckpt_path.
    """
    set_seed(args.seed)

    full_train = datasets.CIFAR100(root=args.data_dir, train=True, download=args.download,
                                   transform=TF_TRAIN)
    n_val = int(0.1 * len(full_train))
    n_train = len(full_train) - n_val
    trainset, _ = random_split(full_train, [n_train, n_val],
                               generator=torch.Generator().manual_seed(args.seed))
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=args.num_workers)

    model = make_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    print(f"[full] training on {n_train} samples (val hold-out {n_val}), seed={args.seed}, "
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
            print(f"[full] epoch {epoch:03d}/{args.num_epochs} | "
                  f"train loss {running_loss / len(trainloader):.4f}", flush=True)
        if epoch % 50 == 0 or epoch == args.num_epochs:
            _, err = evaluate(model, test_cache)  # leaves model in eval(); next epoch re-enters train()
            accuracy_history[epoch] = 100.0 - err
            print(f"[full] epoch {epoch:03d}/{args.num_epochs} | test acc {100.0 - err:.2f}%", flush=True)

    tmp = ckpt_path + ".tmp"  # write-then-rename so a crash never leaves a half-written "final" file
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, ckpt_path)
    with open(os.path.splitext(ckpt_path)[0] + "_metrics.json", "w") as fh:
        json.dump(accuracy_history, fh, indent=4)
    print(f"[full] saved final (epoch {args.num_epochs}) weights -> {ckpt_path}", flush=True)


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


def edges(G, N):
    """1-D paths along the triangle edges; G[i,j], i = weight on ours, j = weight on X."""
    return {"full→ours": [G[i, 0] for i in range(N + 1)],
            "full→X": [G[0, j] for j in range(N + 1)],
            "ours→X": [G[N - j, j] for j in range(N + 1)]}


def barrier(path):
    return max(path) - 0.5 * (path[0] + path[-1])


def plot(results, key, label, path, vmax=None):
    N = results[0]["N"]
    pts = [(i / N, j / N) for i in range(N + 1) for j in range(N + 1 - i)]
    xs, ys = zip(*pts)
    tri = mtri.Triangulation(xs, ys)
    zs = [np.array([r[key][i, j] for i, j in [(int(round(x * N)), int(round(y * N))) for x, y in pts]]) for r in results]
    vmin = min(z.min() for z in zs)
    vmax = vmax or max(z.max() for z in zs)
    levels = np.linspace(vmin, vmax, 16)
    fig, axes = plt.subplots(len(results), 2, figsize=(9.5, 4.3 * len(results)), squeeze=False)
    for r, (res, z) in enumerate(zip(results, zs)):
        z = np.clip(z, vmin, vmax)
        for c in range(2):
            ax = axes[r, c]
            if c == 0:
                im = ax.tripcolor(tri, z, shading="gouraud", vmin=vmin, vmax=vmax, cmap="viridis")
            else:
                im = ax.tricontourf(tri, z, levels=levels, vmin=vmin, vmax=vmax, cmap="viridis")
                ax.tricontour(tri, z, levels=levels, colors="k", linewidths=0.4)
            corners = [((0, 0), "full", res[key][0, 0]), ((1, 0), "ours", res[key][N, 0]), ((0, 1), res["name"], res[key][0, N])]
            for (cx, cy), lab, v in corners:
                ax.scatter([cx], [cy], c="w", edgecolors="k", s=45, zorder=5)
                ax.annotate(f"{lab}\n{v:.2f}", (cx, cy), textcoords="offset points", xytext=(6, 6), fontsize=8,
                            zorder=6, bbox=dict(boxstyle="round,pad=0.15", fc="w", alpha=0.7, lw=0))
            ax.set_xlim(-0.12, 1.15); ax.set_ylim(-0.12, 1.15); ax.set_aspect("equal")
            ax.set_xlabel("a  (weight on ours)"); ax.set_ylabel(f"b  (weight on {res['name']})")
            if c == 0:
                ax.set_title(f"full / ours / {res['name']}", loc="left", fontsize=10)
    fig.colorbar(im, ax=axes.ravel().tolist(), label=label, shrink=0.8)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--download", action="store_true")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--baselines", nargs="+", required=True, help="baseline names as used in the directory layout")
    p.add_argument("--baseline_ckpt", action="append", default=[], metavar="NAME=PATH",
                   help="explicit path override for a baseline (repeatable)")
    p.add_argument("--budget", type=int, default=13500)
    p.add_argument("--seed", type=int, default=42,
                   help="seed for the full-data run (train/val split + init + data order); also the "
                        "seed used in .../seed_{seed}/ directories of ours / baselines")
    p.add_argument("--K", type=int, default=10, help="num checkpoints used in the kl_faithful file name")
    p.add_argument("--scratch", default=os.environ.get("SCRATCH", SCRATCH_DEFAULT))
    p.add_argument("--baseline_root", default=f"{SCRATCH_DEFAULT}/data_selection_baselines")
    p.add_argument("--ours_ckpt", default=None)
    # full-data training (same defaults as train_kl_selection.py / cifar100 config)
    p.add_argument("--num_epochs", type=int, default=300, help="full-data training epochs (cosine T_max)")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--retrain_full", action="store_true",
                   help="retrain the full-data model even if its final checkpoint already exists in out_dir")
    p.add_argument("--grid_res", type=int, default=20, help="N: (N+1)(N+2)/2 evaluations per baseline")
    p.add_argument("--no_bn_recal", action="store_true",
                   help="interpolate BN running stats linearly instead of re-estimating them")
    p.add_argument("--bn_samples", type=int, default=6400, help="unaugmented train images for BN re-estimation")
    p.add_argument("--vmax", type=float, default=None, help="clip colour scale")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── resolve + check ours / baseline checkpoints FIRST, so a wrong path fails now,
    #    not after 300 epochs of full-data training ─────────────────────────────────────
    ours_path = args.ours_ckpt or (f"{args.scratch}/gradmatch_swap/cifar100/{args.budget}/seed_{args.seed}/"
                                   f"kl_faithful_{args.budget}_K{args.K}_model.pth")
    overrides = dict(s.split("=", 1) for s in args.baseline_ckpt)
    paths = {n: overrides.get(n, f"{args.baseline_root}/{n}/cifar100/{args.budget}/seed_{args.seed}/"
                                 f"{n}_{args.budget}_model.pth") for n in args.baselines}
    full_path = os.path.join(args.out_dir, f"full_seed{args.seed}_epoch{args.num_epochs}.pth")
    print(f"{'full':>10s}: {full_path}  (trained by this script)")
    for name, pth in [("ours", ours_path)] + list(paths.items()):
        print(f"{name:>10s}: {pth}")
        if not os.path.exists(pth):
            raise FileNotFoundError(pth)

    kw = dict(root=args.data_dir, download=args.download, transform=TF_TEST)
    test_cache = cache(datasets.CIFAR100(train=False, **kw), device, 1000)
    bn_cache = None
    if not args.no_bn_recal:
        tr = datasets.CIFAR100(train=True, **kw)
        idx = np.random.RandomState(0).choice(len(tr), args.bn_samples, replace=False).tolist()
        bn_cache = cache(Subset(tr, idx), device, 128)

    # ── full-data training (final epoch only) ────────────────────────────────────────
    if os.path.exists(full_path) and not args.retrain_full:
        print(f"[full] found finished epoch-{args.num_epochs} checkpoint; skipping training "
              f"(pass --retrain_full to redo it)")
    else:
        train_full(args, device, full_path, test_cache)

    full, ours = load_sd(full_path, device), load_sd(ours_path, device)
    ev = Evaluator(device, test_cache, bn_cache)
    pkeys = [k for k, _ in ev.model.named_parameters()]
    l2 = lambda a, b: math.sqrt(sum(((a[k] - b[k]) ** 2).sum().item() for k in pkeys))

    results, summary = [], {}
    for name, pth in paths.items():
        X = load_sd(pth, device)
        print(f"[{name}] scanning grid N={args.grid_res}")
        loss, err = scan(ev, [full, ours, X], args.grid_res)
        res = dict(name=name, N=args.grid_res, loss=loss, err=err)
        results.append(res)
        summary[name] = dict(
            l2_full_ours=l2(full, ours), l2_full_X=l2(full, X), l2_ours_X=l2(ours, X),
            **{f"{k}_{m}_barrier": barrier(path) for m, G in (("err", err), ("loss", loss))
               for k, path in edges(G, args.grid_res).items()},
            **{f"{k}_err_path": path for k, path in edges(err, args.grid_res).items()},
            **{f"{k}_loss_path": path for k, path in edges(loss, args.grid_res).items()},
            max_err_in_triangle=float(np.nanmax(err)), max_loss_in_triangle=float(np.nanmax(loss)),
        )
        with open(os.path.join(args.out_dir, "interp_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=1)
        np.savez(os.path.join(args.out_dir, f"grid_{name}.npz"), loss=loss, err=err)

    plot(results, "err", "test error (%)", os.path.join(args.out_dir, "interp_test_error.png"), args.vmax)
    plot(results, "loss", "test loss", os.path.join(args.out_dir, "interp_test_loss.png"), args.vmax)

    print("\n" + "=" * 96)
    print(f"{'baseline':<12}{'L2 full-ours':>13}{'L2 full-X':>11}   test-err barrier (pp):  {'full→ours':>10}{'full→X':>9}{'ours→X':>9}")
    for n, s in summary.items():
        print(f"{n:<12}{s['l2_full_ours']:13.1f}{s['l2_full_X']:11.1f}   {'':<23}"
              f"{s['full→ours_err_barrier']:10.2f}{s['full→X_err_barrier']:9.2f}{s['ours→X_err_barrier']:9.2f}")
    print("=" * 96)


if __name__ == "__main__":
    main()