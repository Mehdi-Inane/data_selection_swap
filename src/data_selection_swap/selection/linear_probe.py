"""
Linear probe on frozen ImageNet features: the setting where gradient matching
provably controls the result.

Only a softmax-regression head W (bias as its last column) is trained, on the
frozen 512-d features of torchvision's ImageNet-1k ResNet-18 (cifar100_in1k,
224 px), with weight decay lam. The objective is lam-strongly convex, so the
subset's solution W_I and the full-data solution W* satisfy

    ||W_I - W*|| <= || (1/m) sum_{i in I} g_i(W*) - gbar(W*) || / lam
                  = sqrt(G(I)) / lam,

g_i being example i's cross-entropy gradient at W*. G at the single point W*
is what the second-order swap minimises, here exactly: g_i = r_i phi_i^T with
r_i = softmax - onehot and phi_i = [features, 1], so every swap quantity
factors through r and phi (``ProbeGradients``) and the 51k-d gradients are
never stored.

One run is one budget: every method and seed. Each probe is fitted to
convergence by full-batch L-BFGS from W = 0, which is deterministic, so the
only randomness is the subset. Features and W* are cached in --out_dir.
"""

import argparse
import fcntl
import json
import logging
import os
import time

import torch
import torch.nn.functional as F

from .. import common, subsets
from .second_order import second_order_swap

DATASET, ARCHITECTURE = 'cifar100_in1k', 'resnet18_in1k'
METHODS = ('random', 'random_stratified', 'herding', 'graphcut',
           'second_order', 'second_order_class_balanced')

logger = logging.getLogger(__name__)


class ProbeGradients:
    """
    The trajectory protocol of ``second_order.second_order_swap`` for the
    probe's exact gradients at one point, z_r = r_r phi_r^T - gbar.

    R [n, C] and Phi [n, F + 1] are float64. Every value the rule acts on
    (u, D, q, swap_block) is exact in float64; only the pool-wide scores s,
    which choose the shortlists, are computed in float32.
    """

    def __init__(self, R, Phi):
        self.R, self.Phi = R, Phi
        self.X = Phi                                     # the protocol reads n and the device
        self.R32, self.Phi32 = R.float(), Phi.float()
        self.gbar = R.T @ Phi / R.shape[0]               # [C, F + 1]
        self.gbar_sq = float((self.gbar ** 2).sum())
        self.h = ((R @ self.gbar) * Phi).sum(1)          # <g_r, gbar>
        self.q = (R ** 2).sum(1) * (Phi ** 2).sum(1) - 2 * self.h + self.gbar_sq  # ||z_r||^2
        self.u = None

    def aggregate(self, index):
        index = torch.as_tensor(index, device=self.R.device)
        return [self.R[index].T @ self.Phi[index] - index.numel() * self.gbar]

    def set_subset(self, index):
        self.u = self.aggregate(index)

    def D(self, m):
        return float((self.u[0] ** 2).sum()) / m ** 2

    def _inner(self, rows, U):
        """<z_r, U> for the given rows, in float64."""
        return ((self.R[rows] @ U) * self.Phi[rows]).sum(1) - (self.gbar * U).sum()

    def gradient(self, m):
        U = self.u[0]
        s = ((self.R32 @ U.float()) * self.Phi32).sum(1).double() - (self.gbar * U).sum()
        return s * (2.0 / m ** 2)

    def diagonal(self):
        return self.q

    def swap_block(self, drop, add, m):
        """Delta[a, b] = G(I - drop[a] + add[b]) - G(I), exact in float64."""
        inv, U = 1.0 / m ** 2, self.u[0]
        d = self.q[drop] * inv - 2 * inv * self._inner(drop, U)
        a = self.q[add] * inv + 2 * inv * self._inner(add, U)
        cross = ((self.R[drop] @ self.R[add].T) * (self.Phi[drop] @ self.Phi[add].T)
                 - self.h[drop][:, None] - self.h[add][None, :] + self.gbar_sq)
        return d[:, None] + a[None, :] - 2 * inv * cross


def fit_probe(Phi, Y, num_classes, lam, max_iter=1000):
    """argmin_W mean CE(Phi W^T, Y) + lam/2 ||W||^2 by L-BFGS from W = 0."""
    W = torch.zeros(num_classes, Phi.shape[1], dtype=Phi.dtype, device=Phi.device,
                    requires_grad=True)
    optimizer = torch.optim.LBFGS([W], lr=1, max_iter=max_iter, history_size=20,
                                  tolerance_grad=1e-10, tolerance_change=1e-14,
                                  line_search_fn='strong_wolfe')

    def objective():
        return F.cross_entropy(Phi @ W.T, Y) + 0.5 * lam * (W ** 2).sum()

    def closure():
        optimizer.zero_grad()
        loss = objective()
        loss.backward()
        return loss

    optimizer.step(closure)
    W = W.detach().requires_grad_(True)
    grad, = torch.autograd.grad(objective(), W)
    return W.detach(), float(grad.norm())


def residuals(Phi, Y, W):
    """r_i = softmax(W phi_i) - onehot(y_i)."""
    R = torch.softmax(Phi @ W.T, dim=1)
    R[torch.arange(len(Y)), Y] -= 1.0
    return R


def herding(features, Y, quota):
    """Per class, greedily keep the running mean of the chosen features
    closest to the class mean (Welling 2009)."""
    chosen = []
    for c, take in enumerate(quota.tolist()):
        members = torch.where(Y == c)[0]
        X = features[members]
        mean, total = X.mean(0), torch.zeros_like(X[0])
        free = torch.ones(len(members), dtype=torch.bool, device=X.device)
        for t in range(1, take + 1):
            gap = ((mean - (total + X) / t) ** 2).sum(1).masked_fill(~free, float('inf'))
            j = int(gap.argmin())
            free[j] = False
            total += X[j]
            chosen.append(int(members[j]))
    return torch.tensor(sorted(chosen))


def graphcut(features, Y, quota, num_classes):
    """GraphCut on per-class feature kernels, as scripts/select/submodular.py."""
    from .submodular import submodular_select
    features, Y = features.cpu(), Y.cpu()
    class_idx = [torch.where(Y == c)[0] for c in range(num_classes)]
    kernels = [features[idx] @ features[idx].T for idx in class_idx]
    return submodular_select('graphcut', kernels, class_idx, quota, 1.0, 'lazy', 0).sort().values


def _cached(path, build):
    """Load ``path``, or build and save it once under a lock (parallel tasks)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(f"{path}.lock", 'a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not os.path.exists(path):
            common.atomic_torch_save(build(), path)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return common.torch_load(path)


def extract_features(args, device):
    """Frozen penultimate features of the pool and the test set (clean views)."""
    cfg = common.get_dataset_config(DATASET, args.data_dir, download=args.download,
                                    architecture=ARCHITECTURE)
    common.set_seed(0)
    model = common.build_model(cfg, cfg['num_classes'], device)
    _, eval_loader, testloader, _ = common.make_loaders(
        cfg, 0, args.batch_size, args.num_workers, scoring_transform='test')
    _, train_phi, train_y = common.extract_probs_features_targets(model, eval_loader, device)
    _, test_phi, test_y = common.extract_probs_features_targets(model, testloader, device)
    return dict(train_phi=train_phi, train_y=train_y, test_phi=test_phi, test_y=test_y,
                num_classes=cfg['num_classes'])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_dir', required=True)
    p.add_argument('--out_dir', required=True,
                   help='Features, W*, results and indices are written here.')
    p.add_argument('--fraction', required=True, type=float)
    p.add_argument('--seeds', default=[0], type=int, nargs='+')
    p.add_argument('--methods', default=list(METHODS), nargs='+', choices=METHODS)
    p.add_argument('--weight_decay', default=1e-3, type=float, help='lam')
    p.add_argument('--shortlist_k', default=16, type=int)
    p.add_argument('--class_shortlist_k', default=16, type=int)
    p.add_argument('--max_iters', default=5000, type=int,
                   help='Cap on second-order swaps.')
    p.add_argument('--batch_size', default=256, type=int)
    p.add_argument('--num_workers', default=4, type=int)
    p.add_argument('--download', action='store_true', default=False)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lam = args.weight_decay

    data = _cached(os.path.join(args.out_dir, 'features.pt'),
                   lambda: extract_features(args, device))
    C = int(data['num_classes'])
    ones = lambda x: torch.cat([x, torch.ones(len(x), 1)], 1).double().to(device)
    Phi, Phi_test = ones(data['train_phi']), ones(data['test_phi'])
    Y, Y_test = data['train_y'].to(device), data['test_y'].to(device)
    n, m = len(Y), int(args.fraction * len(Y))

    def build_full():
        W, grad_norm = fit_probe(Phi, Y, C, lam)
        return dict(W=W.cpu(), grad_norm=grad_norm)
    full = _cached(os.path.join(args.out_dir, f'full_probe_wd{lam:g}.pt'), build_full)
    W_star = full['W'].to(device)
    full_pred = (Phi_test @ W_star.T).argmax(1)
    full_acc = 100.0 * float((full_pred == Y_test).float().mean())
    logger.info(f"Full-data probe (lam={lam:g}): test acc {full_acc:.2f}%, "
                f"gradient norm {full['grad_norm']:.2e}; budget m={m} of n={n}")

    grads = ProbeGradients(residuals(Phi, Y, W_star), Phi)
    quota = common.per_class_budget(Y.cpu(), C, m)
    features = data['train_phi'].to(device)
    deterministic = {'herding', 'graphcut'}         # the same subset for every seed
    index_dir = os.path.join(args.out_dir, 'indices')
    os.makedirs(index_dir, exist_ok=True)

    rows = []
    for method in args.methods:
        for seed in ([None] if method in deterministic else args.seeds):
            t0 = time.perf_counter()
            extra = {}
            if method == 'random':
                index = subsets.initial_subset(n, m, seed=seed)
            elif method == 'random_stratified':
                index = subsets.initial_subset(n, m, seed=seed, labels=Y.cpu(), num_classes=C)
            elif method == 'herding':
                index = herding(features, Y, quota)
            elif method == 'graphcut':
                index = graphcut(features, Y, quota, C)
            else:
                balanced = method == 'second_order_class_balanced'
                index, _, diag = second_order_swap(
                    grads, m, k=args.shortlist_k, max_iters=args.max_iters,
                    labels=Y if balanced else None, num_classes=C,
                    class_k=args.class_shortlist_k, seed=seed)
                extra = {key: diag[key] for key in ('swaps', 'reason', 'one_swap_local_optimum')}
            selection_sec = time.perf_counter() - t0

            index = index.to(device)
            W, grad_norm = fit_probe(Phi[index], Y[index], C, lam)
            pred = (Phi_test @ W.T).argmax(1)
            grads.set_subset(index)
            G = grads.D(m)
            row = dict(method=method, seed=seed, fraction=args.fraction, budget=m,
                       test_acc=100.0 * float((pred == Y_test).float().mean()),
                       agreement=100.0 * float((pred == full_pred).float().mean()),
                       dist_to_full=float((W - W_star).norm()),
                       G=G, bound=G ** 0.5 / lam,
                       classes_covered=int(Y[index].unique().numel()),
                       selection_sec=selection_sec, fit_grad_norm=grad_norm, **extra)
            rows.append(row)
            torch.save(index.cpu(), os.path.join(
                index_dir, f"{method}_f{args.fraction:g}_seed{seed}.pt"))
            logger.info(f"{method:28s} seed={seed}  acc={row['test_acc']:.2f}%  "
                        f"agree={row['agreement']:.2f}%  |W-W*|={row['dist_to_full']:.3f}  "
                        f"G={G:.3e}  ({selection_sec:.1f}s)")

    out = os.path.join(args.out_dir, f"results_f{args.fraction:g}_wd{lam:g}.json")
    common.atomic_json_dump(dict(dataset=DATASET, architecture=ARCHITECTURE,
                                 weight_decay=lam, full_test_acc=full_acc,
                                 full_grad_norm=full['grad_norm'], n=n, rows=rows), out)
    logger.info(f"Wrote {out}")


if __name__ == "__main__":
    main()
