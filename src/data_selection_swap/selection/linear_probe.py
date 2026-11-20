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

Methods are named by the point whose gradients they use (``SELECTORS``):
W = 0, W_1 (one SGD epoch from W = 0) or W*. Herding and GraphCut use the
per-class gradient kernel at their point; at W = 0 that is the feature kernel
up to a constant, so there they are the usual feature-based methods.

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
# method -> (selector, point whose gradients it uses). Points: w0 (W = 0),
# w1 (one SGD epoch from W = 0, ``sgd_epoch``) and wstar (the full-data
# solution). At W = 0 every image of a class has the same residual, so the
# gradient kernel is a constant times the feature kernel: herding_w0 and
# graphcut_w0 are the feature-based Herding and GraphCut. 'herding' and
# 'graphcut' are the names configs/revision/cifar100_linear_probe.yaml uses.
SELECTORS = {
    'random': ('random', None),
    'random_stratified': ('stratified', None),
    'herding': ('herding', 'w0'),
    'graphcut': ('graphcut', 'w0'),
    'second_order': ('swap', 'wstar'),
    'second_order_class_balanced': ('swap_cb', 'wstar'),
    'herding_then_swap': ('herding_swap', 'wstar'),     # swap at W* from Herding's subset
    'swap_two_point': ('swap_cb', 'two_point'),         # G(W = 0) + G(W*)
}
for _point in ('w0', 'w1'):
    SELECTORS.update({f'herding_{_point}': ('herding', _point),
                      f'graphcut_{_point}': ('graphcut', _point),
                      f'second_order_{_point}': ('swap', _point),
                      f'class_balanced_{_point}': ('swap_cb', _point)})
METHODS = tuple(SELECTORS)

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


def two_point_residuals(Phi, Y, W_star):
    """
    Residuals whose gradients r_i phi_i^T stack the gradients at W = 0 and at
    W*, so ProbeGradients on them gives G(W=0) + G(W*): the two points share
    phi, so stacking residuals stacks gradients. Each point is scaled to the
    same mean squared centred gradient, so neither dominates.
    """
    blocks = []
    for W in (torch.zeros_like(W_star), W_star):
        R = residuals(Phi, Y, W)
        blocks.append(R / ProbeGradients(R, Phi).q.mean().sqrt())
    return torch.cat(blocks, dim=1)


def sgd_epoch(Phi, Y, num_classes, lam, lr=0.01, batch_size=128, seed=0):
    """W_1: one epoch of SGD (Nesterov momentum 0.9, as everywhere in the repo)
    from W = 0 on mean CE + lam/2 ||W||^2, with a seeded shuffle."""
    W = torch.zeros(num_classes, Phi.shape[1], dtype=Phi.dtype, device=Phi.device,
                    requires_grad=True)
    optimizer = common.build_optimizer([W], 'sgd', lr, weight_decay=lam)
    order = torch.randperm(len(Y), generator=torch.Generator().manual_seed(seed))
    for start in range(0, len(Y), batch_size):
        batch = order[start:start + batch_size].to(Phi.device)
        optimizer.zero_grad()
        F.cross_entropy(Phi[batch] @ W.T, Y[batch]).backward()
        optimizer.step()
    return W.detach()


def class_kernels(R, Phi, class_idx):
    """Per-class gradient kernels <g_i, g_j> = <r_i, r_j> <phi_i, phi_j>."""
    return [(R[idx] @ R[idx].T) * (Phi[idx] @ Phi[idx].T) for idx in class_idx]


def herding(kernels, class_idx, quota):
    """
    Kernel herding per class (Welling 2009): greedily keep the mean of the
    chosen embeddings closest to the class mean. With S the chosen sum after
    t - 1 picks, j minimises ||mu - (S + x_j)/t||^2, i.e.
    -2 t <mu, x_j> + 2 <S, x_j> + ||x_j||^2.
    """
    chosen = []
    for K, members, take in zip(kernels, class_idx, quota.tolist()):
        mean_sim, picked_sim = K.mean(0), torch.zeros_like(K[0])
        free = torch.ones(len(members), dtype=torch.bool, device=K.device)
        for t in range(1, take + 1):
            score = (-2 * t * mean_sim + 2 * picked_sim + K.diagonal()).masked_fill(
                ~free, float('inf'))
            j = int(score.argmin())
            free[j] = False
            picked_sim += K[j]
            chosen.append(int(members[j]))
    return torch.tensor(sorted(chosen))


def graphcut(kernels, class_idx, quota):
    """GraphCut on per-class kernels, as scripts/select/submodular.py."""
    from .submodular import submodular_select
    return submodular_select('graphcut', [K.cpu() for K in kernels],
                             [idx.cpu() for idx in class_idx], quota, 1.0, 'lazy',
                             0).sort().values


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
                   help='Features, W*, W_1, results and indices are written here.')
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
    accuracy = lambda W: 100.0 * float(((Phi_test @ W.T).argmax(1) == Y_test).float().mean())

    def build_full():
        W, grad_norm = fit_probe(Phi, Y, C, lam)
        return dict(W=W.cpu(), grad_norm=grad_norm)
    full = _cached(os.path.join(args.out_dir, f'full_probe_wd{lam:g}.pt'), build_full)
    W_star = full['W'].to(device)
    full_pred = (Phi_test @ W_star.T).argmax(1)
    full_acc = accuracy(W_star)
    W_1 = _cached(os.path.join(args.out_dir, f'w1_wd{lam:g}.pt'),
                  lambda: dict(W=sgd_epoch(Phi, Y, C, lam).cpu()))['W'].to(device)
    logger.info(f"Full-data probe (lam={lam:g}): test acc {full_acc:.2f}%, gradient norm "
                f"{full['grad_norm']:.2e}; W_1 (one SGD epoch) test acc {accuracy(W_1):.2f}%; "
                f"budget m={m} of n={n}")

    weights = {'w0': torch.zeros_like(W_star), 'w1': W_1, 'wstar': W_star}
    grads = {point: ProbeGradients(residuals(Phi, Y, W), Phi) for point, W in weights.items()}
    quota = common.per_class_budget(Y.cpu(), C, m)
    class_idx = [torch.where(Y == c)[0] for c in range(C)]
    kernels = {}            # per point, built on first use

    def class_kernels_at(point):
        if point not in kernels:
            kernels[point] = class_kernels(grads[point].R, Phi, class_idx)
        return kernels[point]
    index_dir = os.path.join(args.out_dir, 'indices')
    os.makedirs(index_dir, exist_ok=True)

    rows = []
    for method in args.methods:
        kind, point = SELECTORS[method]
        deterministic = kind in ('herding', 'graphcut', 'herding_swap')
        for seed in ([None] if deterministic else args.seeds):
            t0 = time.perf_counter()
            extra = {}
            if kind == 'random':
                index = subsets.initial_subset(n, m, seed=seed)
            elif kind == 'stratified':
                index = subsets.initial_subset(n, m, seed=seed, labels=Y.cpu(), num_classes=C)
            elif kind == 'herding':
                index = herding(class_kernels_at(point), class_idx, quota)
            elif kind == 'graphcut':
                index = graphcut(class_kernels_at(point), class_idx, quota)
            else:
                # swap: plain; swap_cb and herding_swap: class-balanced.
                if point == 'two_point':
                    if 'two_point' not in grads:
                        grads['two_point'] = ProbeGradients(
                            two_point_residuals(Phi, Y, W_star), Phi)
                start = (herding(class_kernels_at('w0'), class_idx, quota)
                         if kind == 'herding_swap' else None)
                index, _, diag = second_order_swap(
                    grads[point], m, k=args.shortlist_k, max_iters=args.max_iters,
                    labels=None if kind == 'swap' else Y, num_classes=C,
                    class_k=args.class_shortlist_k, seed=seed, init_index=start)
                extra = {key: diag[key] for key in ('swaps', 'reason', 'one_swap_local_optimum')}
            selection_sec = time.perf_counter() - t0

            index = index.to(device)
            W, grad_norm = fit_probe(Phi[index], Y[index], C, lam)
            pred = (Phi_test @ W.T).argmax(1)
            G = {}
            for name in ('w0', 'w1', 'wstar'):
                grads[name].set_subset(index)
                G[name] = grads[name].D(m)
            row = dict(method=method, seed=seed, fraction=args.fraction, budget=m,
                       selection_point=point,
                       test_acc=100.0 * float((pred == Y_test).float().mean()),
                       agreement=100.0 * float((pred == full_pred).float().mean()),
                       dist_to_full=float((W - W_star).norm()),
                       G=G['wstar'], bound=G['wstar'] ** 0.5 / lam,
                       G_at_zero=G['w0'], G_at_w1=G['w1'],
                       classes_covered=int(Y[index].unique().numel()),
                       selection_sec=selection_sec, fit_grad_norm=grad_norm, **extra)
            rows.append(row)
            torch.save(index.cpu(), os.path.join(
                index_dir, f"{method}_f{args.fraction:g}_seed{seed}.pt"))
            logger.info(f"{method:28s} seed={seed}  acc={row['test_acc']:.2f}%  "
                        f"agree={row['agreement']:.2f}%  |W-W*|={row['dist_to_full']:.3f}  "
                        f"G(0)={G['w0']:.3e}  G(W1)={G['w1']:.3e}  ({selection_sec:.1f}s)")

    out = os.path.join(args.out_dir, f"results_f{args.fraction:g}_wd{lam:g}.json")
    if os.path.exists(out):     # keep the rows of methods this run did not redo
        with open(out) as fh:
            rows = [r for r in json.load(fh)['rows'] if r['method'] not in args.methods] + rows
    common.atomic_json_dump(dict(dataset=DATASET, architecture=ARCHITECTURE,
                                 weight_decay=lam, full_test_acc=full_acc,
                                 full_grad_norm=full['grad_norm'],
                                 w1_test_acc=accuracy(W_1), n=n, rows=rows), out)
    logger.info(f"Wrote {out}")


if __name__ == "__main__":
    main()
