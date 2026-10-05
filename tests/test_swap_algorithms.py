"""Focused numerical and stopping-rule tests for manuscript Algorithms 2/3.

The production package imports optional vision/experiment dependencies that
are intentionally absent from the lightweight test environment.  These tests
load the two selector modules under a private package with only the shared
``ranked_pairs`` primitive stubbed; the swap arithmetic itself is unmodified.
"""

import importlib.util
import itertools
import sys
import types
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "src" / "data_selection_swap" / "selection"
PACKAGE = "_swap_algorithm_tests"


def _ranked_pairs(drop_idx, drop_h, add_idx, add_h):
    pairs = [
        (float(add_h[b] - drop_h[a]), int(drop_idx[a]), int(add_idx[b]))
        for a in range(len(drop_idx))
        for b in range(len(add_idx))
    ]
    yield from sorted(pairs)


def _load(name):
    full_name = f"{PACKAGE}.selection.{name}"
    spec = importlib.util.spec_from_file_location(full_name, SOURCE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


package = types.ModuleType(PACKAGE)
package.__path__ = []
selection_package = types.ModuleType(f"{PACKAGE}.selection")
selection_package.__path__ = []
common = types.ModuleType(f"{PACKAGE}.common")
common.ranked_pairs = _ranked_pairs
objective = types.ModuleType(f"{PACKAGE}.objective")


def _weighted_sq_norm(v, metric=None):
    sq = v * v
    return float((sq if metric is None else sq * metric).sum().item())


objective.weighted_sq_norm = _weighted_sq_norm
trajectory = types.ModuleType(f"{PACKAGE}.trajectory")
paths = types.ModuleType(f"{PACKAGE}.paths")
paths.storage_variant = lambda *names: ''
paths.tree = lambda scratch, root_name, variant='': scratch
sys.modules.update({
    PACKAGE: package,
    f"{PACKAGE}.selection": selection_package,
    f"{PACKAGE}.common": common,
    f"{PACKAGE}.objective": objective,
    f"{PACKAGE}.trajectory": trajectory,
    f"{PACKAGE}.paths": paths,
})

kl = _load("kl")
kl_full = _load("kl_full")


def _direct_D(traj, selected):
    selected = torch.as_tensor(selected, dtype=torch.long)
    m = selected.numel()
    total = 0.0
    for L, Phi, metric, weight in traj:
        gradients = torch.einsum("nc,nf->ncf", L.double(), Phi.double())
        residual = gradients[selected].mean(dim=0) - gradients.mean(dim=0)
        if metric is not None:
            residual = residual * metric.double().sqrt()
        total += float(weight) * float((residual * residual).sum())
    return total


class FactoredSwapTest(unittest.TestCase):
    @staticmethod
    def trajectory(n=9):
        generator = torch.Generator().manual_seed(913)
        return [
            (torch.randn(n, 3, generator=generator),
             torch.randn(n, 4, generator=generator),
             torch.rand(3, 4, generator=generator) + 0.2,
             0.65),
            (torch.randn(n, 2, generator=generator),
             torch.randn(n, 5, generator=generator),
             None,
             0.35),
        ]

    def test_centered_pair_changes_equal_direct_D_differences(self):
        traj = self.trajectory()
        selected = torch.tensor([0, 2, 5, 7])
        centered, _ = kl._centered_state(traj, selected, selected.numel())
        outside = [i for i in range(9) if i not in selected.tolist()]
        drops = [int(i) for i in selected for _ in outside]
        adds = [j for _ in selected for j in outside]

        actual = kl._pair_changes_fp64(
            traj, centered, selected.numel(), drops, adds).cpu()
        before = _direct_D(traj, selected)
        expected = []
        for i, j in zip(drops, adds):
            after = [j if int(k) == i else int(k) for k in selected]
            expected.append(_direct_D(traj, after) - before)

        torch.testing.assert_close(
            actual, torch.tensor(expected, dtype=torch.float64),
            rtol=2e-12, atol=2e-12)

    def test_algorithm2_returns_exact_one_swap_local_optimum(self):
        traj = self.trajectory(n=8)
        indices, history, diag = kl.best_swap(
            traj, 3, max_iters=100, log_freq=1, device="cpu", seed=5,
            max_pairs=0, batch_pairs=3)

        centered, _ = kl._centered_state(traj, indices, 3)
        selected = indices.tolist()
        outside = [i for i in range(8) if i not in selected]
        drops = [i for i in selected for _ in outside]
        adds = [j for _ in selected for j in outside]
        final_changes = kl._pair_changes_fp64(
            traj, centered, 3, drops, adds)

        self.assertGreaterEqual(float(final_changes.min()), -1e-12)
        self.assertTrue(diag["algorithm_2_exact"])
        self.assertTrue(diag["one_swap_local_optimum"])
        self.assertEqual(diag["iterations"], diag["swaps"])
        self.assertEqual(diag["pair_searches"], diag["swaps"] + 1)
        objectives = [entry["D"] for entry in history if "D" in entry]
        self.assertTrue(all(b <= a + 1e-12
                            for a, b in zip(objectives, objectives[1:])))

    def test_algorithm2_first_exchange_is_the_true_argmin(self):
        traj = self.trajectory(n=8)
        m, seed = 3, 5
        initial = torch.randperm(
            8, generator=torch.Generator().manual_seed(seed))[:m]
        centered, _ = kl._centered_state(traj, initial, m)
        selected = initial.tolist()
        outside = [i for i in range(8) if i not in selected]
        drops = [i for i in selected for _ in outside]
        adds = [j for _ in selected for j in outside]
        changes = kl._pair_changes_fp64(traj, centered, m, drops, adds)
        best = int(torch.argmin(changes))
        self.assertLess(float(changes[best]), 0.0)
        expected = (set(selected) - {drops[best]}) | {adds[best]}

        indices, _, diag = kl.best_swap(
            traj, m, max_iters=1, log_freq=1, device="cpu", seed=seed,
            max_pairs=0, batch_pairs=2)

        self.assertEqual(set(indices.tolist()), expected)
        self.assertEqual(diag["swaps"], 1)
        self.assertEqual(diag["pair_searches"], 1)
        self.assertEqual(diag["reason"], "max_iters")
        self.assertIsNone(diag["final_change"])

    def test_algorithm3_rejection_is_counted_but_not_called_local_optimal(self):
        generator = torch.Generator().manual_seed(3)
        traj = [(torch.randn(8, 3, generator=generator),
                 torch.randn(8, 4, generator=generator),
                 torch.rand(3, 4, generator=generator) + 0.2,
                 0.7)]
        indices, _, diag = kl.gradient_ranked_single_swap(
            traj, 3, max_iters=100, log_freq=1, device="cpu", seed=5)

        centered, _ = kl._centered_state(traj, indices, 3)
        selected = indices.tolist()
        outside = [i for i in range(8) if i not in selected]
        changes = kl._pair_changes_fp64(
            traj, centered, 3,
            [i for i in selected for _ in outside],
            [j for _ in selected for j in outside])

        self.assertEqual(diag["pair_searches"], 1)
        self.assertEqual(diag["swaps"], 0)
        self.assertEqual(diag["iterations"], diag["swaps"])
        self.assertFalse(diag["one_swap_local_optimum"])
        self.assertLess(float(changes.min()), 0.0)
        self.assertGreaterEqual(diag["certificate"], 0.0)
        self.assertTrue(diag["certificate_valid"])
        optimum = min(_direct_D(traj, subset)
                      for subset in itertools.combinations(range(8), 3))
        self.assertLessEqual(diag["D"] - optimum,
                             diag["certificate"] + 1e-12)
        self.assertLessEqual(diag["lower_bound"], optimum + 1e-12)

    def test_float32_false_acceptance_regression(self):
        generator = torch.Generator().manual_seed(1202)
        # Generating in float64 and then storing float32 reproduces the
        # cancellation case documented in troubleshoot.pdf.
        noise = torch.randn(20, 5, generator=generator, dtype=torch.float64).float()
        L = torch.ones(20, 1, dtype=torch.float32)
        Phi = torch.cat((10.0 + noise, torch.ones(20, 1)), dim=1)
        traj = [(L, Phi, None, 1.0)]
        m, n = 6, 20
        selected = torch.tensor([18, 19, 5, 12, 13, 7])
        centered, _ = kl._centered_state(traj, selected, m)
        kl._apply_centered_swap(traj, centered, 19, 11)
        kl._apply_centered_swap(traj, centered, 5, 0)
        exact = float(kl._pair_changes_fp64(
            traj, centered, m, [18], [19])[0])

        alpha = torch.zeros(n)
        alpha[[18, 11, 0, 12, 13, 7]] = 1.0
        c_sum = kl.weighted_kernel_mv(traj, torch.ones(n))
        h = kl.compute_h(traj, alpha, m, n, c_sum)
        diagonal = kl.kernel_diagonal(traj)
        old = float(h[19] - h[18]
                    + (diagonal[18] + diagonal[19]
                       - 2.0 * kl.kernel_pairs(
                           traj, torch.tensor([18]), torch.tensor([19]))[0]) / m ** 2)

        self.assertLess(old, 0.0)       # old float32 path falsely accepted
        self.assertGreater(exact, 0.0)  # centered float64 path correctly rejects


class _DenseTrajectory:
    """Small exact stand-in for the all-parameter trajectory protocol."""

    def __init__(self, gradients):
        self.G = gradients.double()
        self.X = torch.zeros(gradients.shape[0], 1)
        self.u = None

    def set_subset(self, index):
        self.u = [self.G[index].sum(dim=0)]

    def aggregate(self, index):
        return [self.G[index].sum(dim=0)]

    def _residual(self, m):
        return self.u[0] / m - self.G.mean(dim=0)

    def D(self, m):
        return float((self._residual(m) ** 2).sum())

    def gradient(self, m):
        return (2.0 / m) * (self.G @ self._residual(m))

    def exact_change(self, deltas, m):
        delta = deltas[0]
        residual = self._residual(m)
        return float((2.0 / m) * (residual * delta).sum()
                     + (delta * delta).sum() / m ** 2)


class FullNetworkSwapTest(unittest.TestCase):
    def test_full_algorithm3_reports_distinct_sweep_and_swap_counts(self):
        generator = torch.Generator().manual_seed(27)
        traj = _DenseTrajectory(torch.randn(10, 5, generator=generator))
        _, history, diag = kl_full.gradient_ranked_single_swap(
            traj, 4, max_iters=50, seed=9)

        self.assertEqual(diag["algorithm"], 3)
        self.assertEqual(diag["attempted_exchanges"], diag["swaps"] + 1)
        self.assertEqual(diag["sweeps"], diag["attempted_exchanges"])
        self.assertLessEqual(diag["D"], history[0]["D"] + 1e-12)
        estimates = [entry["certificate_estimate"] for entry in history
                     if entry.get("certificate_estimate") is not None]
        self.assertTrue(all(b <= a + 1e-12
                            for a, b in zip(estimates, estimates[1:])))
        self.assertEqual(diag["decision_dtype"], "float64")
        self.assertTrue(diag["centered_acceptance"])
        self.assertEqual(diag["projection"], "none")
        self.assertEqual(diag["proj_dim"], 0)


class CountSketchTest(unittest.TestCase):
    def test_adjoint_linearity_and_determinism(self):
        sketch = kl_full.CountSketch(32, 7, seed=3)
        again = kl_full.CountSketch(32, 7, seed=3)
        other = kl_full.CountSketch(32, 7, seed=4)
        generator = torch.Generator().manual_seed(11)
        x = torch.randn(32, dtype=torch.float64, generator=generator)
        z = torch.randn(32, dtype=torch.float64, generator=generator)
        y = torch.randn(7, dtype=torch.float64, generator=generator)

        torch.testing.assert_close(sketch.apply(x), again.apply(x))
        self.assertFalse(torch.equal(sketch.apply(x), other.apply(x)))
        torch.testing.assert_close(sketch.apply(x + z), sketch.apply(x) + sketch.apply(z))
        left = torch.dot(sketch.apply(x), y)
        right = torch.dot(x, sketch.adjoint(y))
        torch.testing.assert_close(left, right)

    def test_apply_batch_matches_apply(self):
        sketch = kl_full.CountSketch(30, 6, seed=3)
        flats = torch.randn(5, 30)
        batched = sketch.apply_batch(flats)
        for row in range(flats.shape[0]):
            torch.testing.assert_close(batched[row], sketch.apply(flats[row]))


class _Ckpt:
    def __init__(self, path, weight):
        self.path = path
        self.weight = weight
        self.epoch = 1
        self.step = 1


def _per_example_grads(model, state, X, Y, order):
    model.load_state_dict(state)
    grads = []
    for i in range(X.shape[0]):
        model.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(X[i:i + 1]), Y[i:i + 1]).backward()
        named = dict(model.named_parameters())
        grads.append(torch.cat([named[n].grad.reshape(-1) for n in order]).double())
    return torch.stack(grads)


def _linear_pool(n=12, d=4, classes=3, seed=19):
    generator = torch.Generator().manual_seed(seed)
    model = torch.nn.Linear(d, classes)
    with torch.no_grad():
        first = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.weight.add_(0.7)
        model.bias.add_(-0.4)
        second = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bank = {"a": first, "b": second}
    X = torch.randn(n, d, generator=generator)
    Y = torch.randint(0, classes, (n,), generator=generator)
    checkpoints = [_Ckpt("a", 0.25), _Ckpt("b", 0.75)]
    return model, bank, X, Y, checkpoints


class SketchedTrajectoryTest(unittest.TestCase):
    def setUp(self):
        self._saved_loader = getattr(trajectory, "load_weights", None)
        self._saved_metric = getattr(trajectory, "adam_metric_flat", None)

    def tearDown(self):
        if self._saved_loader is None:
            if hasattr(trajectory, "load_weights"):
                del trajectory.load_weights
        else:
            trajectory.load_weights = self._saved_loader
        if self._saved_metric is None:
            if hasattr(trajectory, "adam_metric_flat"):
                del trajectory.adam_metric_flat
        else:
            trajectory.adam_metric_flat = self._saved_metric

    def _trajectory(self, proj_dim, use_metric=False, device="cpu"):
        model, bank, X, Y, checkpoints = _linear_pool()

        def load_weights(path):
            return {k: v.detach().clone() for k, v in bank[path].items()}

        trajectory.load_weights = load_weights
        X = X.to(device)
        Y = Y.to(device)
        traj = kl_full.Trajectory(
            model, checkpoints, use_metric, X, Y, device, batch_size=5,
            proj_dim=proj_dim, proj_seed=2)
        grads = [_per_example_grads(model, bank[c.path], X.cpu(), Y.cpu(), traj.order)
                 for c in checkpoints]
        return traj, grads, checkpoints

    def _explicit_D(self, traj, grads, checkpoints, index, precondition=None):
        m = int(index.numel())
        total = 0.0
        for weight, G in zip((c.weight for c in checkpoints), grads):
            residual = G[index].sum(dim=0) / m - G.mean(dim=0)
            if precondition is not None:
                residual = precondition * residual
            if traj.sketch is not None:
                residual = traj.sketch.apply(residual)
            total += weight * float((residual * residual).sum())
        return total

    def test_sketched_D_matches_explicit_projection_and_descends(self):
        traj, grads, checkpoints = self._trajectory(proj_dim=8)
        index = torch.tensor([0, 2, 5, 7])
        traj.set_subset(index)
        self.assertEqual(tuple(traj.u[0].shape), (8,))
        self.assertEqual(traj.weight_device, "cpu")
        self.assertEqual(
            next(iter(traj.states[0][0].values())).device.type, "cpu")
        sketched = traj.D(index.numel())
        expected = self._explicit_D(traj, grads, checkpoints, index)
        self.assertAlmostEqual(sketched, expected, places=5)

        _, history, diag = kl_full.gradient_ranked_single_swap(
            traj, 4, max_iters=15, seed=1)
        self.assertLess(diag["D"], history[0]["D"])
        self.assertEqual(diag["projection"], "count_sketch")
        self.assertEqual(diag["proj_dim"], 8)
        self.assertEqual(diag["proj_seed"], 2)
        self.assertGreater(diag["swaps"], 0)

    def test_init_index_is_the_subset_when_no_swaps_run(self):
        traj, _, _ = self._trajectory(proj_dim=0)
        index = torch.tensor([8, 1, 4, 3])
        ordered = torch.tensor([1, 3, 4, 8])
        traj.set_subset(ordered)
        expected = traj.D(ordered.numel())
        returned, history, diag = kl_full.gradient_ranked_single_swap(
            traj, 4, max_iters=0, init_index=index)
        self.assertEqual(returned.tolist(), ordered.tolist())
        self.assertAlmostEqual(history[0]["D"], expected, places=6)
        self.assertEqual(diag["swaps"], 0)

        fresh, _, _ = self._trajectory(proj_dim=0)
        with self.assertRaises(ValueError):
            kl_full.gradient_ranked_single_swap(
                fresh, 4, max_iters=0, init_index=torch.tensor([1, 2, 3]))

    def test_unsketched_aggregate_matches_full_residual(self):
        traj, grads, checkpoints = self._trajectory(proj_dim=0)
        index = torch.tensor([1, 3, 4, 8])
        traj.set_subset(index)
        self.assertIsNone(traj.sketch)
        self.assertEqual(tuple(traj.gbar[0].shape), (grads[0].shape[1],))
        self.assertEqual(tuple(traj.u[0].shape), (grads[0].shape[1],))
        self.assertAlmostEqual(
            traj.D(index.numel()),
            self._explicit_D(traj, grads, checkpoints, index),
            places=5)

    def test_preconditioned_sketch_matches_explicit_metric(self):
        def adam_metric_flat(checkpoint, named_params, eps=1e-8, power=2):
            pieces = []
            for name, param in named_params:
                base = 0.5 + (sum(ord(ch) for ch in name) % 5) / 10
                pieces.append(torch.linspace(base, base + 1.5, param.numel(),
                                             dtype=torch.float64))
            return torch.cat(pieces)

        trajectory.adam_metric_flat = adam_metric_flat
        traj, grads, checkpoints = self._trajectory(proj_dim=8, use_metric=True)
        self.assertTrue(traj.preconditioned)
        self.assertTrue(all(M is None for M in traj.metrics))
        index = torch.tensor([0, 4, 6, 9])
        traj.set_subset(index)
        metric = adam_metric_flat(checkpoints[0], traj.named)
        sqrt_m = (metric / traj.metric_scale).sqrt()
        sketched = traj.D(index.numel())
        expected = self._explicit_D(
            traj, grads, checkpoints, index, precondition=sqrt_m)
        self.assertAlmostEqual(sketched, expected, places=5)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA to check CPU offload")
    def test_sketched_weights_stay_on_cpu_when_pool_is_on_gpu(self):
        traj, _, _ = self._trajectory(proj_dim=4, device="cuda")
        self.assertEqual(next(iter(traj.states[0][0].values())).device.type, "cpu")
        self.assertEqual(traj.gbar[0].device.type, "cuda")
        self.assertEqual(traj.gbar[0].numel(), 4)


if __name__ == "__main__":
    unittest.main()
