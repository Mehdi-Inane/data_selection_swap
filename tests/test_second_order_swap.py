"""Verification checks 1-6 for the second-order k x k gradient swap.

Like ``test_swap_algorithms``, the selector modules are loaded under a
private package with their vision-dependent siblings stubbed, so only
torch is needed. Brute-force references are computed from the definition
G_T(I) = sum_t w_t || mean_I g^t - mean g^t ||^2_{M_t}.
"""

import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


SRC = Path(__file__).resolve().parents[1] / "src" / "data_selection_swap"
PACKAGE = "_second_order_swap_tests"


def _load(full_name, path):
    spec = importlib.util.spec_from_file_location(full_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _ranked_pairs(drop_idx, drop_h, add_idx, add_h):
    yield from sorted((float(add_h[b] - drop_h[a]), int(drop_idx[a]), int(add_idx[b]))
                      for a in range(len(drop_idx)) for b in range(len(add_idx)))


def _stub(name, **attributes):
    module = types.ModuleType(f"{PACKAGE}.{name}")
    module.__dict__.update(attributes)
    return module


package = types.ModuleType(PACKAGE)
package.__path__ = []
selection = _stub("selection")
selection.__path__ = []
trajectory = _stub("trajectory")  # load_weights / adam_metric_flat set per test
sys.modules.update({
    PACKAGE: package,
    f"{PACKAGE}.selection": selection,
    f"{PACKAGE}.common": _stub("common", ranked_pairs=_ranked_pairs),
    f"{PACKAGE}.objective": _stub("objective", weighted_sq_norm=lambda v, metric=None: float(
        (v * v if metric is None else v * v * metric).sum())),
    f"{PACKAGE}.trajectory": trajectory,
    f"{PACKAGE}.paths": _stub("paths", storage_variant=lambda *names: '',
                              tree=lambda scratch, root, variant='': scratch,
                              default_results_root=lambda: 'results'),
})
subsets = _load(f"{PACKAGE}.subsets", SRC / "subsets.py")
second_order = _load(f"{PACKAGE}.selection.second_order", SRC / "selection" / "second_order.py")
kl_full = _load(f"{PACKAGE}.selection.kl_full", SRC / "selection" / "kl_full.py")
CachedTrajectory = kl_full.CachedTrajectory


# ─────────────────────────────────────────────────────────────────────────────
# References and fixtures
# ─────────────────────────────────────────────────────────────────────────────

def brute_G(grads, weights, subset, metrics=None):
    subset = torch.as_tensor(sorted(int(i) for i in subset))
    total = 0.0
    for t, (G, w) in enumerate(zip(grads, weights)):
        residual = G.double()[subset].mean(dim=0) - G.double().mean(dim=0)
        sq = residual ** 2 if metrics is None else residual ** 2 * metrics[t]
        total += float(w) * float(sq.sum())
    return total


def brute_block(grads, weights, subset, outside, metrics=None):
    """Delta[a, b] = G(I - subset[a] + outside[b]) - G(I), by recomputation."""
    before = brute_G(grads, weights, subset, metrics)
    return torch.tensor([[brute_G(grads, weights, [j if r == i else r for r in subset],
                                  metrics) - before for j in outside] for i in subset],
                        dtype=torch.float64)


def raw_gradients(n, d, seed, offsets=(3.0, -2.0), scales=(1.0, 0.5)):
    """One per-example gradient matrix per checkpoint, with a non-zero mean."""
    generator = torch.Generator().manual_seed(seed)
    return [scale * torch.randn(n, d, generator=generator, dtype=torch.float64) + offset
            for offset, scale in zip(offsets, scales)]


def accepted(history):
    return [(entry["dropped"], entry["added"]) for entry in history if "dropped" in entry]


def complement(subset, n):
    return [r for r in range(n) if r not in set(subset)]


class _Ckpt:
    def __init__(self, path, weight):
        self.path, self.weight, self.epoch, self.step = path, weight, 1, 1


def _tiny_network(n=16, d=5, hidden=7, classes=3, seed=23):
    """A float64 MLP, two checkpoints of it and a pool, all exact in float64."""
    generator = torch.Generator().manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.Tanh(),
                                torch.nn.Linear(hidden, classes)).double()
    first = {k: v.detach().clone() for k, v in model.state_dict().items()}
    second = {k: v + 0.3 * torch.randn(v.shape, generator=generator, dtype=v.dtype)
              for k, v in first.items()}
    X = torch.randn(n, d, generator=generator, dtype=torch.float64)
    Y = torch.randint(0, classes, (n,), generator=generator)
    return model, {"a": first, "b": second}, X, Y, [_Ckpt("a", 0.4), _Ckpt("b", 0.6)]


def _per_example_grads(model, state, X, Y, order):
    """One backward per example: the reference for every vmap path."""
    model.load_state_dict(state)
    rows = []
    for i in range(X.shape[0]):
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(X[i:i + 1]), Y[i:i + 1]).backward()
        named = dict(model.named_parameters())
        rows.append(torch.cat([named[name].grad.reshape(-1) for name in order]).double())
    return torch.stack(rows)


class _NetworkCase(unittest.TestCase):
    """Builds Trajectory objects over ``_tiny_network`` with a stubbed loader."""

    def setUp(self):
        self._saved = dict(trajectory.__dict__)

    def tearDown(self):
        trajectory.__dict__.clear()
        trajectory.__dict__.update(self._saved)

    def network(self, seed=23, **kwargs):
        model, bank, X, Y, checkpoints = _tiny_network(seed=seed)
        trajectory.load_weights = lambda path: {k: v.clone() for k, v in bank[path].items()}

        def make(use_metric=False, **options):
            return kl_full.Trajectory(model, checkpoints, use_metric, X, Y, "cpu",
                                      batch_size=4, per_sample_batch_size=5,
                                      **{**kwargs, **options})

        return model, bank, X, Y, checkpoints, make


# ─────────────────────────────────────────────────────────────────────────────
# Check 1: the k x k block is G(I - i + j) - G(I), entry by entry
# ─────────────────────────────────────────────────────────────────────────────

class SwapMatrixIdentityTest(_NetworkCase):
    def test_cached_block_equals_brute_force(self):
        n, m, weights = 15, 5, (0.3, 0.7)
        grads = raw_gradients(n, 6, seed=1)
        engine = CachedTrajectory.from_gradients(grads, weights, dtype=torch.float64)
        subset = [1, 4, 6, 9, 13]
        outside = complement(subset, n)
        engine.set_subset(torch.tensor(subset))
        self.assertAlmostEqual(engine.D(m), brute_G(grads, weights, subset), delta=1e-12)
        torch.testing.assert_close(
            engine.swap_block(torch.tensor(subset), torch.tensor(outside), m),
            brute_block(grads, weights, subset, outside), rtol=0, atol=1e-12)

    def test_sketched_cache_is_exact_for_the_sketched_objective(self):
        n, m, d, weights = 15, 5, 12, (0.6, 0.4)
        grads = raw_gradients(n, d, seed=2)
        sketch = kl_full.CountSketch(d, 5, seed=4)
        engine = CachedTrajectory.from_gradients(grads, weights, sketch=sketch,
                                                 dtype=torch.float64)
        subset = [0, 3, 7, 8, 12]
        outside = complement(subset, n)
        engine.set_subset(torch.tensor(subset))
        torch.testing.assert_close(
            engine.swap_block(torch.tensor(subset), torch.tensor(outside), m),
            brute_block([sketch.apply_batch(G) for G in grads], weights, subset, outside),
            rtol=0, atol=1e-12)

    def test_exact_mode_block_and_curvature_match_brute_force(self):
        model, bank, X, Y, checkpoints, make = self.network()
        traj = make()
        grads = [_per_example_grads(model, bank[c.path], X, Y, traj.order)
                 for c in checkpoints]
        weights = [c.weight for c in checkpoints]
        m, subset = 5, [2, 3, 7, 11, 14]
        outside = complement(subset, X.shape[0])
        traj.set_subset(torch.tensor(subset))
        torch.testing.assert_close(
            traj.swap_block(torch.tensor(subset), torch.tensor(outside), m),
            brute_block(grads, weights, subset, outside), rtol=0, atol=1e-10)
        torch.testing.assert_close(
            traj.diagonal(),
            sum(w * ((G - G.mean(dim=0)) ** 2).sum(dim=1) for G, w in zip(grads, weights)),
            rtol=0, atol=1e-10)

    def test_adam_metric_is_applied_in_both_modes(self):
        model, bank, X, Y, checkpoints, make = self.network(seed=31)

        def adam_metric_flat(checkpoint, named_params, eps=1e-8, power=2):
            return torch.cat([torch.linspace(0.5 + len(name + checkpoint.path) % 7 / 10, 1.8,
                                             p.numel(), dtype=torch.float64)
                              for name, p in named_params])

        trajectory.adam_metric_flat = adam_metric_flat
        exact = make(use_metric=True)
        grads = [_per_example_grads(model, bank[c.path], X, Y, exact.order)
                 for c in checkpoints]
        m, subset = 4, [0, 5, 9, 12]
        outside = complement(subset, X.shape[0])
        expected = brute_block(grads, [c.weight for c in checkpoints], subset, outside,
                               metrics=exact.metrics)  # metrics are already rescaled
        exact.set_subset(torch.tensor(subset))
        cached = kl_full.build_cached_trajectory(exact, dtype=torch.float64)
        cached.set_subset(torch.tensor(subset))
        for engine in (exact, cached):
            torch.testing.assert_close(
                engine.swap_block(torch.tensor(subset), torch.tensor(outside), m),
                expected, rtol=0, atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# Check 2: a full shortlist ends at an exact one-swap local optimum
# ─────────────────────────────────────────────────────────────────────────────

class FullShortlistLocalOptimumTest(unittest.TestCase):
    def test_full_shortlist_returns_exact_one_swap_local_optimum(self):
        n, m, weights = 14, 5, (0.35, 0.65)
        grads = raw_gradients(n, 4, seed=8)
        engine = CachedTrajectory.from_gradients(grads, weights, dtype=torch.float64)
        index, history, diag = second_order.second_order_swap(
            engine, m, k=n, max_iters=1000, seed=3)

        selected = index.tolist()
        changes = brute_block(grads, weights, selected, complement(selected, n))
        self.assertEqual(changes.numel(), m * (n - m))
        self.assertGreaterEqual(float(changes.min()), -1e-12)
        self.assertGreater(diag["swaps"], 0)
        self.assertTrue(diag["one_swap_local_optimum"])
        values = [entry["D"] for entry in history[:-1]]
        self.assertTrue(all(b < a for a, b in zip(values, values[1:])))


# ─────────────────────────────────────────────────────────────────────────────
# Check 3: float32 storage, float64 decisions, no aggregate drift
# ─────────────────────────────────────────────────────────────────────────────

class Float32CacheRegressionTest(unittest.TestCase):
    def test_large_offset_float32_cache(self):
        n, m, d, weights, seed = 60, 12, 8, (0.45, 0.55), 11
        generator = torch.Generator().manual_seed(7)
        offset = 1e4 * torch.randn(1, d, generator=generator, dtype=torch.float64)
        grads = [torch.randn(n, d, generator=generator, dtype=torch.float64) + offset,
                 0.5 * torch.randn(n, d, generator=generator, dtype=torch.float64)
                 - 0.7 * offset]
        engine = CachedTrajectory.from_gradients(grads, weights, dtype=torch.float32)
        exact = CachedTrajectory.from_gradients(grads, weights, dtype=torch.float64).Z

        # The offset is removed before the cast. Centring after it would keep
        # only the offset's rounding error, which this would catch.
        scale = exact.abs().max()
        self.assertLess(float((engine.Z.double() - exact).abs().max() / scale), 1e-6)
        naive = torch.cat([math.sqrt(w) * (G.float() - G.float().mean(dim=0)).double()
                           for G, w in zip(grads, weights)], dim=1)
        self.assertGreater(float((naive - exact).abs().max() / scale), 1e-4)

        index, history, diag = second_order.second_order_swap(
            engine, m, k=8, max_iters=500, seed=seed)
        self.assertEqual(diag["reason"], "no improving swap in the shortlist")
        self.assertGreater(diag["swaps"], 3)

        # (a) G recomputed from the stored rows strictly decreases at every step.
        start = subsets.initial_subset(n, m, seed=seed)
        G_stored = [float(engine.Z[subsets.replay_swaps(start, history, t)].double()
                          .sum(dim=0).square().sum()) / m ** 2
                    for t in range(diag["swaps"] + 1)]
        self.assertTrue(all(b < a for a, b in zip(G_stored, G_stored[1:])), G_stored)
        self.assertEqual(subsets.replay_swaps(start, history).tolist(), index.tolist())

        # (b) the incrementally updated aggregate equals a fresh sum.
        self.assertLessEqual(diag["aggregate_drift_max_abs"], 1e-10)
        replay = CachedTrajectory.from_gradients(grads, weights, dtype=torch.float32)
        replay.set_subset(start)
        for dropped, added in accepted(history):
            replay.u[0] += (replay.aggregate(torch.tensor(added))[0]
                            - replay.aggregate(torch.tensor(dropped))[0])
        self.assertLessEqual(float((replay.u[0] - replay.aggregate(index)[0]).abs().max()),
                             1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# Check 4: high-norm outliers stop Algorithm 1 but not the second-order rule
# ─────────────────────────────────────────────────────────────────────────────

class _LinearOnlyShortlist(CachedTrajectory):
    """The bug check 4 targets: shortlists ranked by s alone (q left out)."""

    def diagonal(self):
        return torch.zeros_like(self.q)


class HeavyTailOutlierTrapTest(unittest.TestCase):
    """
    15 unselected 10x-norm outliers next to 185 regular examples, p = 32.

    The swap count alone does not tell a correct shortlist from one ranked by
    s alone, because the exact block still screens the outliers: both make
    about a dozen swaps (9 to 14 to the local optimum over six pool draws;
    this draw needs 14). What does tell them apart is that the
    diagonal-aware k = 16 shortlist always holds the exact best exchange, so
    it reproduces the full-shortlist search swap for swap.
    """

    def test_diagonal_aware_shortlist_escapes_the_outlier_trap(self):
        n, m, p, n_out = 200, 20, 32, 15
        generator = torch.Generator().manual_seed(3)
        G = torch.cat([torch.randn(n - n_out, p, generator=generator, dtype=torch.float64),
                       10.0 * torch.randn(n_out, p, generator=generator, dtype=torch.float64)])
        outliers = set(range(n - n_out, n))
        regular = torch.arange(n - n_out)
        start = regular[torch.randperm(regular.numel(),
                                       generator=torch.Generator().manual_seed(5))[:m]]

        def engine(cls=CachedTrajectory):
            return cls(CachedTrajectory.from_gradients([G], [1.0], dtype=torch.float64).Z)

        _, _, alg1 = kl_full.gradient_ranked_single_swap(
            engine(), m, max_iters=200, init_index=start)
        index, history, diag = second_order.second_order_swap(
            engine(), m, k=16, max_iters=1500, init_index=start)
        _, full_history, full = second_order.second_order_swap(
            engine(), m, k=n, max_iters=1500, init_index=start)
        _, buggy_history, _ = second_order.second_order_swap(
            engine(_LinearOnlyShortlist), m, k=16, max_iters=1500, init_index=start)

        self.assertLessEqual(alg1["swaps"], 4)
        self.assertGreaterEqual(diag["swaps"], 12)
        self.assertLess(diag["D"], 0.1 * alg1["D"])
        self.assertFalse(set(index.tolist()) & outliers)
        self.assertTrue(full["one_swap_local_optimum"])
        self.assertEqual(accepted(history), accepted(full_history))
        self.assertNotEqual(accepted(buggy_history), accepted(full_history))


# ─────────────────────────────────────────────────────────────────────────────
# Check 5: Mode A (identity sketch) and Mode B take identical swaps, and
#          Mode A makes no model call during the search
# ─────────────────────────────────────────────────────────────────────────────

def _search_counting_calls(model, run):
    calls = [0]

    def count(*_):
        calls[0] += 1

    handle = model.register_forward_hook(count)
    try:
        return run(), calls[0]
    finally:
        handle.remove()


class ModeEquivalenceTest(_NetworkCase):
    def test_cached_and_exact_modes_take_the_same_swaps(self):
        model, _, _, _, _, make = self.network()
        m, k, start = 5, 3, torch.tensor([0, 3, 6, 10, 13])
        cached = kl_full.build_cached_trajectory(make(), dtype=torch.float64)
        exact = make()
        exact.diagonal()

        cached.set_subset(start)
        exact.set_subset(start)
        drop, add = torch.tensor([0, 6, 13]), torch.tensor([1, 8, 9, 15])
        torch.testing.assert_close(cached.swap_block(drop, add, m),
                                   exact.swap_block(drop, add, m), rtol=0, atol=1e-12)

        def search(engine):
            return lambda: second_order.second_order_swap(engine, m, k=k, max_iters=200,
                                                          init_index=start)

        (index_a, history_a, diag_a), calls_a = _search_counting_calls(model, search(cached))
        (index_b, history_b, diag_b), calls_b = _search_counting_calls(model, search(exact))
        self.assertEqual(calls_a, 0)
        self.assertGreater(calls_b, 0)
        self.assertGreater(diag_a["swaps"], 0)
        self.assertEqual(accepted(history_a), accepted(history_b))
        self.assertEqual(index_a.tolist(), index_b.tolist())
        self.assertAlmostEqual(diag_a["D"], diag_b["D"], delta=1e-12)

    def test_algorithm1_on_the_sketch_cache_makes_no_model_call(self):
        model, _, _, _, _, make = self.network(seed=41, proj_dim=6, proj_seed=1)
        cached = kl_full.build_cached_trajectory(make())
        (_, history, diag), calls = _search_counting_calls(
            model, lambda: kl_full.gradient_ranked_single_swap(cached, 5, max_iters=20,
                                                                seed=2))
        self.assertEqual(calls, 0)
        self.assertLessEqual(diag["D"], history[0]["D"])

    def test_cache_from_the_network_matches_explicit_gradients(self):
        model, bank, X, Y, checkpoints, make = self.network(seed=37, proj_dim=9, proj_seed=2)
        traj = make()
        grads = [_per_example_grads(model, bank[c.path], X, Y, traj.order)
                 for c in checkpoints]
        explicit = CachedTrajectory.from_gradients(
            grads, [c.weight for c in checkpoints], sketch=traj.sketch, dtype=torch.float64)
        torch.testing.assert_close(kl_full.build_cached_trajectory(traj, dtype=torch.float64).Z,
                                   explicit.Z, rtol=0, atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# Check 6: class-balanced swaps keep the per_class_budget quotas exactly
# ─────────────────────────────────────────────────────────────────────────────

class ClassBalancedQuotaTest(unittest.TestCase):
    n, C, m = 97, 7, 19

    def setUp(self):
        self.labels = torch.randint(0, self.C, (self.n,),
                                    generator=torch.Generator().manual_seed(5))
        self.quota = subsets.per_class_budget(self.labels, self.C, self.m)
        self.engine = CachedTrajectory.from_gradients(
            raw_gradients(self.n, 6, seed=12), (0.5, 0.5), dtype=torch.float64)

    def test_quotas_are_met_and_every_swap_stays_in_its_class(self):
        index, history, diag = second_order.second_order_swap(
            self.engine, self.m, max_iters=500, labels=self.labels, num_classes=self.C,
            class_k=4, seed=2)
        self.assertGreater(int(self.quota.max() - self.quota.min()), 0)  # unequal quotas
        self.assertEqual(index.unique().numel(), self.m)
        self.assertTrue(torch.equal(torch.bincount(self.labels[index], minlength=self.C),
                                    self.quota))
        self.assertGreater(diag["swaps"], 0)
        for entry in history:
            if "dropped" in entry:
                classes = {int(self.labels[entry["dropped"][0]]),
                           int(self.labels[entry["added"][0]]), entry["class"]}
                self.assertEqual(len(classes), 1)

    def test_long_tail_pool_has_the_standard_class_counts(self):
        targets = torch.arange(10).repeat_interleave(5000)[torch.randperm(
            50000, generator=torch.Generator().manual_seed(0))]
        kept = subsets.long_tail_indices(targets, 10, ratio=50, seed=0)
        counts = torch.bincount(targets[kept], minlength=10).tolist()
        self.assertEqual(counts, [5000, 3237, 2096, 1357, 878, 568, 368, 238, 154, 100])
        self.assertEqual(len(kept), 13996)
        self.assertEqual(kept, sorted(set(kept)))
        self.assertEqual(kept, subsets.long_tail_indices(targets, 10, ratio=50, seed=0))
        hundred = torch.arange(100).repeat_interleave(500)
        self.assertEqual(len(subsets.long_tail_indices(hundred, 100, ratio=50)), 12608)

    def test_stratified_start_meets_quotas_and_a_bad_start_is_rejected(self):
        start = subsets.initial_subset(self.n, self.m, seed=4, labels=self.labels,
                                       num_classes=self.C)
        self.assertTrue(torch.equal(torch.bincount(self.labels[start], minlength=self.C),
                                    self.quota))
        with self.assertRaises(ValueError):
            second_order.second_order_swap(self.engine, self.m, labels=self.labels,
                                           num_classes=self.C,
                                           init_index=torch.arange(self.m))


if __name__ == "__main__":
    unittest.main()
