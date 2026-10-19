"""Numerical tests for the LLM selector's objective and swap math.

These run on the real ``data_selection_swap.llm`` package in pure CPU/torch, so
they gate the ported algorithm *before* any GPU or LESS-data work. They check
three things the science depends on:

1. the targeted/faithful objective ``f(I)`` and the manuscript discrepancy
   ``D(I)`` differ by a constant (identical argmin),
2. the closed-form single-swap change is exact against re-evaluating ``f``,
3. ``best_swap`` reaches the true optimum on tiny pools and is a one-swap local
   optimum on larger ones; ``top-k`` on the targeted score reproduces LESS and is
   dominated by the swap on the full objective.
"""

import itertools
import unittest

import torch

from data_selection_swap.llm.config import SelectionConfig
from data_selection_swap.llm.features import TrajectoryFeatures
from data_selection_swap.llm.objective import SwapObjective, select_topk
from data_selection_swap.llm.swap import best_swap, ranked_swap, select
from data_selection_swap.llm.targets import build_score, targeted_score


def _normalize_rows(x, eps=1e-12):
    return x / x.norm(dim=1, keepdim=True).clamp_min(eps)


def _normalize_vec(x, eps=1e-12):
    return x / x.norm().clamp_min(eps)


def _toy(n=12, d=6, K=2, seed=7, normalize="none", metrics=True):
    g = torch.Generator().manual_seed(seed)
    feats = [torch.randn(n, d, generator=g) for _ in range(K)]
    weights = [0.6, 0.4][:K]
    mets = [torch.rand(d, generator=g) + 0.1 for _ in range(K)] if metrics else [None] * K
    return TrajectoryFeatures(feats, weights=weights, metrics=mets, normalize=normalize)


def _toy_targets(n=12, d=6, K=2, seed=99, normalize="none", metrics=True):
    g = torch.Generator().manual_seed(seed)
    base = _toy(n=n, d=d, K=K, seed=seed, normalize=normalize, metrics=metrics)
    # two subtasks, each with one target per checkpoint
    subtasks = [[torch.rand(d, generator=g) + 0.2 for _ in range(K)] for _ in range(2)]
    return base, subtasks


def _weighted_sq_norm(x, metric):
    return float((x * x).sum() if metric is None else (metric * x * x).sum())


class FaithfulnessIdentityTest(unittest.TestCase):
    def test_objective_is_discrepancy_minus_constant(self):
        feats = _toy(normalize="none")
        gbar = feats.pool_mean_targets()
        c = build_score(feats, "faithful")
        obj = SwapObjective(feats, c)
        # const = sum_k w_k ||gbar_k||^2_{M_k}
        const = sum(feats.weights[k] * _weighted_sq_norm(gbar[k], feats.metrics[k].values)
                    for k in range(len(gbar)))
        sel = torch.tensor([0, 2, 5, 7, 9])
        self.assertAlmostEqual(obj.value(sel), feats.D(sel, gbar) - const, places=9)

    def test_faithful_swap_reduces_discrepancy(self):
        feats = _toy(normalize="none", seed=3)
        gbar = feats.pool_mean_targets()
        m = 4
        sel, _, diag = select(feats, "faithful",
                              SelectionConfig(setting="faithful", fraction=m / feats.n,
                                              rule="best", max_iters=100, seed=1))
        init = torch.tensor([0, 1, 2, 3])
        self.assertLessEqual(feats.D(sel, gbar), feats.D(init, gbar) + 1e-12)
        self.assertTrue(diag["one_swap_local_optimum"])


class SwapDeltaExactnessTest(unittest.TestCase):
    def test_closed_form_matches_reevaluation(self):
        feats = _toy(seed=5, metrics=True)
        c = build_score(feats, "targeted",
                        subtask_targets=[[torch.ones(feats.d), torch.ones(feats.d) * 2]],
                        aggregation="max")
        obj = SwapObjective(feats, c)
        sel = torch.tensor([1, 3, 4, 8])
        A = feats.aggregate(sel)
        drop = sel.clone()
        add = torch.tensor([0, 6, 11])
        deltas = obj.swap_deltas(sel, A, drop, add)
        before = obj.value(sel)
        for r in range(drop.numel()):
            for cc in range(add.numel()):
                i = int(drop[r])
                j = int(add[cc])
                after = torch.cat([sel[sel != i], torch.tensor([j])])
                self.assertAlmostEqual(before + float(deltas[r, cc]), obj.value(after),
                                       places=9, msg=f"swap {i}->{j}")


class BestSwapOptimalityTest(unittest.TestCase):
    def test_matches_brute_force_optimum(self):
        feats, subtasks = _toy_targets(n=8, d=4, seed=11)
        c = targeted_score(feats, subtasks, aggregation="mean")
        obj = SwapObjective(feats, c)
        m = 3
        sel, _, diag = best_swap(obj, m, max_iters=100, seed=2)
        brute = min(obj.value(torch.tensor(list(sub)))
                    for sub in itertools.combinations(range(8), m))
        self.assertAlmostEqual(obj.value(sel), brute, places=8)
        self.assertTrue(diag["one_swap_local_optimum"])

    def test_is_one_swap_local_optimum(self):
        feats = _toy(n=14, d=6, seed=13)
        c = build_score(feats, "faithful")
        obj = SwapObjective(feats, c)
        m = 5
        sel, _, diag = best_swap(obj, m, max_iters=200, seed=4)
        A = feats.aggregate(sel)
        outside = torch.tensor([i for i in range(14) if i not in sel.tolist()])
        deltas = obj.swap_deltas(sel, A, sel, outside)
        self.assertGreaterEqual(float(deltas.min()), -1e-9)


class LessParityTest(unittest.TestCase):
    def test_topk_reproduces_less_ordering(self):
        feats, subtasks = _toy_targets(normalize="l2")
        c = targeted_score(feats, subtasks, aggregation="max")
        m = 4
        top = select_topk(c, m)
        self.assertEqual(top.tolist(),
                         torch.sort(torch.argsort(c, descending=True)[:m]).values.tolist())

    def test_swap_dominates_topk_on_full_objective(self):
        feats, subtasks = _toy_targets(normalize="l2")
        c = targeted_score(feats, subtasks, aggregation="max")
        obj = SwapObjective(feats, c)
        m = 4
        top = select_topk(c, m)
        sel, _, _ = ranked_swap(obj, m, max_iters=200, seed=0)
        self.assertLessEqual(obj.value(sel), obj.value(top) + 1e-12)

    def test_max_vs_mean_aggregation_differ(self):
        feats, subtasks = _toy_targets()
        c_max = targeted_score(feats, subtasks, aggregation="max")
        c_mean = targeted_score(feats, subtasks, aggregation="mean")
        self.assertFalse(torch.allclose(c_max, c_mean))


class ConfigTest(unittest.TestCase):
    def test_subset_size(self):
        self.assertEqual(SelectionConfig(fraction=0.05).subset_size(270000), 13500)
        self.assertEqual(SelectionConfig(fraction=1.0).subset_size(10), 10)

    def test_rejects_bad_values(self):
        for bad in ({"setting": "nope"}, {"fraction": 0.0}, {"normalize": "x"},
                    {"aggregation": "x"}, {"rule": "x"}):
            with self.assertRaises(ValueError):
                SelectionConfig(**bad)

    def test_select_end_to_end_faithful(self):
        feats = _toy(normalize="none", seed=17)
        sel, hist, diag = select(feats, "faithful",
                                 SelectionConfig(setting="faithful", fraction=0.3,
                                                 rule="ranked", max_iters=50, seed=0))
        self.assertEqual(sel.numel(), int(round(0.3 * feats.n)))
        self.assertLessEqual(diag["value"], hist[0]["value"] + 1e-12)


if __name__ == "__main__":
    unittest.main()
