"""The linear-probe experiment: factored exact gradients and the convex bound."""

import unittest

import torch

from data_selection_swap import subsets
from data_selection_swap.selection.kl_full import CachedTrajectory
from data_selection_swap.selection.linear_probe import (
    ProbeGradients, class_kernels, fit_probe, graphcut, herding, residuals, sgd_epoch,
    two_point_residuals)
from data_selection_swap.selection.second_order import second_order_swap


def problem(n=40, features=5, classes=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    Y = torch.randint(0, classes, (n,), generator=g)
    centres = 2.0 * torch.randn(classes, features, generator=g, dtype=torch.float64)
    Phi = centres[Y] + torch.randn(n, features, generator=g, dtype=torch.float64)
    return torch.cat([Phi, torch.ones(n, 1, dtype=torch.float64)], 1), Y


class ProbeGradientsTest(unittest.TestCase):
    """Against CachedTrajectory on the explicit gradients g_i = r_i phi_i^T."""

    def setUp(self):
        self.Phi, self.Y = problem()
        W = torch.randn(4, self.Phi.shape[1], generator=torch.Generator().manual_seed(1),
                        dtype=torch.float64)
        R = residuals(self.Phi, self.Y, W)
        self.probe = ProbeGradients(R, self.Phi)
        explicit = (R[:, :, None] * self.Phi[:, None, :]).reshape(len(R), -1)
        self.cached = CachedTrajectory.from_gradients([explicit], [1.0], dtype=torch.float64)

    def test_every_quantity_matches(self):
        index, m = torch.arange(0, 40, 3), 14
        for traj in (self.probe, self.cached):
            traj.set_subset(index)
        torch.testing.assert_close(self.probe.diagonal(), self.cached.diagonal())
        self.assertAlmostEqual(self.probe.D(m), self.cached.D(m), places=12)
        torch.testing.assert_close(self.probe.gradient(m), self.cached.gradient(m),
                                   rtol=1e-5, atol=1e-7)   # float32 pool-wide scores
        drop, add = index[:5], torch.tensor([1, 2, 4, 5])
        torch.testing.assert_close(self.probe.swap_block(drop, add, m),
                                   self.cached.swap_block(drop, add, m))

    def test_the_swap_takes_the_same_path(self):
        # Shortlists that hold every example: no ranking enters the choice.
        runs = [second_order_swap(traj, 14, k=40, seed=3) for traj in (self.probe, self.cached)]
        torch.testing.assert_close(runs[0][0], runs[1][0])
        self.assertEqual([h.get('added') for h in runs[0][1]],
                         [h.get('added') for h in runs[1][1]])
        self.assertLess(runs[0][2]['D'], runs[0][1][0]['D'])


class TwoPointTest(unittest.TestCase):
    def test_stacked_residuals_give_the_sum_of_both_points(self):
        Phi, Y = problem()
        W_star, _ = fit_probe(Phi, Y, 4, 0.05)
        index = torch.arange(0, 40, 4)
        parts = []
        for W in (torch.zeros_like(W_star), W_star):
            R = residuals(Phi, Y, W)
            traj = ProbeGradients(R, Phi)
            scale = float(traj.q.mean())
            traj.set_subset(index)
            parts.append(traj.D(10) / scale)
        both = ProbeGradients(two_point_residuals(Phi, Y, W_star), Phi)
        both.set_subset(index)
        self.assertAlmostEqual(both.D(10), sum(parts), places=12)


class SelectionPointTest(unittest.TestCase):
    """Herding and GraphCut on the gradient kernel at W = 0 are the feature versions."""

    def setUp(self):
        self.Phi, self.Y = problem(n=80)
        self.class_idx = [torch.where(self.Y == c)[0] for c in range(4)]
        self.quota = torch.tensor([3, 4, 2, 5])
        R0 = residuals(self.Phi, self.Y, torch.zeros(4, self.Phi.shape[1], dtype=torch.float64))
        self.at_zero = class_kernels(R0, self.Phi, self.class_idx)
        self.features = [self.Phi[i] @ self.Phi[i].T for i in self.class_idx]

    def test_kernel_herding_is_feature_herding(self):
        explicit = []
        for idx, take in zip(self.class_idx, self.quota.tolist()):
            X, total, free = self.Phi[idx], torch.zeros(self.Phi.shape[1], dtype=torch.float64), \
                torch.ones(len(idx), dtype=torch.bool)
            for t in range(1, take + 1):
                gap = ((X.mean(0) - (total + X) / t) ** 2).sum(1).masked_fill(~free, float('inf'))
                j = int(gap.argmin())
                free[j] = False
                total += X[j]
                explicit.append(int(idx[j]))
        torch.testing.assert_close(herding(self.features, self.class_idx, self.quota),
                                   torch.tensor(sorted(explicit)))
        torch.testing.assert_close(herding(self.at_zero, self.class_idx, self.quota),
                                   torch.tensor(sorted(explicit)))

    def test_graphcut_at_zero_is_feature_graphcut(self):
        torch.testing.assert_close(graphcut(self.at_zero, self.class_idx, self.quota),
                                   graphcut(self.features, self.class_idx, self.quota))

    def test_one_sgd_epoch_moves_between_zero_and_the_solution(self):
        W_star, _ = fit_probe(self.Phi, self.Y, 4, 0.05)
        W_1 = sgd_epoch(self.Phi, self.Y, 4, 0.05, batch_size=8)
        self.assertGreater(float(W_1.norm()), 0.0)
        self.assertLess(float((W_1 - W_star).norm()), float(W_star.norm()))


class ConvexBoundTest(unittest.TestCase):
    def test_subset_solution_lies_within_sqrt_G_over_lambda(self):
        Phi, Y = problem(n=120)
        lam = 0.05
        W_star, grad_norm = fit_probe(Phi, Y, 4, lam)
        self.assertLess(grad_norm, 1e-6)
        probe = ProbeGradients(residuals(Phi, Y, W_star), Phi)
        for seed in range(5):
            index = subsets.initial_subset(120, 30, seed=seed)
            W, _ = fit_probe(Phi[index], Y[index], 4, lam)
            probe.set_subset(index)
            self.assertLessEqual(float((W - W_star).norm()),
                                 probe.D(30) ** 0.5 / lam + 1e-6)


if __name__ == '__main__':
    unittest.main()
