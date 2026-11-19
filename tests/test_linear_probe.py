"""The linear-probe experiment: factored exact gradients and the convex bound."""

import unittest

import torch

from data_selection_swap import subsets
from data_selection_swap.selection.kl_full import CachedTrajectory
from data_selection_swap.selection.linear_probe import (
    ProbeGradients, fit_probe, residuals, two_point_residuals)
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
