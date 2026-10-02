"""D from the stored kernel matches the sketched residual."""

import unittest

import torch

from data_selection_swap.analysis.kernel import accumulate_gram, budget_q, subset_d


class KernelQuadraticTest(unittest.TestCase):
    def test_subset_d_matches_sketched_residual(self):
        torch.manual_seed(0)
        n, p = 12, 4
        sketches = [torch.randn(n, p), torch.randn(n, p)]
        weights = [0.25, 0.75]
        P = torch.zeros(n, n, dtype=torch.float32)
        r = torch.zeros(n, dtype=torch.float64)
        c = 0.0
        for S, w in zip(sketches, weights):
            c = accumulate_gram(P, r, c, S, w, row_block=5)

        idx = torch.tensor([0, 3, 5, 7])
        m = int(idx.numel())
        expected = 0.0
        for S, w in zip(sketches, weights):
            sbar = S.mean(0)
            residual = S.index_select(0, idx).sum(0) / m - sbar
            expected += w * float(residual.double().pow(2).sum())
        got = subset_d(P, r, c, idx)
        scale = max(abs(expected), 1e-6)
        self.assertLess(abs(got - expected) / scale, 1e-4)

        q = budget_q(r, m)
        quad = P.index_select(0, idx).index_select(1, idx).double().sum()
        alt = float(quad / (m * m) + q.index_select(0, idx).double().sum() + c)
        self.assertAlmostEqual(got, alt, places=6)

    def test_example_sketches_match_autograd(self):
        try:
            import torchvision  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest('torchvision is not installed')
        import torch.nn.functional as F

        from data_selection_swap.analysis.kernel import example_sketches
        from data_selection_swap.selection.kl_full import CountSketch

        torch.manual_seed(1)
        model = torch.nn.Linear(4, 3).eval()
        X = torch.randn(5, 4)
        Y = torch.randint(0, 3, (5,))
        dim = sum(param.numel() for param in model.parameters())
        sketch = CountSketch(dim, 7, seed=1)
        sketched = example_sketches(
            model, model.state_dict(), X, Y, sketch, batch_size=2)

        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(X[0:1]), Y[0:1]).backward()
        flat = torch.cat([param.grad.reshape(-1) for param in model.parameters()])
        self.assertTrue(torch.allclose(
            sketched[0].cpu(), sketch.apply(flat.cpu()), atol=1e-5))
