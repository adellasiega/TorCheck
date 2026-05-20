#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for torcheck.stl quantitative robustness computation.

Coverage:
  - Hand-computed expected values for Until, Globally, Eventually
  - Consistency between evaluate_at_all_times=True/False
  - Until variants: bounded, unbound, non-zero left bound
"""

import pytest
import torch
from torcheck import stl

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def signal(*rows):
    """Build a (1, C, T) signal from row vectors."""
    t = torch.tensor(rows, dtype=torch.float32)  # (C, T)
    return t.unsqueeze(0)                         # (1, C, T)


def quant(phi, x, all_times=True):
    return phi.quantitative(x, evaluate_at_all_times=all_times)


# ---------------------------------------------------------------------------
# Atom
# ---------------------------------------------------------------------------

class TestAtom:
    def test_gte(self):
        x = signal([0.0, 1.0, -1.0, 2.0])
        phi = stl.Atom(0, 0.5, lte=False)   # x >= 0.5
        z = quant(phi, x)
        expected = torch.tensor([[[-0.5, 0.5, -1.5, 1.5]]])
        assert torch.allclose(z, expected)

    def test_lte(self):
        x = signal([0.0, 1.0, -1.0, 2.0])
        phi = stl.Atom(0, 0.5, lte=True)    # x <= 0.5
        z = quant(phi, x)
        expected = torch.tensor([[[0.5, -0.5, 1.5, -1.5]]])
        assert torch.allclose(z, expected)


# ---------------------------------------------------------------------------
# Not / And / Or
# ---------------------------------------------------------------------------

class TestLogical:
    def test_not(self):
        x = signal([1.0, -2.0, 3.0])
        phi = stl.Not(stl.Atom(0, 0.0))
        z = quant(phi, x)
        expected = torch.tensor([[[-1.0, 2.0, -3.0]]])
        assert torch.allclose(z, expected)

    def test_and(self):
        x = signal([2.0, 1.0, -1.0], [0.5, -0.5, 1.0])
        a = stl.Atom(0, 0.0)   # z0 = x0
        b = stl.Atom(1, 0.0)   # z1 = x1
        phi = stl.And(a, b)
        z = quant(phi, x)
        expected = torch.tensor([[[0.5, -0.5, -1.0]]])
        assert torch.allclose(z, expected)

    def test_or(self):
        x = signal([2.0, 1.0, -1.0], [0.5, -0.5, 1.0])
        a = stl.Atom(0, 0.0)
        b = stl.Atom(1, 0.0)
        phi = stl.Or(a, b)
        z = quant(phi, x)
        expected = torch.tensor([[[2.0, 1.0, 1.0]]])
        assert torch.allclose(z, expected)


# ---------------------------------------------------------------------------
# Globally / Eventually
# ---------------------------------------------------------------------------

class TestGlobally:
    def test_bounded_hand(self):
        # G[0,1] x>=0: min over windows of size 2
        x = signal([3.0, -1.0, 2.0, 4.0])
        phi = stl.Globally(stl.Atom(0, 0.0), left_time_bound=0, right_time_bound=1)
        z = quant(phi, x)
        # window [0,1]: min(3,-1)=-1; [1,2]: min(-1,2)=-1; [2,3]: min(2,4)=2
        expected = torch.tensor([[[-1.0, -1.0, 2.0]]])
        assert torch.allclose(z, expected)

    def test_unbound_adapt(self):
        # G(unbound, adapt): cummin from right
        x = signal([3.0, 1.0, 4.0, 2.0])
        phi = stl.Globally(stl.Atom(0, 0.0), unbound=True, adapt_unbound=True)
        z = quant(phi, x)
        # cummin from the right: [1, 1, 2, 2]
        expected = torch.tensor([[[1.0, 1.0, 2.0, 2.0]]])
        assert torch.allclose(z, expected)


class TestEventually:
    def test_bounded_hand(self):
        # F[0,1] x>=0: max over windows of size 2
        x = signal([-2.0, 1.0, -1.0, 3.0])
        phi = stl.Eventually(stl.Atom(0, 0.0), left_time_bound=0, right_time_bound=1)
        z = quant(phi, x)
        # [0,1]: max(-2,1)=1; [1,2]: max(1,-1)=1; [2,3]: max(-1,3)=3
        expected = torch.tensor([[[1.0, 1.0, 3.0]]])
        assert torch.allclose(z, expected)

    def test_unbound_adapt(self):
        x = signal([3.0, 1.0, 4.0, 2.0])
        phi = stl.Eventually(stl.Atom(0, 0.0), unbound=True, adapt_unbound=True)
        z = quant(phi, x)
        # cummax from the right: [4, 4, 4, 2]
        expected = torch.tensor([[[4.0, 4.0, 4.0, 2.0]]])
        assert torch.allclose(z, expected)


# ---------------------------------------------------------------------------
# Until — hand-computed
# ---------------------------------------------------------------------------

class TestUntilBounded:
    """
    Signal: z1 = [1, 1, 0, 1], z2 = [-1, 2, -1, -1]
    U[0,1]: result[t] = max_{d=0,1} min(min_{k=t}^{t+d} z1[k], z2[t+d])

    t=0: max(min(1,-1), min(min(1,1),2)) = max(-1, 1) = 1
    t=1: max(min(1,2),  min(min(1,0),-1)) = max(1, -1) = 1
    t=2: max(min(0,-1), min(min(0,1),-1)) = max(-1, -1) = -1
    t=3: max(min(1,-1), [d=1 out of range→-inf]) = -1
    """

    @pytest.fixture
    def signals(self):
        return signal([1.0, 1.0, 0.0, 1.0], [-1.0, 2.0, -1.0, -1.0])

    @pytest.fixture
    def phi(self):
        return stl.Until(stl.Atom(0, 0.0), stl.Atom(1, 0.0),
                         left_time_bound=0, right_time_bound=1)

    def test_values(self, phi, signals):
        z = quant(phi, signals)
        expected = torch.tensor([[[1.0, 1.0, -1.0, -1.0]]])
        assert torch.allclose(z, expected, atol=1e-6)

    def test_shape(self, phi, signals):
        z = quant(phi, signals)
        assert z.shape == (1, 1, 4)  # valid_len = T - left_time_bound = 4 - 0

    def test_nonzero_left_bound(self):
        # U[1,2]: witnesses must be at offset 1 or 2 from t
        # z1=[1,1,0,1], z2=[-1,2,-1,-1]
        # t=0: max(min(min(1,1),2), min(min(1,1,0),-1)) = max(1,-1)=1 → wait
        # d=1: min(min_{k=0}^{1} z1, z2[1]) = min(min(1,1), 2) = min(1,2) = 1
        # d=2: min(min_{k=0}^{2} z1, z2[2]) = min(min(1,1,0), -1) = min(0,-1) = -1
        # t=0: max(1, -1) = 1
        # t=1: d=1: min(min(1,0),z2[2])=min(0,-1)=-1; d=2: min(min(1,0,1),z2[3])=min(0,-1)=-1 → -1
        # t=2: d=1: min(min(0,1),z2[3])=min(0,-1)=-1; d=2: out of range → -1
        # valid_len = T - left_time_bound = 4 - 1 = 3
        x = signal([1.0, 1.0, 0.0, 1.0], [-1.0, 2.0, -1.0, -1.0])
        phi = stl.Until(stl.Atom(0, 0.0), stl.Atom(1, 0.0),
                        left_time_bound=1, right_time_bound=2)
        z = quant(phi, x)
        expected = torch.tensor([[[1.0, -1.0, -1.0]]])
        assert torch.allclose(z, expected, atol=1e-6)

    def test_window_one(self):
        # U[2,2]: only d=2 contributes
        # z1=[1,1,0,1], z2=[-1,2,-1,-1]
        # t=0: min(min(1,1,0), z2[2]) = min(0,-1) = -1
        # t=1: min(min(1,0,1), z2[3]) = min(0,-1) = -1
        # valid_len = 4 - 2 = 2
        x = signal([1.0, 1.0, 0.0, 1.0], [-1.0, 2.0, -1.0, -1.0])
        phi = stl.Until(stl.Atom(0, 0.0), stl.Atom(1, 0.0),
                        left_time_bound=2, right_time_bound=2)
        z = quant(phi, x)
        expected = torch.tensor([[[-1.0, -1.0]]])
        assert torch.allclose(z, expected, atol=1e-6)


class TestUntilUnbound:
    """
    z1 = [1, 0, 1], z2 = [-1, 2, -1]
    M[2] = min(1, -1) = -1
    M[1] = min(0, max(2, -1)) = min(0, 2) = 0
    M[0] = min(1, max(-1, 0)) = min(1, 0) = 0
    """

    @pytest.fixture
    def signals(self):
        return signal([1.0, 0.0, 1.0], [-1.0, 2.0, -1.0])

    @pytest.fixture
    def phi(self):
        return stl.Until(stl.Atom(0, 0.0), stl.Atom(1, 0.0), unbound=True)

    def test_values(self, phi, signals):
        z = quant(phi, signals)
        expected = torch.tensor([[[0.0, 0.0, -1.0]]])
        assert torch.allclose(z, expected, atol=1e-6)

    def test_shape(self, phi, signals):
        assert quant(phi, signals).shape == (1, 1, 3)

    def test_adapt_unbound_false(self, signals):
        # Should collapse to global max across time → scalar per sample
        phi = stl.Until(stl.Atom(0, 0.0), stl.Atom(1, 0.0),
                        unbound=True, adapt_unbound=False)
        z = quant(phi, signals)
        assert z.shape == (1, 1, 1)
        assert torch.allclose(z, torch.tensor([[[0.0]]]), atol=1e-6)


# ---------------------------------------------------------------------------
# Consistency: all_times=False vs all_times=True at t=0
# ---------------------------------------------------------------------------

class TestTimeZeroConsistency:
    """quantitative(x, evaluate_at_all_times=False) == quantitative(x)[..., 0]."""

    @pytest.fixture(params=[
        stl.Atom(0, 0.5),
        stl.Not(stl.Atom(0, 0.0)),
        stl.And(stl.Atom(0, 0.0), stl.Atom(1, 0.0)),
        stl.Or(stl.Atom(0, 0.0), stl.Atom(1, 0.0)),
        stl.Globally(stl.Atom(0, 0.0), left_time_bound=0, right_time_bound=4),
        stl.Eventually(stl.Atom(0, 0.0), left_time_bound=0, right_time_bound=4),
        stl.Until(stl.Atom(0, 0.0), stl.Atom(1, 0.0),
                  left_time_bound=0, right_time_bound=4),
        stl.Until(stl.Atom(0, 0.0), stl.Atom(1, 0.0), unbound=True),
    ], ids=[
        "Atom", "Not", "And", "Or", "Globally", "Eventually",
        "Until_bounded", "Until_unbound",
    ])
    def phi(self, request):
        return request.param

    @pytest.fixture
    def x(self):
        torch.manual_seed(1)
        return torch.randn(8, 2, 30)

    def test_t0_equals_all_times_index0(self, phi, x):
        z_all = phi.quantitative(x, evaluate_at_all_times=True)
        z_t0 = phi.quantitative(x, evaluate_at_all_times=False)
        # _extract_semantics_at_time_zero returns z[:, 0, 0] reshaped to (N,)
        assert torch.allclose(z_t0, z_all[:, 0, 0], atol=1e-6)


# ---------------------------------------------------------------------------
# Until correctness vs reference (random signals, multiple shapes)
# ---------------------------------------------------------------------------

class TestUntilVsReference:
    """Cross-validate optimized Until against a simple reference loop."""

    @staticmethod
    def _reference_bounded(z1, z2, a, b):
        """Brute-force O(N*T^2) reference — correct but slow, fine for small T."""
        N, C, T = z1.shape
        valid_len = max(0, T - a)
        result = torch.full((N, C, valid_len), float('-inf'))
        for t in range(valid_len):
            for d in range(a, b + 1):
                s = t + d
                if s >= T:
                    break
                cmin = z1[:, :, t:s + 1].min(dim=2)[0]
                contrib = torch.minimum(cmin, z2[:, :, s])
                result[:, :, t] = torch.maximum(result[:, :, t], contrib)
        return result

    @pytest.mark.parametrize("a,b", [(0, 3), (0, 0), (1, 4), (2, 2), (0, 9)])
    def test_bounded_matches_reference(self, a, b):
        torch.manual_seed(42)
        x = torch.randn(4, 2, 20)
        phi = stl.Until(stl.Atom(0, 0.0), stl.Atom(1, 0.0),
                        left_time_bound=a, right_time_bound=b)
        z_fast = phi.quantitative(x, evaluate_at_all_times=True)

        z1 = stl.Atom(0, 0.0)._quantitative(x)
        z2 = stl.Atom(1, 0.0)._quantitative(x)
        z_ref = self._reference_bounded(z1, z2, a, b)

        assert torch.allclose(z_fast, z_ref, atol=1e-5), \
            f"U[{a},{b}] mismatch: max diff {(z_fast - z_ref).abs().max():.2e}"

    @staticmethod
    def _reference_unbound(z1, z2):
        N, C, T = z1.shape
        result = torch.empty(N, C, T)
        result[:, :, T - 1] = torch.minimum(z1[:, :, T - 1], z2[:, :, T - 1])
        for t in range(T - 2, -1, -1):
            result[:, :, t] = torch.minimum(
                z1[:, :, t], torch.maximum(z2[:, :, t], result[:, :, t + 1])
            )
        return result

    def test_unbound_matches_reference(self):
        torch.manual_seed(7)
        x = torch.randn(4, 2, 20)
        phi = stl.Until(stl.Atom(0, 0.0), stl.Atom(1, 0.0), unbound=True)
        z_fast = phi.quantitative(x, evaluate_at_all_times=True)

        z1 = stl.Atom(0, 0.0)._quantitative(x)
        z2 = stl.Atom(1, 0.0)._quantitative(x)
        z_ref = self._reference_unbound(z1, z2)
        assert torch.allclose(z_fast, z_ref, atol=1e-5)
