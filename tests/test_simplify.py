#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for torcheck.simplify.

Each rule is covered by:
  - a structural test  : verifies the output formula tree has the expected shape
  - a semantic test    : verifies Boolean and quantitative outputs are identical
                         on random signals (both semantics must be preserved exactly).
"""

import pytest
import torch
from torcheck import stl
from torcheck.simplify import simplify, _structurally_equal

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_SAMPLES, N_VARS, N_STEPS = 50, 4, 60

@pytest.fixture
def x():
    torch.manual_seed(0)
    return torch.randn(N_SAMPLES, N_VARS, N_STEPS)


def _check_semantics(original, simplified, x):
    """Assert Boolean and quantitative outputs match on all time steps."""
    assert torch.equal(
        original.boolean(x, evaluate_at_all_times=True),
        simplified.boolean(x, evaluate_at_all_times=True),
    ), "Boolean semantics differ"
    assert torch.allclose(
        original.quantitative(x, evaluate_at_all_times=True),
        simplified.quantitative(x, evaluate_at_all_times=True),
        atol=1e-5,
    ), "Quantitative semantics differ"


# ---------------------------------------------------------------------------
# Helper atoms
# ---------------------------------------------------------------------------

def phi():
    return stl.Atom(0, 0.5, lte=False)  # x_0 >= 0.5

def psi():
    return stl.Atom(1, 1.0, lte=True)   # x_1 <= 1.0


# ---------------------------------------------------------------------------
# Structural equality helper tests
# ---------------------------------------------------------------------------

class TestStructuralEquality:
    def test_same_atom(self):
        assert _structurally_equal(phi(), phi())

    def test_different_atom_threshold(self):
        assert not _structurally_equal(stl.Atom(0, 0.5, lte=True), stl.Atom(0, 0.6, lte=True))

    def test_different_types(self):
        assert not _structurally_equal(phi(), stl.Not(phi()))

    def test_nested_equal(self):
        a = stl.And(phi(), stl.Not(psi()))
        b = stl.And(phi(), stl.Not(psi()))
        assert _structurally_equal(a, b)


# ---------------------------------------------------------------------------
# Rule 1: double negation  ¬¬φ → φ
# ---------------------------------------------------------------------------

class TestDoubleNegation:
    def test_structure(self):
        f = stl.Not(stl.Not(phi()))
        s = simplify(f)
        assert _structurally_equal(s, phi())

    def test_semantics(self, x):
        f = stl.Not(stl.Not(phi()))
        _check_semantics(f, simplify(f), x)

    def test_triple_negation(self):
        # ¬¬¬(x_0 >= 0.5) simplifies fully to x_0 <= 0.5 (atom negation applied after double-neg)
        f = stl.Not(stl.Not(stl.Not(phi())))
        s = simplify(f)
        assert _structurally_equal(s, stl.Atom(0, 0.5, lte=True))

    def test_triple_negation_semantics(self, x):
        f = stl.Not(stl.Not(stl.Not(phi())))
        _check_semantics(f, simplify(f), x)


# ---------------------------------------------------------------------------
# Rule 2: atom negation  ¬(x_i <= c) → x_i >= c
# ---------------------------------------------------------------------------

class TestAtomNegation:
    def test_lte_to_gte(self):
        atom = stl.Atom(0, 1.0, lte=True)
        s = simplify(stl.Not(atom))
        assert isinstance(s, stl.Atom)
        assert s.var_index == 0 and s.threshold == 1.0 and s.lte is False

    def test_gte_to_lte(self):
        atom = stl.Atom(2, 0.3, lte=False)
        s = simplify(stl.Not(atom))
        assert isinstance(s, stl.Atom)
        assert s.var_index == 2 and s.threshold == 0.3 and s.lte is True

    def test_semantics_lte(self, x):
        f = stl.Not(stl.Atom(0, 0.5, lte=True))
        _check_semantics(f, simplify(f), x)

    def test_semantics_gte(self, x):
        f = stl.Not(stl.Atom(1, 1.0, lte=False))
        _check_semantics(f, simplify(f), x)


# ---------------------------------------------------------------------------
# Rules 3–4: De Morgan
# ---------------------------------------------------------------------------

class TestDeMorgan:
    def test_not_and_structure(self):
        # ¬(φ ∧ ψ) → ¬φ ∨ ¬ψ; since φ,ψ are Atoms, ¬Atom is further simplified to a flipped Atom
        f = stl.Not(stl.And(phi(), psi()))
        s = simplify(f)
        assert isinstance(s, stl.Or)
        # ¬(x_0 >= 0.5) → x_0 <= 0.5;  ¬(x_1 <= 1.0) → x_1 >= 1.0
        assert _structurally_equal(s.left_child, stl.Atom(0, 0.5, lte=True))
        assert _structurally_equal(s.right_child, stl.Atom(1, 1.0, lte=False))

    def test_not_or_structure(self):
        f = stl.Not(stl.Or(phi(), psi()))
        s = simplify(f)
        assert isinstance(s, stl.And)

    def test_not_and_semantics(self, x):
        f = stl.Not(stl.And(phi(), psi()))
        _check_semantics(f, simplify(f), x)

    def test_not_or_semantics(self, x):
        f = stl.Not(stl.Or(phi(), psi()))
        _check_semantics(f, simplify(f), x)


# ---------------------------------------------------------------------------
# Rules 5–6: temporal negation  ¬G → F¬  and  ¬F → G¬
# ---------------------------------------------------------------------------

class TestTemporalNegation:
    def test_not_globally_structure(self):
        # ¬G[1,5](x_0>=0.5) → F[1,5](x_0<=0.5)  (¬Atom further simplified to flipped Atom)
        g = stl.Globally(phi(), left_time_bound=1, right_time_bound=5)
        f = stl.Not(g)
        s = simplify(f)
        assert isinstance(s, stl.Eventually)
        assert _structurally_equal(s.child, stl.Atom(0, 0.5, lte=True))
        assert s.left_time_bound == 1
        assert s.right_time_bound == 6  # stored as right+1

    def test_not_eventually_structure(self):
        # ¬F[2,4](x_1<=1.0) → G[2,4](x_1>=1.0)
        e = stl.Eventually(phi(), left_time_bound=2, right_time_bound=4)
        f = stl.Not(e)
        s = simplify(f)
        assert isinstance(s, stl.Globally)
        assert _structurally_equal(s.child, stl.Atom(0, 0.5, lte=True))

    def test_not_globally_semantics(self, x):
        g = stl.Globally(phi(), left_time_bound=0, right_time_bound=3)
        _check_semantics(stl.Not(g), simplify(stl.Not(g)), x)

    def test_not_eventually_semantics(self, x):
        e = stl.Eventually(phi(), left_time_bound=1, right_time_bound=4)
        _check_semantics(stl.Not(e), simplify(stl.Not(e)), x)

    def test_not_globally_unbound(self, x):
        g = stl.Globally(phi(), unbound=True)
        f = stl.Not(g)
        s = simplify(f)
        assert isinstance(s, stl.Eventually)
        assert s.unbound is True
        _check_semantics(f, s, x)

    def test_not_eventually_unbound(self, x):
        e = stl.Eventually(phi(), unbound=True)
        f = stl.Not(e)
        s = simplify(f)
        assert isinstance(s, stl.Globally)
        assert s.unbound is True
        _check_semantics(f, s, x)


# ---------------------------------------------------------------------------
# Rules 7–8: degenerate windows  G[0,0] φ → φ  and  F[0,0] φ → φ
# ---------------------------------------------------------------------------

class TestDegenerateWindow:
    def test_globally_zero_zero_structure(self):
        f = stl.Globally(phi(), left_time_bound=0, right_time_bound=0)
        s = simplify(f)
        assert _structurally_equal(s, phi())

    def test_eventually_zero_zero_structure(self):
        f = stl.Eventually(phi(), left_time_bound=0, right_time_bound=0)
        s = simplify(f)
        assert _structurally_equal(s, phi())

    def test_globally_zero_zero_semantics(self, x):
        f = stl.Globally(phi(), left_time_bound=0, right_time_bound=0)
        _check_semantics(f, simplify(f), x)

    def test_eventually_zero_zero_semantics(self, x):
        f = stl.Eventually(phi(), left_time_bound=0, right_time_bound=0)
        _check_semantics(f, simplify(f), x)

    def test_non_zero_globally_unchanged(self):
        # G[0,1] φ should NOT be simplified away
        f = stl.Globally(phi(), left_time_bound=0, right_time_bound=1)
        s = simplify(f)
        assert isinstance(s, stl.Globally)


# ---------------------------------------------------------------------------
# Rules 9–10: temporal composition
# ---------------------------------------------------------------------------

class TestTemporalComposition:
    def test_globally_composition_structure(self):
        inner = stl.Globally(phi(), left_time_bound=1, right_time_bound=3)
        outer = stl.Globally(inner, left_time_bound=2, right_time_bound=5)
        s = simplify(outer)
        assert isinstance(s, stl.Globally)
        assert s.left_time_bound == 3        # 2 + 1
        assert s.right_time_bound == 9       # stored: (5+1) + (3+1) - 1 = 9; new right = 5+3=8, stored = 9
        assert _structurally_equal(s.child, phi())

    def test_eventually_composition_structure(self):
        inner = stl.Eventually(phi(), left_time_bound=0, right_time_bound=2)
        outer = stl.Eventually(inner, left_time_bound=1, right_time_bound=3)
        s = simplify(outer)
        assert isinstance(s, stl.Eventually)
        assert s.left_time_bound == 1        # 1 + 0
        assert s.right_time_bound == 6       # stored: (3+1) + (2+1) - 1 = 6; new right = 3+2=5, stored = 6

    def test_globally_composition_semantics(self, x):
        inner = stl.Globally(phi(), left_time_bound=1, right_time_bound=3)
        outer = stl.Globally(inner, left_time_bound=0, right_time_bound=2)
        _check_semantics(outer, simplify(outer), x)

    def test_eventually_composition_semantics(self, x):
        inner = stl.Eventually(phi(), left_time_bound=0, right_time_bound=2)
        outer = stl.Eventually(inner, left_time_bound=1, right_time_bound=2)
        _check_semantics(outer, simplify(outer), x)


# ---------------------------------------------------------------------------
# Rule 11: idempotency  φ ∧ φ → φ
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_and_idempotency_structure(self):
        f = stl.And(phi(), phi())
        s = simplify(f)
        assert _structurally_equal(s, phi())

    def test_or_idempotency_structure(self):
        f = stl.Or(phi(), phi())
        s = simplify(f)
        assert _structurally_equal(s, phi())

    def test_and_semantics(self, x):
        f = stl.And(phi(), phi())
        _check_semantics(f, simplify(f), x)

    def test_or_semantics(self, x):
        f = stl.Or(phi(), phi())
        _check_semantics(f, simplify(f), x)


# ---------------------------------------------------------------------------
# Rules 13–14: absorption  φ ∧ (φ ∨ ψ) → φ  and  φ ∨ (φ ∧ ψ) → φ
# ---------------------------------------------------------------------------

class TestAbsorption:
    def test_and_absorption_left(self):
        f = stl.And(phi(), stl.Or(phi(), psi()))
        s = simplify(f)
        assert _structurally_equal(s, phi())

    def test_and_absorption_right(self):
        # (φ ∨ ψ) ∧ φ → φ
        f = stl.And(stl.Or(phi(), psi()), phi())
        s = simplify(f)
        assert _structurally_equal(s, phi())

    def test_and_absorption_inner_right(self):
        # φ ∧ (ψ ∨ φ) → φ
        f = stl.And(phi(), stl.Or(psi(), phi()))
        s = simplify(f)
        assert _structurally_equal(s, phi())

    def test_or_absorption_left(self):
        f = stl.Or(phi(), stl.And(phi(), psi()))
        s = simplify(f)
        assert _structurally_equal(s, phi())

    def test_or_absorption_right(self):
        # (φ ∧ ψ) ∨ φ → φ
        f = stl.Or(stl.And(phi(), psi()), phi())
        s = simplify(f)
        assert _structurally_equal(s, phi())

    def test_and_absorption_semantics(self, x):
        f = stl.And(phi(), stl.Or(phi(), psi()))
        _check_semantics(f, simplify(f), x)

    def test_or_absorption_semantics(self, x):
        f = stl.Or(phi(), stl.And(phi(), psi()))
        _check_semantics(f, simplify(f), x)


# ---------------------------------------------------------------------------
# Rules 15–16: nested idempotency  φ ∧ (φ ∧ ψ) → φ ∧ ψ
# ---------------------------------------------------------------------------

class TestNestedIdempotency:
    def test_and_nested_left(self):
        # φ ∧ (φ ∧ ψ) → φ ∧ ψ
        inner = stl.And(phi(), psi())
        f = stl.And(phi(), inner)
        s = simplify(f)
        assert _structurally_equal(s, inner)

    def test_and_nested_right(self):
        # (φ ∧ ψ) ∧ φ → φ ∧ ψ
        inner = stl.And(phi(), psi())
        f = stl.And(inner, phi())
        s = simplify(f)
        assert _structurally_equal(s, inner)

    def test_and_nested_inner_right(self):
        # φ ∧ (ψ ∧ φ) → ψ ∧ φ
        inner = stl.And(psi(), phi())
        f = stl.And(phi(), inner)
        s = simplify(f)
        assert _structurally_equal(s, inner)

    def test_or_nested_left(self):
        inner = stl.Or(phi(), psi())
        f = stl.Or(phi(), inner)
        s = simplify(f)
        assert _structurally_equal(s, inner)

    def test_or_nested_right(self):
        inner = stl.Or(phi(), psi())
        f = stl.Or(inner, phi())
        s = simplify(f)
        assert _structurally_equal(s, inner)

    def test_and_nested_semantics(self, x):
        f = stl.And(phi(), stl.And(phi(), psi()))
        _check_semantics(f, simplify(f), x)

    def test_or_nested_semantics(self, x):
        f = stl.Or(phi(), stl.Or(phi(), psi()))
        _check_semantics(f, simplify(f), x)


# ---------------------------------------------------------------------------
# Rule 17: φ U φ → φ (unbounded)
# ---------------------------------------------------------------------------

class TestUntilSelf:
    def test_structure_unbound(self):
        f = stl.Until(phi(), phi(), unbound=True)
        s = simplify(f)
        assert _structurally_equal(s, phi())

    def test_semantics_unbound(self, x):
        f = stl.Until(phi(), phi(), unbound=True)
        _check_semantics(f, simplify(f), x)

    def test_bounded_not_simplified(self):
        # φ U[1,5] φ should NOT simplify to φ
        f = stl.Until(phi(), phi(), left_time_bound=1, right_time_bound=5)
        s = simplify(f)
        assert isinstance(s, stl.Until)


# ---------------------------------------------------------------------------
# Fixpoint / idempotency of simplify
# ---------------------------------------------------------------------------

class TestFixpoint:
    def test_simplify_is_idempotent(self):
        # simplify(simplify(f)) must equal simplify(f) structurally
        formulas = [
            stl.Not(stl.Not(stl.Not(phi()))),
            stl.And(phi(), stl.Or(phi(), psi())),
            stl.Not(stl.Globally(stl.And(phi(), psi()), left_time_bound=1, right_time_bound=4)),
            stl.Globally(stl.Globally(phi(), left_time_bound=1, right_time_bound=3),
                         left_time_bound=0, right_time_bound=2),
        ]
        for f in formulas:
            once = simplify(f)
            twice = simplify(once)
            assert _structurally_equal(once, twice), f"Not idempotent for: {f}"

    def test_already_simplified_unchanged(self):
        # A formula with no applicable rules should return structurally equal
        f = stl.And(phi(), psi())
        s = simplify(f)
        assert _structurally_equal(s, f)


# ---------------------------------------------------------------------------
# Regression: example from example_stl_usage.py
# ---------------------------------------------------------------------------

class TestRegression:
    def test_example_formula_semantics(self, x):
        n0 = stl.Atom(0, 1, lte=False)
        n1 = stl.Atom(1, 2, lte=True)
        ng = stl.Globally(n1, unbound=True)
        formula = stl.And(n0, ng)
        # No rule fires; robustness must be identical
        _check_semantics(formula, simplify(formula), x)
