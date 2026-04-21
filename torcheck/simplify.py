#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================================================
# Copyright 2020-* Luca Bortolussi. All Rights Reserved.
# Copyright 2020-* Laura Nenzi.     All Rights Reserved.
# Copyright 2020-* AI-CPS Group @ University of Trieste. All Rights Reserved.
# ==============================================================================

"""Simplification of STL formula trees via structure-preserving rewriting rules.

All rules in the default mode are valid for both Boolean and quantitative
(robustness) semantics.  Each rule is an exact identity:
    rho(original, x, t) == rho(simplified, x, t)  for all x, t.
"""

from torcheck.stl import Atom, Eventually, Globally, Node, Not, And, Or, Until


# ---------------------------------------------------------------------------
# Structural equality
# ---------------------------------------------------------------------------

def _structurally_equal(a: Node, b: Node) -> bool:
    """Return True iff the two formula trees are structurally identical."""
    if type(a) is not type(b):
        return False
    if isinstance(a, Atom):
        return a.var_index == b.var_index and a.threshold == b.threshold and a.lte == b.lte
    if isinstance(a, Not):
        return _structurally_equal(a.child, b.child)
    if isinstance(a, (And, Or)):
        return (_structurally_equal(a.left_child, b.left_child)
                and _structurally_equal(a.right_child, b.right_child))
    if isinstance(a, (Globally, Eventually)):
        return (a.unbound == b.unbound
                and a.right_unbound == b.right_unbound
                and a.left_time_bound == b.left_time_bound
                and a.right_time_bound == b.right_time_bound  # stored value (+1 applied at init)
                and a.adapt_unbound == b.adapt_unbound
                and _structurally_equal(a.child, b.child))
    if isinstance(a, Until):
        return (a.unbound == b.unbound
                and a.right_unbound == b.right_unbound
                and a.left_time_bound == b.left_time_bound
                and a.right_time_bound == b.right_time_bound
                and _structurally_equal(a.left_child, b.left_child)
                and _structurally_equal(a.right_child, b.right_child))
    # Unknown node type: compare by identity
    return a is b


# ---------------------------------------------------------------------------
# Child reconstruction (bottom-up pass helper)
# ---------------------------------------------------------------------------

def _simplify_children(node: Node) -> Node:
    """Return a copy of *node* with all children replaced by their simplified forms."""
    if isinstance(node, Atom):
        return node
    if isinstance(node, Not):
        return Not(_simplify_pass(node.child))
    if isinstance(node, And):
        return And(_simplify_pass(node.left_child), _simplify_pass(node.right_child))
    if isinstance(node, Or):
        return Or(_simplify_pass(node.left_child), _simplify_pass(node.right_child))
    if isinstance(node, Globally):
        return Globally(
            _simplify_pass(node.child),
            unbound=node.unbound,
            right_unbound=node.right_unbound,
            left_time_bound=node.left_time_bound,
            right_time_bound=node.right_time_bound - 1,  # undo the +1 stored at init
            adapt_unbound=node.adapt_unbound,
        )
    if isinstance(node, Eventually):
        return Eventually(
            _simplify_pass(node.child),
            unbound=node.unbound,
            right_unbound=node.right_unbound,
            left_time_bound=node.left_time_bound,
            right_time_bound=node.right_time_bound - 1,
            adapt_unbound=node.adapt_unbound,
        )
    if isinstance(node, Until):
        return Until(
            _simplify_pass(node.left_child),
            _simplify_pass(node.right_child),
            unbound=node.unbound,
            right_unbound=node.right_unbound,
            left_time_bound=node.left_time_bound,
            right_time_bound=node.right_time_bound - 1,
        )
    return node  # unknown node type: pass through unchanged


# ---------------------------------------------------------------------------
# Rewriting rules (applied at a single node after children are simplified)
# ---------------------------------------------------------------------------

def _apply_rules(node: Node) -> Node:
    """Try every simplification rule in priority order; return first match."""

    # ------------------------------------------------------------------
    # Negation rules
    # ------------------------------------------------------------------

    # Double negation: ¬¬φ → φ
    # Proof: rho(¬¬φ) = -(-rho(φ)) = rho(φ)
    if isinstance(node, Not) and isinstance(node.child, Not):
        return node.child.child

    # Atom negation: ¬(x_i <= c) → x_i >= c  and vice versa
    # Proof: rho(¬(x<=c)) = -(c-x) = x-c = rho(x>=c)
    if isinstance(node, Not) and isinstance(node.child, Atom):
        a = node.child
        return Atom(a.var_index, a.threshold, lte=not a.lte)

    # De Morgan: ¬(φ ∧ ψ) → ¬φ ∨ ¬ψ
    # Proof: -min(a,b) = max(-a,-b)
    if isinstance(node, Not) and isinstance(node.child, And):
        return Or(Not(node.child.left_child), Not(node.child.right_child))

    # De Morgan: ¬(φ ∨ ψ) → ¬φ ∧ ¬ψ
    # Proof: -max(a,b) = min(-a,-b)
    if isinstance(node, Not) and isinstance(node.child, Or):
        return And(Not(node.child.left_child), Not(node.child.right_child))
    
    # Distributivity: (φ ∧ ψ) ∨ (φ ∧ χ) → φ ∧ (ψ ∨ χ)
    # Proof: max(min(a,b), min(a,c)) = min(a, max(b,c))
    if isinstance(node, Or) and isinstance(l, And) and isinstance(r, And):
        if _structurally_equal(l.left_child, r.left_child):
            return And(l.left_child, Or(l.right_child, r.right_child))
        if _structurally_equal(l.right_child, r.right_child):
            return And(Or(l.left_child, r.left_child), l.right_child)
        if _structurally_equal(l.left_child, r.right_child):
            return And(l.left_child, Or(l.right_child, r.left_child))
        if _structurally_equal(l.right_child, r.left_child):
            return And(l.right_child, Or(l.left_child, r.right_child))
    

    # Temporal negation: ¬G[a,b] φ → F[a,b] ¬φ
    # Proof: -min_{[a,b]} rho(φ) = max_{[a,b]} -rho(φ) = max_{[a,b]} rho(¬φ)
    if isinstance(node, Not) and isinstance(node.child, Globally):
        g = node.child
        return Eventually(
            Not(g.child),
            unbound=g.unbound,
            right_unbound=g.right_unbound,
            left_time_bound=g.left_time_bound,
            right_time_bound=g.right_time_bound - 1,
            adapt_unbound=g.adapt_unbound,
        )

    # Temporal negation: ¬F[a,b] φ → G[a,b] ¬φ
    # Proof: symmetric
    if isinstance(node, Not) and isinstance(node.child, Eventually):
        e = node.child
        return Globally(
            Not(e.child),
            unbound=e.unbound,
            right_unbound=e.right_unbound,
            left_time_bound=e.left_time_bound,
            right_time_bound=e.right_time_bound - 1,
            adapt_unbound=e.adapt_unbound,
        )

    # ------------------------------------------------------------------
    # Degenerate temporal windows
    # ------------------------------------------------------------------

    # G[0,0] φ → φ
    # Proof: -max_pool1d(-rho, kernel=1) = identity
    if (isinstance(node, Globally)
            and not node.unbound and not node.right_unbound
            and node.left_time_bound == 0 and node.right_time_bound == 1):
        return node.child

    # F[0,0] φ → φ
    if (isinstance(node, Eventually)
            and not node.unbound and not node.right_unbound
            and node.left_time_bound == 0 and node.right_time_bound == 1):
        return node.child

    # ------------------------------------------------------------------
    # Temporal composition (bounded operators only)
    # ------------------------------------------------------------------

    # G[a,b](G[c,d] φ) → G[a+c, b+d] φ
    # Proof: min_{s∈[a,b]} min_{u∈[c,d]} rho(t+s+u) = min_{v∈[a+c,b+d]} rho(t+v)
    if (isinstance(node, Globally)
            and not node.unbound and not node.right_unbound
            and isinstance(node.child, Globally)
            and not node.child.unbound and not node.child.right_unbound):
        outer, inner = node, node.child
        return Globally(
            inner.child,
            left_time_bound=outer.left_time_bound + inner.left_time_bound,
            right_time_bound=(outer.right_time_bound - 1) + (inner.right_time_bound - 1),
        )

    # F[a,b](F[c,d] φ) → F[a+c, b+d] φ
    # Proof: symmetric (max of max)
    if (isinstance(node, Eventually)
            and not node.unbound and not node.right_unbound
            and isinstance(node.child, Eventually)
            and not node.child.unbound and not node.child.right_unbound):
        outer, inner = node, node.child
        return Eventually(
            inner.child,
            left_time_bound=outer.left_time_bound + inner.left_time_bound,
            right_time_bound=(outer.right_time_bound - 1) + (inner.right_time_bound - 1),
        )

    # ------------------------------------------------------------------
    # Boolean/quantitative idempotency and absorption (And / Or)
    # ------------------------------------------------------------------

    if isinstance(node, And):
        l, r = node.left_child, node.right_child

        # Idempotency: φ ∧ φ → φ
        # Proof: min(a, a) = a
        if _structurally_equal(l, r):
            return l

        # Absorption: φ ∧ (φ ∨ ψ) → φ  and  (φ ∨ ψ) ∧ φ → φ
        # Proof: min(a, max(a, b)) = a
        if isinstance(r, Or) and (
                _structurally_equal(l, r.left_child) or _structurally_equal(l, r.right_child)):
            return l
        if isinstance(l, Or) and (
                _structurally_equal(r, l.left_child) or _structurally_equal(r, l.right_child)):
            return r

        # Nested idempotency: φ ∧ (φ ∧ ψ) → φ ∧ ψ  and  (φ ∧ ψ) ∧ φ → φ ∧ ψ
        # Proof: min(a, min(a, b)) = min(a, b)
        if isinstance(r, And) and (
                _structurally_equal(l, r.left_child) or _structurally_equal(l, r.right_child)):
            return r
        if isinstance(l, And) and (
                _structurally_equal(r, l.left_child) or _structurally_equal(r, l.right_child)):
            return l

    if isinstance(node, Or):
        l, r = node.left_child, node.right_child

        # Idempotency: φ ∨ φ → φ
        # Proof: max(a, a) = a
        if _structurally_equal(l, r):
            return l

        # Absorption: φ ∨ (φ ∧ ψ) → φ  and  (φ ∧ ψ) ∨ φ → φ
        # Proof: max(a, min(a, b)) = a
        if isinstance(r, And) and (
                _structurally_equal(l, r.left_child) or _structurally_equal(l, r.right_child)):
            return l
        if isinstance(l, And) and (
                _structurally_equal(r, l.left_child) or _structurally_equal(r, l.right_child)):
            return r

        # Nested idempotency: φ ∨ (φ ∨ ψ) → φ ∨ ψ  and  (φ ∨ ψ) ∨ φ → φ ∨ ψ
        # Proof: max(a, max(a, b)) = max(a, b)
        if isinstance(r, Or) and (
                _structurally_equal(l, r.left_child) or _structurally_equal(l, r.right_child)):
            return r
        if isinstance(l, Or) and (
                _structurally_equal(r, l.left_child) or _structurally_equal(r, l.right_child)):
            return l

    # ------------------------------------------------------------------
    # Until
    # ------------------------------------------------------------------

    # φ U φ → φ  (unbounded Until only)
    # Proof: max_{s>=t} min(min_{t'<s} rho(t'), rho(s)) achieves its max at s=t
    #        (empty prefix gives +inf, so min(+inf, rho(t)) = rho(t)); all s>t give <= rho(t).
    # NOTE: does NOT hold for bounded Until with left_time_bound > 0.
    if isinstance(node, Until) and node.unbound:
        if _structurally_equal(node.left_child, node.right_child):
            return node.left_child

    return node  # no rule matched


# ---------------------------------------------------------------------------
# Single pass and fixpoint
# ---------------------------------------------------------------------------

def _simplify_pass(node: Node) -> Node:
    node = _simplify_children(node)
    node = _apply_rules(node)
    return node


def simplify(formula: Node, max_iterations: int = 100) -> Node:
    """Return a logically equivalent simplified form of *formula*.

    All rewriting rules preserve both Boolean and quantitative (robustness)
    semantics exactly.  The function iterates until a fixpoint is reached or
    *max_iterations* is exceeded.

    Parameters
    ----------
    formula:
        Root of the STL formula tree to simplify.
    max_iterations:
        Safety cap on the number of rewriting passes (default 100).
        With the current rule set termination is guaranteed; this cap exists
        only as a safeguard against future rule additions.

    Returns
    -------
    Node
        A new formula tree that is logically equivalent to *formula*.
    """
    for _ in range(max_iterations):
        new_formula = _simplify_pass(formula)
        if _structurally_equal(new_formula, formula):
            return new_formula
        formula = new_formula
    return formula
