#!/usr/bin/env python3
"""Benchmarks for quantitative robustness computation."""

import torch
import torch.utils.benchmark as benchmark
from torcheck import stl

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}\n")


def make_signal(n_samples, n_vars, n_steps):
    return torch.randn(n_samples, n_vars, n_steps, device=DEVICE)


# ---------------------------------------------------------------------------
# Formula builders
# ---------------------------------------------------------------------------

def formula_globally(window):
    atom = stl.Atom(0, 0.0, lte=False)
    return stl.Globally(atom, left_time_bound=0, right_time_bound=window - 1)


def formula_eventually(window):
    atom = stl.Atom(0, 0.0, lte=False)
    return stl.Eventually(atom, left_time_bound=0, right_time_bound=window - 1)


def formula_and():
    a = stl.Atom(0, 0.0, lte=False)
    b = stl.Atom(1, 0.0, lte=True)
    return stl.And(a, b)


def formula_until_bounded(window):
    a = stl.Atom(0, 0.0, lte=False)
    b = stl.Atom(1, 0.0, lte=True)
    return stl.Until(a, b, left_time_bound=0, right_time_bound=window - 1)


def formula_until_unbound():
    a = stl.Atom(0, 0.0, lte=False)
    b = stl.Atom(1, 0.0, lte=True)
    return stl.Until(a, b, unbound=True)


def formula_nested():
    """G[0,9](x0>=0 And F[0,4](x1<=0))"""
    a = stl.Atom(0, 0.0, lte=False)
    b = stl.Atom(1, 0.0, lte=True)
    inner = stl.And(a, stl.Eventually(b, left_time_bound=0, right_time_bound=4))
    return stl.Globally(inner, left_time_bound=0, right_time_bound=9)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def run_bench(label, fn, n_repeats=20):
    t = benchmark.Timer(stmt="fn()", globals={"fn": fn})
    result = t.timeit(n_repeats)
    print(f"  {label:<55s}  {result.median * 1e3:8.3f} ms  (median of {n_repeats})")


def bench_operator(name, formula_fn, n_samples, n_steps, n_vars=2, window=10):
    phi = formula_fn(window) if "window" in formula_fn.__code__.co_varnames else formula_fn()
    x = make_signal(n_samples, n_vars, n_steps)
    label = f"{name} | N={n_samples}, T={n_steps}"
    run_bench(label, lambda: phi.quantitative(x, evaluate_at_all_times=True))


# ---------------------------------------------------------------------------
# Sweep over signal lengths for Until (the expensive operator)
# ---------------------------------------------------------------------------

def bench_until_sweep():
    print("=== Until (bounded, window=10) — varying T ===")
    phi = formula_until_bounded(10)
    for T in [50, 100, 200, 500, 1000]:
        x = make_signal(64, 2, T)
        run_bench(f"Until[0,9] N=64, T={T}", lambda x=x: phi.quantitative(x, evaluate_at_all_times=True))

    print("\n=== Until (bounded) — varying batch size, T=100 ===")
    phi = formula_until_bounded(10)
    for N in [1, 16, 64, 256, 1024]:
        x = make_signal(N, 2, 100)
        run_bench(f"Until[0,9] N={N}, T=100", lambda x=x: phi.quantitative(x, evaluate_at_all_times=True))

    print("\n=== Until (bounded) — varying window size, N=64, T=100 ===")
    for W in [5, 10, 20, 50]:
        phi = formula_until_bounded(W)
        x = make_signal(64, 2, 100)
        run_bench(f"Until[0,{W-1}] N=64, T=100", lambda x=x, phi=phi: phi.quantitative(x, evaluate_at_all_times=True))

    print("\n=== Until (unbound) — varying T, N=64 ===")
    phi = formula_until_unbound()
    for T in [50, 100, 200, 500]:
        x = make_signal(64, 2, T)
        run_bench(f"Until(unbound) N=64, T={T}", lambda x=x: phi.quantitative(x, evaluate_at_all_times=True))


# ---------------------------------------------------------------------------
# Operator comparison at fixed size
# ---------------------------------------------------------------------------

def bench_operator_comparison():
    N, T = 64, 100
    print(f"\n=== Operator comparison (N={N}, T={T}) ===")
    cases = [
        ("Atom",           stl.Atom(0, 0.0, lte=False),             make_signal(N, 1, T)),
        ("Not(Atom)",      stl.Not(stl.Atom(0, 0.0)),               make_signal(N, 1, T)),
        ("And",            formula_and(),                            make_signal(N, 2, T)),
        ("Or",             stl.Or(stl.Atom(0,0),stl.Atom(1,0)),    make_signal(N, 2, T)),
        ("Globally[0,9]",  formula_globally(10),                    make_signal(N, 1, T)),
        ("Eventually[0,9]",formula_eventually(10),                  make_signal(N, 1, T)),
        ("Until[0,9]",     formula_until_bounded(10),               make_signal(N, 2, T)),
        ("Until(unbound)", formula_until_unbound(),                 make_signal(N, 2, T)),
        ("Nested formula", formula_nested(),                        make_signal(N, 2, T)),
    ]
    for name, phi, x in cases:
        run_bench(name, lambda phi=phi, x=x: phi.quantitative(x, evaluate_at_all_times=True))


# ---------------------------------------------------------------------------
# Memory usage for Until matrix
# ---------------------------------------------------------------------------

def report_until_memory():
    print("\n=== Until matrix memory footprint (float32) ===")
    for T in [50, 100, 200, 500, 1000]:
        # _build_until_matrix allocates two (N, 1, T, T) tensors
        N = 64
        bytes_per = N * 1 * T * T * 4  # float32 = 4 bytes, two tensors
        mb = 2 * bytes_per / 1e6
        print(f"  T={T:4d}, N={N}: 2 tensors of shape ({N}, 1, {T}, {T}) = {mb:.1f} MB")


# ---------------------------------------------------------------------------
# t=0 only vs all time steps
# ---------------------------------------------------------------------------

def bench_time_zero():
    print("\n=== evaluate_at_all_times=False (t=0 only) vs True (all steps) ===")
    print("  NOTE: current implementation always computes the full signal;")
    print("  evaluate_at_all_times only controls how many values are *returned*.\n")

    cases = [
        ("Atom",            stl.Atom(0, 0.0),                                  1),
        ("Globally[0,9]",   formula_globally(10),                              1),
        ("Eventually[0,9]", formula_eventually(10),                            1),
        ("Until[0,9]",      formula_until_bounded(10),                         2),
        ("Until(unbound)",  formula_until_unbound(),                           2),
        ("Nested",          formula_nested(),                                  2),
    ]

    N, T = 64, 200
    for name, phi, n_vars in cases:
        x = make_signal(N, n_vars, T)
        t_all = run_bench(f"{name} all_times=True ",
                          lambda x=x, phi=phi: phi.quantitative(x, evaluate_at_all_times=True))
        t_t0  = run_bench(f"{name} all_times=False",
                          lambda x=x, phi=phi: phi.quantitative(x, evaluate_at_all_times=False))
        ratio = t_all / t_t0
        print(f"  {name:<20s}  all={t_all:7.3f}ms   t0={t_t0:7.3f}ms   ratio={ratio:.2f}x")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bench_operator_comparison()
    bench_until_sweep()
    report_until_memory()
    bench_time_zero()
