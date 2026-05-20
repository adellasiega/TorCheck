#!/usr/bin/env python3
"""Compare current vs optimized Until quantitative robustness."""

import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark
from torcheck import stl
from torcheck.stl import Node, Tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}\n")


# ---------------------------------------------------------------------------
# Optimized Until implementations
# ---------------------------------------------------------------------------

def _until_bounded_fast(z1: Tensor, z2: Tensor, left_time_bound: int, right_time_bound: int) -> Tensor:
    """O(T * W) bounded Until via unfold + cummin instead of T×T matrix.

    Computes: result[t] = max_{d=a}^{b} min(min_{k=t}^{t+d} z1[k], z2[t+d])

    Strategy: pad z1 with +inf and z2 with -inf so out-of-range accesses are
    automatically neutral. Then use unfold to build all windows at once.
    """
    a = left_time_bound
    b = right_time_bound - 1  # stored as b+1 in the node
    W = b - a + 1

    size = min(z1.size(2), z2.size(2))
    z1 = z1[:, :, :size]
    z2 = z2[:, :, :size]

    valid_len = max(0, size - a)
    if valid_len == 0 or W <= 0:
        return torch.full((z1.size(0), z1.size(1), valid_len),
                          float('-inf'), device=z1.device, dtype=z1.dtype)

    # Pad so out-of-bounds accesses in the window are safe:
    # z1 padded with +inf (neutral for min), z2 with -inf (masks invalid witnesses)
    pad_len = b
    inf_pos = torch.full((*z1.shape[:2], pad_len), float('inf'), device=z1.device, dtype=z1.dtype)
    inf_neg = torch.full((*z2.shape[:2], pad_len), float('-inf'), device=z2.device, dtype=z2.dtype)
    z1_p = torch.cat([z1, inf_pos], dim=2)   # (N, C, size+b)
    z2_p = torch.cat([z2, inf_neg], dim=2)   # (N, C, size+b)

    # Rolling min of size (a+1): min_{k=t}^{t+a} z1_p[k]
    # Output length: size+b - a ≥ valid_len
    roll_a = -F.max_pool1d(-z1_p, kernel_size=a + 1, stride=1)  # (N, C, size+b-a)

    if W == 1:
        prefix = roll_a[:, :, :valid_len]                      # (N, C, valid_len)
        z2_at = z2_p[:, :, a: a + valid_len]
        return torch.minimum(prefix, z2_at)

    # Build full_cummin[n, c, t, j] = min_{k=t}^{t+a+j} z1_p[k] for j in [0, W-1]
    # j=0 → roll_a[t]
    # j>0 → min(roll_a[t], cummin of z1_p[t+a+1 .. t+a+j])
    tail = z1_p[:, :, a + 1:]                                   # length: size+b-a-1
    # unfold(2, W-1, 1): produces windows [t+a+1 .. t+a+W-1] for each t
    tail_w = tail.unfold(2, W - 1, 1)[:, :, :valid_len, :]     # (N, C, valid_len, W-1)
    tail_cm = torch.cummin(tail_w, dim=-1)[0]                   # (N, C, valid_len, W-1)

    prefix_exp = roll_a[:, :, :valid_len].unsqueeze(-1)         # (N, C, valid_len, 1)
    suffix_cm = torch.minimum(prefix_exp, tail_cm)              # (N, C, valid_len, W-1)
    full_cm = torch.cat([prefix_exp, suffix_cm], dim=-1)        # (N, C, valid_len, W)

    # z2_windows[n, c, t, j] = z2_p[t+a+j]
    z2_w = z2_p[:, :, a:].unfold(2, W, 1)[:, :, :valid_len, :] # (N, C, valid_len, W)

    inner = torch.minimum(full_cm, z2_w)                        # (N, C, valid_len, W)
    return torch.max(inner, dim=-1)[0]                          # (N, C, valid_len)


def _until_unbound_scan(z1: Tensor, z2: Tensor, adapt_unbound: bool) -> Tensor:
    """O(T) unbound Until via backward scan instead of T×T matrix.

    Recurrence: M[t] = min(z1[t], max(z2[t], M[t+1]))  with M[T] = -inf
    """
    N, C, T = z1.shape
    result = torch.empty_like(z1)
    result[:, :, T - 1] = torch.minimum(z1[:, :, T - 1], z2[:, :, T - 1])
    for t in range(T - 2, -1, -1):
        result[:, :, t] = torch.minimum(
            z1[:, :, t],
            torch.maximum(z2[:, :, t], result[:, :, t + 1])
        )
    if not adapt_unbound:
        result = result.max(dim=2, keepdim=True)[0]
    return result


# ---------------------------------------------------------------------------
# Wrappers for correctness checking
# ---------------------------------------------------------------------------

class UntilFast(stl.Until):
    """Drop-in replacement for Until using optimized quantitative semantics."""

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z1 = self.left_child._quantitative(x, normalize)
        z2 = self.right_child._quantitative(x, normalize)
        size = min(z1.size(2), z2.size(2))
        z1 = z1[:, :, :size]
        z2 = z2[:, :, :size]

        if self.unbound:
            return _until_unbound_scan(z1, z2, self.adapt_unbound)

        if self.right_unbound:
            # Fall back to original for right_unbound (rare)
            return super()._quantitative(x, normalize)

        return _until_bounded_fast(
            z1, z2,
            self.left_time_bound, self.right_time_bound
        )


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def check_correctness():
    print("=== Correctness check ===")
    torch.manual_seed(42)
    for a, b in [(0, 4), (0, 9), (2, 7), (0, 0), (3, 3)]:
        N, C, T = 16, 1, 50
        x = torch.randn(N, 2, T)
        atom_l = stl.Atom(0, 0.0)
        atom_r = stl.Atom(1, 0.0)

        orig = stl.Until(atom_l, atom_r, left_time_bound=a, right_time_bound=b)
        fast = UntilFast(atom_l, atom_r, left_time_bound=a, right_time_bound=b)

        z_orig = orig.quantitative(x, evaluate_at_all_times=True)
        z_fast = fast.quantitative(x, evaluate_at_all_times=True)

        match = torch.allclose(z_orig, z_fast, atol=1e-5)
        print(f"  U[{a},{b}]  shapes: orig={tuple(z_orig.shape)} fast={tuple(z_fast.shape)}  match={match}")
        if not match:
            diff = (z_orig - z_fast).abs().max().item()
            print(f"    max diff = {diff:.2e}")

    # Unbound
    for _ in range(3):
        x = torch.randn(8, 2, 30)
        atom_l = stl.Atom(0, 0.0)
        atom_r = stl.Atom(1, 0.0)
        orig = stl.Until(atom_l, atom_r, unbound=True)
        fast = UntilFast(atom_l, atom_r, unbound=True)
        z_orig = orig.quantitative(x, evaluate_at_all_times=True)
        z_fast = fast.quantitative(x, evaluate_at_all_times=True)
        match = torch.allclose(z_orig, z_fast, atol=1e-5)
        print(f"  U(unbound) shapes: orig={tuple(z_orig.shape)} fast={tuple(z_fast.shape)}  match={match}")


# ---------------------------------------------------------------------------
# Speed comparison
# ---------------------------------------------------------------------------

def make_signal(N, C, T):
    return torch.randn(N, C, T, device=DEVICE)


def run_bench(label, fn, n_repeats=30):
    t = benchmark.Timer(stmt="fn()", globals={"fn": fn})
    result = t.timeit(n_repeats)
    return result.median * 1e3


def compare(label, orig_fn, fast_fn, n_repeats=30):
    t_orig = run_bench(label, orig_fn, n_repeats)
    t_fast = run_bench(label, fast_fn, n_repeats)
    speedup = t_orig / t_fast
    print(f"  {label:<45s}  orig={t_orig:8.3f}ms  fast={t_fast:7.3f}ms  speedup={speedup:.1f}x")


def bench_bounded():
    print("\n=== Bounded Until: orig vs fast ===")
    for T in [50, 100, 200, 500, 1000]:
        N = 64
        x = make_signal(N, 2, T)
        orig = stl.Until(stl.Atom(0, 0.), stl.Atom(1, 0.), left_time_bound=0, right_time_bound=9)
        fast = UntilFast(stl.Atom(0, 0.), stl.Atom(1, 0.), left_time_bound=0, right_time_bound=9)
        compare(f"U[0,9] N={N}, T={T}",
                lambda x=x, phi=orig: phi.quantitative(x, evaluate_at_all_times=True),
                lambda x=x, phi=fast: phi.quantitative(x, evaluate_at_all_times=True))

    print()
    for W in [5, 10, 20, 50, 99]:
        N, T = 64, 100
        x = make_signal(N, 2, T)
        orig = stl.Until(stl.Atom(0, 0.), stl.Atom(1, 0.), left_time_bound=0, right_time_bound=W - 1)
        fast = UntilFast(stl.Atom(0, 0.), stl.Atom(1, 0.), left_time_bound=0, right_time_bound=W - 1)
        compare(f"U[0,{W-1}] N={N}, T={T}",
                lambda x=x, phi=orig: phi.quantitative(x, evaluate_at_all_times=True),
                lambda x=x, phi=fast: phi.quantitative(x, evaluate_at_all_times=True))


def bench_unbound():
    print("\n=== Unbound Until: orig vs scan ===")
    for T in [50, 100, 200, 500]:
        N = 64
        x = make_signal(N, 2, T)
        orig = stl.Until(stl.Atom(0, 0.), stl.Atom(1, 0.), unbound=True)
        fast = UntilFast(stl.Atom(0, 0.), stl.Atom(1, 0.), unbound=True)
        compare(f"U(unbound) N={N}, T={T}",
                lambda x=x, phi=orig: phi.quantitative(x, evaluate_at_all_times=True),
                lambda x=x, phi=fast: phi.quantitative(x, evaluate_at_all_times=True))


if __name__ == "__main__":
    check_correctness()
    bench_bounded()
    bench_unbound()
