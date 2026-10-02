# `TorCheck` 🔥✅
A fully-differentiable implementation of Signal Temporal Logic semantic trees based on PyTorch

## Install
```console
pip install git+https://github.com/ailab-units/TorCheck.git
```

## Semantics

Signals are tensors of shape `[n_samples, n_vars, T]`, sampled at discrete steps $t = 0, \dots, T-1$.
Time bounds are integers (number of steps) and intervals $[a, b]$ are closed on both ends, with $a \le b$.

`boolean(x)` returns the truth value and `quantitative(x)` the robustness $\rho$, both at $t = 0$
(pass `evaluate_at_all_times=True` to get every $t$ at which the formula can be evaluated).
If $\rho > 0$ the formula is true, if $\rho < 0$ it is false.

In the table, $a$ and $b$ stand for the keyword arguments `left_time_bound` and `right_time_bound`
(they are not the 2nd and 3rd positional arguments).

| Node | Boolean | Robustness $\rho(\cdot, t)$ |
|---|---|---|
| `Atom(i, c, lte=False)` | $x_i(t) \ge c$ | $x_i(t) - c$ |
| `Atom(i, c, lte=True)` | $x_i(t) \le c$ | $c - x_i(t)$ |
| `Not(φ)` | $\lnot φ$ | $-\rho(φ, t)$ |
| `And(φ, ψ)` | $φ \land ψ$ | $\min(\rho(φ, t), \rho(ψ, t))$ |
| `Or(φ, ψ)` | $φ \lor ψ$ | $\max(\rho(φ, t), \rho(ψ, t))$ |
| `Globally(φ, a, b)` | $φ$ at every $t' \in [t+a, t+b]$ | $\min_{t' \in [t+a, t+b]} \rho(φ, t')$ |
| `Eventually(φ, a, b)` | $φ$ at some $t' \in [t+a, t+b]$ | $\max_{t' \in [t+a, t+b]} \rho(φ, t')$ |
| `Until(φ, ψ, a, b)` | $ψ$ at some $t' \in [t+a, t+b]$ and $φ$ at every $t'' \in [t, t')$ | $\max_{t' \in [t+a, t+b]} \min\big(\rho(ψ, t'), \min_{t'' \in [t, t')} \rho(φ, t'')\big)$ |

Notes:
- Atoms are non-strict: a sample exactly equal to the threshold satisfies both `lte=False` and `lte=True`.
- In `Until`, $φ$ is required on $[t, t')$, not at $t'$; the empty minimum (when $t' = t$) is $+\infty$.
  Hence, with $a = 0$, `Until(φ, ψ)` holds whenever $ψ$ holds at $t$, even if $φ$ never does.
- With `normalize=True`, the robustness of each atom is passed through $\tanh$.

**Unbounded operators** use finite-trace semantics, i.e. "until the end of the signal":
- `right_unbound=True`: the interval is $[t+a, T-1]$.
- `unbound=True`: the right bound is ignored for `Globally`/`Eventually` (interval $[t+a, T-1]$, default $a = 0$);
  for `Until` both bounds are ignored (interval $[t, T-1]$).
- `adapt_unbound=False` (`Globally`/`Eventually` only): returns only the value at $t = 0$.

**Time depth.** `time_depth()` is the number of steps after $t$ a formula needs to look at: $0$ for atoms,
the maximum over children for `Not`, `And`, `Or`, plus $b$ for each bounded `Globally`, `Eventually`, `Until`
(plus $a$ when `right_unbound=True`). A signal needs at least `time_depth() + 1` samples to evaluate a bounded formula at $t = 0$.
