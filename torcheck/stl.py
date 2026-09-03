#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================================================
# Copyright 2020-* Luca Bortolussi. All Rights Reserved.
# Copyright 2020-* Laura Nenzi.     All Rights Reserved.
# Copyright 2020-* AI-CPS Group @ University of Trieste. All Rights Reserved.
# ==============================================================================

"""A fully-differentiable implementation of Signal Temporal Logic semantic trees."""

from typing import Union

# For custom type-hints
# For tensor functions
import torch
import torch.nn.functional as F
from torch import Tensor

# Custom types
realnum = Union[float, int]


# TODO: automatic check of timespan when evaluating robustness? (should be done only at root node)

def eventually(x: Tensor, time_span: int) -> Tensor:
    # TODO: as of this implementation, the time_span must be int (we are working with steps,
    #  not exactly points in the time axis)
    # TODO: maybe converter from resolution to steps, if one has different setting
    """
    STL operator 'eventually' in 1D.

    Parameters
    ----------
    x: torch.Tensor
        Signal
    time_span: any numeric type
        Timespan duration

    Returns
    -------
    torch.Tensor
    A tensor containing the result of the operation.
    """
    return F.max_pool1d(x, kernel_size=time_span, stride=1)


class Node:
    """Abstract node class for STL semantics tree."""

    def __init__(self) -> None:
        # Must be overloaded.
        pass

    def __str__(self) -> str:
        # Must be overloaded.
        pass

    def boolean(self, x: Tensor, evaluate_at_all_times: bool = False) -> Tensor:
        """
        Evaluates the boolean semantics at the node.

        Parameters
        ----------
        x : torch.Tensor, of size N_samples x N_vars x N_sampling_points
            The input signals, stored as a batch tensor with trhee dimensions.
        evaluate_at_all_times: bool
            Whether to evaluate the semantics at all times (True) or
            just at t=0 (False).

        Returns
        -------
        torch.Tensor
        A tensor with the boolean semantics for the node.
        """
        z: Tensor = self._boolean(x)
        if evaluate_at_all_times:
            return z
        else:
            return self._extract_semantics_at_time_zero(z)

    def quantitative(
            self,
            x: Tensor,
            normalize: bool = False,
            evaluate_at_all_times: bool = False,
    ) -> Tensor:
        """
        Evaluates the quantitative semantics at the node.

        Parameters
        ----------
        x : torch.Tensor, of size N_samples x N_vars x N_sampling_points
            The input signals, stored as a batch tensor with three dimensions.
        normalize: bool
            Whether the measure of robustness if normalized (True) or
            not (False). Currently not in use.
        evaluate_at_all_times: bool
            Whether to evaluate the semantics at all times (True) or
            just at t=0 (False).

        Returns
        -------
        torch.Tensor
        A tensor with the quantitative semantics for the node.
        """
        if not evaluate_at_all_times:
            needed = self._signal_depth_for_t0()
            if needed != float('inf') and needed + 1 < x.size(2):
                x = x[:, :, :needed + 1]
        z: Tensor = self._quantitative(x, normalize)
        if evaluate_at_all_times:
            return z
        else:
            return self._extract_semantics_at_time_zero(z)

    def set_normalizing_flag(self, value: bool = True) -> None:
        """
        Setter for the 'normalization of robustness of the formula' flag.
        Currently not in use.
        """

    def time_depth(self) -> int:
        """Returns time depth of bounded temporal operators only."""
        # Must be overloaded.

    def _signal_depth_for_t0(self) -> Union[int, float]:
        """Steps of signal needed to evaluate at t=0.

        Returns float('inf') if the sub-formula contains an unbound temporal
        operator (whose semantics depends on the whole trace).  Otherwise
        equals time_depth().
        """
        return self.time_depth()

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        """Private method equivalent to public one for inner call."""
        # Must be overloaded.

    def _boolean(self, x: Tensor) -> Tensor:
        """Private method equivalent to public one for inner call."""
        # Must be overloaded.

    @staticmethod
    def _extract_semantics_at_time_zero(x: Tensor) -> Tensor:
        """Extrapolates the vector of truth values at time zero"""
        return torch.reshape(x[:, 0, 0], (-1,))


class Atom(Node):
    """Atomic formula node; for now of the form X<=t or X>=t"""

    def __init__(self, var_index: int, threshold: realnum, lte: bool = False) -> None:
        super().__init__()
        self.var_index: int = var_index
        self.threshold: realnum = threshold
        self.lte: bool = lte

    def __str__(self) -> str:
        s: str = (
                "(x_"
                + str(self.var_index)
                + (" <= " if self.lte else " => ")
                + str(round(self.threshold, 4))
                + ")"
        )
        return s

    def time_depth(self) -> int:
        return 0

    def _boolean(self, x: Tensor) -> Tensor:
        # extract tensor of the same dimension as data, but with only one variable
        xj: Tensor = x[:, self.var_index, :]
        xj: Tensor = xj.view(xj.size()[0], 1, -1)
        if self.lte:
            z: Tensor = torch.le(xj, self.threshold)
        else:
            z: Tensor = torch.ge(xj, self.threshold)
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        # extract tensor of the same dimension as data, but with only one variable
        xj: Tensor = x[:, self.var_index, :]
        xj: Tensor = xj.view(xj.size()[0], 1, -1)
        if self.lte:
            z: Tensor = -xj + self.threshold
        else:
            z: Tensor = xj - self.threshold
        if normalize:
            z: Tensor = torch.tanh(z)
        return z


class Not(Node):
    """Negation node."""

    def __init__(self, child: Node) -> None:
        super().__init__()
        self.child: Node = child

    def __str__(self) -> str:
        s: str = "( \u00AC" + self.child.__str__() + " )"
        return s

    def time_depth(self) -> int:
        return self.child.time_depth()

    def _signal_depth_for_t0(self) -> Union[int, float]:
        return self.child._signal_depth_for_t0()

    def _boolean(self, x: Tensor) -> Tensor:
        z: Tensor = ~self.child._boolean(x)
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z: Tensor = -self.child._quantitative(x, normalize)
        return z


class And(Node):
    """Conjunction node."""

    def __init__(self, left_child: Node, right_child: Node) -> None:
        super().__init__()
        self.left_child: Node = left_child
        self.right_child: Node = right_child

    def __str__(self) -> str:
        s: str = (
                "( "
                + self.left_child.__str__()
                + " \u2227 "
                + self.right_child.__str__()
                + " )"
        )
        return s

    def time_depth(self) -> int:
        return max(self.left_child.time_depth(), self.right_child.time_depth())

    def _signal_depth_for_t0(self) -> Union[int, float]:
        return max(self.left_child._signal_depth_for_t0(),
                   self.right_child._signal_depth_for_t0())

    def _boolean(self, x: Tensor) -> Tensor:
        z1: Tensor = self.left_child._boolean(x)
        z2: Tensor = self.right_child._boolean(x)
        size: int = min(z1.size()[2], z2.size()[2])
        z1: Tensor = z1[:, :, :size]
        z2: Tensor = z2[:, :, :size]
        z: Tensor = torch.logical_and(z1, z2)
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z1: Tensor = self.left_child._quantitative(x, normalize)
        z2: Tensor = self.right_child._quantitative(x, normalize)
        size: int = min(z1.size()[2], z2.size()[2])
        z1: Tensor = z1[:, :, :size]
        z2: Tensor = z2[:, :, :size]
        z: Tensor = torch.min(z1, z2)
        return z


class Or(Node):
    """Disjunction node."""

    def __init__(self, left_child: Node, right_child: Node) -> None:
        super().__init__()
        self.left_child: Node = left_child
        self.right_child: Node = right_child

    def __str__(self) -> str:
        s: str = (
                "( "
                + self.left_child.__str__()
                + " \u2228 "
                + self.right_child.__str__()
                + " )"
        )
        return s

    def time_depth(self) -> int:
        return max(self.left_child.time_depth(), self.right_child.time_depth())

    def _signal_depth_for_t0(self) -> Union[int, float]:
        return max(self.left_child._signal_depth_for_t0(),
                   self.right_child._signal_depth_for_t0())

    def _boolean(self, x: Tensor) -> Tensor:
        z1: Tensor = self.left_child._boolean(x)
        z2: Tensor = self.right_child._boolean(x)
        size: int = min(z1.size()[2], z2.size()[2])
        z1: Tensor = z1[:, :, :size]
        z2: Tensor = z2[:, :, :size]
        z: Tensor = torch.logical_or(z1, z2)
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z1: Tensor = self.left_child._quantitative(x, normalize)
        z2: Tensor = self.right_child._quantitative(x, normalize)
        size: int = min(z1.size()[2], z2.size()[2])
        z1: Tensor = z1[:, :, :size]
        z2: Tensor = z2[:, :, :size]
        z: Tensor = torch.max(z1, z2)
        return z


class Globally(Node):
    """Globally node."""

    def __init__(
            self,
            child: Node,
            unbound: bool = False,
            right_unbound: bool = False,
            left_time_bound: int = 0,
            right_time_bound: int = 1,
            adapt_unbound: bool = True,
    ) -> None:
        super().__init__()
        self.child: Node = child
        self.unbound: bool = unbound
        self.right_unbound: bool = right_unbound
        self.left_time_bound: int = left_time_bound
        self.right_time_bound: int = right_time_bound + 1
        self.adapt_unbound: bool = adapt_unbound

        if self.unbound and self.right_unbound:
            raise ValueError("Cannot set both unbound=True and right_unbound=True")
        if (self.unbound is False) and (self.right_unbound is False) and \
                (self.right_time_bound <= self.left_time_bound):
            raise ValueError("Temporal thresholds are incorrect: right parameter is not higher than left parameter")

    def __str__(self) -> str:
        s_left = "[" + str(self.left_time_bound) + ","
        s_right = str(self.right_time_bound - 1) if not self.right_unbound else "inf"
        s0: str = s_left + s_right + "]" if not self.unbound else ""
        s: str = "( G" + s0 + self.child.__str__() + " )"
        return s

    def time_depth(self) -> int:
        if self.unbound:
            return self.child.time_depth() + self.left_time_bound
        elif self.right_unbound:
            return self.child.time_depth() + self.left_time_bound
        else:
            # diff = torch.le(torch.tensor([self.left_time_bound]), 0).float()
            return self.child.time_depth() + self.right_time_bound - 1
            # (self.right_time_bound - self.left_time_bound + 1) - diff

    def _signal_depth_for_t0(self) -> Union[int, float]:
        if self.unbound or self.right_unbound:
            return float('inf')
        return self.child._signal_depth_for_t0() + self.right_time_bound - 1

    def _boolean(self, x: Tensor) -> Tensor:
        z1: Tensor = self.child._boolean(x[:, :, self.left_time_bound:])  # nested temporal parameters
        # z1 = z1[:, :, self.left_time_bound:]
        if self.unbound or self.right_unbound:
            if self.adapt_unbound:
                z: Tensor
                _: Tensor
                z, _ = torch.cummin(torch.flip(z1, [2]), dim=2)
                z: Tensor = torch.flip(z, [2])
            else:
                z: Tensor
                _: Tensor
                z, _ = torch.min(z1, 2, keepdim=True)
        else:
            z: Tensor = eventually((~z1).float(), self.right_time_bound - self.left_time_bound) < 0.5
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z1: Tensor = self.child._quantitative(x[:, :, self.left_time_bound:], normalize)
        # z1 = z1[:, :, self.left_time_bound:]
        if self.unbound or self.right_unbound:
            if self.adapt_unbound:
                z: Tensor
                _: Tensor
                z, _ = torch.cummin(torch.flip(z1, [2]), dim=2)
                z: Tensor = torch.flip(z, [2])
            else:
                z: Tensor
                _: Tensor
                z, _ = torch.min(z1, 2, keepdim=True)
        else:
            z: Tensor = -eventually(-z1, self.right_time_bound - self.left_time_bound)
        return z


class Eventually(Node):
    """Eventually node."""

    def __init__(
            self,
            child: Node,
            unbound: bool = False,
            right_unbound: bool = False,
            left_time_bound: int = 0,
            right_time_bound: int = 1,
            adapt_unbound: bool = True,
    ) -> None:
        super().__init__()
        self.child: Node = child
        self.unbound: bool = unbound
        self.right_unbound: bool = right_unbound
        self.left_time_bound: int = left_time_bound
        self.right_time_bound: int = right_time_bound + 1
        self.adapt_unbound: bool = adapt_unbound

        if self.unbound and self.right_unbound:
            raise ValueError("Cannot set both unbound=True and right_unbound=True")
        if (self.unbound is False) and (self.right_unbound is False) and \
                (self.right_time_bound <= self.left_time_bound):
            raise ValueError("Temporal thresholds are incorrect: right parameter is not higher than left parameter")

    def __str__(self) -> str:
        s_left = "[" + str(self.left_time_bound) + ","
        s_right = str(self.right_time_bound - 1) if not self.right_unbound else "inf"
        s0: str = s_left + s_right + "]" if not self.unbound else ""
        s: str = "( F" + s0 + self.child.__str__() + " )"
        return s

    # TODO: coherence between computation of time depth and time span given when computing eventually 1d
    def time_depth(self) -> int:
        if self.unbound:
            return self.child.time_depth() + self.left_time_bound
        elif self.right_unbound:
            return self.child.time_depth() + self.left_time_bound
        else:
            # diff = torch.le(torch.tensor([self.left_time_bound]), 0).float()
            return self.child.time_depth() + self.right_time_bound - 1
            # (self.right_time_bound - self.left_time_bound + 1) - diff

    def _signal_depth_for_t0(self) -> Union[int, float]:
        if self.unbound or self.right_unbound:
            return float('inf')
        return self.child._signal_depth_for_t0() + self.right_time_bound - 1

    def _boolean(self, x: Tensor) -> Tensor:
        z1: Tensor = self.child._boolean(x[:, :, self.left_time_bound:])
        if self.unbound or self.right_unbound:
            if self.adapt_unbound:
                z: Tensor
                _: Tensor
                z, _ = torch.cummax(torch.flip(z1, [2]), dim=2)
                z: Tensor = torch.flip(z, [2])
            else:
                z: Tensor
                _: Tensor
                z, _ = torch.max(z1, 2, keepdim=True)
        else:
            z: Tensor = eventually(z1.float(), self.right_time_bound - self.left_time_bound) >= 0.5
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z1: Tensor = self.child._quantitative(x[:, :, self.left_time_bound:], normalize)
        if self.unbound or self.right_unbound:
            if self.adapt_unbound:
                z: Tensor
                _: Tensor
                z, _ = torch.cummax(torch.flip(z1, [2]), dim=2)
                z: Tensor = torch.flip(z, [2])
            else:
                z: Tensor
                _: Tensor
                z, _ = torch.max(z1, 2, keepdim=True)
        else:
            z: Tensor = eventually(z1, self.right_time_bound - self.left_time_bound)
        return z


class Until(Node):
    # TODO: maybe define timed and untimed until, and use this class to wrap them
    # TODO: maybe faster implementation (of untimed until especially)
    """Until node."""

    def __init__(
            self,
            left_child: Node,
            right_child: Node,
            unbound: bool = False,
            right_unbound: bool = False,
            left_time_bound: int = 0,
            right_time_bound: int = 1,
            adapt_unbound: bool = True,
    ) -> None:
        super().__init__()
        self.left_child: Node = left_child
        self.right_child: Node = right_child
        self.unbound: bool = unbound
        self.right_unbound: bool = right_unbound
        self.left_time_bound: int = left_time_bound
        self.right_time_bound: int = right_time_bound + 1
        self.adapt_unbound: bool = adapt_unbound

        if self.unbound and self.right_unbound:
            raise ValueError("Cannot set both unbound=True and right_unbound=True")
        if (self.unbound is False) and (self.right_unbound is False) and \
                (self.right_time_bound <= self.left_time_bound):
            raise ValueError("Temporal thresholds are incorrect: right parameter is not higher than left parameter")

    def __str__(self) -> str:
        s_left = "[" + str(self.left_time_bound) + ","
        s_right = str(self.right_time_bound - 1) if not self.right_unbound else "inf"
        s0: str = s_left + s_right + "]" if not self.unbound else ""
        s: str = "( " + self.left_child.__str__() + " U" + s0 + " " + self.right_child.__str__() + " )"
        return s

    def time_depth(self) -> int:
        sum_children_depth: int = self.left_child.time_depth() + self.right_child.time_depth()
        if self.unbound:
            return sum_children_depth
        elif self.right_unbound:
            return sum_children_depth + self.left_time_bound
        else:
            return sum_children_depth + self.right_time_bound - 1

    def _signal_depth_for_t0(self) -> Union[int, float]:
        if self.unbound or self.right_unbound:
            return float('inf')
        return (self.left_child._signal_depth_for_t0() +
                self.right_child._signal_depth_for_t0() +
                self.right_time_bound - 1)

    def _build_until_matrix(self, z1: Tensor, z2: Tensor, size: int):
        """Build z1_def (cumulative-min matrix) and z2_def for Until semantics.

        z1_def[t, s] = min_{k in [t,s]} z1[k]  for s >= t
        z2_def[t, s] = z2[s]                    for s >= t
        (entries for s < t are filled with z1[t]/z2[t] respectively but are
        masked out by the callers before taking the final max/any.)
        """
        z1_rep = torch.repeat_interleave(z1.unsqueeze(2), size, 2)
        z1_tril = torch.tril(z1_rep.transpose(2, 3), diagonal=-1)
        z1_triu = torch.triu(z1_rep)
        z1_def: Tensor = torch.cummin(z1_tril + z1_triu, dim=3)[0]

        z2_rep = torch.repeat_interleave(z2.unsqueeze(2), size, 2)
        z2_tril = torch.tril(z2_rep.transpose(2, 3), diagonal=-1)
        z2_triu = torch.triu(z2_rep)
        z2_def: Tensor = z2_tril + z2_triu
        return z1_def, z2_def

    def _band_mask(self, size: int, device) -> Tensor:
        """Return a (1,1,T,T) boolean mask selecting columns s where
        left_time_bound <= s-t (and s-t <= right_time_bound-1 for bounded)."""
        idx = torch.arange(size, device=device)
        diff = idx.view(1, -1) - idx.view(-1, 1)  # (T, T): diff[t,s] = s - t
        band = diff >= self.left_time_bound
        if not self.right_unbound:
            band = band & (diff <= self.right_time_bound - 1)
        return band.view(1, 1, size, size)

    def _boolean(self, x: Tensor) -> Tensor:
        z1: Tensor = self.left_child._boolean(x)
        z2: Tensor = self.right_child._boolean(x)
        size: int = min(z1.size()[2], z2.size()[2])
        z1 = z1[:, :, :size]
        z2 = z2[:, :, :size]

        z1_def, z2_def = self._build_until_matrix(z1, z2, size)
        inner: Tensor = torch.min(
            torch.cat([z1_def.unsqueeze(-1), z2_def.unsqueeze(-1)], dim=-1), dim=-1
        )[0]  # (N, 1, T, T)

        if self.unbound:
            z: Tensor = torch.max(inner, dim=-1)[0]
            if not self.adapt_unbound:
                z = z.max(dim=2, keepdim=True)[0]
        else:
            band: Tensor = self._band_mask(size, z1.device)
            # Zero out out-of-band entries, then check if any valid entry is satisfied
            z = (inner * band.long()).bool().any(dim=-1)
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z1: Tensor = self.left_child._quantitative(x, normalize)
        z2: Tensor = self.right_child._quantitative(x, normalize)
        size: int = min(z1.size()[2], z2.size()[2])
        z1 = z1[:, :, :size]
        z2 = z2[:, :, :size]

        if self.unbound:
            return self._quantitative_unbound(z1, z2)

        if self.right_unbound:
            return self._quantitative_right_unbound(z1, z2)

        return self._quantitative_bounded(z1, z2)

    def _quantitative_bounded(self, z1: Tensor, z2: Tensor) -> Tensor:
        """O(T * W) bounded Until via unfold + cummin.

        Computes result[t] = max_{d=a}^{b} min(min_{k=t}^{t+d} z1[k], z2[t+d])
        using sliding windows instead of a full T×T matrix.
        """
        a: int = self.left_time_bound
        b: int = self.right_time_bound - 1
        W: int = b - a + 1
        size: int = z1.size(2)

        valid_len: int = max(0, size - a)
        if valid_len == 0 or W <= 0:
            return torch.full((z1.size(0), z1.size(1), valid_len),
                              float('-inf'), device=z1.device, dtype=z1.dtype)

        # Pad z1 with +inf (neutral for min), z2 with -inf (masks out-of-range witnesses)
        inf_pos = torch.full((*z1.shape[:2], b), float('inf'), device=z1.device, dtype=z1.dtype)
        inf_neg = torch.full((*z2.shape[:2], b), float('-inf'), device=z2.device, dtype=z2.dtype)
        z1_p: Tensor = torch.cat([z1, inf_pos], dim=2)
        z2_p: Tensor = torch.cat([z2, inf_neg], dim=2)

        # rolling min of length (a+1): min_{k=t}^{t+a} z1_p[k]
        roll_a: Tensor = -F.max_pool1d(-z1_p, kernel_size=a + 1, stride=1)

        if W == 1:
            prefix: Tensor = roll_a[:, :, :valid_len]
            z2_at: Tensor = z2_p[:, :, a: a + valid_len]
            return torch.minimum(prefix, z2_at)

        # tail_w[t, k] = z1_p[t+a+1+k]  for k in [0, W-2], t in [0, valid_len-1]
        tail: Tensor = z1_p[:, :, a + 1:]
        tail_w: Tensor = tail.unfold(2, W - 1, 1)[:, :, :valid_len, :]
        tail_cm: Tensor = torch.cummin(tail_w, dim=-1)[0]

        prefix_exp: Tensor = roll_a[:, :, :valid_len].unsqueeze(-1)
        full_cm: Tensor = torch.cat(
            [prefix_exp, torch.minimum(prefix_exp, tail_cm)], dim=-1
        )  # (N, C, valid_len, W)

        # z2_w[t, j] = z2_p[t+a+j]
        z2_w: Tensor = z2_p[:, :, a:].unfold(2, W, 1)[:, :, :valid_len, :]

        return torch.max(torch.minimum(full_cm, z2_w), dim=-1)[0]

    @staticmethod
    def _until_backward_scan(z1: Tensor, z2: Tensor) -> Tensor:
        """O(T) untimed-Until backward scan.

        M[t] = max_{e >= t} min(min_{k in [t,e]} z1[k], z2[e]), computed by the
        recurrence M[t] = min(z1[t], max(z2[t], M[t+1])) with M[T] = -inf.
        """
        T: int = z1.size(2)
        result: Tensor = torch.empty_like(z1)
        result[:, :, T - 1] = torch.minimum(z1[:, :, T - 1], z2[:, :, T - 1])
        for t in range(T - 2, -1, -1):
            result[:, :, t] = torch.minimum(
                z1[:, :, t],
                torch.maximum(z2[:, :, t], result[:, :, t + 1])
            )
        return result

    def _quantitative_unbound(self, z1: Tensor, z2: Tensor) -> Tensor:
        """O(T) unbound Until via backward scan."""
        result: Tensor = self._until_backward_scan(z1, z2)
        if not self.adapt_unbound:
            result = result.max(dim=2, keepdim=True)[0]
        return result

    def _quantitative_right_unbound(self, z1: Tensor, z2: Tensor) -> Tensor:
        """O(T) right-unbounded Until, i.e. phi U_[a, inf) psi.

        result[t] = max_{d >= a} min(min_{k in [t,t+d]} z1[k], z2[t+d])

        which factors into a rolling min of z1 over the prefix [t, t+a] and an
        untimed Until starting at t+a.  Replaces a (N, C, T, T) matrix with two
        O(T) passes; the matrix form is quadratic in memory and OOMs on long
        signals (e.g. 165 GB at N=128, T=17984).
        """
        a: int = self.left_time_bound
        size: int = z1.size(2)
        valid_len: int = max(0, size - a)
        if valid_len == 0:
            return torch.full((z1.size(0), z1.size(1), 0),
                              float('-inf'), device=z1.device, dtype=z1.dtype)

        tail: Tensor = self._until_backward_scan(z1, z2)
        if a == 0:
            return tail[:, :, :valid_len]

        # min_{k in [t, t+a]} z1[k]; pad with +inf, the neutral element for min.
        inf_pos = torch.full((*z1.shape[:2], a), float('inf'),
                             device=z1.device, dtype=z1.dtype)
        z1_p: Tensor = torch.cat([z1, inf_pos], dim=2)
        prefix: Tensor = -F.max_pool1d(-z1_p, kernel_size=a + 1, stride=1)
        return torch.minimum(prefix[:, :, :valid_len], tail[:, :, a:a + valid_len])
