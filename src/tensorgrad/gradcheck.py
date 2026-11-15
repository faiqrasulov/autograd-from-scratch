"""Finite-difference gradient checking: the framework-independent ground truth for backward()."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .tensor import Tensor


def numerical_grads(
    fn: Callable[..., Tensor], inputs: Sequence[np.ndarray], eps: float = 1e-6
) -> list[np.ndarray]:
    """Central differences of the scalar ``fn(*tensors)`` w.r.t. every element of every input."""
    arrays = [np.array(x, dtype=np.float64) for x in inputs]
    grads = []
    for x in arrays:
        g = np.zeros_like(x)
        for idx in np.ndindex(x.shape):
            orig = x[idx]
            x[idx] = orig + eps
            plus = fn(*[Tensor(a) for a in arrays]).item()
            x[idx] = orig - eps
            minus = fn(*[Tensor(a) for a in arrays]).item()
            x[idx] = orig
            g[idx] = (plus - minus) / (2 * eps)
        grads.append(g)
    return grads


def analytic_grads(fn: Callable[..., Tensor], inputs: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Gradients of the scalar ``fn(*tensors)`` from :meth:`Tensor.backward`."""
    tensors = [Tensor(np.array(x, dtype=np.float64), requires_grad=True) for x in inputs]
    fn(*tensors).backward()
    return [t.grad if t.grad is not None else np.zeros_like(t.data) for t in tensors]


def gradcheck(
    op: Callable[..., Tensor], inputs: Sequence[np.ndarray], seed: int = 0, eps: float = 1e-6
) -> float:
    """Max relative error between analytic and numerical gradients of ``op``.

    A non-scalar output is reduced with a fixed random projection ``sum(op(*x) * R)`` so the
    check covers the full vector-Jacobian product, not just ``sum``.
    """
    sample = op(*[Tensor(np.asarray(x, dtype=np.float64)) for x in inputs])
    proj = np.random.default_rng(seed).standard_normal(sample.shape)

    def scalar_fn(*tensors: Tensor) -> Tensor:
        return (op(*tensors) * proj).sum()

    worst = 0.0
    for a, n in zip(analytic_grads(scalar_fn, inputs), numerical_grads(scalar_fn, inputs, eps),
                    strict=True):
        err = np.abs(a - n) / np.maximum(1e-3, np.abs(a) + np.abs(n))
        worst = max(worst, float(err.max(initial=0.0)))
    return worst
