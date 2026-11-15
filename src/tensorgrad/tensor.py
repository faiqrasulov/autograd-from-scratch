"""The ``Tensor`` class and the reverse-mode autodiff engine.

Every differentiable operation builds its output with :meth:`Tensor._make`, passing the parent
tensors and a *backward function*. The backward function receives the upstream gradient
``dL/d(out)`` as a NumPy array and returns one gradient per parent (``dL/d(parent)``), already
reduced to the parent's shape. :meth:`Tensor.backward` topologically sorts the graph and runs
those functions from the output back to the leaves, summing gradients where a tensor is used more
than once. Only leaf tensors (created by the user, e.g. parameters) keep a ``.grad``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from typing import Any, Union

import numpy as np

DEFAULT_DTYPE = np.float64

_grad_enabled = True

ArrayLike = Union["Tensor", np.ndarray, float, int, Sequence[Any]]
BackwardFn = Callable[[np.ndarray], Sequence[np.ndarray | None]]


@contextmanager
def no_grad():
    """Context manager that disables graph construction (inference, optimizer updates)."""
    global _grad_enabled
    previous = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = previous


def is_grad_enabled() -> bool:
    """Return True when operations currently record the computation graph."""
    return _grad_enabled


def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum ``grad`` down to ``shape``: the adjoint of NumPy broadcasting.

    If ``x`` of shape ``shape`` was broadcast to ``grad.shape`` in the forward pass, each element
    of ``x`` contributed to several outputs, so its gradient is the sum over the broadcast axes:
    leading axes that were added, and axes where ``x`` had size 1.
    """
    if grad.shape == shape:
        return grad
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    stretched = tuple(i for i, n in enumerate(shape) if n == 1 and grad.shape[i] != 1)
    if stretched:
        grad = grad.sum(axis=stretched, keepdims=True)
    return grad


def _normalize_axes(axis: int | Iterable[int] | None, ndim: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    return tuple(sorted(a % ndim if ndim else 0 for a in axes))


def _expand_reduced(
    grad: np.ndarray, axes: tuple[int, ...], keepdims: bool, shape: tuple[int, ...]
) -> np.ndarray:
    """Broadcast the gradient of a reduction back to the input shape."""
    if not keepdims and axes:
        grad = np.expand_dims(grad, axes)
    return np.broadcast_to(grad, shape)


def _is_basic_index(index: Any) -> bool:
    items = index if isinstance(index, tuple) else (index,)
    return all(
        isinstance(i, (int, np.integer, slice)) or i is None or i is Ellipsis for i in items
    )


class Tensor:
    """An n-dimensional array that records operations for reverse-mode differentiation."""

    __slots__ = ("data", "grad", "requires_grad", "_parents", "_backward", "_op")
    # Make ``ndarray (op) Tensor`` dispatch to Tensor's reflected operators.
    __array_ufunc__ = None

    def __init__(self, data: ArrayLike, requires_grad: bool = False, dtype: Any = None) -> None:
        if isinstance(data, Tensor):
            data = data.data
        arr = np.asarray(data, dtype=dtype)
        if dtype is None and not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(DEFAULT_DTYPE)
        self.data: np.ndarray = arr
        self.grad: np.ndarray | None = None
        self.requires_grad = bool(requires_grad)
        self._parents: tuple[Tensor, ...] = ()
        self._backward: BackwardFn | None = None
        self._op = ""

    # ------------------------------------------------------------------ graph construction
    @staticmethod
    def _make(
        data: np.ndarray, parents: tuple[Tensor, ...], backward: BackwardFn, op: str
    ) -> Tensor:
        """Wrap an op result; record parents + backward only if a gradient is needed."""
        out = Tensor(data)
        if _grad_enabled and any(p.requires_grad for p in parents):
            out.requires_grad = True
            out._parents = parents
            out._backward = backward
            out._op = op
        return out

    def _lift(self, other: ArrayLike) -> Tensor:
        """Convert a Python scalar / array operand to a constant Tensor of this dtype."""
        if isinstance(other, Tensor):
            return other
        return Tensor(np.asarray(other, dtype=self.data.dtype))

    # ------------------------------------------------------------------ autodiff engine
    def backward(self, grad: ArrayLike | None = None) -> None:
        """Back-propagate from this tensor and accumulate ``.grad`` on every leaf."""
        if not self.requires_grad:
            raise RuntimeError("backward() called on a tensor that does not require grad")
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be given for non-scalar outputs")
            seed = np.ones_like(self.data)
        else:
            seed = np.asarray(grad.data if isinstance(grad, Tensor) else grad, self.data.dtype)
            if seed.shape != self.data.shape:
                raise ValueError(f"grad shape {seed.shape} != tensor shape {self.data.shape}")

        grads: dict[int, np.ndarray] = {id(self): seed}
        for node in reversed(self._topological_order()):
            g = grads.pop(id(node), None)
            if g is None:
                continue
            if node._backward is None:  # leaf: accumulate, as PyTorch does
                g = np.array(g, dtype=node.data.dtype)  # owned, writable, contiguous copy
                node.grad = g if node.grad is None else node.grad + g
                continue
            for parent, pg in zip(node._parents, node._backward(g), strict=True):
                if pg is None or not parent.requires_grad:
                    continue
                key = id(parent)
                grads[key] = grads[key] + pg if key in grads else pg

    def _topological_order(self) -> list[Tensor]:
        """Iterative post-order DFS: every node appears after all of its parents."""
        order: list[Tensor] = []
        visited: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for parent in node._parents:
                if parent.requires_grad and id(parent) not in visited:
                    stack.append((parent, False))
        return order

    def zero_grad(self) -> None:
        self.grad = None

    def detach(self) -> Tensor:
        """Return a tensor sharing the data but cut from the graph."""
        return Tensor(self.data)

    # ------------------------------------------------------------------ convenience
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def size(self) -> int:
        return self.data.size

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self) -> float:
        return self.data.item()

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        grad = f", grad_fn={self._op}" if self._op else ""
        flag = ", requires_grad=True" if self.requires_grad and not self._op else ""
        return f"Tensor({np.array2string(self.data, precision=4)}{flag}{grad})"

    # ------------------------------------------------------------------ elementwise arithmetic
    def __add__(self, other: ArrayLike) -> Tensor:
        a, b = self, self._lift(other)

        def backward(g: np.ndarray):
            return unbroadcast(g, a.shape), unbroadcast(g, b.shape)

        return Tensor._make(a.data + b.data, (a, b), backward, "add")

    def __sub__(self, other: ArrayLike) -> Tensor:
        a, b = self, self._lift(other)

        def backward(g: np.ndarray):
            return unbroadcast(g, a.shape), unbroadcast(-g, b.shape)

        return Tensor._make(a.data - b.data, (a, b), backward, "sub")

    def __neg__(self) -> Tensor:
        return Tensor._make(-self.data, (self,), lambda g: (-g,), "neg")

    def __mul__(self, other: ArrayLike) -> Tensor:
        a, b = self, self._lift(other)

        def backward(g: np.ndarray):
            return unbroadcast(g * b.data, a.shape), unbroadcast(g * a.data, b.shape)

        return Tensor._make(a.data * b.data, (a, b), backward, "mul")

    def __truediv__(self, other: ArrayLike) -> Tensor:
        a, b = self, self._lift(other)

        def backward(g: np.ndarray):
            ga = g / b.data
            return unbroadcast(ga, a.shape), unbroadcast(-ga * a.data / b.data, b.shape)

        return Tensor._make(a.data / b.data, (a, b), backward, "div")

    def __pow__(self, exponent: float) -> Tensor:
        if isinstance(exponent, Tensor) or np.ndim(exponent) != 0:
            raise TypeError("only scalar (non-Tensor) exponents are supported")
        a, p = self, float(exponent)

        def backward(g: np.ndarray):
            return (g * p * a.data ** (p - 1.0),)

        return Tensor._make(a.data**p, (a,), backward, "pow")

    def __radd__(self, other: ArrayLike) -> Tensor:
        return self + other

    def __rsub__(self, other: ArrayLike) -> Tensor:
        return self._lift(other) - self

    def __rmul__(self, other: ArrayLike) -> Tensor:
        return self * other

    def __rtruediv__(self, other: ArrayLike) -> Tensor:
        return self._lift(other) / self

    def __matmul__(self, other: ArrayLike) -> Tensor:
        a, b = self, self._lift(other)
        if a.ndim == 0 or b.ndim == 0:
            raise ValueError("matmul does not accept 0-d operands")

        def backward(g: np.ndarray):
            # Promote 1-D operands to matrices exactly as np.matmul does, then undo it.
            a2 = a.data[None, :] if a.ndim == 1 else a.data
            b2 = b.data[:, None] if b.ndim == 1 else b.data
            g2 = g[..., None] if b.ndim == 1 else g  # re-insert the dropped column axis
            g2 = g2[..., None, :] if a.ndim == 1 else g2  # ... and the dropped row axis
            ga = g2 @ np.swapaxes(b2, -1, -2)
            gb = np.swapaxes(a2, -1, -2) @ g2
            if a.ndim == 1:
                ga = ga.squeeze(-2)
            if b.ndim == 1:
                gb = gb.squeeze(-1)
            return unbroadcast(ga, a.shape), unbroadcast(gb, b.shape)

        return Tensor._make(a.data @ b.data, (a, b), backward, "matmul")

    def __rmatmul__(self, other: ArrayLike) -> Tensor:
        return self._lift(other) @ self

    # ------------------------------------------------------------------ unary math
    def exp(self) -> Tensor:
        out = np.exp(self.data)
        return Tensor._make(out, (self,), lambda g: (g * out,), "exp")

    def log(self) -> Tensor:
        a = self.data
        return Tensor._make(np.log(a), (self,), lambda g: (g / a,), "log")

    def relu(self) -> Tensor:
        mask = self.data > 0
        return Tensor._make(np.maximum(self.data, 0), (self,), lambda g: (g * mask,), "relu")

    def tanh(self) -> Tensor:
        out = np.tanh(self.data)
        return Tensor._make(out, (self,), lambda g: (g * (1.0 - out * out),), "tanh")

    def sigmoid(self) -> Tensor:
        x = self.data
        # Numerically stable: never exponentiate a large positive number.
        e = np.exp(-np.abs(x))
        out = np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e)).astype(x.dtype, copy=False)
        return Tensor._make(out, (self,), lambda g: (g * out * (1.0 - out),), "sigmoid")

    # ------------------------------------------------------------------ reductions
    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        axes, shape = _normalize_axes(axis, self.ndim), self.shape

        def backward(g: np.ndarray):
            return (_expand_reduced(g, axes, keepdims, shape),)

        return Tensor._make(self.data.sum(axis=axes, keepdims=keepdims), (self,), backward, "sum")

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        axes, shape = _normalize_axes(axis, self.ndim), self.shape
        count = int(np.prod([shape[a] for a in axes])) if axes else 1

        def backward(g: np.ndarray):
            return (_expand_reduced(g / count, axes, keepdims, shape),)

        out = self.data.mean(axis=axes, keepdims=keepdims)
        return Tensor._make(out, (self,), backward, "mean")

    def max(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        """Maximum over ``axis``; on ties the gradient is split evenly (``torch.amax``)."""
        axes, shape = _normalize_axes(axis, self.ndim), self.shape
        kept = self.data.max(axis=axes, keepdims=True)
        mask = self.data == kept
        count = mask.sum(axis=axes, keepdims=True)

        def backward(g: np.ndarray):
            g = g if keepdims or not axes else np.expand_dims(g, axes)
            return (mask * (g / count),)

        out = kept if keepdims else kept.reshape([n for i, n in enumerate(shape) if i not in axes])
        return Tensor._make(out, (self,), backward, "max")

    # ------------------------------------------------------------------ shape ops
    def reshape(self, *shape: int | tuple[int, ...]) -> Tensor:
        target = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        src = self.shape
        return Tensor._make(
            self.data.reshape(target), (self,), lambda g: (g.reshape(src),), "reshape"
        )

    def transpose(self, *axes: int | tuple[int, ...]) -> Tensor:
        """Permute axes (NumPy semantics); no arguments reverses them."""
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        perm = tuple(axes) if axes else tuple(reversed(range(self.ndim)))
        inverse = tuple(np.argsort(perm))
        return Tensor._make(
            self.data.transpose(perm), (self,), lambda g: (g.transpose(inverse),), "transpose"
        )

    @property
    def T(self) -> Tensor:
        return self.transpose()

    def __getitem__(self, index: Any) -> Tensor:
        if isinstance(index, Tensor):
            index = index.data.astype(np.intp)
        shape, dtype, basic = self.shape, self.dtype, _is_basic_index(index)

        def backward(g: np.ndarray):
            full = np.zeros(shape, dtype=dtype)
            if basic:  # basic slicing selects every element at most once
                full[index] = g
            else:  # advanced indexing may repeat elements: scatter-add
                np.add.at(full, index, g)
            return (full,)

        return Tensor._make(self.data[index], (self,), backward, "getitem")
