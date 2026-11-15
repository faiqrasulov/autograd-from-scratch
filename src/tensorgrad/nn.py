"""Layers with a PyTorch-like ``Module`` / ``Parameter`` registry.

Parameter names and layouts deliberately match ``torch.nn`` (``Linear.weight`` is
(out_features, in_features), ``Sequential`` children are named "0", "1", ...), so a
``state_dict`` can be copied 1:1 into the equivalent PyTorch model for parity checks.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from . import functional as F
from .tensor import DEFAULT_DTYPE, Tensor


class Parameter(Tensor):
    """A leaf tensor that a :class:`Module` registers as trainable."""

    __slots__ = ()

    def __init__(self, data: Any, dtype: Any = None) -> None:
        super().__init__(data, requires_grad=True, dtype=dtype)


class Module:
    """Base class: tracks child modules and parameters assigned as attributes."""

    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.extra_repr()})"

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        for name, p in self._parameters.items():
            yield prefix + name, p
        for name, m in self._modules.items():
            yield from m.named_parameters(prefix + name + ".")

    def parameters(self) -> list[Parameter]:
        return [p for _, p in self.named_parameters()]

    def num_parameters(self) -> int:
        return sum(p.size for p in self.parameters())

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def train(self, mode: bool = True) -> Module:
        object.__setattr__(self, "training", mode)
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self) -> Module:
        return self.train(False)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {name: p.data.copy() for name, p in self.named_parameters()}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        params = dict(self.named_parameters())
        if params.keys() != state.keys():
            raise KeyError(f"state_dict keys mismatch: {sorted(params)} vs {sorted(state)}")
        for name, p in params.items():
            value = np.asarray(state[name])
            if value.shape != p.shape:
                raise ValueError(f"{name}: shape {value.shape} != {p.shape}")
            p.data = value.astype(p.dtype, copy=True)

    def to(self, dtype: Any) -> Module:
        """Cast all parameters to ``dtype`` in place (e.g. np.float32 for faster training)."""
        for p in self.parameters():
            p.data = p.data.astype(dtype)
            p.grad = None
        return self


def kaiming_normal(
    shape: tuple[int, ...], fan_in: int, rng: np.random.Generator, dtype: Any = DEFAULT_DTYPE
) -> np.ndarray:
    """He et al. (2015) init for ReLU nets: N(0, 2 / fan_in) keeps activation variance stable."""
    return (rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(dtype)


def _rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


class Linear(Module):
    """``y = x @ W.T + b`` with Kaiming-normal weights and zero bias."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rng: np.random.Generator | int | None = None,
        dtype: Any = DEFAULT_DTYPE,
    ) -> None:
        super().__init__()
        shape = (out_features, in_features)
        self.weight = Parameter(kaiming_normal(shape, in_features, _rng(rng), dtype))
        self.bias = Parameter(np.zeros(out_features, dtype=dtype)) if bias else None

    def extra_repr(self) -> str:
        out_f, in_f = self.weight.shape
        return f"{in_f}, {out_f}" + ("" if self.bias is not None else ", bias=False")

    def forward(self, x: Tensor) -> Tensor:
        y = x @ self.weight.T
        return y + self.bias if self.bias is not None else y


class Conv2d(Module):
    """2-D convolution (im2col) with square kernels, stride and zero padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        rng: np.random.Generator | int | None = None,
        dtype: Any = DEFAULT_DTYPE,
    ) -> None:
        super().__init__()
        self.stride, self.padding = stride, padding
        shape = (out_channels, in_channels, kernel_size, kernel_size)
        fan_in = in_channels * kernel_size * kernel_size
        self.weight = Parameter(kaiming_normal(shape, fan_in, _rng(rng), dtype))
        self.bias = Parameter(np.zeros(out_channels, dtype=dtype)) if bias else None

    def extra_repr(self) -> str:
        f, c, k, _ = self.weight.shape
        return f"{c}, {f}, k={k}, stride={self.stride}, pad={self.padding}"

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding)


class MaxPool2d(Module):
    def __init__(self, kernel_size: int, stride: int | None = None) -> None:
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride

    def extra_repr(self) -> str:
        return f"{self.kernel_size}" + (f", stride={self.stride}" if self.stride else "")

    def forward(self, x: Tensor) -> Tensor:
        return F.max_pool2d(x, self.kernel_size, self.stride)


class Flatten(Module):
    """Flatten all dimensions except the batch dimension."""

    def forward(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], -1)


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Dropout(Module):
    """Inverted dropout, active only in ``train()`` mode."""

    def __init__(self, p: float = 0.5, rng: np.random.Generator | int | None = None) -> None:
        super().__init__()
        self.p = p
        self.rng = _rng(rng)

    def extra_repr(self) -> str:
        return f"p={self.p}"

    def forward(self, x: Tensor) -> Tensor:
        return F.dropout(x, self.p, self.training, self.rng)


class Sequential(Module):
    """Chain modules; children are registered as "0", "1", ... like ``torch.nn.Sequential``."""

    def __init__(self, *layers: Module) -> None:
        super().__init__()
        for i, layer in enumerate(layers):
            setattr(self, str(i), layer)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def __getitem__(self, i: int) -> Module:
        return list(self._modules.values())[i]

    def __len__(self) -> int:
        return len(self._modules)

    def __repr__(self) -> str:
        return " -> ".join(repr(m) for m in self._modules.values())

    def forward(self, x: Tensor) -> Tensor:
        for layer in self._modules.values():
            x = layer(x)
        return x
