"""Op-by-op and model-level comparison against PyTorch (the reference implementation).

PyTorch is imported lazily: the library itself never needs it, only these checks do.
``OP_CASES`` is the single registry used by the unit tests *and* by the report in ``make run``,
so the README table and the test-suite check exactly the same things.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import functional as F
from . import nn
from .gradcheck import gradcheck
from .tensor import Tensor


def require_torch() -> Any:
    try:
        import torch
    except ImportError as err:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "PyTorch is needed for parity checks: pip install -e '.[dev]' "
            "(CPU wheel: pip install torch --index-url https://download.pytorch.org/whl/cpu)"
        ) from err
    return torch


def _tF() -> Any:
    return require_torch().nn.functional


def _ce_targets_torch() -> Any:
    return require_torch().from_numpy(_CE_TARGETS)


# ---------------------------------------------------------------------------- op registry
InputMaker = Callable[[np.random.Generator], list[np.ndarray]]


@dataclass(frozen=True)
class OpCase:
    """One op under test: how to make inputs, and the same computation in both frameworks."""

    name: str
    make_inputs: InputMaker
    tg: Callable[..., Tensor]
    pt: Callable[..., Any]


def _normal(*shapes: tuple[int, ...]) -> InputMaker:
    return lambda rng: [rng.standard_normal(s) for s in shapes]


def _positive(*shapes: tuple[int, ...]) -> InputMaker:
    return lambda rng: [rng.uniform(0.5, 2.0, s) for s in shapes]


def _away_from_zero(*shapes: tuple[int, ...]) -> InputMaker:
    """Keep inputs off the ReLU kink so finite differences are well defined."""
    return lambda rng: [
        rng.choice([-1.0, 1.0], s) * rng.uniform(0.1, 2.0, s) for s in shapes
    ]


def _normal_then_positive(a: tuple[int, ...], b: tuple[int, ...]) -> InputMaker:
    return lambda rng: [rng.standard_normal(a), rng.uniform(0.5, 2.0, b)]


_CE_TARGETS = np.array([0, 3, 1, 2, 3])

OP_CASES: list[OpCase] = [
    OpCase("add", _normal((3, 4), (3, 4)), lambda a, b: a + b, lambda a, b: a + b),
    OpCase("add (broadcast 2x3x4 + 4)", _normal((2, 3, 4), (4,)),
           lambda a, b: a + b, lambda a, b: a + b),
    OpCase("sub (broadcast 3x1 - 1x4)", _normal((3, 1), (1, 4)),
           lambda a, b: a - b, lambda a, b: a - b),
    OpCase("neg", _normal((3, 4)), lambda a: -a, lambda a: -a),
    OpCase("mul (broadcast 2x3x4 * 3x1)", _normal((2, 3, 4), (3, 1)),
           lambda a, b: a * b, lambda a, b: a * b),
    OpCase("div (broadcast 3x4 / 4)", _normal_then_positive((3, 4), (4,)),
           lambda a, b: a / b, lambda a, b: a / b),
    OpCase("scalar ops (2*x - x/3 + 1, 1/x)", _positive((3, 4)),
           lambda a: 2 * a - a / 3 + 1 + 1 / a, lambda a: 2 * a - a / 3 + 1 + 1 / a),
    OpCase("pow (x**2)", _normal((3, 4)), lambda a: a**2, lambda a: a**2),
    OpCase("pow (x**-1.5)", _positive((3, 4)), lambda a: a**-1.5, lambda a: a**-1.5),
    OpCase("matmul (2-D)", _normal((3, 4), (4, 5)), lambda a, b: a @ b, lambda a, b: a @ b),
    OpCase("matmul (batched, broadcast)", _normal((2, 1, 3, 4), (5, 4, 2)),
           lambda a, b: a @ b, lambda a, b: a @ b),
    OpCase("matmul (vector @ matrix)", _normal((4,), (4, 3)),
           lambda a, b: a @ b, lambda a, b: a @ b),
    OpCase("exp", _normal((3, 4)), lambda a: a.exp(), lambda a: a.exp()),
    OpCase("log", _positive((3, 4)), lambda a: a.log(), lambda a: a.log()),
    OpCase("sum (all)", _normal((3, 4)), lambda a: a.sum(), lambda a: a.sum()),
    OpCase("sum (axis=(0, 2))", _normal((2, 3, 4)),
           lambda a: a.sum(axis=(0, 2)), lambda a: a.sum(dim=(0, 2))),
    OpCase("sum (axis=1, keepdims)", _normal((2, 3, 4)),
           lambda a: a.sum(axis=1, keepdims=True), lambda a: a.sum(dim=1, keepdim=True)),
    OpCase("mean (all)", _normal((3, 4)), lambda a: a.mean(), lambda a: a.mean()),
    OpCase("mean (axis=-1)", _normal((2, 3, 4)),
           lambda a: a.mean(axis=-1), lambda a: a.mean(dim=-1)),
    OpCase("max (all)", _normal((3, 4)), lambda a: a.max(), lambda a: a.amax()),
    OpCase("max (axis=1)", _normal((2, 3, 4)),
           lambda a: a.max(axis=1), lambda a: a.amax(dim=1)),
    OpCase("max (axis=0, keepdims)", _normal((3, 4)),
           lambda a: a.max(axis=0, keepdims=True), lambda a: a.amax(dim=0, keepdim=True)),
    OpCase("reshape", _normal((2, 3, 4)),
           lambda a: a.reshape(6, -1), lambda a: a.reshape(6, -1)),
    OpCase("transpose (permute 2,0,1)", _normal((2, 3, 4)),
           lambda a: a.transpose(2, 0, 1), lambda a: a.permute(2, 0, 1)),
    OpCase("getitem (slices + None)", _normal((4, 5)),
           lambda a: a[1:, ::2, None], lambda a: a[1:, ::2, None]),
    OpCase("getitem (negative step, int)", _normal((4, 5)),
           lambda a: a[::-1, 1], lambda a: a.flip(0)[:, 1]),
    OpCase("relu", _away_from_zero((3, 4)), lambda a: a.relu(), lambda a: a.relu()),
    OpCase("tanh", _normal((3, 4)), lambda a: a.tanh(), lambda a: a.tanh()),
    OpCase("sigmoid", lambda rng: [rng.standard_normal((3, 4)) * 5],
           lambda a: a.sigmoid(), lambda a: a.sigmoid()),
    OpCase("log_softmax (axis=-1)", _normal((3, 5)),
           lambda a: F.log_softmax(a, axis=-1), lambda a: a.log_softmax(dim=-1)),
    OpCase("log_softmax (axis=0)", _normal((3, 5)),
           lambda a: F.log_softmax(a, axis=0), lambda a: a.log_softmax(dim=0)),
    OpCase("cross_entropy (mean)", _normal((5, 4)),
           lambda a: F.cross_entropy(a, _CE_TARGETS),
           lambda a: _tF().cross_entropy(a, _ce_targets_torch())),
    OpCase("cross_entropy (sum)", _normal((5, 4)),
           lambda a: F.cross_entropy(a, _CE_TARGETS, reduction="sum"),
           lambda a: _tF().cross_entropy(a, _ce_targets_torch(), reduction="sum")),
    OpCase("conv2d (stride 1, pad 1, bias)", _normal((2, 3, 5, 5), (4, 3, 3, 3), (4,)),
           lambda x, w, b: F.conv2d(x, w, b, stride=1, padding=1),
           lambda x, w, b: _tF().conv2d(x, w, b, stride=1, padding=1)),
    OpCase("conv2d (stride 2, no pad, no bias)", _normal((2, 2, 7, 6), (3, 2, 3, 2)),
           lambda x, w: F.conv2d(x, w, stride=2),
           lambda x, w: _tF().conv2d(x, w, stride=2)),
    OpCase("max_pool2d (2x2, stride 2)", _normal((2, 3, 4, 6)),
           lambda x: F.max_pool2d(x, 2), lambda x: _tF().max_pool2d(x, 2)),
    OpCase("max_pool2d (3x3, stride 1, overlapping)", _normal((2, 2, 5, 5)),
           lambda x: F.max_pool2d(x, 3, 1), lambda x: _tF().max_pool2d(x, 3, 1)),
]


def compare_op(case: OpCase, seed: int = 0) -> tuple[float, float]:
    """Run ``case`` in both frameworks (float64) with the same inputs and upstream gradient.

    Returns (max |forward difference|, max |input-gradient difference|).
    """
    torch = require_torch()
    rng = np.random.default_rng(seed)
    inputs = case.make_inputs(rng)

    tg_in = [Tensor(x, requires_grad=True) for x in inputs]
    tg_out = case.tg(*tg_in)
    upstream = np.asarray(rng.standard_normal(tg_out.shape))
    tg_out.backward(upstream)

    pt_in = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in inputs]
    pt_out = case.pt(*pt_in)
    pt_out.backward(torch.from_numpy(upstream))

    fwd = float(np.abs(tg_out.data - pt_out.detach().numpy()).max())
    grad = max(float(np.abs(t.grad - p.grad.numpy()).max()) for t, p in zip(tg_in, pt_in,
                                                                             strict=True))
    return fwd, grad


def op_report(seed: int = 0) -> list[dict[str, Any]]:
    """Finite-difference and PyTorch checks for every registered op (used by ``make run``)."""
    rows = []
    for case in OP_CASES:
        fwd, grad = compare_op(case, seed)
        rel = gradcheck(case.tg, case.make_inputs(np.random.default_rng(seed)), seed=seed)
        rows.append({"op": case.name, "max_abs_forward_diff_vs_torch": fwd,
                     "max_abs_grad_diff_vs_torch": grad, "max_rel_err_vs_finite_diff": rel})
    return rows


# ---------------------------------------------------------------------------- model mirror
def to_torch(model: nn.Sequential, dtype: Any = np.float64) -> Any:
    """Build the identical ``torch.nn.Sequential`` and copy every weight into it."""
    torch = require_torch()
    tnn = torch.nn
    layers = []
    for layer in model:
        if isinstance(layer, nn.Linear):
            out_f, in_f = layer.weight.shape
            layers.append(tnn.Linear(in_f, out_f, bias=layer.bias is not None))
        elif isinstance(layer, nn.Conv2d):
            f, c, k, _ = layer.weight.shape
            layers.append(tnn.Conv2d(c, f, k, stride=layer.stride, padding=layer.padding,
                                     bias=layer.bias is not None))
        elif isinstance(layer, nn.MaxPool2d):
            layers.append(tnn.MaxPool2d(layer.kernel_size, layer.stride))
        elif isinstance(layer, nn.Flatten):
            layers.append(tnn.Flatten())
        elif isinstance(layer, nn.ReLU):
            layers.append(tnn.ReLU())
        elif isinstance(layer, nn.Tanh):
            layers.append(tnn.Tanh())
        else:
            raise TypeError(f"no deterministic PyTorch mirror for {type(layer).__name__}")
    tdtype = torch.float64 if np.dtype(dtype) == np.float64 else torch.float32
    tmodel = tnn.Sequential(*layers).to(tdtype)
    state = {k: torch.from_numpy(v.astype(dtype)) for k, v in model.state_dict().items()}
    tmodel.load_state_dict(state, strict=True)
    return tmodel
