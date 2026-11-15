"""Neural-network operations built on :class:`~tensorgrad.tensor.Tensor`.

``log_softmax``, ``cross_entropy``, ``conv2d`` and ``max_pool2d`` are *fused* primitives with
hand-written backward passes: composing them from smaller ops would also be correct, but slower
and (for the softmax family) less numerically stable.
"""

from __future__ import annotations

import numpy as np

from .tensor import Tensor


def relu(x: Tensor) -> Tensor:
    return x.relu()


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


def sigmoid(x: Tensor) -> Tensor:
    return x.sigmoid()


def _log_softmax(z: np.ndarray, axis: int) -> np.ndarray:
    shifted = z - z.max(axis=axis, keepdims=True)  # log-sum-exp trick: exp() never overflows
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    """``x - logsumexp(x)`` along ``axis``, computed stably."""
    out = _log_softmax(x.data, axis)

    def backward(g: np.ndarray):
        # d/dx_j of sum_i g_i (x_i - lse) = g_j - softmax_j * sum_i g_i
        return (g - np.exp(out) * g.sum(axis=axis, keepdims=True),)

    return Tensor._make(out, (x,), backward, "log_softmax")


def cross_entropy(logits: Tensor, targets: np.ndarray, reduction: str = "mean") -> Tensor:
    """Softmax cross-entropy between ``logits`` (N, C) and integer class ``targets`` (N,).

    The gradient w.r.t. the logits is the famous ``softmax(logits) - one_hot(targets)``.
    """
    targets = np.asarray(targets.data if isinstance(targets, Tensor) else targets)
    if logits.ndim != 2 or targets.shape != (logits.shape[0],):
        raise ValueError("expected logits (N, C) and targets (N,)")
    if not np.issubdtype(targets.dtype, np.integer):
        raise TypeError("targets must be integer class indices")
    if reduction not in ("mean", "sum"):
        raise ValueError("reduction must be 'mean' or 'sum'")
    n = logits.shape[0]
    rows = np.arange(n)
    logp = _log_softmax(logits.data, axis=1)
    total = -logp[rows, targets].sum()
    scale = 1.0 / n if reduction == "mean" else 1.0

    def backward(g: np.ndarray):
        grad = np.exp(logp)
        grad[rows, targets] -= 1.0
        return (grad * (g * scale),)

    out = np.asarray(total * scale, dtype=logits.dtype)
    return Tensor._make(out, (logits,), backward, "cross_entropy")


# ---------------------------------------------------------------------- convolution machinery
def conv_output_size(size: int, kernel: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - kernel) // stride + 1


def im2col(
    x: np.ndarray, kh: int, kw: int, stride: int = 1, padding: int = 0
) -> tuple[np.ndarray, int, int]:
    """Unfold every receptive field of ``x`` (N, C, H, W) into a row.

    Returns ``cols`` of shape (N*OH*OW, C*kh*kw) with rows ordered (n, oh, ow) and columns
    ordered (c, i, j), plus the output height/width. Convolution then becomes one matrix
    multiply. The loop runs over the kh*kw kernel offsets only, never over pixels.
    """
    n, c, h, w = x.shape
    oh, ow = conv_output_size(h, kh, stride, padding), conv_output_size(w, kw, stride, padding)
    if oh <= 0 or ow <= 0:
        raise ValueError("kernel larger than the (padded) input")
    if padding:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    cols = np.empty((n, c, kh, kw, oh, ow), dtype=x.dtype)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j] = x[:, :, i : i + stride * oh : stride, j : j + stride * ow : stride]
    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(n * oh * ow, c * kh * kw), oh, ow


def col2im(
    cols: np.ndarray,
    x_shape: tuple[int, int, int, int],
    kh: int,
    kw: int,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """Adjoint of :func:`im2col`: scatter-add rows back into an (N, C, H, W) image.

    Overlapping receptive fields receive the sum of their contributions, which is exactly the
    gradient of the unfold operation.
    """
    n, c, h, w = x_shape
    oh, ow = conv_output_size(h, kh, stride, padding), conv_output_size(w, kw, stride, padding)
    cols = cols.reshape(n, oh, ow, c, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    out = np.zeros((n, c, h + 2 * padding, w + 2 * padding), dtype=cols.dtype)
    for i in range(kh):
        rows = slice(i, i + stride * oh, stride)
        for j in range(kw):
            out[:, :, rows, j : j + stride * ow : stride] += cols[:, :, i, j]
    return out[:, :, padding : padding + h, padding : padding + w] if padding else out


def conv2d(
    x: Tensor, weight: Tensor, bias: Tensor | None = None, stride: int = 1, padding: int = 0
) -> Tensor:
    """2-D cross-correlation (PyTorch's ``conv2d``) via im2col + matmul.

    x: (N, C, H, W), weight: (F, C, kh, kw), bias: (F,) -> (N, F, OH, OW).
    """
    n, c, _, _ = x.shape
    f, wc, kh, kw = weight.shape
    if c != wc:
        raise ValueError(f"input has {c} channels but weight expects {wc}")
    cols, oh, ow = im2col(x.data, kh, kw, stride, padding)
    wmat = weight.data.reshape(f, -1)
    out = cols @ wmat.T  # (N*OH*OW, F)
    if bias is not None:
        out += bias.data
    out = np.ascontiguousarray(out.reshape(n, oh, ow, f).transpose(0, 3, 1, 2))
    x_shape = x.shape

    def backward(g: np.ndarray):
        g2 = g.transpose(0, 2, 3, 1).reshape(-1, f)  # (N*OH*OW, F), same row order as cols
        dx = col2im(g2 @ wmat, x_shape, kh, kw, stride, padding)
        dw = (g2.T @ cols).reshape(weight.shape)
        return (dx, dw) if bias is None else (dx, dw, g2.sum(axis=0))

    parents = (x, weight) if bias is None else (x, weight, bias)
    return Tensor._make(out, parents, backward, "conv2d")


def max_pool2d(x: Tensor, kernel_size: int, stride: int | None = None) -> Tensor:
    """Max pooling over ``kernel_size`` windows (no padding).

    The gradient flows only to the arg-max of each window; on ties the first element in
    row-major window order wins, which is also what PyTorch's CPU kernel does.
    """
    k, s = kernel_size, stride or kernel_size
    n, c, h, w = x.shape
    flat_shape = (n * c, 1, h, w)
    cols, oh, ow = im2col(x.data.reshape(flat_shape), k, k, s, 0)  # (N*C*OH*OW, k*k)
    rows = np.arange(cols.shape[0])
    argmax = cols.argmax(axis=1)
    out = cols[rows, argmax].reshape(n, c, oh, ow)

    def backward(g: np.ndarray):
        dcols = np.zeros_like(cols)
        dcols[rows, argmax] = g.reshape(-1)
        return (col2im(dcols, flat_shape, k, k, s, 0).reshape(n, c, h, w),)

    return Tensor._make(out, (x,), backward, "max_pool2d")


def dropout(
    x: Tensor, p: float, training: bool, rng: np.random.Generator | None = None
) -> Tensor:
    """Inverted dropout: zero each element with probability ``p`` and scale the rest by
    ``1/(1-p)`` so the expected activation is unchanged; identity in eval mode."""
    if not 0.0 <= p < 1.0:
        raise ValueError("dropout probability must be in [0, 1)")
    if not training or p == 0.0:
        return x
    rng = rng if rng is not None else np.random.default_rng()
    mask = (rng.random(x.shape) >= p).astype(x.dtype) / (1.0 - p)
    return x * Tensor(mask)
