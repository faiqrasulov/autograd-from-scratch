"""NN ops against independent references (naive loops, closed forms) and edge cases."""

import numpy as np
import pytest

from tensorgrad import Tensor
from tensorgrad import functional as F


def naive_conv2d(x, w, b, stride, padding):
    """Direct 6-loop cross-correlation, the textbook definition."""
    n, c, h, wd = x.shape
    f, _, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    oh, ow = (h + 2 * padding - kh) // stride + 1, (wd + 2 * padding - kw) // stride + 1
    out = np.zeros((n, f, oh, ow))
    for i in range(n):
        for o in range(f):
            for r in range(oh):
                for s in range(ow):
                    patch = xp[i, :, r * stride : r * stride + kh, s * stride : s * stride + kw]
                    out[i, o, r, s] = np.sum(patch * w[o]) + b[o]
    return out


@pytest.mark.parametrize("stride, padding", [(1, 0), (1, 1), (2, 1), (3, 2)])
def test_conv2d_matches_naive_loops(stride, padding):
    rng = np.random.default_rng(0)
    x, w, b = rng.standard_normal((2, 3, 7, 6)), rng.standard_normal((4, 3, 3, 3)), rng.random(4)
    out = F.conv2d(Tensor(x), Tensor(w), Tensor(b), stride=stride, padding=padding)
    np.testing.assert_allclose(out.data, naive_conv2d(x, w, b, stride, padding), atol=1e-12)


@pytest.mark.parametrize("k, stride, padding", [(3, 1, 0), (3, 2, 1), (2, 2, 0), (4, 3, 2)])
def test_col2im_is_the_adjoint_of_im2col(k, stride, padding):
    rng = np.random.default_rng(0)
    shape = (2, 3, 7, 8)
    x = rng.standard_normal(shape)
    cols, _, _ = F.im2col(x, k, k, stride, padding)
    y = rng.standard_normal(cols.shape)
    # <im2col(x), y> == <x, col2im(y)>  (dot-product test)
    np.testing.assert_allclose(np.sum(cols * y),
                               np.sum(x * F.col2im(y, shape, k, k, stride, padding)))


def test_max_pool_matches_reshape_reference():
    x = np.random.default_rng(0).standard_normal((2, 3, 6, 4))
    ref = x.reshape(2, 3, 3, 2, 2, 2).max(axis=(3, 5))
    np.testing.assert_array_equal(F.max_pool2d(Tensor(x), 2).data, ref)


def test_max_pool_tie_sends_gradient_to_first_element():
    x = Tensor(np.zeros((1, 1, 2, 2)), requires_grad=True)
    F.max_pool2d(x, 2).sum().backward()
    np.testing.assert_array_equal(x.grad[0, 0], [[1.0, 0.0], [0.0, 0.0]])


def test_log_softmax_is_stable_for_huge_logits():
    z = Tensor(np.array([[1000.0, 0.0, -1000.0]]), requires_grad=True)
    out = F.log_softmax(z)
    assert np.all(np.isfinite(out.data))
    np.testing.assert_allclose(out.data[0, 0], 0.0, atol=1e-12)
    out.sum().backward()
    assert np.all(np.isfinite(z.grad))


def test_cross_entropy_closed_form_and_uniform_logits():
    n, c = 4, 10
    loss = F.cross_entropy(Tensor(np.zeros((n, c))), np.arange(n))
    np.testing.assert_allclose(loss.item(), np.log(c))  # uniform prediction -> log(C)


def test_cross_entropy_validates_inputs():
    with pytest.raises(ValueError):
        F.cross_entropy(Tensor(np.zeros((3, 2))), np.array([0, 1]))
    with pytest.raises(TypeError):
        F.cross_entropy(Tensor(np.zeros((2, 2))), np.array([0.0, 1.0]))


def test_sigmoid_extremes_do_not_overflow():
    out = Tensor(np.array([-1000.0, 0.0, 1000.0])).sigmoid()
    np.testing.assert_allclose(out.data, [0.0, 0.5, 1.0])


def test_dropout_train_and_eval():
    rng = np.random.default_rng(0)
    x = Tensor(np.ones((200, 50)), requires_grad=True)
    assert F.dropout(x, 0.3, training=False) is x
    y = F.dropout(x, 0.3, training=True, rng=rng)
    kept = y.data != 0
    np.testing.assert_allclose(y.data[kept], 1 / 0.7)
    assert abs(kept.mean() - 0.7) < 0.02  # keep-rate ~ 1 - p
    y.sum().backward()
    np.testing.assert_allclose(x.grad, y.data)  # d(x * mask)/dx = mask
    with pytest.raises(ValueError):
        F.dropout(x, 1.0, training=True)
