"""Autodiff engine semantics: chain rule, accumulation, graph bookkeeping, dtypes."""

import numpy as np
import pytest

from tensorgrad import Tensor, is_grad_enabled, no_grad
from tensorgrad.tensor import unbroadcast


def test_shared_node_gets_summed_gradient():
    x = Tensor([1.0, 2.0, -3.0], requires_grad=True)
    y = x * x + x  # x used three times: dy/dx = 2x + 1
    y.sum().backward()
    np.testing.assert_allclose(x.grad, 2 * x.data + 1)


def test_diamond_graph():
    x = Tensor(2.0, requires_grad=True)
    a, b = x * 3, x.exp()
    (a * b).backward()  # d/dx (3x e^x) = 3 e^x (1 + x)
    np.testing.assert_allclose(x.grad, 3 * np.exp(2.0) * 3)


def test_gradients_accumulate_until_zeroed():
    x = Tensor([1.0, 2.0], requires_grad=True)
    (x * 2).sum().backward()
    (x * 3).sum().backward()
    np.testing.assert_allclose(x.grad, [5.0, 5.0])
    x.zero_grad()
    assert x.grad is None


def test_only_leaves_keep_grad():
    x = Tensor(np.ones(3), requires_grad=True)
    h = x * 2
    h.sum().backward()
    assert h.grad is None and x.grad is not None


def test_constants_receive_no_gradient():
    x = Tensor([1.0, 2.0], requires_grad=True)
    c = Tensor([5.0, 6.0])
    (x * c).sum().backward()
    assert c.grad is None
    np.testing.assert_allclose(x.grad, c.data)


def test_no_grad_records_nothing():
    x = Tensor([1.0, 2.0], requires_grad=True)
    with no_grad():
        assert not is_grad_enabled()
        y = (x * 2).exp()
    assert is_grad_enabled()
    assert not y.requires_grad and y._parents == ()


def test_backward_errors():
    with pytest.raises(RuntimeError):
        Tensor([1.0]).backward()  # does not require grad
    x = Tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(RuntimeError):
        (x * 2).backward()  # non-scalar without explicit grad
    with pytest.raises(ValueError):
        (x * 2).backward(np.ones(3))


def test_backward_with_explicit_upstream_gradient():
    x = Tensor([1.0, 2.0], requires_grad=True)
    (x * 3).backward(np.array([1.0, 10.0]))
    np.testing.assert_allclose(x.grad, [3.0, 30.0])


def test_deep_graph_does_not_hit_recursion_limit():
    x = Tensor(1.0, requires_grad=True)
    y = x
    for _ in range(5000):
        y = y + 1.0
    y.backward()
    assert x.grad == 1.0


@pytest.mark.parametrize(
    "src, dst",
    [((2, 3, 4), (3, 4)), ((2, 3, 4), (4,)), ((2, 3, 4), (3, 1)), ((3, 4), (1, 1)),
     ((3, 4), (3, 4)), ((5,), ())],
)
def test_unbroadcast_is_adjoint_of_broadcasting(src, dst):
    rng = np.random.default_rng(0)
    g, x = rng.standard_normal(src), rng.standard_normal(dst)
    reduced = unbroadcast(g, dst)
    assert reduced.shape == dst
    # <broadcast(x), g> == <x, unbroadcast(g)>
    np.testing.assert_allclose(np.sum(np.broadcast_to(x, src) * g), np.sum(x * reduced))


def test_float32_stays_float32():
    x = Tensor(np.ones((2, 3), dtype=np.float32), requires_grad=True)
    y = ((2.5 * x - 1) ** 2 / 3.0).relu().mean()
    y.backward()
    assert y.dtype == np.float32 and x.grad.dtype == np.float32


def test_python_scalars_default_to_float64_and_ints_are_cast():
    assert Tensor(3).dtype == np.float64
    assert Tensor([1, 2]).dtype == np.float64


def test_numpy_array_on_the_left_dispatches_to_tensor():
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = np.array([3.0, 4.0]) * x + np.array([1.0, 1.0])
    assert isinstance(y, Tensor)
    y.sum().backward()
    np.testing.assert_allclose(x.grad, [3.0, 4.0])


def test_tensor_exponent_rejected():
    with pytest.raises(TypeError):
        Tensor([2.0]) ** Tensor([2.0])


def test_detach_cuts_the_graph():
    x = Tensor([1.0], requires_grad=True)
    d = (x * 2).detach()
    assert not d.requires_grad
    (x * d).sum().backward()
    np.testing.assert_allclose(x.grad, [2.0])


def test_advanced_indexing_scatter_adds_repeated_indices():
    x = Tensor(np.arange(4.0), requires_grad=True)
    x[np.array([0, 0, 3])].sum().backward()
    np.testing.assert_allclose(x.grad, [2.0, 0.0, 0.0, 1.0])
