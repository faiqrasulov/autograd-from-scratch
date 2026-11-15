"""Forward + backward parity with torch.autograd, op by op and layer by layer (float64)."""

import numpy as np
import pytest

from tensorgrad import Tensor, nn
from tensorgrad import functional as F
from tensorgrad.parity import OP_CASES, compare_op, to_torch

torch = pytest.importorskip("torch")
ATOL = 1e-12


@pytest.mark.parametrize("case", OP_CASES, ids=[c.name for c in OP_CASES])
def test_op_matches_torch(case):
    fwd, grad = compare_op(case, seed=2)
    assert fwd < ATOL and grad < ATOL


def _module_parity(model: nn.Sequential, x: np.ndarray) -> None:
    """Same weights + input + upstream grad -> same output, input grad and weight grads."""
    tmodel = to_torch(model)
    xt = Tensor(x, requires_grad=True)
    out = model(xt)
    upstream = np.random.default_rng(1).standard_normal(out.shape)
    out.backward(upstream)

    xp = torch.tensor(x, requires_grad=True)
    tout = tmodel(xp)
    tout.backward(torch.from_numpy(upstream))

    np.testing.assert_allclose(out.data, tout.detach().numpy(), atol=ATOL)
    np.testing.assert_allclose(xt.grad, xp.grad.numpy(), atol=ATOL)
    tgrads = {k: p.grad.numpy() for k, p in tmodel.named_parameters()}
    for name, p in model.named_parameters():
        np.testing.assert_allclose(p.grad, tgrads[name], atol=ATOL, err_msg=name)


def test_linear_layer_parity():
    x = np.random.default_rng(0).standard_normal((7, 5))
    _module_parity(nn.Sequential(nn.Linear(5, 3, rng=0)), x)


@pytest.mark.parametrize("stride, padding", [(1, 1), (2, 0), (2, 2)])
def test_conv2d_layer_parity(stride, padding):
    x = np.random.default_rng(0).standard_normal((3, 2, 8, 7))
    _module_parity(nn.Sequential(nn.Conv2d(2, 4, 3, stride, padding, rng=0)), x)


@pytest.mark.parametrize("k, stride", [(2, None), (3, 1), (3, 2)])
def test_maxpool_layer_parity(k, stride):
    x = np.random.default_rng(0).standard_normal((2, 3, 7, 7))
    _module_parity(nn.Sequential(nn.MaxPool2d(k, stride)), x)


def test_full_cnn_parity_including_relu_zero_ties():
    # After ReLU many pooling windows are all-zero ties: arg-max conventions must agree.
    model = nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1, rng=0), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
        nn.Linear(4 * 4 * 4, 10, rng=1), nn.Tanh(),
    )
    x = np.random.default_rng(0).standard_normal((5, 1, 8, 8))
    _module_parity(model, x)


def test_cross_entropy_training_step_parity():
    rng = np.random.default_rng(0)
    model = nn.Sequential(nn.Linear(6, 8, rng=0), nn.ReLU(), nn.Linear(8, 3, rng=1))
    tmodel = to_torch(model)
    x, y = rng.standard_normal((10, 6)), rng.integers(0, 3, 10)
    loss = F.cross_entropy(model(Tensor(x)), y)
    loss.backward()
    tloss = torch.nn.functional.cross_entropy(tmodel(torch.from_numpy(x)), torch.from_numpy(y))
    tloss.backward()
    assert abs(loss.item() - tloss.item()) < ATOL
    for (name, p), (_, tp) in zip(model.named_parameters(), tmodel.named_parameters(),
                                  strict=True):
        np.testing.assert_allclose(p.grad, tp.grad.numpy(), atol=ATOL, err_msg=name)


def test_to_torch_rejects_layers_without_a_deterministic_mirror():
    with pytest.raises(TypeError):
        to_torch(nn.Sequential(nn.Dropout(0.5)))
