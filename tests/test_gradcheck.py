"""Finite-difference gradient checks: every registered op plus composites (no PyTorch needed)."""

import numpy as np
import pytest

from tensorgrad import Tensor
from tensorgrad import functional as F
from tensorgrad.gradcheck import gradcheck
from tensorgrad.parity import OP_CASES

TOL = 1e-5


@pytest.mark.parametrize("case", OP_CASES, ids=[c.name for c in OP_CASES])
def test_op_matches_finite_differences(case):
    inputs = case.make_inputs(np.random.default_rng(1))
    assert gradcheck(case.tg, inputs, seed=1) < TOL


def test_two_layer_network_end_to_end():
    rng = np.random.default_rng(0)
    x, w1, b1, w2 = (rng.standard_normal(s) for s in [(5, 4), (4, 6), (6,), (6, 3)])
    targets = np.array([0, 2, 1, 1, 0])

    def net(x, w1, b1, w2):
        return F.cross_entropy((x @ w1 + b1).tanh() @ w2, targets)

    assert gradcheck(net, [x, w1, b1, w2]) < TOL


def test_cross_entropy_equals_composed_log_softmax():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((6, 4))
    targets = np.array([0, 1, 2, 3, 3, 1])
    rows = np.arange(6)

    def composed(z):
        return -F.log_softmax(z, axis=1)[rows, targets].mean()

    np.testing.assert_allclose(F.cross_entropy(Tensor(logits), targets).item(),
                               composed(Tensor(logits)).item(), rtol=1e-12)
    assert gradcheck(composed, [logits]) < TOL


def test_conv_pool_stack():
    rng = np.random.default_rng(0)
    x, w = rng.standard_normal((2, 1, 6, 6)), rng.standard_normal((3, 1, 3, 3))

    def stack(x, w):
        return F.max_pool2d(F.conv2d(x, w, padding=1).tanh(), 2)

    assert gradcheck(stack, [x, w]) < TOL
