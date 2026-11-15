"""Optimizer update rules vs torch.optim, step by step, plus framework-free sanity checks."""

import numpy as np
import pytest

from tensorgrad import Tensor, optim
from tensorgrad.nn import Parameter

STEPS = 6


def _trajectories(make_tg, make_torch):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    p0 = [rng.standard_normal((4, 3)), rng.standard_normal(3)]
    grads = [[rng.standard_normal(p.shape) for p in p0] for _ in range(STEPS)]

    params = [Parameter(p.copy()) for p in p0]
    tparams = [torch.tensor(p.copy(), requires_grad=True) for p in p0]
    opt, topt = make_tg(params), make_torch(tparams)
    for step_grads in grads:
        for p, tp, g in zip(params, tparams, step_grads, strict=True):
            p.grad, tp.grad = g.copy(), torch.from_numpy(g.copy())
        opt.step()
        topt.step()
        for p, tp in zip(params, tparams, strict=True):
            np.testing.assert_allclose(p.data, tp.detach().numpy(), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(
    "kwargs",
    [dict(lr=0.1), dict(lr=0.05, momentum=0.9), dict(lr=0.05, momentum=0.9, weight_decay=1e-2),
     dict(lr=0.05, momentum=0.8, nesterov=True, weight_decay=1e-3)],
)
def test_sgd_matches_torch(kwargs):
    import torch

    _trajectories(lambda ps: optim.SGD(ps, **kwargs), lambda ps: torch.optim.SGD(ps, **kwargs))


@pytest.mark.parametrize(
    "kwargs",
    [dict(lr=1e-2), dict(lr=3e-3, betas=(0.8, 0.95), eps=1e-6, weight_decay=1e-2)],
)
def test_adam_matches_torch(kwargs):
    import torch

    _trajectories(lambda ps: optim.Adam(ps, **kwargs), lambda ps: torch.optim.Adam(ps, **kwargs))


def test_parameters_without_grad_are_skipped():
    a, b = Parameter(np.ones(2)), Parameter(np.ones(2))
    opt = optim.Adam([a, b], lr=0.1)
    a.grad = np.ones(2)
    opt.step()
    np.testing.assert_array_equal(b.data, 1.0)
    assert np.all(a.data < 1.0)


@pytest.mark.parametrize("make", [lambda ps: optim.SGD(ps, lr=0.1, momentum=0.9),
                                  lambda ps: optim.Adam(ps, lr=0.1)])
def test_minimises_a_quadratic(make):
    target = np.array([3.0, -2.0])
    w = Parameter(np.zeros(2))
    opt = make([w])
    for _ in range(300):
        opt.zero_grad()
        ((w - Tensor(target)) ** 2).sum().backward()
        opt.step()
    np.testing.assert_allclose(w.data, target, atol=1e-2)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        optim.SGD([], lr=0.1)
    with pytest.raises(ValueError):
        optim.SGD([Parameter(np.ones(1))], lr=0.0)
    with pytest.raises(ValueError):
        optim.SGD([Parameter(np.ones(1))], lr=0.1, nesterov=True)
