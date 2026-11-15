"""Module / Parameter registry, initialisation and layer behaviour."""

import numpy as np
import pytest

from tensorgrad import Tensor, nn


def small_cnn() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1, rng=0), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Dropout(0.2, rng=0), nn.Linear(64, 10, rng=1),
    )


def test_parameter_names_follow_torch_convention():
    names = [n for n, _ in small_cnn().named_parameters()]
    assert names == ["0.weight", "0.bias", "5.weight", "5.bias"]


def test_parameter_count():
    assert small_cnn().num_parameters() == (4 * 9 + 4) + (64 * 10 + 10)


def test_forward_shapes():
    out = small_cnn()(Tensor(np.zeros((3, 1, 8, 8))))
    assert out.shape == (3, 10)
    conv = nn.Conv2d(2, 5, 3, stride=2, padding=1, rng=0)
    assert conv(Tensor(np.zeros((1, 2, 9, 9)))).shape == (1, 5, 5, 5)


def test_train_eval_propagates_to_children():
    model = small_cnn()
    model.eval()
    assert all(not m.training for m in model)
    model.train()
    assert all(m.training for m in model)


def test_dropout_is_identity_in_eval_mode():
    model = small_cnn().eval()
    x = Tensor(np.random.default_rng(0).standard_normal((2, 1, 8, 8)))
    np.testing.assert_array_equal(model(x).data, model(x).data)


def test_state_dict_round_trip_and_validation():
    a, b = small_cnn(), nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1, rng=5), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, 10, rng=6),
    )
    b.load_state_dict(a.state_dict())
    for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        np.testing.assert_array_equal(pa.data, pb.data)
        assert pa.data is not pb.data  # copied, not aliased
    with pytest.raises(KeyError):
        b.load_state_dict({"0.weight": a.state_dict()["0.weight"]})
    bad = a.state_dict()
    bad["0.bias"] = np.zeros(7)
    with pytest.raises(ValueError):
        b.load_state_dict(bad)


def test_kaiming_init_variance():
    layer = nn.Linear(500, 400, rng=0)
    assert abs(layer.weight.data.std() - np.sqrt(2 / 500)) < 0.002
    np.testing.assert_array_equal(layer.bias.data, 0.0)
    conv = nn.Conv2d(16, 64, 3, rng=0)
    assert abs(conv.weight.data.std() - np.sqrt(2 / (16 * 9))) < 0.005


def test_cast_to_float32():
    model = small_cnn().to(np.float32)
    assert all(p.dtype == np.float32 for p in model.parameters())
    out = model(Tensor(np.zeros((2, 1, 8, 8), dtype=np.float32)))
    assert out.dtype == np.float32


def test_zero_grad_clears_all_parameters():
    model = small_cnn()
    model(Tensor(np.ones((2, 1, 8, 8)))).sum().backward()
    assert all(p.grad is not None for p in model.parameters())
    model.zero_grad()
    assert all(p.grad is None for p in model.parameters())


def test_linear_without_bias():
    layer = nn.Linear(3, 2, bias=False, rng=0)
    assert [n for n, _ in layer.named_parameters()] == ["weight"]
    x = np.ones((1, 3))
    np.testing.assert_allclose(layer(Tensor(x)).data, x @ layer.weight.data.T)
