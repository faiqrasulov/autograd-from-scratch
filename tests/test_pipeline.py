"""End-to-end smoke tests on tiny synthetic data (no download, a few seconds)."""

import json

import numpy as np
import pytest

from tensorgrad.data import Splits, batch_schedule, load_digit_splits, make_splits
from tensorgrad.models import make_cnn, make_mlp
from tensorgrad.train import TrainConfig, evaluate, train_tensorgrad


@pytest.fixture(scope="module")
def synthetic() -> Splits:
    """Ten noisy copies of ten random 8x8 'prototype digits' per class (known ground truth)."""
    rng = np.random.default_rng(0)
    prototypes = rng.random((10, 1, 8, 8))
    y = np.repeat(np.arange(10), 20)
    X = np.clip(prototypes[y] + 0.25 * rng.standard_normal((len(y), 1, 8, 8)), 0, 1)
    return make_splits(X, y, seed=0)


@pytest.mark.parametrize("make, cfg", [
    (make_mlp, TrainConfig(epochs=8, batch_size=16, optimizer="sgd", lr=0.05, momentum=0.9)),
    (make_cnn, TrainConfig(epochs=8, batch_size=16, optimizer="adam", lr=3e-3)),
])
def test_training_reduces_loss(synthetic, make, cfg):
    model = make(seed=0)
    before, _, _ = evaluate(model, synthetic.X_val, synthetic.y_val)
    schedule = batch_schedule(len(synthetic.y_train), cfg.epochs, cfg.batch_size, seed=0)
    hist = train_tensorgrad(model, synthetic, cfg, schedule)
    assert hist.val_loss[-1] < 0.5 * before
    assert np.mean(hist.train_loss[-5:]) < np.mean(hist.train_loss[:5])
    assert hist.test_acc > 0.5  # chance is 0.1


def test_float32_training_runs(synthetic):
    cfg = TrainConfig(epochs=2, batch_size=16, optimizer="adam", lr=3e-3)
    model = make_cnn(seed=0)
    schedule = batch_schedule(len(synthetic.y_train), cfg.epochs, cfg.batch_size, seed=0)
    train_tensorgrad(model, synthetic, cfg, schedule, dtype=np.float32)
    assert all(p.dtype == np.float32 for p in model.parameters())


def test_batch_schedule_covers_every_sample_once_per_epoch():
    schedule = batch_schedule(50, epochs=3, batch_size=16, seed=0)
    for epoch in schedule:
        assert sorted(np.concatenate(epoch).tolist()) == list(range(50))
    assert not np.array_equal(schedule[0][0], schedule[1][0])  # reshuffled each epoch


def test_digit_splits_are_stratified_and_disjoint():
    s = load_digit_splits(seed=0)
    assert s.sizes() == {"train": 1077, "val": 360, "test": 360}
    assert s.X_train.shape[1:] == (1, 8, 8) and 0.0 <= s.X_train.min() <= s.X_train.max() <= 1.0
    for y in (s.y_train, s.y_val, s.y_test):
        frac = np.bincount(y, minlength=10) / len(y)
        assert np.all(np.abs(frac - 0.1) < 0.02)


def test_full_pipeline_end_to_end(synthetic, tmp_path):
    pytest.importorskip("torch")
    from tensorgrad.cli import run_experiment

    configs = {
        "mlp": TrainConfig(epochs=2, batch_size=16, optimizer="sgd", lr=0.05, momentum=0.9),
        "cnn": TrainConfig(epochs=2, batch_size=16, optimizer="adam", lr=2e-3),
    }
    metrics = run_experiment(synthetic, tmp_path, configs=configs, seeds=(0, 1))

    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["models"].keys() == {"mlp", "cnn"}
    for name in ("mlp", "cnn"):
        parity = metrics["models"][name]["float64"]["parity"]
        assert parity["max_abs_train_loss_diff"] < 1e-12
        assert parity["test_prediction_agreement"] == 1.0
        assert len(metrics["models"][name]["seed_sweep_float64"]["test_accuracy"]) == 2
    assert metrics["op_checks"]["summary"]["max_abs_grad_diff_vs_torch"] < 1e-12
    for fig in ("parity_curves.png", "confusion_matrix_cnn.png", "conv1_filters.png"):
        assert (tmp_path / "figures" / fig).stat().st_size > 10_000
