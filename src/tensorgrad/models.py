"""The two architectures trained on 8x8 digits. Both take (N, 1, 8, 8) images."""

from __future__ import annotations

import numpy as np

from . import nn


def make_mlp(seed: int = 0, hidden: tuple[int, int] = (128, 64)) -> nn.Sequential:
    """64 -> 128 -> 64 -> 10 multilayer perceptron with ReLU."""
    rng = np.random.default_rng(seed)
    h1, h2 = hidden
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(64, h1, rng=rng), nn.ReLU(),
        nn.Linear(h1, h2, rng=rng), nn.ReLU(),
        nn.Linear(h2, 10, rng=rng),
    )


def make_cnn(seed: int = 0, channels: tuple[int, int] = (16, 32)) -> nn.Sequential:
    """Two conv(3x3, pad 1) + ReLU + 2x2 max-pool blocks (8x8 -> 4x4 -> 2x2), then an MLP head."""
    rng = np.random.default_rng(seed)
    c1, c2 = channels
    return nn.Sequential(
        nn.Conv2d(1, c1, 3, padding=1, rng=rng), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(c1, c2, 3, padding=1, rng=rng), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(c2 * 2 * 2, 64, rng=rng), nn.ReLU(),
        nn.Linear(64, 10, rng=rng),
    )


MODELS = {"mlp": make_mlp, "cnn": make_cnn}
