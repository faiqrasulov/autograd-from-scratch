"""The UCI handwritten-digits data bundled with scikit-learn (no download needed)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Splits:
    """Images as (N, 1, 8, 8) float64 in [0, 1]; labels as int64."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.y_train), "val": len(self.y_val), "test": len(self.y_test)}


def make_splits(
    X: np.ndarray, y: np.ndarray, val_size: float = 0.2, test_size: float = 0.2, seed: int = 0
) -> Splits:
    """Stratified train / validation / test split of images ``X`` (N, 1, H, W) and labels ``y``."""
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_size / (1 - test_size), stratify=y_rest, random_state=seed
    )
    return Splits(X_train, y_train, X_val, y_val, X_test, y_test)


def load_digit_splits(seed: int = 0) -> Splits:
    """1,797 8x8 digit images, pixel values 0-16 scaled to [0, 1], split 60 / 20 / 20."""
    digits = load_digits()
    X = (digits.images / 16.0).astype(np.float64)[:, None, :, :]
    return make_splits(X, digits.target.astype(np.int64), seed=seed)


def batch_schedule(n: int, epochs: int, batch_size: int, seed: int) -> list[list[np.ndarray]]:
    """Pre-computed shuffled mini-batch indices per epoch, shared by both frameworks so they
    see exactly the same data in exactly the same order."""
    rng = np.random.default_rng(seed)
    schedule = []
    for _ in range(epochs):
        perm = rng.permutation(n)
        schedule.append([perm[i : i + batch_size] for i in range(0, n, batch_size)])
    return schedule


if __name__ == "__main__":
    splits = load_digit_splits()
    print("load_digits (bundled with scikit-learn, no download):", splits.sizes())
    for name, y in [("train", splits.y_train), ("val", splits.y_val), ("test", splits.y_test)]:
        print(f"  {name:5s} class counts: {np.bincount(y).tolist()}")
