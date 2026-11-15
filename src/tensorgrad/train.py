"""Mirrored training loops for tensorgrad and PyTorch.

Both loops consume the same pre-computed batch schedule, start from the same weights and use the
same optimizer settings, so any divergence between their loss curves is numerical, not random.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import functional as F
from . import nn, optim
from .data import Splits
from .parity import require_torch
from .tensor import Tensor, no_grad


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    optimizer: str  # "sgd" or "adam"
    lr: float
    momentum: float = 0.0
    weight_decay: float = 0.0

    def describe(self) -> str:
        if self.optimizer == "sgd":
            return (f"SGD(lr={self.lr}, momentum={self.momentum}, "
                    f"weight_decay={self.weight_decay})")
        return f"Adam(lr={self.lr}, weight_decay={self.weight_decay})"


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)  # one value per optimizer step
    val_loss: list[float] = field(default_factory=list)  # one value per epoch
    val_acc: list[float] = field(default_factory=list)
    train_time_s: float = 0.0
    test_loss: float = float("nan")
    test_acc: float = float("nan")
    test_pred: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    final_params: dict[str, np.ndarray] = field(default_factory=dict)


def _make_optimizer(params: list, cfg: TrainConfig, torch_mod: Any = None) -> Any:
    kwargs: dict[str, Any] = {"lr": cfg.lr, "weight_decay": cfg.weight_decay}
    if cfg.optimizer == "sgd":
        kwargs["momentum"] = cfg.momentum
        return (torch_mod.optim.SGD if torch_mod else optim.SGD)(params, **kwargs)
    if cfg.optimizer == "adam":
        return (torch_mod.optim.Adam if torch_mod else optim.Adam)(params, **kwargs)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


# ------------------------------------------------------------------------------ tensorgrad
def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Loss, accuracy and predictions in eval mode without building a graph."""
    model.eval()
    with no_grad():
        logits = model(Tensor(X))
        loss = F.cross_entropy(logits, y).item()
    model.train()
    pred = logits.data.argmax(axis=1)
    return loss, float((pred == y).mean()), pred


def train_tensorgrad(
    model: nn.Module,
    data: Splits,
    cfg: TrainConfig,
    schedule: list[list[np.ndarray]],
    dtype: Any = np.float64,
) -> History:
    """Mini-batch training with tensorgrad; returns per-step and per-epoch curves."""
    model.to(dtype).train()
    X_train, X_val = data.X_train.astype(dtype), data.X_val.astype(dtype)
    opt = _make_optimizer(model.parameters(), cfg)
    hist = History()
    start = time.perf_counter()
    for epoch_batches in schedule:
        for idx in epoch_batches:
            loss = F.cross_entropy(model(Tensor(X_train[idx])), data.y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            hist.train_loss.append(loss.item())
        val_loss, val_acc, _ = evaluate(model, X_val, data.y_val)
        hist.val_loss.append(val_loss)
        hist.val_acc.append(val_acc)
    hist.train_time_s = time.perf_counter() - start
    hist.test_loss, hist.test_acc, hist.test_pred = evaluate(
        model, data.X_test.astype(dtype), data.y_test
    )
    hist.final_params = model.state_dict()
    return hist


# ------------------------------------------------------------------------------ PyTorch
def train_torch(
    tmodel: Any, data: Splits, cfg: TrainConfig, schedule: list[list[np.ndarray]],
) -> History:
    """The same loop in PyTorch (dtype is taken from ``tmodel``'s parameters)."""
    torch = require_torch()
    tF = torch.nn.functional
    dtype = next(tmodel.parameters()).dtype

    def as_tensor(a: np.ndarray) -> Any:
        return torch.from_numpy(np.ascontiguousarray(a)).to(dtype)

    def evaluate_t(X: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
        tmodel.eval()
        with torch.no_grad():
            logits = tmodel(as_tensor(X))
            loss = tF.cross_entropy(logits, torch.from_numpy(y)).item()
        tmodel.train()
        pred = logits.argmax(dim=1).numpy()
        return loss, float((pred == y).mean()), pred

    tmodel.train()
    opt = _make_optimizer(list(tmodel.parameters()), cfg, torch_mod=torch)
    X_train, y_train = as_tensor(data.X_train), torch.from_numpy(data.y_train)
    hist = History()
    start = time.perf_counter()
    for epoch_batches in schedule:
        for idx in epoch_batches:
            ti = torch.from_numpy(idx)
            loss = tF.cross_entropy(tmodel(X_train[ti]), y_train[ti])
            opt.zero_grad()
            loss.backward()
            opt.step()
            hist.train_loss.append(loss.item())
        val_loss, val_acc, _ = evaluate_t(data.X_val, data.y_val)
        hist.val_loss.append(val_loss)
        hist.val_acc.append(val_acc)
    hist.train_time_s = time.perf_counter() - start
    hist.test_loss, hist.test_acc, hist.test_pred = evaluate_t(data.X_test, data.y_test)
    hist.final_params = {k: v.detach().numpy().copy() for k, v in tmodel.state_dict().items()}
    return hist


def parity_stats(a: History, b: History) -> dict[str, float]:
    """How far two training runs drifted apart."""
    return {
        "max_abs_train_loss_diff": float(np.max(np.abs(np.subtract(a.train_loss, b.train_loss)))),
        "max_abs_val_loss_diff": float(np.max(np.abs(np.subtract(a.val_loss, b.val_loss)))),
        "max_abs_final_param_diff": max(
            float(np.max(np.abs(a.final_params[k] - b.final_params[k]))) for k in a.final_params
        ),
        "test_prediction_agreement": float(np.mean(a.test_pred == b.test_pred)),
    }
