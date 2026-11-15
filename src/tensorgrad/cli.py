"""Reproduce every number and figure in the README: ``python -m tensorgrad.cli --out reports``.

1. Op-by-op checks: finite differences and PyTorch autograd for every registered op.
2. MLP and CNN on scikit-learn's digits, trained by tensorgrad and, from the same initial
   weights and the same batch order, by PyTorch, in float64 and float32.
3. A seed sweep (init + batch order) for the tensorgrad test accuracy.
4. ``metrics.json`` and the figures.

Training runs with single-threaded BLAS and PyTorch (faster for these small matrices, and
reproducible), and with PyTorch's NNPACK convolution backend disabled for the parity runs (see
``nnpack_ablation`` for why).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from . import __version__
from .data import Splits, batch_schedule, load_digit_splits
from .models import MODELS
from .parity import op_report, require_torch, to_torch
from .plots import plot_confusion_matrix, plot_filters, plot_parity_curves
from .tensor import Tensor, no_grad
from .train import History, TrainConfig, parity_stats, train_tensorgrad, train_torch

CONFIGS: dict[str, TrainConfig] = {
    "mlp": TrainConfig(epochs=40, batch_size=32, optimizer="sgd", lr=0.05, momentum=0.9,
                       weight_decay=1e-3),
    "cnn": TrainConfig(epochs=30, batch_size=32, optimizer="adam", lr=2e-3, weight_decay=1e-4),
}
TITLES = {"mlp": "MLP", "cnn": "CNN"}
DTYPES = {"64": np.float64, "32": np.float32}


def _summary(h: History) -> dict[str, float]:
    return {"test_accuracy": h.test_acc, "test_loss": h.test_loss,
            "final_val_accuracy": h.val_acc[-1], "final_val_loss": h.val_loss[-1],
            "train_time_s": round(h.train_time_s, 3)}


def train_pair(
    name: str, data: Splits, cfg: TrainConfig, seed: int, dtype: Any
) -> tuple[History, History, Any]:
    """Train the same initial model with tensorgrad and with PyTorch on the same batches."""
    model = MODELS[name](seed=seed)
    tmodel = to_torch(model, dtype)  # copies the initial weights before training starts
    schedule = batch_schedule(len(data.y_train), cfg.epochs, cfg.batch_size, seed=seed)
    h_tg = train_tensorgrad(model, data, cfg, schedule, dtype)
    h_pt = train_torch(tmodel, data, cfg, schedule)
    return h_tg, h_pt, model


def blas_thread_benchmark(data: Splits, cfg: TrainConfig, epochs: int = 5) -> dict[str, Any]:
    """Time a few tensorgrad CNN epochs with the default BLAS thread pool vs one thread."""

    def timed(n_epochs: int) -> float:
        schedule = batch_schedule(len(data.y_train), n_epochs, cfg.batch_size, seed=0)
        model = MODELS["cnn"](seed=0)
        return train_tensorgrad(model, data, replace(cfg, epochs=n_epochs), schedule).train_time_s

    default = max((i["num_threads"] for i in threadpool_info() if i["user_api"] == "blas"),
                  default=1)
    timed(1)  # warm-up: first-call allocations and thread-pool start-up
    t_default = timed(epochs)
    with threadpool_limits(limits=1, user_api="blas"):
        t_single = timed(epochs)
    return {"model": "cnn", "epochs": epochs, "default_blas_threads": default,
            "train_time_s_default_threads": round(t_default, 3),
            "train_time_s_single_thread": round(t_single, 3)}


def nnpack_ablation(data: Splits, cfg: TrainConfig, seed: int) -> dict[str, Any]:
    """Float32 CNN parity with PyTorch's NNPACK conv backend switched back on.

    PyTorch routes float32 CPU convolutions with batch >= 16 and grad mode on (i.e. training)
    to NNPACK, whose fast algorithm returns tiny non-zero values where the exact result is 0
    (blank image regions times zero-initialised bias). ReLU then passes gradient where it should
    not, and the two training runs drift apart from the first step.
    """
    torch = require_torch()
    with torch.backends.nnpack.flags(enabled=True):
        h_tg, h_pt, _ = train_pair("cnn", data, cfg, seed, np.float32)
        model = MODELS["cnn"](seed=seed).to(np.float32)  # initial weights: zero bias
        x = data.X_train[:cfg.batch_size].astype(np.float32)
        exact = model[0](Tensor(x)).data
        tconv = to_torch(model, np.float32)[0]
        fast = tconv(torch.from_numpy(x)).detach().numpy()  # grad mode, batch >= 16 -> NNPACK
    with torch.backends.nnpack.flags(enabled=False):
        direct = tconv(torch.from_numpy(x)).detach().numpy()
    zeros = exact == 0
    return {"torch_nnpack_available": bool(torch.backends.nnpack.is_available()),
            "float32_cnn_parity_with_nnpack": parity_stats(h_tg, h_pt),
            "probe_images": len(x),
            "conv1_exact_zero_outputs": int(zeros.sum()),
            "of_which_nonzero_in_torch_nnpack": int((fast[zeros] != 0).sum()),
            "of_which_positive_in_torch_nnpack": int((fast[zeros] > 0).sum()),
            "conv1_max_abs_output_diff_nnpack": float(np.abs(exact - fast).max()),
            "conv1_max_abs_output_diff_nnpack_off": float(np.abs(exact - direct).max())}


def _op_checks() -> dict[str, Any]:
    ops = op_report()
    keys = ["max_abs_forward_diff_vs_torch", "max_abs_grad_diff_vs_torch",
            "max_rel_err_vs_finite_diff"]
    return {"summary": {"n_cases": len(ops), **{k: max(r[k] for r in ops) for k in keys}},
            "cases": ops}


def _model_metrics(
    name: str, data: Splits, cfg: TrainConfig, seeds: Sequence[int]
) -> tuple[dict[str, Any], dict[str, History], Any]:
    runs: dict[str, History] = {}
    for tag, dtype in DTYPES.items():
        h_tg, h_pt, model = train_pair(name, data, cfg, seeds[0], dtype)
        runs[f"tg{tag}"], runs[f"pt{tag}"] = h_tg, h_pt
        if tag == "64":
            trained = model
    sweep = [runs["tg64"].test_acc] + [
        train_tensorgrad(MODELS[name](seed=s), data, cfg,
                         batch_schedule(len(data.y_train), cfg.epochs, cfg.batch_size, s)).test_acc
        for s in seeds[1:]
    ]
    metrics = {
        "architecture": repr(trained), "n_parameters": trained.num_parameters(),
        "optimizer": cfg.describe(), "epochs": cfg.epochs, "batch_size": cfg.batch_size,
        "optimizer_steps": len(runs["tg64"].train_loss),
        "float64": {"tensorgrad": _summary(runs["tg64"]), "pytorch": _summary(runs["pt64"]),
                    "parity": parity_stats(runs["tg64"], runs["pt64"])},
        "float32": {"tensorgrad": _summary(runs["tg32"]), "pytorch": _summary(runs["pt32"]),
                    "parity": parity_stats(runs["tg32"], runs["pt32"])},
        "seed_sweep_float64": {"seeds": list(seeds), "test_accuracy": sweep,
                               "mean": float(np.mean(sweep)),
                               "std": float(np.std(sweep, ddof=1)) if len(sweep) > 1 else 0.0},
    }
    return metrics, runs, trained


def run_experiment(
    data: Splits,
    out_dir: Path,
    configs: dict[str, TrainConfig] = CONFIGS,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> dict[str, Any]:
    """Run everything, write ``metrics.json`` + figures into ``out_dir``, return the metrics."""
    torch = require_torch()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        "dataset": {"name": "scikit-learn load_digits (UCI optdigits test set)",
                    "image_shape": list(data.X_train.shape[1:]), "n_classes": 10,
                    **{f"n_{k}": v for k, v in data.sizes().items()}},
        "models": {},
    }
    if "cnn" in configs:
        print("[0/4] BLAS threading benchmark")
        metrics["blas_thread_benchmark"] = blas_thread_benchmark(data, configs["cnn"])

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with threadpool_limits(limits=1, user_api="blas"), \
                torch.backends.nnpack.flags(enabled=False):
            print("[1/4] op-by-op checks against finite differences and torch.autograd")
            metrics["op_checks"] = _op_checks()
            runs, trained = {}, {}
            for name, cfg in configs.items():
                print(f"[2/4] {name}: tensorgrad vs PyTorch, float64 + float32, "
                      f"{len(seeds)} seeds ({cfg.describe()})")
                metrics["models"][name], runs[name], trained[name] = _model_metrics(
                    name, data, cfg, seeds)
            if "cnn" in configs:
                print("[3/4] NNPACK ablation (float32 CNN)")
                metrics["nnpack_ablation"] = nnpack_ablation(data, configs["cnn"], seeds[0])
    finally:
        torch.set_num_threads(threads)

    print("[4/4] figures + metrics.json")
    plot_parity_curves(runs, TITLES, fig_dir / "parity_curves.png")
    if "cnn" in configs:
        cnn, h = trained["cnn"], runs["cnn"]["tg64"]
        plot_confusion_matrix(data.y_test, h.test_pred,
                              f"tensorgrad CNN, test set (accuracy {h.test_acc:.1%})",
                              fig_dir / "confusion_matrix_cnn.png")
        with no_grad():
            maps = cnn[0](Tensor(data.X_test[:1])).relu().data[0]
        plot_filters(cnn[0].weight.data, data.X_test[0, 0], maps, int(data.y_test[0]),
                     fig_dir / "conv1_filters.png")

    metrics["environment"] = {
        "tensorgrad": __version__, "python": platform.python_version(), "numpy": np.__version__,
        "torch": torch.__version__, "machine": f"{platform.system()} {platform.machine()}",
        "threads": "1 (BLAS and torch)", "torch_nnpack_in_parity_runs": False,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)
    metrics = run_experiment(load_digit_splits(seed=0), args.out)
    for name, m in metrics["models"].items():
        f64 = m["float64"]
        print(f"{name}: test acc {f64['tensorgrad']['test_accuracy']:.4f} "
              f"(PyTorch {f64['pytorch']['test_accuracy']:.4f}), "
              f"max |loss diff| float64 {f64['parity']['max_abs_train_loss_diff']:.2e}, "
              f"float32 {m['float32']['parity']['max_abs_train_loss_diff']:.2e}")
    print(f"wrote {args.out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
