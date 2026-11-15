"""Report figures (static PNGs for the README)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .train import History  # noqa: E402

TG_COLOR, PT_COLOR = "#2a78d6", "#eb6834"  # categorical slots 1-2: tensorgrad, PyTorch
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a32b2a"]
)

STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.labelsize": 9.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.frameon": False, "legend.fontsize": 8.5, "font.size": 9.5,
}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _epoch_means(h: History) -> np.ndarray:
    return np.array([chunk.mean() for chunk in np.array_split(h.train_loss, len(h.val_loss))])


def plot_parity_curves(
    runs: dict[str, dict[str, History]], titles: dict[str, str], path: Path
) -> None:
    """Top: loss curves of both frameworks (float64). Bottom: per-step |loss difference|.

    ``runs[model]`` holds "tg64", "pt64", "tg32", "pt32" histories.
    """
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, len(runs), figsize=(5.4 * len(runs), 6.6), squeeze=False,
                                 gridspec_kw={"height_ratios": [1.25, 1]})
        for col, (name, h) in enumerate(runs.items()):
            top, bottom = axes[0, col], axes[1, col]
            epochs = np.arange(1, len(h["tg64"].val_loss) + 1)
            for kind, tg, pt in [("train", _epoch_means(h["tg64"]), _epoch_means(h["pt64"])),
                                 ("validation", h["tg64"].val_loss, h["pt64"].val_loss)]:
                top.plot(epochs, tg, color=TG_COLOR, lw=2,
                         label="tensorgrad (line)" if kind == "train" else None)
                top.plot(epochs, pt, color=PT_COLOR, lw=0, marker="x", ms=5.5, mew=1.4,
                         label="PyTorch (markers)" if kind == "train" else None)
                top.annotate(kind, (epochs[-1], tg[-1]), xytext=(6, 0),
                             textcoords="offset points", va="center", color=INK_2, fontsize=9)
            top.set_yscale("log")
            top.set_xlim(0, epochs[-1] * 1.13)
            top.set_title(f"{titles[name]}: loss per epoch, float64")
            top.set_xlabel("epoch")
            top.set_ylabel("cross-entropy loss")
            top.legend(loc="upper right")

            steps = np.arange(1, len(h["tg64"].train_loss) + 1)
            for key, color, label, eps in [("64", INK, "float64", np.finfo(np.float64).eps),
                                           ("32", MUTED, "float32", np.finfo(np.float32).eps)]:
                diff = np.abs(np.subtract(h[f"tg{key}"].train_loss, h[f"pt{key}"].train_loss))
                nz = diff > 0  # identical losses (diff exactly 0) cannot be drawn on a log axis
                bottom.plot(steps[nz], diff[nz], ls="none", marker=".", ms=2.2, color=color,
                            alpha=0.7)
                bottom.axhline(eps, color=AXIS, lw=0.8, zorder=0)
                bottom.annotate(f"{label}\nmachine eps", (1, eps),
                                xycoords=("axes fraction", "data"), xytext=(4, 0),
                                textcoords="offset points", va="center", color=MUTED,
                                fontsize=7.5)
                bottom.annotate(f"{label} runs: max {diff.max():.1e}", (steps[0], diff.max()),
                                xytext=(0, 5), textcoords="offset points", color=color,
                                fontsize=8.5, fontweight="bold")
            bottom.set_yscale("log")
            bottom.set_ylim(1e-18, 1)
            bottom.set_title("|tensorgrad - PyTorch| training loss, per step")
            bottom.set_xlabel("optimizer step")
            bottom.set_ylabel("absolute difference")
        fig.tight_layout()
        _save(fig, path)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str, path: Path) -> None:
    n = int(max(y_true.max(), y_pred.max())) + 1
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(5.0, 4.6))
        ax.imshow(cm, cmap=BLUES, vmin=0)
        for i in range(n):
            for j in range(n):
                if cm[i, j]:
                    dark = cm[i, j] > 0.55 * cm.max()
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8.5,
                            color="white" if dark else INK)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xlabel("predicted digit")
        ax.set_ylabel("true digit")
        ax.set_title(title)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout()
        _save(fig, path)


def plot_filters(
    weights: np.ndarray, image: np.ndarray, feature_maps: np.ndarray, label: int, path: Path
) -> None:
    """First-layer 3x3 kernels and the ReLU feature maps they produce for one test digit."""
    n, cols = weights.shape[0], 8
    rows = int(np.ceil(n / cols))
    lim, fmax = np.abs(weights).max(), feature_maps.max() or 1.0
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig = plt.figure(figsize=(10.5, 3.2 * rows / 2 + 1.6))
        left, right = fig.subfigures(1, 2, width_ratios=[1, 3.6], wspace=0.02)
        ax = left.subplots()
        ax.imshow(image, cmap="gray_r")
        ax.set_title(f"input: test digit '{label}'", fontsize=10)
        ax.axis("off")
        kern_fig, map_fig = right.subfigures(2, 1, hspace=0.06)
        kern_fig.suptitle("16 learned conv1 kernels, 3x3 (blue < 0 < red)", fontsize=10,
                          color=INK)
        map_fig.suptitle("their ReLU feature maps for this input, 8x8 (darker = stronger)",
                         fontsize=10, color=INK)
        k_axes = kern_fig.subplots(rows, cols)
        m_axes = map_fig.subplots(rows, cols)
        for k in range(n):
            ka, ma = k_axes.flat[k], m_axes.flat[k]
            ka.imshow(weights[k, 0], cmap=DIVERGING, vmin=-lim, vmax=lim)
            ma.imshow(feature_maps[k], cmap=BLUES, vmin=0, vmax=fmax)
            ka.axis("off")
            ma.axis("off")
        _save(fig, path)
