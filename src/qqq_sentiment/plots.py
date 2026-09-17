from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


INK = "#26332E"
FOREST = "#2F6B5F"
SAGE = "#A7C4A0"
EARTH = "#9C6644"
GOLD = "#C49A54"
PARCHMENT = "#F4F1E8"
GRID = "#D8DDD6"

DIVERGING = LinearSegmentedColormap.from_list(
    "earth_to_forest", [EARTH, PARCHMENT, FOREST], N=256
)


def _finish(ax, title: str, subtitle: str | None = None):
    ax.set_title(title, loc="left", color=INK, fontsize=13, fontweight="bold", pad=30)
    if subtitle:
        ax.text(0, 1.01, subtitle, transform=ax.transAxes, color=INK, fontsize=9, va="bottom")
    ax.tick_params(colors=INK)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)


def plot_ic_heatmap(
    grid: pd.DataFrame,
    lookback_order: list[str] | tuple[str, ...],
    horizon_order: list[str] | tuple[str, ...],
    output: str | Path,
    metric: str = "mean_spearman",
    subtitle: str = "Mean walk-forward Spearman IC; earth = negative, forest = positive",
) -> Path:
    matrix = (
        grid.pivot(index="lookback", columns="horizon", values=metric)
        .reindex(index=lookback_order, columns=horizon_order)
    )
    values = matrix.to_numpy(dtype=float)
    finite = np.abs(values[np.isfinite(values)])
    vmax = max(float(finite.max()) if finite.size else 0.05, 0.05)

    fig, ax = plt.subplots(
        figsize=(max(11.0, 0.9 * len(matrix.columns)), max(6.5, 0.55 * len(matrix.index)))
    )
    im = ax.imshow(values, cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_xlabel("Forward abnormal-return horizon", color=INK)
    ax.set_ylabel("News sentiment lookback", color=INK)
    _finish(
        ax,
        "News sentiment × forward return correlation",
        subtitle,
    )

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8, color=INK)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Spearman IC", color=INK)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_reaction_curve(curve: pd.DataFrame, metrics: dict[str, float], output: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(curve["horizon_minutes"], curve["spread_car"], marker="o", color=FOREST, linewidth=2)
    ax.axhline(0, color=GRID, linewidth=1)

    for label, key, color in (("T50", "T50_min", GOLD), ("T80", "T80_min", EARTH)):
        value = metrics.get(key, np.nan)
        if np.isfinite(value):
            ax.axvline(value, color=color, linestyle="--", linewidth=1.5, label=f"{label} = {value:g} min")

    ax.set_xscale("log")
    ax.set_xlabel("Minutes after news", color=INK)
    ax.set_ylabel("Q5 − Q1 mean abnormal return", color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.7)
    _finish(
        ax,
        "Market reaction speed",
        "Cumulative abnormal-return spread between the most positive and most negative sentiment quintiles",
    )
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output
