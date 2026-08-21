#!/usr/bin/env python3
"""Shared plotting style and utilities for target_ancestry publication figures."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "figures" / "publication_v5"
MAIN_DIR = OUT_ROOT / "main"
SUPP_DIR = OUT_ROOT / "supplement"
TABLE_DIR = OUT_ROOT / "tables"

for directory in (MAIN_DIR, SUPP_DIR, TABLE_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.0,
            "figure.titlesize": 13.5,
            "figure.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.5,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.24,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, stem: str, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            directory / f"{stem}.{suffix}",
            dpi=600 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")

    suffixes = "".join(path.suffixes).lower()

    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)

    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path)

    if suffixes.endswith(".tsv") or suffixes.endswith(".tsv.gz"):
        return pd.read_csv(path, sep="\t")

    raise ValueError(f"Unsupported input table: {path}")


def find_first(directory: Path, patterns: Sequence[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    return None


def pick_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    *,
    label: str,
    required: bool = True,
) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    if required:
        raise KeyError(
            f"Could not find {label}. Tried {list(candidates)}. "
            f"Available columns: {list(frame.columns)}"
        )

    return None


def normalize_pool(values: pd.Series) -> pd.Series:
    def convert(value: object) -> str:
        text = str(value).strip().casefold()

        if text in {"a", "pool a", "launched", "approved", "1", "true"}:
            return "A"

        if text in {
            "b",
            "pool b",
            "phase i",
            "phase ii",
            "phase iii",
            "phase i-iii",
            "phase i–iii",
            "not launched",
            "not-approved",
            "not approved",
            "0",
            "false",
        }:
            return "B"

        if "launch" in text and "not" not in text:
            return "A"

        if "phase" in text:
            return "B"

        return str(value).strip()

    return values.map(convert)


def panel_label(
    ax: plt.Axes,
    label: str,
    *,
    x: float = -0.12,
    y: float = 1.06,
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="top",
    )


def clean_axis(ax: plt.Axes, grid_axis: str | None = None) -> None:
    if grid_axis:
        ax.grid(axis=grid_axis)
        ax.set_axisbelow(True)


def jitter(
    n: int,
    *,
    center: float,
    width: float = 0.08,
    seed: int = 1,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return center + rng.uniform(-width, width, size=n)


def raincloud(
    ax: plt.Axes,
    groups: dict[str, np.ndarray],
    positions: dict[str, float],
    *,
    width: float = 0.42,
    seed: int = 10,
) -> None:
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for index, (group, values) in enumerate(groups.items()):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            continue

        position = positions[group]
        color = cycle[index % len(cycle)]

        violin = ax.violinplot(
            values,
            positions=[position],
            widths=width,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )

        for body in violin["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.18)

        q1, median, q3 = np.percentile(values, [25, 50, 75])

        ax.plot(
            [position, position],
            [q1, q3],
            linewidth=5,
            solid_capstyle="round",
            color=color,
        )

        ax.plot(
            [position - 0.08, position + 0.08],
            [median, median],
            linewidth=2.3,
            color="white",
        )

        ax.scatter(
            jitter(
                len(values),
                center=position,
                width=0.10,
                seed=seed + index,
            ),
            values,
            s=17,
            alpha=0.42,
            linewidths=0,
            color=color,
            zorder=3,
        )


def export_table(
    frame: pd.DataFrame,
    stem: str,
    *,
    caption: str,
    label: str,
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    frame.to_csv(TABLE_DIR / f"{stem}.csv", index=False)

    latex = frame.to_latex(
        index=False,
        escape=True,
        caption=caption,
        label=label,
        float_format=lambda value: f"{value:.3f}",
    )

    (TABLE_DIR / f"{stem}.tex").write_text(latex, encoding="utf-8")


def value_label(
    estimate: float,
    lower: float,
    upper: float,
    *,
    digits: int = 2,
) -> str:
    return (
        f"{estimate:.{digits}f} "
        f"[{lower:.{digits}f}, {upper:.{digits}f}]"
    )
