#!/usr/bin/env python3
"""
Main Figure 5: cross-ancestry mechanism results by launch pool.

Panel a:
    Unit-level portability distributions for launched (Pool A) and
    Phase I–III (Pool B) targets.

Panel b:
    Direct Pool A minus Pool B portability contrasts.

Panel c:
    Final colocalization classes by launch pool. Counts are shown inside
    the bars because the powered colocalization sample is small.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from loaders import normalize_coloc_category
from style import (
    ROOT,
    MAIN_DIR,
    clean_axis,
    normalize_pool,
    panel_label,
    pick_column,
    read_table,
    save_figure,
    set_style,
    value_label,
)


ESTIMATORS = ["naive", "fiqt", "deming"]

ESTIMATOR_LABELS = {
    "naive": "Naive",
    "fiqt": "FIQT",
    "deming": "Deming",
}

CATEGORY_ORDER = [
    "robust shared",
    "prior-sensitive",
    "robust distinct",
    "ancestry-discordant",
    "mixed with prior sensitivity",
]

CATEGORY_LABELS = {
    "robust shared": "Robust shared",
    "prior-sensitive": "Prior-sensitive",
    "robust distinct": "Robust distinct",
    "ancestry-discordant": "Ancestry-discordant",
    "mixed with prior sensitivity": "Mixed / prior-sensitive",
}


def as_boolean(values: pd.Series) -> pd.Series:
    """Robustly normalize boolean-like columns."""
    if values.dtype == bool:
        return values

    normalized = (
        values.astype(str)
        .str.strip()
        .str.casefold()
    )

    return normalized.isin(
        {
            "true",
            "1",
            "yes",
            "y",
            "t",
        }
    )


def load_portability_units() -> pd.DataFrame:
    """
    Load the actual Step 15F unit table used elsewhere in this project.
    """
    path = (
        ROOT
        / "step15/output/15f_mechanistic_units.parquet"
    )

    frame = read_table(path).copy()

    required = [
        "approval_pool",
        "portability_final_available",
        "portability_naive_slope",
        "portability_fiqt_slope",
        "portability_deming_slope",
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise KeyError(
            f"Missing expected portability columns: {missing}. "
            f"Available columns: {list(frame.columns)}"
        )

    frame["approval_pool"] = normalize_pool(
        frame["approval_pool"]
    )

    frame = frame[
        as_boolean(
            frame["portability_final_available"]
        )
        & frame["approval_pool"].isin(["A", "B"])
    ].copy()

    return frame


def load_portability_contrasts() -> pd.DataFrame:
    path = (
        ROOT
        / "step15/output/15f_portability_pool_contrasts.csv"
    )

    frame = read_table(path).copy()

    estimator_col = pick_column(
        frame,
        ["estimator", "method", "slope_estimator"],
        label="portability estimator",
    )

    estimate_col = pick_column(
        frame,
        [
            "median_difference_A_minus_B",
            "median_difference",
            "difference",
            "estimate",
            "pool_a_minus_b",
        ],
        label="Pool A minus Pool B contrast",
    )

    lower_col = pick_column(
        frame,
        [
            "difference_ci_lower",
            "ci_low",
            "ci_lower",
            "bootstrap_ci_low",
            "lower",
            "lower_ci",
        ],
        label="contrast lower confidence bound",
    )

    upper_col = pick_column(
        frame,
        [
            "difference_ci_upper",
            "ci_high",
            "ci_upper",
            "bootstrap_ci_high",
            "upper",
            "upper_ci",
        ],
        label="contrast upper confidence bound",
    )

    p_col = pick_column(
        frame,
        [
            "exploratory_permutation_p_two_sided",
            "permutation_p",
            "permutation_p_value",
            "p_value",
            "pvalue",
            "p",
        ],
        label="permutation p-value",
        required=False,
    )

    columns = [
        estimator_col,
        estimate_col,
        lower_col,
        upper_col,
    ]

    if p_col:
        columns.append(p_col)

    data = frame[columns].copy()

    data.columns = [
        "estimator",
        "estimate",
        "lower",
        "upper",
    ] + (["p_value"] if p_col else [])

    data["estimator"] = (
        data["estimator"]
        .astype(str)
        .str.casefold()
    )

    data["_order"] = data["estimator"].map(
        {
            name: index
            for index, name in enumerate(ESTIMATORS)
        }
    )

    data = (
        data.sort_values("_order")
        .reset_index(drop=True)
    )

    data["label"] = data["estimator"].map(
        ESTIMATOR_LABELS
    )

    return data


def load_coloc_units(
    mechanistic_units: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load final unit-level colocalization classes and attach approval pool.
    """
    path = (
        ROOT
        / "step15/output/15e2_coloc_unit_stability_final.parquet"
    )

    frame = read_table(path).copy()

    category_col = pick_column(
        frame,
        [
            "unit_stability_final",
            "stability_final",
            "final_category",
            "category",
        ],
        label="final colocalization category",
    )

    if "approval_pool" not in frame.columns:
        if (
            "gene_trait_uid" in frame.columns
            and "gene_trait_uid" in mechanistic_units.columns
        ):
            frame = frame.merge(
                mechanistic_units[
                    [
                        "gene_trait_uid",
                        "approval_pool",
                    ]
                ].drop_duplicates(
                    subset=["gene_trait_uid"]
                ),
                on="gene_trait_uid",
                how="left",
            )

        elif {
            "gene",
            "candidate_trait_name",
        }.issubset(frame.columns) and {
            "gene",
            "candidate_trait_name",
        }.issubset(mechanistic_units.columns):
            frame = frame.merge(
                mechanistic_units[
                    [
                        "gene",
                        "candidate_trait_name",
                        "approval_pool",
                    ]
                ].drop_duplicates(
                    subset=[
                        "gene",
                        "candidate_trait_name",
                    ]
                ),
                on=[
                    "gene",
                    "candidate_trait_name",
                ],
                how="left",
            )

        else:
            raise KeyError(
                "Could not attach approval pool to colocalization units. "
                f"Colocalization columns: {list(frame.columns)}; "
                f"mechanistic-unit columns: {list(mechanistic_units.columns)}"
            )

    frame["approval_pool"] = normalize_pool(
        frame["approval_pool"]
    )

    frame["category"] = frame[
        category_col
    ].map(normalize_coloc_category)

    frame = frame[
        frame["approval_pool"].isin(["A", "B"])
        & frame["category"].isin(CATEGORY_ORDER)
    ].copy()

    return frame


def draw_portability_distributions(
    ax: plt.Axes,
    units: pd.DataFrame,
) -> None:
    base_positions = np.arange(
        len(ESTIMATORS)
    ) * 2.0

    cycle = plt.rcParams[
        "axes.prop_cycle"
    ].by_key()["color"]

    for estimator_index, estimator in enumerate(
        ESTIMATORS
    ):
        column = (
            f"portability_{estimator}_slope"
        )

        for pool_index, pool in enumerate(
            ["A", "B"]
        ):
            values = pd.to_numeric(
                units.loc[
                    units["approval_pool"].eq(pool),
                    column,
                ],
                errors="coerce",
            ).dropna().to_numpy()

            center = (
                base_positions[estimator_index]
                + (
                    -0.24
                    if pool == "A"
                    else 0.24
                )
            )

            color = cycle[pool_index]

            violin = ax.violinplot(
                values,
                positions=[center],
                widths=0.40,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )

            for body in violin["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.18)

            if len(values):
                q1, median, q3 = np.percentile(
                    values,
                    [25, 50, 75],
                )

                ax.plot(
                    [center, center],
                    [q1, q3],
                    linewidth=5,
                    solid_capstyle="round",
                    color=color,
                )

                ax.plot(
                    [
                        center - 0.08,
                        center + 0.08,
                    ],
                    [median, median],
                    linewidth=2.2,
                    color="white",
                )

                rng = np.random.default_rng(
                    100
                    + estimator_index * 10
                    + pool_index
                )

                jitter = rng.uniform(
                    -0.10,
                    0.10,
                    size=len(values),
                )

                ax.scatter(
                    center + jitter,
                    values,
                    s=18,
                    alpha=0.42,
                    linewidths=0,
                    color=color,
                    zorder=3,
                )

                ax.text(
                    center,
                    ax.get_ylim()[0] + 0.02,
                    f"n={len(values)}",
                    ha="center",
                    va="bottom",
                    fontsize=8.1,
                )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_xticks(
        base_positions,
        [
            ESTIMATOR_LABELS[name]
            for name in ESTIMATORS
        ],
    )

    ax.set_ylabel(
        "Cross-ancestry portability slope"
    )

    ax.set_title(
        "Portability distributions overlap across launch pools",
        loc="left",
    )

    panel_label(
        ax,
        "a",
        x=-0.10,
    )

    clean_axis(
        ax,
        "y",
    )

    ax.text(
        0.02,
        0.97,
        "1.0 = equal EUR and non-EUR effect magnitude",
        transform=ax.transAxes,
        va="top",
        fontsize=8.6,
    )

    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=cycle[0],
                label="Launched (Pool A)",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=cycle[1],
                label="Phase I–III (Pool B)",
            ),
        ],
        frameon=False,
        loc="lower right",
    )


def draw_portability_contrasts(
    ax: plt.Axes,
    contrasts: pd.DataFrame,
) -> None:
    plot = contrasts.reset_index(
        drop=True
    )

    y = np.arange(
        len(plot)
    )[::-1]

    estimate = pd.to_numeric(
        plot["estimate"],
        errors="coerce",
    ).to_numpy()

    lower = pd.to_numeric(
        plot["lower"],
        errors="coerce",
    ).to_numpy()

    upper = pd.to_numeric(
        plot["upper"],
        errors="coerce",
    ).to_numpy()

    ax.errorbar(
        estimate,
        y,
        xerr=np.vstack(
            [
                estimate - lower,
                upper - estimate,
            ]
        ),
        fmt="o",
        capsize=3.2,
        elinewidth=1.9,
        markeredgewidth=0,
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_yticks(
        y,
        plot["label"],
    )

    ax.set_xlabel(
        "Median portability difference\n"
        "(Pool A minus Pool B)"
    )

    ax.set_title(
        "Direct portability contrasts are centered near zero",
        loc="left",
    )

    panel_label(
        ax,
        "b",
        x=-0.15,
    )

    clean_axis(
        ax,
        "x",
    )

    ax.set_xlim(
        -0.60,
        0.60,
    )

    for yi, row in zip(
        y,
        plot.itertuples(index=False),
    ):
        p_text = ""

        if (
            hasattr(row, "p_value")
            and pd.notna(row.p_value)
        ):
            p_text = (
                f"; P={row.p_value:.2f}"
            )

        ax.text(
            0.58,
            yi,
            value_label(
                row.estimate,
                row.lower,
                row.upper,
                digits=2,
            )
            + p_text,
            ha="right",
            va="center",
            fontsize=8.3,
        )


def draw_coloc_by_pool(
    ax: plt.Axes,
    coloc: pd.DataFrame,
) -> None:
    counts = (
        coloc.groupby(
            [
                "approval_pool",
                "category",
            ]
        )
        .size()
        .unstack(fill_value=0)
        .reindex(
            index=["A", "B"],
            columns=CATEGORY_ORDER,
            fill_value=0,
        )
    )

    totals = counts.sum(axis=1)

    proportions = counts.div(
        totals,
        axis=0,
    ).fillna(0)

    y_positions = {
        "A": 1,
        "B": 0,
    }

    cycle = plt.rcParams[
        "axes.prop_cycle"
    ].by_key()["color"]

    category_colors = {
        category: cycle[
            index % len(cycle)
        ]
        for index, category in enumerate(
            CATEGORY_ORDER
        )
    }

    for pool in ["A", "B"]:
        left = 0.0

        for category in CATEGORY_ORDER:
            width = float(
                proportions.loc[
                    pool,
                    category,
                ]
            )

            count = int(
                counts.loc[
                    pool,
                    category,
                ]
            )

            if width <= 0:
                continue

            ax.barh(
                y_positions[pool],
                width,
                left=left,
                height=0.52,
                color=category_colors[category],
                edgecolor="white",
                linewidth=0.8,
            )

            if width >= 0.10:
                ax.text(
                    left + width / 2,
                    y_positions[pool],
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=9.0,
                    fontweight="bold",
                    color="white",
                )

            left += width

    ax.set_xlim(
        0,
        1,
    )

    ax.set_xticks(
        [
            0,
            0.25,
            0.50,
            0.75,
            1.0,
        ],
        [
            "0",
            "25",
            "50",
            "75",
            "100",
        ],
    )

    ax.set_xlabel(
        "Share of powered colocalization units (%)"
    )

    ax.set_yticks(
        [
            1,
            0,
        ],
        [
            (
                f"Launched (Pool A)\n"
                f"n={int(totals.get('A', 0))}"
            ),
            (
                f"Phase I–III (Pool B)\n"
                f"n={int(totals.get('B', 0))}"
            ),
        ],
    )

    ax.set_title(
        "Colocalization classes do not show a clean launch-pool split",
        loc="left",
    )

    panel_label(
        ax,
        "c",
        x=-0.06,
    )

    clean_axis(
        ax,
        "x",
    )

    ax.legend(
        handles=[
            Patch(
                facecolor=category_colors[category],
                label=CATEGORY_LABELS[category],
            )
            for category in CATEGORY_ORDER
        ],
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            -0.25,
        ),
    )

    ax.text(
        0.99,
        0.03,
        (
            "Descriptive only: "
            f"{len(coloc)} powered units."
        ),
        transform=ax.transAxes,
        ha="right",
        fontsize=8.5,
    )


def main() -> None:
    set_style()

    units = load_portability_units()
    contrasts = load_portability_contrasts()
    coloc = load_coloc_units(units)

    fig = plt.figure(
        figsize=(11.8, 7.8),
        constrained_layout=True,
    )

    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[
            1.03,
            0.97,
        ],
        width_ratios=[
            1.18,
            0.82,
        ],
    )

    ax_a = fig.add_subplot(
        grid[0, 0]
    )

    ax_b = fig.add_subplot(
        grid[0, 1]
    )

    ax_c = fig.add_subplot(
        grid[1, :]
    )

    draw_portability_distributions(
        ax_a,
        units,
    )

    draw_portability_contrasts(
        ax_b,
        contrasts,
    )

    draw_coloc_by_pool(
        ax_c,
        coloc,
    )

    save_figure(
        fig,
        "Figure5_mechanisms_by_pool",
        MAIN_DIR,
    )

    plt.close(fig)

    print(
        "Wrote Figure 5: mechanisms by launch pool."
    )


if __name__ == "__main__":
    main()
