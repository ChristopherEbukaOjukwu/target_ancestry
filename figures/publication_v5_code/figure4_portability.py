#!/usr/bin/env python3
"""Main Figure 4: cross-ancestry portability and launch status."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from style import (
    ROOT,
    MAIN_DIR,
    clean_axis,
    normalize_pool,
    panel_label,
    pick_column,
    raincloud,
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


def load_contrasts() -> pd.DataFrame:
    frame = read_table(
        ROOT
        / "step15/output/15f_portability_pool_contrasts.csv"
    )

    estimator = pick_column(
        frame,
        ["estimator", "method", "slope_estimator"],
        label="portability estimator",
    )

    estimate = pick_column(
        frame,
        [
            "median_difference_A_minus_B",
            "median_difference",
            "difference",
            "estimate",
            "pool_a_minus_b",
        ],
        label="Pool A minus Pool B difference",
    )

    lower = pick_column(
        frame,
        [
            "difference_ci_lower",
            "ci_low",
            "ci_lower",
            "bootstrap_ci_low",
            "lower",
            "lower_ci",
        ],
        label="contrast lower CI",
    )

    upper = pick_column(
        frame,
        [
            "difference_ci_upper",
            "ci_high",
            "ci_upper",
            "bootstrap_ci_high",
            "upper",
            "upper_ci",
        ],
        label="contrast upper CI",
    )

    pvalue = pick_column(
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

    columns = [estimator, estimate, lower, upper]

    if pvalue:
        columns.append(pvalue)

    data = frame[columns].copy()

    data.columns = [
        "estimator",
        "estimate",
        "lower",
        "upper",
    ] + (["p_value"] if pvalue else [])

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

    data = data.sort_values("_order")

    data["label"] = data["estimator"].map(
        ESTIMATOR_LABELS
    )

    return data


def main():
    set_style()

    units = read_table(
        ROOT
        / "step15/output/15f_mechanistic_units.parquet"
    ).copy()

    units["approval_pool"] = normalize_pool(
        units["approval_pool"]
    )

    units = units[
        units["portability_final_available"].astype(bool)
        & units["approval_pool"].isin(["A", "B"])
    ].copy()

    contrasts = load_contrasts()

    comparisons_path = (
        ROOT
        / "step15/output/15f_mechanistic_comparisons.parquet"
    )

    comparisons = (
        read_table(comparisons_path)
        if comparisons_path.exists()
        else pd.DataFrame()
    )

    fig = plt.figure(
        figsize=(11.5, 7.6),
        constrained_layout=True,
    )

    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.03, 0.97],
        width_ratios=[1.12, 0.88],
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    base = np.arange(len(ESTIMATORS)) * 2.0

    for index, estimator in enumerate(ESTIMATORS):
        column = f"portability_{estimator}_slope"

        raincloud(
            ax_a,
            {
                "A": pd.to_numeric(
                    units.loc[
                        units["approval_pool"].eq("A"),
                        column,
                    ],
                    errors="coerce",
                ).dropna().to_numpy(),
                "B": pd.to_numeric(
                    units.loc[
                        units["approval_pool"].eq("B"),
                        column,
                    ],
                    errors="coerce",
                ).dropna().to_numpy(),
            },
            positions={
                "A": base[index] - 0.24,
                "B": base[index] + 0.24,
            },
            width=0.40,
            seed=50 + index,
        )

    ax_a.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax_a.set_xticks(
        base,
        [
            ESTIMATOR_LABELS[name]
            for name in ESTIMATORS
        ],
    )

    ax_a.set_ylabel(
        "Cross-ancestry portability slope"
    )

    ax_a.set_title(
        "Portability is moderate and overlaps across launch status",
        loc="left",
    )

    panel_label(ax_a, "a")
    clean_axis(ax_a, "y")

    ax_a.text(
        0.02,
        0.97,
        "1.0 = equal EUR and non-EUR effect magnitude",
        transform=ax_a.transAxes,
        va="top",
        fontsize=8.8,
    )

    cycle = plt.rcParams[
        "axes.prop_cycle"
    ].by_key()["color"]

    ax_a.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=cycle[0],
                label="Launched",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=cycle[1],
                label="Phase I–III",
            ),
        ],
        frameon=False,
        loc="lower right",
    )

    y = np.arange(len(contrasts))[::-1]

    estimate = contrasts["estimate"].to_numpy()
    lower = contrasts["lower"].to_numpy()
    upper = contrasts["upper"].to_numpy()

    ax_b.errorbar(
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

    ax_b.axvline(
        0,
        linestyle="--",
        linewidth=1.0,
    )

    ax_b.set_yticks(
        y,
        contrasts["label"],
    )

    ax_b.set_xlabel(
        "Median portability difference\n"
        "(Launched minus Phase I–III)"
    )

    ax_b.set_title(
        "Direct pool contrasts are centered near zero",
        loc="left",
    )

    panel_label(ax_b, "b")
    clean_axis(ax_b, "x")

    ax_b.set_xlim(-0.60, 0.60)

    for yi, row in zip(
        y,
        contrasts.itertuples(index=False),
    ):
        p_text = (
            f"; P = {row.p_value:.2f}"
            if hasattr(row, "p_value")
            and pd.notna(row.p_value)
            else ""
        )

        ax_b.text(
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
            fontsize=8.4,
        )

    if not comparisons.empty:
        population = pick_column(
            comparisons,
            [
                "comparison_population",
                "comparison_ancestry",
                "ancestry",
            ],
            label="comparison ancestry",
        )

        comparisons["comparison_population"] = (
            comparisons[population].astype(str)
        )

        rows = []

        for ancestry, subset in comparisons.groupby(
            "comparison_population"
        ):
            for estimator in ESTIMATORS:
                column = f"portability_{estimator}_slope"

                if column not in subset.columns:
                    continue

                values = pd.to_numeric(
                    subset[column],
                    errors="coerce",
                ).dropna().to_numpy()

                if len(values):
                    rows.append(
                        {
                            "ancestry": ancestry,
                            "estimator": estimator,
                            "median": np.median(values),
                            "q1": np.percentile(values, 25),
                            "q3": np.percentile(values, 75),
                        }
                    )

        ancestry_summary = pd.DataFrame(rows)

        ancestries = sorted(
            ancestry_summary["ancestry"].unique()
        )

        base_x = np.arange(len(ancestries))

        for index, estimator in enumerate(ESTIMATORS):
            subset = (
                ancestry_summary[
                    ancestry_summary["estimator"].eq(estimator)
                ]
                .set_index("ancestry")
                .reindex(ancestries)
            )

            offset = (index - 1) * 0.18

            ax_c.errorbar(
                base_x + offset,
                subset["median"],
                yerr=np.vstack(
                    [
                        subset["median"] - subset["q1"],
                        subset["q3"] - subset["median"],
                    ]
                ),
                fmt="o",
                capsize=3,
                label=ESTIMATOR_LABELS[estimator],
            )

        ax_c.axhline(
            1.0,
            linestyle="--",
            linewidth=1.0,
        )

        ax_c.set_xticks(
            base_x,
            ancestries,
        )

        ax_c.set_xlabel(
            "Comparison ancestry"
        )

        ax_c.set_ylabel(
            "Median portability slope (IQR)"
        )

        ax_c.set_title(
            "Attenuation appears across comparison ancestries",
            loc="left",
        )

        panel_label(
            ax_c,
            "c",
            x=-0.06,
        )

        clean_axis(
            ax_c,
            "y",
        )

        ax_c.legend(
            frameon=False,
            ncol=3,
        )

    else:
        ax_c.axis("off")

        panel_label(
            ax_c,
            "c",
            x=-0.06,
        )

        ax_c.text(
            0.5,
            0.5,
            "Comparison-level portability file was not found.",
            ha="center",
            va="center",
            transform=ax_c.transAxes,
        )

    save_figure(
        fig,
        "Figure4_cross_ancestry_portability",
        MAIN_DIR,
    )

    plt.close(fig)

    print("Wrote Figure 4.")


if __name__ == "__main__":
    main()
