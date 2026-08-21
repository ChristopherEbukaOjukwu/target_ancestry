#!/usr/bin/env python3
"""Supplementary figures for selection, feasibility, coverage, and power."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from style import (
    ROOT,
    SUPP_DIR,
    clean_axis,
    panel_label,
    pick_column,
    read_table,
    save_figure,
    set_style,
)


def figure_s1_selection_retention() -> None:
    candidates = [
        ROOT / "step9/output/09_retention_by_phase.csv",
        ROOT / "step9/output/retention_by_phase.csv",
    ]

    path = next(
        (
            candidate
            for candidate in candidates
            if candidate.exists()
        ),
        None,
    )

    if path is None:
        print(
            "Skipping Figure S1: "
            "no retention-by-phase table found."
        )
        return

    frame = read_table(path)

    group = pick_column(
        frame,
        [
            "phase",
            "clinical_phase",
            "stage",
            "ccat",
        ],
        label="clinical phase or category",
    )

    retention = pick_column(
        frame,
        [
            "retention_rate",
            "retention",
            "included_fraction",
        ],
        label="retention rate",
    )

    data = frame[
        [
            group,
            retention,
        ]
    ].copy()

    data[retention] = pd.to_numeric(
        data[retention],
        errors="coerce",
    )

    data = data.dropna(
        subset=[retention]
    )

    fig, ax = plt.subplots(
        figsize=(6.8, 4.3),
        constrained_layout=True,
    )

    bars = ax.bar(
        data[group].astype(str),
        data[retention],
    )

    ax.set_ylabel(
        "Retention rate"
    )

    ax.set_title(
        "Retention across clinical categories",
        loc="left",
    )

    panel_label(
        ax,
        "a",
        x=-0.08,
    )

    clean_axis(
        ax,
        "y",
    )

    for bar, value in zip(
        bars,
        data[retention],
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.015,
            f"{value:.2f}",
            ha="center",
            fontsize=8.5,
        )

    save_figure(
        fig,
        "FigureS1_selection_retention",
        SUPP_DIR,
    )

    plt.close(fig)


def figure_s2_direct_disease_feasibility() -> None:
    stages = [
        "Directly mapped\npairs",
        "Ancestry-specific\nunits",
        "Any usable EUR-\nsignificant SNP",
        "Passed raw\n10-SNP gate",
        "Unique target-\nindication units",
    ]

    counts = np.array(
        [140, 169, 24, 11, 8]
    )

    x = np.arange(
        len(stages)
    )

    fig, ax = plt.subplots(
        figsize=(8.0, 4.6),
        constrained_layout=True,
    )

    ax.plot(
        x,
        counts,
        marker="o",
        linewidth=2.2,
    )

    ax.fill_between(
        x,
        counts,
        alpha=0.10,
    )

    ax.set_xticks(
        x,
        stages,
    )

    ax.set_ylabel(
        "Count"
    )

    ax.set_title(
        (
            "Direct disease endpoints were too sparse "
            "for an approval comparison"
        ),
        loc="left",
    )

    panel_label(
        ax,
        "a",
        x=-0.08,
    )

    clean_axis(
        ax,
        "y",
    )

    for xi, value in zip(
        x,
        counts,
    ):
        ax.text(
            xi,
            value + 5,
            str(value),
            ha="center",
            fontweight="bold",
        )

    ax.text(
        0.02,
        0.05,
        (
            "Final set: 3 launched units and 5 Phase I–III units; "
            "no gene occurred in both pools."
        ),
        transform=ax.transAxes,
        fontsize=9.0,
    )

    save_figure(
        fig,
        "FigureS2_direct_disease_feasibility",
        SUPP_DIR,
    )

    plt.close(fig)


def figure_s3_mechanistic_coverage() -> None:
    path = (
        ROOT
        / "step15/output/15f_mechanistic_coverage.csv"
    )

    if not path.exists():
        print(
            "Skipping Figure S3: "
            "mechanistic coverage table not found."
        )
        return

    frame = read_table(path)

    label = pick_column(
        frame,
        [
            "analysis_stage",
            "stage",
            "universe",
            "metric",
        ],
        label="coverage stage",
    )

    count = pick_column(
        frame,
        [
            "n_units",
            "count",
            "n",
            "value",
        ],
        label="coverage count",
    )

    data = frame[
        [
            label,
            count,
        ]
    ].copy()

    data[count] = pd.to_numeric(
        data[count],
        errors="coerce",
    )

    data = (
        data.dropna()
        .sort_values(count)
    )

    fig, ax = plt.subplots(
        figsize=(7.0, 4.8),
        constrained_layout=True,
    )

    ax.barh(
        data[label].astype(str),
        data[count],
    )

    ax.set_xlabel(
        "Units or comparisons"
    )

    ax.set_title(
        (
            "Mechanistic analyses were restricted "
            "by data availability"
        ),
        loc="left",
    )

    panel_label(
        ax,
        "a",
        x=-0.08,
    )

    clean_axis(
        ax,
        "x",
    )

    save_figure(
        fig,
        "FigureS3_mechanistic_coverage",
        SUPP_DIR,
    )

    plt.close(fig)


def figure_s4_equivalence_and_power() -> None:
    values = pd.DataFrame(
        {
            "quantity": [
                "Exploratory equivalence bound",
                "Minimum detectable OR per SD",
            ],
            "value": [
                1.25,
                1.575,
            ],
        }
    )

    fig, ax = plt.subplots(
        figsize=(6.5, 3.8),
        constrained_layout=True,
    )

    ax.barh(
        values["quantity"],
        values["value"],
    )

    ax.axvline(
        1.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_xlabel(
        "Odds ratio"
    )

    ax.set_title(
        "The study could not establish tight equivalence",
        loc="left",
    )

    panel_label(
        ax,
        "a",
        x=-0.08,
    )

    clean_axis(
        ax,
        "x",
    )

    ax.text(
        0.99,
        0.05,
        (
            "OR 1.20 equivalence failed; "
            "MDE ≈ 1.575 per SD."
        ),
        transform=ax.transAxes,
        ha="right",
        fontsize=8.8,
    )

    save_figure(
        fig,
        "FigureS4_equivalence_and_power",
        SUPP_DIR,
    )

    plt.close(fig)


def run_optional(name, function) -> None:
    try:
        function()
    except Exception as error:
        print(
            f"Skipping {name} after error: "
            f"{type(error).__name__}: {error}"
        )


def main():
    set_style()

    run_optional(
        "Figure S1",
        figure_s1_selection_retention,
    )

    run_optional(
        "Figure S2",
        figure_s2_direct_disease_feasibility,
    )

    run_optional(
        "Figure S3",
        figure_s3_mechanistic_coverage,
    )

    run_optional(
        "Figure S4",
        figure_s4_equivalence_and_power,
    )

    print(
        "Supplementary figure pass completed."
    )


if __name__ == "__main__":
    main()
