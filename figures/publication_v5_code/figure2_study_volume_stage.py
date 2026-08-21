#!/usr/bin/env python3
"""Main Figure 2: study volume and evidence stage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from loaders import load_trails
from style import (
    MAIN_DIR,
    clean_axis,
    panel_label,
    raincloud,
    save_figure,
    set_style,
)


BIN_ORDER = ["1–2", "3–5", "6–10", "11+"]


def main():
    set_style()

    trails = load_trails()

    trails["study_bin"] = pd.cut(
        trails["n_studies_total"],
        bins=[0, 2, 5, 10, np.inf],
        labels=BIN_ORDER,
        include_lowest=True,
    )

    summary = (
        trails.groupby(
            ["study_bin", "pool"],
            observed=True,
        )
        .agg(
            mean_all=("n_ancestries_all", "mean"),
            mean_initial=("n_ancestries_initial", "mean"),
            mean_replication=("n_ancestries_replication", "mean"),
            n=("n_ancestries_all", "size"),
        )
        .reset_index()
    )

    gap_rows = []

    for index, study_bin in enumerate(BIN_ORDER):
        subset = trails[trails["study_bin"].eq(study_bin)]

        a = subset.loc[
            subset["pool"].eq("A"),
            "n_ancestries_all",
        ].dropna().to_numpy()

        b = subset.loc[
            subset["pool"].eq("B"),
            "n_ancestries_all",
        ].dropna().to_numpy()

        if len(a) == 0 or len(b) == 0:
            continue

        rng = np.random.default_rng(900 + index)

        draws = np.array(
            [
                np.mean(rng.choice(a, len(a), replace=True))
                - np.mean(rng.choice(b, len(b), replace=True))
                for _ in range(5000)
            ]
        )

        gap_rows.append(
            {
                "study_bin": study_bin,
                "difference": np.mean(a) - np.mean(b),
                "lower": np.percentile(draws, 2.5),
                "upper": np.percentile(draws, 97.5),
            }
        )

    gaps = pd.DataFrame(gap_rows)

    fig = plt.figure(
        figsize=(11.4, 8.0),
        constrained_layout=True,
    )

    gs = fig.add_gridspec(2, 2)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    raincloud(
        ax_a,
        {
            "A": trails.loc[
                trails["pool"].eq("A"),
                "n_studies_total",
            ].dropna().to_numpy(),
            "B": trails.loc[
                trails["pool"].eq("B"),
                "n_studies_total",
            ].dropna().to_numpy(),
        },
        positions={"A": 0, "B": 1},
        width=0.50,
    )

    ax_a.set_yscale("log")

    ax_a.set_xticks(
        [0, 1],
        ["Launched\nPool A", "Phase I–III\nPool B"],
    )

    ax_a.set_ylabel(
        "Supporting studies per pair (log scale)"
    )

    ax_a.set_title(
        "Launched pairs have larger evidence bases",
        loc="left",
    )

    panel_label(ax_a, "a")
    clean_axis(ax_a, "y")

    x = np.arange(len(BIN_ORDER))

    for pool in ["A", "B"]:
        subset = (
            summary[summary["pool"].eq(pool)]
            .set_index("study_bin")
            .reindex(BIN_ORDER)
        )

        ax_b.plot(
            x,
            subset["mean_all"],
            marker="o",
            label=(
                "Launched"
                if pool == "A"
                else "Phase I–III"
            ),
        )

    ax_b.set_xticks(x, BIN_ORDER)
    ax_b.set_xlabel("Supporting studies per pair")
    ax_b.set_ylabel(
        "Mean number of represented ancestries"
    )
    ax_b.set_title(
        "Ancestry breadth rises with study volume",
        loc="left",
    )
    panel_label(ax_b, "b")
    clean_axis(ax_b, "y")
    ax_b.legend(frameon=False)

    y = np.arange(len(gaps))[::-1]

    ax_c.errorbar(
        gaps["difference"],
        y,
        xerr=np.vstack(
            [
                gaps["difference"] - gaps["lower"],
                gaps["upper"] - gaps["difference"],
            ]
        ),
        fmt="o",
        capsize=3,
        elinewidth=1.8,
    )

    ax_c.axvline(0, linestyle="--", linewidth=1.0)

    ax_c.set_yticks(
        y,
        gaps["study_bin"].astype(str),
    )

    ax_c.set_xlabel(
        "Mean ancestry difference\n"
        "(Launched minus Phase I–III)"
    )

    ax_c.set_ylabel("Supporting studies per pair")

    ax_c.set_title(
        "The gap is concentrated among heavily studied pairs",
        loc="left",
    )

    panel_label(ax_c, "c")
    clean_axis(ax_c, "x")

    line_specs = [
        (
            "mean_initial",
            "A",
            "-",
            "o",
            "Discovery – launched",
        ),
        (
            "mean_initial",
            "B",
            "-",
            "o",
            "Discovery – Phase I–III",
        ),
        (
            "mean_replication",
            "A",
            "--",
            "s",
            "Replication – launched",
        ),
        (
            "mean_replication",
            "B",
            "--",
            "s",
            "Replication – Phase I–III",
        ),
    ]

    for column, pool, linestyle, marker, label in line_specs:
        subset = (
            summary[summary["pool"].eq(pool)]
            .set_index("study_bin")
            .reindex(BIN_ORDER)
        )

        ax_d.plot(
            x,
            subset[column],
            linestyle=linestyle,
            marker=marker,
            label=label,
        )

    ax_d.set_xticks(x, BIN_ORDER)
    ax_d.set_xlabel("Supporting studies per pair")
    ax_d.set_ylabel(
        "Mean number of represented ancestries"
    )

        # Explicit panel-D label and title to prevent overlap with panel C.
    ax_d.text(
        0.00,
        1.015,
        "d",
        transform=ax_d.transAxes,
        ha="left",
        va="bottom",
        fontsize=15,
        fontweight="bold",
        clip_on=False,
    )

    ax_d.text(
        0.075,
        1.015,
        "Discovery and replication contribute differently",
        transform=ax_d.transAxes,
        ha="left",
        va="bottom",
        fontsize=13.5,
        fontweight="bold",
        clip_on=False,
    )

    clean_axis(ax_d, "y")

    ax_d.legend(
        frameon=False,
        fontsize=8.1,
    )

    save_figure(
        fig,
        "Figure2_study_volume_and_stage",
        MAIN_DIR,
    )

    plt.close(fig)

    print("Wrote Figure 2.")


if __name__ == "__main__":
    main()
