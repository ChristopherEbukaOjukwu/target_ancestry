#!/usr/bin/env python3
"""
Step 15G2 — Sample-size audit for cross-ancestry portability.

Purpose
-------
Step 15G originally reported zero sample-size ratios because its resolver did
not include the sample-size column names actually written by Step 15B4.

This audit:
1. reads the Step 15B4 ancestry-specific sample-size metadata;
2. reports TOTAL N separately from EFFECTIVE N;
3. derives case-control effective N as 4 / (1/Ncase + 1/Ncontrol);
4. uses total N for descriptive cohort-size summaries;
5. uses effective N for sample-size-restriction sensitivity analyses;
6. does not alter the primary Step 15D portability estimates.

Inputs
------
output/15d_primary_comparison_portability.parquet
output/15b4_qc_feasibility.parquet

Outputs
-------
output/15g2_sample_size_audit.csv
output/15g2_sample_size_summary.csv
output/15g2_portability_sample_size_sensitivity.csv
output/15g2_sample_size_audit_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ESTIMATORS = ("naive", "fiqt", "deming")
Z975 = 1.959963984540054


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--portability",
        type=Path,
        default=Path("output/15d_primary_comparison_portability.parquet"),
    )
    p.add_argument(
        "--feasibility",
        type=Path,
        default=Path("output/15b4_qc_feasibility.parquet"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
    )
    p.add_argument(
        "--primary-r2",
        type=float,
        default=0.10,
    )
    return p.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")


def require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise SystemExit(f"{label} missing columns: {missing}")


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def coalesce_positive(frame: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for column in candidates:
        if column not in frame.columns:
            continue
        values = numeric(frame, column)
        usable = values.notna() & np.isfinite(values) & (values > 0)
        fill = out.isna() & usable
        out.loc[fill] = values.loc[fill]
    return out


def sample_candidates(prefix: str) -> dict[str, tuple[str, ...]]:
    if prefix == "eur":
        return {
            "total": (
                "eur_n_total",
                "eur_n_total_median",
                "eur_n",
                "n_eur",
                "eur_sample_size",
            ),
            "cases": (
                "eur_n_cases",
                "eur_n_cases_median",
                "eur_n_case",
                "eur_cases",
                "eur_ncase",
            ),
            "controls": (
                "eur_n_controls",
                "eur_n_controls_median",
                "eur_n_control",
                "eur_controls",
                "eur_nctrl",
            ),
            "neff": (
                "eur_neff",
                "eur_effective_n",
                "eur_effective_sample_size",
            ),
        }
    return {
        "total": (
            "comparison_n_total",
            "comparison_n_total_median",
            "comparison_n",
            "n_comparison",
            "comparison_sample_size",
            "non_eur_n",
        ),
        "cases": (
            "comparison_n_cases",
            "comparison_n_cases_median",
            "comparison_n_case",
            "comparison_cases",
            "comparison_ncase",
        ),
        "controls": (
            "comparison_n_controls",
            "comparison_n_controls_median",
            "comparison_n_control",
            "comparison_controls",
            "comparison_nctrl",
        ),
        "neff": (
            "comparison_neff",
            "non_eur_neff",
            "comparison_effective_n",
            "comparison_effective_sample_size",
        ),
    }


def resolve_sample_metadata(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    c = sample_candidates(prefix)

    total = coalesce_positive(frame, c["total"])
    cases = coalesce_positive(frame, c["cases"])
    controls = coalesce_positive(frame, c["controls"])

    # Backfill total N from cases + controls when an explicit total is absent.
    case_control_total = cases + controls
    fill_total = total.isna() & cases.notna() & controls.notna()
    total.loc[fill_total] = case_control_total.loc[fill_total]

    # Prefer an explicit Neff if present. Otherwise derive it for case-control
    # traits; quantitative traits fall back to total N.
    effective = coalesce_positive(frame, c["neff"])
    valid_cc = cases.notna() & controls.notna() & (cases > 0) & (controls > 0)
    derived_neff = pd.Series(np.nan, index=frame.index, dtype=float)
    derived_neff.loc[valid_cc] = 4.0 / (
        1.0 / cases.loc[valid_cc] + 1.0 / controls.loc[valid_cc]
    )
    fill_neff = effective.isna() & derived_neff.notna()
    effective.loc[fill_neff] = derived_neff.loc[fill_neff]
    fill_neff_total = effective.isna() & total.notna()
    effective.loc[fill_neff_total] = total.loc[fill_neff_total]

    return pd.DataFrame(
        {
            f"{prefix}_total_n": total,
            f"{prefix}_cases": cases,
            f"{prefix}_controls": controls,
            f"{prefix}_effective_n": effective,
        },
        index=frame.index,
    )


def meta_summary(effect: np.ndarray, se: np.ndarray) -> dict[str, float | int]:
    effect = np.asarray(effect, dtype=float)
    se = np.asarray(se, dtype=float)
    keep = np.isfinite(effect) & np.isfinite(se) & (se > 0)
    effect = effect[keep]
    se = se[keep]
    n = len(effect)

    if n == 0:
        return {
            "n": 0,
            "random_mean": math.nan,
            "random_ci_lower": math.nan,
            "random_ci_upper": math.nan,
            "tau2": math.nan,
            "i2_percent": math.nan,
        }

    w = 1.0 / np.square(se)
    fixed_mean = float(np.sum(w * effect) / np.sum(w))
    q = float(np.sum(w * np.square(effect - fixed_mean)))
    q_df = max(n - 1, 0)
    c = float(np.sum(w) - np.sum(np.square(w)) / np.sum(w))

    tau2 = 0.0
    if n > 1 and c > 0:
        tau2 = max(0.0, (q - q_df) / c)

    wr = 1.0 / (np.square(se) + tau2)
    random_mean = float(np.sum(wr * effect) / np.sum(wr))
    random_se = float(math.sqrt(1.0 / np.sum(wr)))

    i2 = (
        max(0.0, (q - q_df) / q) * 100.0
        if n > 1 and q > 0
        else 0.0
    )

    return {
        "n": int(n),
        "random_mean": random_mean,
        "random_ci_lower": random_mean - Z975 * random_se,
        "random_ci_upper": random_mean + Z975 * random_se,
        "tau2": float(tau2),
        "i2_percent": float(i2),
    }


def main() -> int:
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)

    require_file(a.portability, "Step 15D portability file")
    require_file(a.feasibility, "Step 15B4 feasibility file")

    portability = pd.read_parquet(a.portability)
    feasibility = pd.read_parquet(a.feasibility)

    require_columns(
        portability,
        [
            "comparison_uid",
            "gene_trait_uid",
            "estimator",
            "slope",
            "ci_lower",
            "ci_upper",
            "n_variants",
        ],
        "Step 15D portability",
    )
    require_columns(
        feasibility,
        ["comparison_uid", "comparison_population", "candidate_source"],
        "Step 15B4 feasibility",
    )

    if "r2_threshold" in portability.columns:
        portability = portability[
            np.isclose(
                pd.to_numeric(portability["r2_threshold"], errors="coerce"),
                a.primary_r2,
                rtol=0.0,
                atol=1e-12,
            )
        ].copy()

    portability = portability[
        portability["estimator"].astype(str).isin(ESTIMATORS)
    ].copy()

    for col in ("slope", "ci_lower", "ci_upper", "n_variants"):
        portability[col] = pd.to_numeric(portability[col], errors="coerce")

    portability["slope_se_approx"] = (
        portability["ci_upper"] - portability["ci_lower"]
    ) / (2.0 * Z975)
    portability.loc[
        ~np.isfinite(portability["slope_se_approx"])
        | (portability["slope_se_approx"] <= 0),
        "slope_se_approx",
    ] = np.nan

    feasibility = feasibility.drop_duplicates("comparison_uid").copy()

    feasibility = feasibility[
        feasibility["comparison_uid"].isin(portability["comparison_uid"])
    ].copy()

    eur = resolve_sample_metadata(feasibility, "eur")
    comp = resolve_sample_metadata(feasibility, "comparison")
    feasibility = pd.concat(
        [feasibility.reset_index(drop=True), eur.reset_index(drop=True), comp.reset_index(drop=True)],
        axis=1,
    )

    feasibility["total_n_ratio"] = (
        feasibility["comparison_total_n"] / feasibility["eur_total_n"]
    )
    feasibility["effective_n_ratio"] = (
        feasibility["comparison_effective_n"] / feasibility["eur_effective_n"]
    )

    keep = [
        "comparison_uid",
        "gene_trait_uid",
        "gene",
        "candidate_trait_name",
        "candidate_source",
        "comparison_population",
        "eur_coloc_type",
        "comparison_coloc_type",
        "eur_total_n",
        "comparison_total_n",
        "total_n_ratio",
        "eur_cases",
        "eur_controls",
        "comparison_cases",
        "comparison_controls",
        "eur_effective_n",
        "comparison_effective_n",
        "effective_n_ratio",
    ]
    keep = [c for c in keep if c in feasibility.columns]

    audit = feasibility[keep].copy()
    audit.to_csv(a.output_dir / "15g2_sample_size_audit.csv", index=False)
    audit.to_parquet(
        a.output_dir / "15g2_sample_size_audit.parquet",
        index=False,
        compression="zstd",
    )

    summary_rows = []
    groups = [("ALL", audit)] + list(
        audit.groupby("comparison_population", dropna=False)
    )
    for label, group in groups:
        total_ratio = pd.to_numeric(group["total_n_ratio"], errors="coerce").dropna()
        effective_ratio = pd.to_numeric(
            group["effective_n_ratio"], errors="coerce"
        ).dropna()
        eur_total = pd.to_numeric(group["eur_total_n"], errors="coerce").dropna()
        cmp_total = pd.to_numeric(
            group["comparison_total_n"], errors="coerce"
        ).dropna()

        summary_rows.append(
            {
                "comparison_population": label,
                "n_comparisons": int(len(group)),
                "n_with_total_n": int(
                    group[["eur_total_n", "comparison_total_n"]]
                    .notna()
                    .all(axis=1)
                    .sum()
                ),
                "median_eur_total_n": (
                    float(eur_total.median()) if len(eur_total) else math.nan
                ),
                "median_comparison_total_n": (
                    float(cmp_total.median()) if len(cmp_total) else math.nan
                ),
                "median_total_n_ratio": (
                    float(total_ratio.median()) if len(total_ratio) else math.nan
                ),
                "q25_total_n_ratio": (
                    float(total_ratio.quantile(0.25)) if len(total_ratio) else math.nan
                ),
                "q75_total_n_ratio": (
                    float(total_ratio.quantile(0.75)) if len(total_ratio) else math.nan
                ),
                "n_with_effective_n": int(
                    group[["eur_effective_n", "comparison_effective_n"]]
                    .notna()
                    .all(axis=1)
                    .sum()
                ),
                "median_effective_n_ratio": (
                    float(effective_ratio.median())
                    if len(effective_ratio)
                    else math.nan
                ),
                "q25_effective_n_ratio": (
                    float(effective_ratio.quantile(0.25))
                    if len(effective_ratio)
                    else math.nan
                ),
                "q75_effective_n_ratio": (
                    float(effective_ratio.quantile(0.75))
                    if len(effective_ratio)
                    else math.nan
                ),
            }
        )

    sample_summary = pd.DataFrame(summary_rows)
    sample_summary.to_csv(
        a.output_dir / "15g2_sample_size_summary.csv", index=False
    )

    merged = portability.merge(
        audit,
        on="comparison_uid",
        how="left",
        validate="many_to_one",
        suffixes=("", "_sample"),
    )

    sensitivity_rows = []
    for estimator, group in merged.groupby("estimator", sort=False):
        rules = [
            ("all", pd.Series(True, index=group.index)),
            (
                "effective_N_ratio_ge_0.01",
                group["effective_n_ratio"] >= 0.01,
            ),
            (
                "effective_N_ratio_ge_0.05",
                group["effective_n_ratio"] >= 0.05,
            ),
            (
                "effective_N_ratio_ge_0.10",
                group["effective_n_ratio"] >= 0.10,
            ),
        ]

        available_ratio = group["effective_n_ratio"].dropna()
        if len(available_ratio):
            median_ratio = float(available_ratio.median())
            rules.append(
                (
                    "more_balanced_half",
                    group["effective_n_ratio"] >= median_ratio,
                )
            )

        for rule_name, mask in rules:
            subset = group[mask.fillna(False)].copy()
            stats = meta_summary(
                subset["slope"].to_numpy(),
                subset["slope_se_approx"].to_numpy(),
            )
            sensitivity_rows.append(
                {
                    "estimator": estimator,
                    "restriction": rule_name,
                    "n_comparisons": int(len(subset)),
                    "n_units": int(subset["gene_trait_uid"].nunique()),
                    "median_slope": (
                        float(subset["slope"].median())
                        if len(subset)
                        else math.nan
                    ),
                    "median_effective_n_ratio": (
                        float(subset["effective_n_ratio"].median())
                        if subset["effective_n_ratio"].notna().any()
                        else math.nan
                    ),
                    **stats,
                }
            )

    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(
        a.output_dir / "15g2_portability_sample_size_sensitivity.csv",
        index=False,
    )

    n_comparisons = int(portability["comparison_uid"].nunique())
    n_with_total = int(
        audit[["eur_total_n", "comparison_total_n"]].notna().all(axis=1).sum()
    )
    n_with_effective = int(
        audit[["eur_effective_n", "comparison_effective_n"]]
        .notna()
        .all(axis=1)
        .sum()
    )

    summary_json = {
        "step": "15G2",
        "status": "PASS",
        "primary_r2": a.primary_r2,
        "n_portability_comparisons": n_comparisons,
        "n_with_total_sample_sizes": n_with_total,
        "n_with_effective_sample_sizes": n_with_effective,
        "interpretation": {
            "total_n": "Descriptive ancestry-specific cohort sample size.",
            "effective_n": (
                "For case-control traits, 4/(1/Ncase+1/Ncontrol); "
                "for quantitative traits, total N. Used only for precision sensitivity."
            ),
            "sample_size_sensitivity": (
                "Exploratory robustness analysis; does not replace the original "
                "bootstrap portability estimates."
            ),
        },
        "outputs": {
            "audit": "15g2_sample_size_audit.csv",
            "summary": "15g2_sample_size_summary.csv",
            "sensitivity": "15g2_portability_sample_size_sensitivity.csv",
        },
    }
    (a.output_dir / "15g2_sample_size_audit_summary.json").write_text(
        json.dumps(summary_json, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=" * 78)
    print("STEP 15G2 — SAMPLE-SIZE AUDIT")
    print("=" * 78)
    print(f"Portability comparisons:              {n_comparisons}")
    print(f"With total N in both ancestries:      {n_with_total}")
    print(f"With effective N in both ancestries:  {n_with_effective}")
    print()
    print("Sample-size summary:")
    print(sample_summary.to_string(index=False))
    print()
    print("Portability sensitivity to effective-N balance:")
    print(
        sensitivity[
            [
                "estimator",
                "restriction",
                "n_comparisons",
                "n_units",
                "median_slope",
                "random_mean",
                "random_ci_lower",
                "random_ci_upper",
            ]
        ].to_string(index=False)
    )
    print()
    print("Outputs written to:", a.output_dir)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
