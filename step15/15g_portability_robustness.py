#!/usr/bin/env python3
"""
Step 15G — Portability robustness analyses.

This script adds three analyses that were not part of the original Step 15D/15E
workflow:

1. Precision-aware portability summaries
   - reports EUR and comparison-ancestry sample-size asymmetry when metadata exist;
   - approximates each comparison-level slope SE from its bootstrap 95% CI;
   - computes fixed- and random-effects inverse-variance summaries;
   - repeats the summary after transparent precision/sample-size/variant-count filters.

2. Colocalization-stratified portability
   - joins portability and powered colocalization by the exact comparison_uid;
   - summarizes portability separately for ROBUST_SHARED, ROBUST_DISTINCT,
     and PRIOR_SENSITIVE comparisons;
   - remains descriptive because the powered overlap is expected to be small.

3. Locus/unit heterogeneity
   - first combines multiple ancestry comparisons within each gene-trait-source unit;
   - then computes a random-effects summary across units;
   - reports tau^2, Cochran Q, and I^2;
   - writes locus-specific estimates and a FIQT forest plot.

Important interpretation boundary
---------------------------------
The comparison-level CIs are percentile-bootstrap intervals. Their SEs are
approximated as (upper - lower)/(2*1.96) solely for inverse-variance sensitivity
analysis. This does not replace the original bootstrap summaries.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


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
        "--variant-effects",
        type=Path,
        default=Path("output/15d_variant_effects_primary.parquet"),
    )
    p.add_argument(
        "--feasibility",
        type=Path,
        default=Path("output/15b4_qc_feasibility.parquet"),
    )
    p.add_argument(
        "--unit-universe",
        type=Path,
        default=Path("output/15b5_unit_analysis_universe.parquet"),
    )
    p.add_argument(
        "--coloc-stability",
        type=Path,
        default=Path("output/15e2_coloc_comparison_stability.parquet"),
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
        out.loc[out.isna() & usable] = values.loc[out.isna() & usable]
    return out


def resolve_sample_size(frame: pd.DataFrame, prefix: str) -> pd.Series:
    """Prefer explicit Neff, then derive case-control Neff, then use total N."""
    if prefix == "eur":
        neff = ("eur_neff", "eur_effective_n", "eur_effective_sample_size")
        total = ("eur_n", "n_eur", "eur_sample_size")
        cases = ("eur_n_case", "eur_n_cases", "eur_cases", "eur_ncase")
        controls = (
            "eur_n_control",
            "eur_n_controls",
            "eur_controls",
            "eur_nctrl",
        )
    else:
        neff = (
            "comparison_neff",
            "non_eur_neff",
            "comparison_effective_n",
            "comparison_effective_sample_size",
        )
        total = (
            "comparison_n",
            "n_comparison",
            "comparison_sample_size",
            "non_eur_n",
        )
        cases = (
            "comparison_n_case",
            "comparison_n_cases",
            "comparison_cases",
            "comparison_ncase",
        )
        controls = (
            "comparison_n_control",
            "comparison_n_controls",
            "comparison_controls",
            "comparison_nctrl",
        )

    result = coalesce_positive(frame, neff)
    n_case = coalesce_positive(frame, cases)
    n_control = coalesce_positive(frame, controls)
    valid_cc = n_case.notna() & n_control.notna()
    derived = pd.Series(np.nan, index=frame.index, dtype=float)
    derived.loc[valid_cc] = 4.0 / (
        1.0 / n_case.loc[valid_cc] + 1.0 / n_control.loc[valid_cc]
    )
    result.loc[result.isna() & derived.notna()] = derived.loc[
        result.isna() & derived.notna()
    ]

    total_n = coalesce_positive(frame, total)
    result.loc[result.isna() & total_n.notna()] = total_n.loc[
        result.isna() & total_n.notna()
    ]
    return result


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
            "fixed_mean": math.nan,
            "fixed_se": math.nan,
            "fixed_ci_lower": math.nan,
            "fixed_ci_upper": math.nan,
            "random_mean": math.nan,
            "random_se": math.nan,
            "random_ci_lower": math.nan,
            "random_ci_upper": math.nan,
            "tau2": math.nan,
            "q": math.nan,
            "q_df": 0,
            "i2_percent": math.nan,
        }

    w = 1.0 / np.square(se)
    fixed_mean = float(np.sum(w * effect) / np.sum(w))
    fixed_se = float(math.sqrt(1.0 / np.sum(w)))
    q = float(np.sum(w * np.square(effect - fixed_mean)))
    q_df = max(n - 1, 0)

    c = float(np.sum(w) - np.sum(np.square(w)) / np.sum(w))
    tau2 = 0.0
    if n > 1 and c > 0:
        tau2 = max(0.0, (q - q_df) / c)

    wr = 1.0 / (np.square(se) + tau2)
    random_mean = float(np.sum(wr * effect) / np.sum(wr))
    random_se = float(math.sqrt(1.0 / np.sum(wr)))

    if n > 1 and q > 0:
        i2 = max(0.0, (q - q_df) / q) * 100.0
    else:
        i2 = 0.0

    return {
        "n": int(n),
        "fixed_mean": fixed_mean,
        "fixed_se": fixed_se,
        "fixed_ci_lower": fixed_mean - Z975 * fixed_se,
        "fixed_ci_upper": fixed_mean + Z975 * fixed_se,
        "random_mean": random_mean,
        "random_se": random_se,
        "random_ci_lower": random_mean - Z975 * random_se,
        "random_ci_upper": random_mean + Z975 * random_se,
        "tau2": float(tau2),
        "q": q,
        "q_df": int(q_df),
        "i2_percent": float(i2),
    }


def unique_metadata(frame: pd.DataFrame, key: str, columns: list[str]) -> pd.DataFrame:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return frame[[key]].drop_duplicates()
    rows: list[dict[str, object]] = []
    for value, group in frame.groupby(key, sort=False, dropna=False):
        row: dict[str, object] = {key: value}
        for column in available:
            observed = group[column].dropna().drop_duplicates()
            row[column] = observed.iloc[0] if len(observed) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_portability(args: argparse.Namespace) -> pd.DataFrame:
    require_file(args.portability, "portability estimates")
    frame = pd.read_parquet(args.portability)
    require_columns(
        frame,
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

    if "r2_threshold" in frame.columns:
        frame = frame[
            np.isclose(
                numeric(frame, "r2_threshold"),
                args.primary_r2,
                rtol=0.0,
                atol=1e-12,
            )
        ].copy()

    frame = frame[frame["estimator"].astype(str).isin(ESTIMATORS)].copy()
    frame["slope"] = numeric(frame, "slope")
    frame["ci_lower"] = numeric(frame, "ci_lower")
    frame["ci_upper"] = numeric(frame, "ci_upper")
    frame["n_variants"] = numeric(frame, "n_variants")
    frame["slope_se_approx"] = (
        frame["ci_upper"] - frame["ci_lower"]
    ) / (2.0 * Z975)
    frame.loc[
        ~np.isfinite(frame["slope_se_approx"])
        | (frame["slope_se_approx"] <= 0),
        "slope_se_approx",
    ] = np.nan

    duplicate = frame.duplicated(["comparison_uid", "estimator"], keep=False)
    if duplicate.any():
        raise SystemExit(
            "Duplicate comparison_uid × estimator rows in portability estimates."
        )
    return frame


def attach_sample_metadata(
    frame: pd.DataFrame,
    feasibility_path: Path,
    variant_path: Path,
) -> pd.DataFrame:
    result = frame.copy()

    if feasibility_path.exists():
        feasibility = pd.read_parquet(feasibility_path)
        require_columns(feasibility, ["comparison_uid"], "Step 15B4 feasibility")
        feasibility = feasibility.drop_duplicates("comparison_uid")
        feasibility["eur_analysis_n"] = resolve_sample_size(feasibility, "eur")
        feasibility["comparison_analysis_n"] = resolve_sample_size(
            feasibility, "comparison"
        )
        keep = [
            "comparison_uid",
            "eur_analysis_n",
            "comparison_analysis_n",
        ]
        for optional in [
            "comparison_population",
            "gene",
            "candidate_trait_name",
            "candidate_source",
        ]:
            if optional in feasibility.columns:
                keep.append(optional)
        result = result.merge(
            feasibility[keep],
            on="comparison_uid",
            how="left",
            validate="many_to_one",
            suffixes=("", "_feasibility"),
        )
    else:
        result["eur_analysis_n"] = np.nan
        result["comparison_analysis_n"] = np.nan

    result["sample_size_ratio"] = (
        result["comparison_analysis_n"] / result["eur_analysis_n"]
    )

    if variant_path.exists():
        variants = pd.read_parquet(variant_path)
        if {
            "comparison_uid",
            "eur_se",
            "comparison_se",
        }.issubset(variants.columns):
            variants["eur_se"] = numeric(variants, "eur_se")
            variants["comparison_se"] = numeric(variants, "comparison_se")
            variants["variant_se_ratio"] = (
                variants["comparison_se"] / variants["eur_se"]
            )
            valid = (
                np.isfinite(variants["variant_se_ratio"])
                & (variants["variant_se_ratio"] > 0)
            )
            precision = (
                variants.loc[valid]
                .groupby("comparison_uid", as_index=False)
                .agg(
                    median_variant_se_ratio=("variant_se_ratio", "median"),
                    median_eur_variant_se=("eur_se", "median"),
                    median_comparison_variant_se=("comparison_se", "median"),
                    n_variant_effect_rows=("variant_se_ratio", "size"),
                )
            )
            result = result.merge(
                precision,
                on="comparison_uid",
                how="left",
                validate="many_to_one",
            )

    return result


def attach_unit_metadata(frame: pd.DataFrame, unit_path: Path) -> pd.DataFrame:
    if not unit_path.exists():
        return frame
    units = pd.read_parquet(unit_path)
    require_columns(units, ["gene_trait_uid"], "Step 15B5 unit universe")
    fields = [
        "gene_trait_uid",
        "gene",
        "candidate_trait_name",
        "candidate_source",
        "sensitivity_pool_assignment",
    ]
    fields = [field for field in fields if field in units.columns]
    units = units[fields].drop_duplicates("gene_trait_uid")
    return frame.merge(
        units,
        on="gene_trait_uid",
        how="left",
        validate="many_to_one",
        suffixes=("", "_unit"),
    )


def precision_reporting(frame: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    comparison_columns = [
        "comparison_uid",
        "gene_trait_uid",
        "comparison_population",
        "eur_analysis_n",
        "comparison_analysis_n",
        "sample_size_ratio",
        "median_variant_se_ratio",
        "median_eur_variant_se",
        "median_comparison_variant_se",
        "n_variant_effect_rows",
    ]
    comparison_columns = [column for column in comparison_columns if column in frame]
    report = frame[comparison_columns].drop_duplicates("comparison_uid").copy()
    report.to_csv(outdir / "15g_sample_size_precision_report.csv", index=False)
    report.to_parquet(
        outdir / "15g_sample_size_precision_report.parquet", index=False
    )

    rows: list[dict[str, object]] = []
    if "comparison_population" in report.columns:
        groups = [("ALL", report)] + list(report.groupby("comparison_population"))
    else:
        groups = [("ALL", report)]

    for label, group in groups:
        ratios = numeric(group, "sample_size_ratio").dropna()
        se_ratios = (
            numeric(group, "median_variant_se_ratio").dropna()
            if "median_variant_se_ratio" in group
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "comparison_population": label,
                "n_comparisons": int(len(group)),
                "n_with_sample_size_ratio": int(len(ratios)),
                "sample_size_ratio_median": (
                    float(ratios.median()) if len(ratios) else math.nan
                ),
                "sample_size_ratio_q25": (
                    float(ratios.quantile(0.25)) if len(ratios) else math.nan
                ),
                "sample_size_ratio_q75": (
                    float(ratios.quantile(0.75)) if len(ratios) else math.nan
                ),
                "n_with_variant_se_ratio": int(len(se_ratios)),
                "variant_se_ratio_median": (
                    float(se_ratios.median()) if len(se_ratios) else math.nan
                ),
                "variant_se_ratio_q25": (
                    float(se_ratios.quantile(0.25)) if len(se_ratios) else math.nan
                ),
                "variant_se_ratio_q75": (
                    float(se_ratios.quantile(0.75)) if len(se_ratios) else math.nan
                ),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "15g_sample_size_precision_summary.csv", index=False)
    return summary


def run_precision_meta(frame: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for estimator, group in frame.groupby("estimator", sort=False):
        populations: list[tuple[str, pd.DataFrame]] = [("ALL", group)]
        if "comparison_population" in group.columns:
            populations.extend(list(group.groupby("comparison_population")))
        for population, subset in populations:
            stats = meta_summary(
                subset["slope"].to_numpy(),
                subset["slope_se_approx"].to_numpy(),
            )
            rows.append(
                {
                    "estimator": estimator,
                    "comparison_population": population,
                    **stats,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(outdir / "15g_precision_weighted_meta.csv", index=False)
    return result


def run_threshold_sensitivity(frame: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for estimator, group in frame.groupby("estimator", sort=False):
        valid_precision = group["slope_se_approx"].dropna()
        q50 = float(valid_precision.quantile(0.50)) if len(valid_precision) else math.nan
        q75 = float(valid_precision.quantile(0.75)) if len(valid_precision) else math.nan

        rules: list[tuple[str, pd.Series]] = [
            ("all", pd.Series(True, index=group.index)),
            ("exclude_worst_precision_quartile", group["slope_se_approx"] <= q75),
            ("most_precise_half", group["slope_se_approx"] <= q50),
            ("at_least_5_ld_pruned_variants", group["n_variants"] >= 5),
            ("at_least_10_ld_pruned_variants", group["n_variants"] >= 10),
        ]

        if group["sample_size_ratio"].notna().any():
            for threshold in (0.01, 0.05, 0.10):
                rules.append(
                    (
                        f"comparison_over_eur_N_ge_{threshold:g}",
                        group["sample_size_ratio"] >= threshold,
                    )
                )

        for rule_name, mask in rules:
            subset = group[mask.fillna(False)].copy()
            stats = meta_summary(
                subset["slope"].to_numpy(),
                subset["slope_se_approx"].to_numpy(),
            )
            rows.append(
                {
                    "estimator": estimator,
                    "restriction": rule_name,
                    "n_rows_before_se_filter": int(len(subset)),
                    "median_slope": (
                        float(subset["slope"].median()) if len(subset) else math.nan
                    ),
                    "median_slope_se_approx": (
                        float(subset["slope_se_approx"].median())
                        if len(subset)
                        else math.nan
                    ),
                    **stats,
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv(outdir / "15g_precision_threshold_sensitivity.csv", index=False)
    return result


def run_coloc_stratification(
    frame: pd.DataFrame,
    coloc_path: Path,
    outdir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not coloc_path.exists():
        empty = pd.DataFrame()
        empty.to_csv(outdir / "15g_coloc_stratified_portability.csv", index=False)
        return empty, empty

    coloc = pd.read_parquet(coloc_path)
    require_columns(
        coloc,
        [
            "comparison_uid",
            "comparison_prior_stability",
            "primary_pp_h3",
            "primary_pp_h4",
        ],
        "Step 15E2 coloc stability",
    )
    coloc = coloc.drop_duplicates("comparison_uid")
    keep = [
        "comparison_uid",
        "comparison_prior_stability",
        "primary_pp_h3",
        "primary_pp_h4",
        "primary_conditional_h4",
        "primary_category",
    ]
    keep = [column for column in keep if column in coloc.columns]
    merged = frame.merge(
        coloc[keep],
        on="comparison_uid",
        how="inner",
        validate="many_to_one",
    )
    merged["shared_vs_other"] = np.where(
        merged["comparison_prior_stability"].eq("ROBUST_SHARED"),
        "ROBUST_SHARED",
        "NOT_ROBUST_SHARED",
    )
    merged.to_csv(outdir / "15g_coloc_portability_overlap.csv", index=False)
    merged.to_parquet(outdir / "15g_coloc_portability_overlap.parquet", index=False)

    rows: list[dict[str, object]] = []
    for grouping in ("comparison_prior_stability", "shared_vs_other"):
        for (estimator, category), group in merged.groupby(
            ["estimator", grouping], sort=True
        ):
            stats = meta_summary(
                group["slope"].to_numpy(),
                group["slope_se_approx"].to_numpy(),
            )
            rows.append(
                {
                    "grouping": grouping,
                    "estimator": estimator,
                    "coloc_group": category,
                    "n_comparisons_total": int(len(group)),
                    "n_units": int(group["gene_trait_uid"].nunique()),
                    "median_slope": float(group["slope"].median()),
                    "q25_slope": float(group["slope"].quantile(0.25)),
                    "q75_slope": float(group["slope"].quantile(0.75)),
                    **stats,
                }
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "15g_coloc_stratified_portability.csv", index=False)
    return merged, summary


def unit_level_estimates(frame: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    metadata_columns = [
        "gene",
        "candidate_trait_name",
        "candidate_source",
        "sensitivity_pool_assignment",
    ]
    metadata = unique_metadata(frame, "gene_trait_uid", metadata_columns)

    rows: list[dict[str, object]] = []
    for (unit, estimator), group in frame.groupby(
        ["gene_trait_uid", "estimator"], sort=True
    ):
        stats = meta_summary(
            group["slope"].to_numpy(),
            group["slope_se_approx"].to_numpy(),
        )
        populations = (
            " || ".join(sorted(group["comparison_population"].dropna().astype(str).unique()))
            if "comparison_population" in group.columns
            else ""
        )
        rows.append(
            {
                "gene_trait_uid": unit,
                "estimator": estimator,
                "n_comparison_ancestries": int(group["comparison_uid"].nunique()),
                "comparison_ancestries": populations,
                "unit_iv_fixed_slope": stats["fixed_mean"],
                "unit_iv_fixed_se": stats["fixed_se"],
                "unit_iv_fixed_ci_lower": stats["fixed_ci_lower"],
                "unit_iv_fixed_ci_upper": stats["fixed_ci_upper"],
                "within_unit_tau2": stats["tau2"],
                "within_unit_i2_percent": stats["i2_percent"],
            }
        )

    units = pd.DataFrame(rows).merge(
        metadata,
        on="gene_trait_uid",
        how="left",
        validate="many_to_one",
    )
    units.to_csv(outdir / "15g_locus_specific_portability.csv", index=False)
    units.to_parquet(outdir / "15g_locus_specific_portability.parquet", index=False)
    return units


def random_effects_across_units(units: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for estimator, group in units.groupby("estimator", sort=False):
        stats = meta_summary(
            group["unit_iv_fixed_slope"].to_numpy(),
            group["unit_iv_fixed_se"].to_numpy(),
        )
        rows.append(
            {
                "estimator": estimator,
                "level": "gene_trait_source_unit",
                **stats,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(outdir / "15g_random_effects_unit_summary.csv", index=False)
    return result


def make_plots(
    threshold: pd.DataFrame,
    coloc_overlap: pd.DataFrame,
    units: pd.DataFrame,
    outdir: Path,
) -> None:
    if plt is None:
        return

    fiqt_threshold = threshold[threshold["estimator"].eq("fiqt")].copy()
    fiqt_threshold = fiqt_threshold[
        fiqt_threshold["random_mean"].notna()
    ].reset_index(drop=True)
    if len(fiqt_threshold):
        y = np.arange(len(fiqt_threshold))
        x = fiqt_threshold["random_mean"].to_numpy()
        lower = x - fiqt_threshold["random_ci_lower"].to_numpy()
        upper = fiqt_threshold["random_ci_upper"].to_numpy() - x
        fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.48 * len(y) + 1.5)))
        ax.errorbar(x, y, xerr=np.vstack([lower, upper]), fmt="o", capsize=3)
        ax.axvline(1.0, linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(fiqt_threshold["restriction"].str.replace("_", " "))
        ax.invert_yaxis()
        ax.set_xlabel("FIQT portability slope (random-effects summary)")
        ax.set_title("Portability under precision and sample-size restrictions")
        fig.tight_layout()
        fig.savefig(outdir / "15g_fiqt_precision_sensitivity.pdf")
        fig.savefig(outdir / "15g_fiqt_precision_sensitivity.png", dpi=300)
        plt.close(fig)

    fiqt_coloc = coloc_overlap[coloc_overlap["estimator"].eq("fiqt")].copy()
    if len(fiqt_coloc):
        categories = [
            value
            for value in ("ROBUST_SHARED", "PRIOR_SENSITIVE", "ROBUST_DISTINCT")
            if value in set(fiqt_coloc["comparison_prior_stability"])
        ]
        data = [
            fiqt_coloc.loc[
                fiqt_coloc["comparison_prior_stability"].eq(category), "slope"
            ].dropna().to_numpy()
            for category in categories
        ]
        if categories and all(len(values) for values in data):
            fig, ax = plt.subplots(figsize=(7.5, 4.8))
            ax.boxplot(data, labels=[c.replace("_", " ") for c in categories])
            ax.axhline(1.0, linestyle="--", linewidth=1)
            ax.set_ylabel("FIQT portability slope")
            ax.set_title("Portability stratified by colocalization stability")
            fig.tight_layout()
            fig.savefig(outdir / "15g_fiqt_coloc_stratified.pdf")
            fig.savefig(outdir / "15g_fiqt_coloc_stratified.png", dpi=300)
            plt.close(fig)

    fiqt_units = units[units["estimator"].eq("fiqt")].copy()
    fiqt_units = fiqt_units[
        fiqt_units["unit_iv_fixed_slope"].notna()
        & fiqt_units["unit_iv_fixed_se"].notna()
    ].sort_values("unit_iv_fixed_slope")
    if len(fiqt_units):
        labels = []
        for _, row in fiqt_units.iterrows():
            gene = str(row.get("gene", row["gene_trait_uid"]))
            trait = str(row.get("candidate_trait_name", ""))
            labels.append(f"{gene} — {trait}" if trait and trait != "nan" else gene)
        y = np.arange(len(fiqt_units))
        x = fiqt_units["unit_iv_fixed_slope"].to_numpy()
        se = fiqt_units["unit_iv_fixed_se"].to_numpy()
        fig, ax = plt.subplots(figsize=(10, max(8, 0.27 * len(y) + 2)))
        ax.errorbar(x, y, xerr=Z975 * se, fmt="o", markersize=3, capsize=2)
        ax.axvline(1.0, linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("FIQT portability slope (unit-level IV estimate)")
        ax.set_title("Locus-specific cross-ancestry portability")
        fig.tight_layout()
        fig.savefig(outdir / "15g_fiqt_locus_forest.pdf")
        fig.savefig(outdir / "15g_fiqt_locus_forest.png", dpi=300)
        plt.close(fig)


def write_notes(
    outdir: Path,
    n_comparisons: int,
    n_units: int,
    coloc_overlap: int,
) -> None:
    text = "# Step 15G portability robustness\n\n"
    text += f"- Primary portability comparisons analyzed: **{n_comparisons}**\n"
    text += f"- Gene–trait–source units represented: **{n_units}**\n"
    text += f"- Comparisons with both portability and powered colocalization: **{coloc_overlap}**\n\n"
    text += "## Interpretation rules\n\n"
    text += "1. Inverse-variance weighting gives more influence to comparisons with narrower slope confidence intervals; it does not create information for underpowered ancestries.\n"
    text += "2. Sample-size-ratio restrictions are sensitivity analyses, not corrections for missing non-European data.\n"
    text += "3. Colocalization-stratified portability is descriptive because the overlap is small.\n"
    text += "4. The random-effects result estimates average portability while allowing true variation across gene–trait–source units. I² describes heterogeneity, not biological ancestry effects by itself.\n"
    text += "5. Slope SEs are approximated from percentile-bootstrap CI widths and should not replace the original Step 15D bootstrap intervals.\n"
    (outdir / "15g_results_notes.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    portability = prepare_portability(args)
    portability = attach_sample_metadata(
        portability,
        args.feasibility,
        args.variant_effects,
    )
    portability = attach_unit_metadata(portability, args.unit_universe)

    # Preserve the analysis-ready joined table.
    portability.to_parquet(
        args.output_dir / "15g_portability_analysis_table.parquet", index=False
    )
    portability.to_csv(
        args.output_dir / "15g_portability_analysis_table.csv", index=False
    )

    sample_summary = precision_reporting(portability, args.output_dir)
    precision_meta = run_precision_meta(portability, args.output_dir)
    threshold = run_threshold_sensitivity(portability, args.output_dir)
    coloc_overlap, coloc_summary = run_coloc_stratification(
        portability,
        args.coloc_stability,
        args.output_dir,
    )
    units = unit_level_estimates(portability, args.output_dir)
    unit_random = random_effects_across_units(units, args.output_dir)
    make_plots(threshold, coloc_overlap, units, args.output_dir)

    n_comparisons = int(portability["comparison_uid"].nunique())
    n_units = int(portability["gene_trait_uid"].nunique())
    n_coloc = int(coloc_overlap["comparison_uid"].nunique()) if len(coloc_overlap) else 0
    write_notes(args.output_dir, n_comparisons, n_units, n_coloc)

    summary = {
        "step": "15G",
        "status": "PASS",
        "primary_r2": args.primary_r2,
        "n_portability_comparisons": n_comparisons,
        "n_portability_units": n_units,
        "n_portability_coloc_overlap": n_coloc,
        "sample_size_metadata_available_for": int(
            portability[["comparison_uid", "sample_size_ratio"]]
            .drop_duplicates("comparison_uid")["sample_size_ratio"]
            .notna()
            .sum()
        ),
        "outputs": {
            "analysis_table": "15g_portability_analysis_table.parquet",
            "sample_size_report": "15g_sample_size_precision_report.csv",
            "precision_weighted_meta": "15g_precision_weighted_meta.csv",
            "threshold_sensitivity": "15g_precision_threshold_sensitivity.csv",
            "coloc_stratified": "15g_coloc_stratified_portability.csv",
            "locus_specific": "15g_locus_specific_portability.csv",
            "random_effects_units": "15g_random_effects_unit_summary.csv",
        },
    }
    (args.output_dir / "15g_portability_robustness_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=" * 78)
    print("STEP 15G — PORTABILITY ROBUSTNESS")
    print("=" * 78)
    print(f"Comparisons: {n_comparisons}")
    print(f"Gene–trait–source units: {n_units}")
    print(f"Portability × powered-coloc overlap: {n_coloc}")
    print()
    print("Random-effects summary across units:")
    print(
        unit_random[
            [
                "estimator",
                "n",
                "random_mean",
                "random_ci_lower",
                "random_ci_upper",
                "tau2",
                "i2_percent",
            ]
        ].to_string(index=False)
    )
    print()
    if len(coloc_summary):
        print("Colocalization-stratified counts:")
        print(
            coloc_summary[
                [
                    "grouping",
                    "estimator",
                    "coloc_group",
                    "n_comparisons_total",
                    "n_units",
                    "median_slope",
                ]
            ].to_string(index=False)
        )
        print()
    print("Outputs written to:", args.output_dir)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
