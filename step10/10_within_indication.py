#!/usr/bin/env python3
"""
Step 10: Within-indication approval analysis.

Purpose
-------
Compare approved (Pool A) and not-approved (Pool B) target-indication pairs
within the same exact MeSH indication. This complements the existing within-gene
analysis by holding disease/indication constant.

Run from target_ancestry/step10.

Inputs
------
../step4/output/04_trails.parquet
../step2/output/02_pair_studies.parquet

Outputs
-------
output/10_within_indication_dataset.parquet
output/10_indication_counts.csv
output/10_within_indication_differences.csv
output/10_within_indication_regression.csv
output/10_within_indication_coefficients.csv
output/10_within_indication_diagnostics.csv
output/10_within_indication.json

Models
------
W1: approval ~ exposure | exact indication
W2: approval ~ exposure + log1p(studies) | exact indication
W3: approval ~ exposure + log1p(studies) + earliest evidence year | exact indication

The primary estimator is conditional logistic regression stratified by exact
MeSH indication. A conventional logistic regression with exact-indication fixed
effects and gene-clustered standard errors is reported as a sensitivity check.
Only indications containing both Pool A and Pool B pairs are informative.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.discrete.conditional_models import ConditionalLogit


COUNT_EXPOSURES = [
    "n_ancestries_all",
    "n_ancestries_initial",
    "n_ancestries_replication",
]

BINARY_EXPOSURES = [
    "has_replication",
    "has_african",
    "has_east_asian",
    "has_south_asian",
]

EXPOSURES = COUNT_EXPOSURES + BINARY_EXPOSURES
MODEL_IDS = ["W1", "W2", "W3"]

REQUIRED_TRAIL_COLUMNS = {
    "ti_uid",
    "gene",
    "pool",
    "indication_mesh_id",
    "indication_mesh_term",
    "all_set",
    "n_ancestries_all",
    "n_ancestries_initial",
    "n_ancestries_replication",
    "n_studies_total",
    "has_replication",
}
REQUIRED_STUDY_COLUMNS = {"ti_uid", "assoc_year"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exact-indication-stratified approval analyses."
    )
    parser.add_argument(
        "--trails",
        type=Path,
        default=Path("../step4/output/04_trails.parquet"),
    )
    parser.add_argument(
        "--pair-studies",
        type=Path,
        default=Path("../step2/output/02_pair_studies.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=500,
    )
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        fail(f"{label} is missing required columns: {missing}")


def ancestry_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, set):
        return {str(item) for item in value if pd.notna(item)}
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return {str(item) for item in value if pd.notna(item)}
    try:
        if pd.isna(value):
            return set()
    except (TypeError, ValueError):
        pass
    return {str(value)}


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def predictors_for(model_id: str, exposure: str) -> list[str]:
    if model_id == "W1":
        return [exposure]
    if model_id == "W2":
        return [exposure, "log_studies"]
    if model_id == "W3":
        return [exposure, "log_studies", "earliest_year_centered"]
    raise ValueError(model_id)


def build_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    for path in [args.trails, args.pair_studies]:
        if not path.exists():
            fail(f"Required input not found: {path}")

    print(f"Loading Step 4 trails: {args.trails}")
    trails = pd.read_parquet(args.trails)
    require_columns(trails, REQUIRED_TRAIL_COLUMNS, "Step 4 trails")
    if trails["ti_uid"].duplicated().any():
        fail("Step 4 trails must contain one row per ti_uid")

    print(f"Loading pair-study years: {args.pair_studies}")
    studies = pd.read_parquet(args.pair_studies)
    require_columns(studies, REQUIRED_STUDY_COLUMNS, "Step 2 pair studies")
    studies = studies.copy()
    studies["assoc_year"] = pd.to_numeric(studies["assoc_year"], errors="coerce")
    years = (
        studies.groupby("ti_uid", as_index=False)
        .agg(
            earliest_assoc_year=("assoc_year", "min"),
            latest_assoc_year=("assoc_year", "max"),
            n_distinct_assoc_years=("assoc_year", "nunique"),
        )
    )

    data = trails.merge(years, on="ti_uid", how="left", validate="one_to_one")
    data["approval"] = (data["pool"] == "A").astype("int8")
    data["n_studies_total"] = pd.to_numeric(data["n_studies_total"], errors="raise")
    data["log_studies"] = np.log1p(data["n_studies_total"])

    if data["earliest_assoc_year"].isna().any():
        fail("Final trails contain missing earliest association years")
    year_center = float(data["earliest_assoc_year"].median())
    data["earliest_year_centered"] = data["earliest_assoc_year"] - year_center

    all_ancestries = data["all_set"].apply(ancestry_set)
    data["has_african"] = all_ancestries.apply(lambda x: "African" in x).astype("int8")
    data["has_east_asian"] = all_ancestries.apply(
        lambda x: "East Asian" in x
    ).astype("int8")
    data["has_south_asian"] = all_ancestries.apply(
        lambda x: "South Asian" in x
    ).astype("int8")
    data["has_replication"] = data["has_replication"].astype("int8")

    # Exact MeSH descriptor is the stratum. Missing IDs cannot be conditioned on.
    data["indication_mesh_id"] = data["indication_mesh_id"].astype("string")
    nonmissing = data[data["indication_mesh_id"].notna()].copy()

    indication_counts = (
        nonmissing.groupby(["indication_mesh_id", "indication_mesh_term"], dropna=False)
        .agg(
            n_pairs=("ti_uid", "size"),
            n_A=("approval", "sum"),
            n_genes=("gene", "nunique"),
        )
        .reset_index()
    )
    indication_counts["n_B"] = indication_counts["n_pairs"] - indication_counts["n_A"]
    indication_counts["mixed_pool"] = (
        (indication_counts["n_A"] > 0) & (indication_counts["n_B"] > 0)
    )

    mixed_ids = set(
        indication_counts.loc[indication_counts["mixed_pool"], "indication_mesh_id"].astype(str)
    )
    analysis = nonmissing[
        nonmissing["indication_mesh_id"].astype(str).isin(mixed_ids)
    ].copy()

    analysis["indication_mesh_id"] = analysis["indication_mesh_id"].astype(str)
    analysis["indication_mesh_term"] = analysis["indication_mesh_term"].astype(str)

    qa = {
        "n_all_pairs": int(len(data)),
        "n_all_indications": int(data["indication_mesh_id"].nunique(dropna=True)),
        "n_missing_indication_id": int(data["indication_mesh_id"].isna().sum()),
        "n_mixed_indications": int(indication_counts["mixed_pool"].sum()),
        "n_analysis_pairs": int(len(analysis)),
        "n_analysis_A": int(analysis["approval"].sum()),
        "n_analysis_B": int((1 - analysis["approval"]).sum()),
        "n_analysis_genes": int(analysis["gene"].nunique()),
        "year_center": year_center,
    }

    print(f"  All final pairs: {qa['n_all_pairs']:,}")
    print(f"  Exact indications: {qa['n_all_indications']:,}")
    print(f"  Mixed A/B indications: {qa['n_mixed_indications']:,}")
    print(
        f"  Analysis pairs: {qa['n_analysis_pairs']:,} "
        f"(A={qa['n_analysis_A']}, B={qa['n_analysis_B']})"
    )
    print(f"  Unique genes in analysis: {qa['n_analysis_genes']:,}")

    return analysis, indication_counts, qa


def within_indication_differences(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["indication_mesh_id", "indication_mesh_term"]
    for (mesh_id, term), group in data.groupby(group_cols, dropna=False):
        a = group[group["approval"] == 1]
        b = group[group["approval"] == 0]
        if a.empty or b.empty:
            continue
        base = {
            "indication_mesh_id": mesh_id,
            "indication_mesh_term": term,
            "n_pairs": int(len(group)),
            "n_A": int(len(a)),
            "n_B": int(len(b)),
        }
        for exposure in EXPOSURES:
            a_mean = float(pd.to_numeric(a[exposure], errors="coerce").mean())
            b_mean = float(pd.to_numeric(b[exposure], errors="coerce").mean())
            rows.append(
                {
                    **base,
                    "exposure": exposure,
                    "A_mean": a_mean,
                    "B_mean": b_mean,
                    "A_minus_B": a_mean - b_mean,
                }
            )
    return pd.DataFrame(rows)


def within_rank(data: pd.DataFrame, predictors: list[str]) -> tuple[int, int]:
    x = data[predictors].astype(float)
    centered = x - x.groupby(data["indication_mesh_id"]).transform("mean")
    matrix = centered.to_numpy(dtype=float)
    return int(np.linalg.matrix_rank(matrix)), int(matrix.shape[1])


def fit_conditional(
    data: pd.DataFrame,
    exposure: str,
    model_id: str,
    maxiter: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    predictors = predictors_for(model_id, exposure)
    needed = ["approval", "gene", "indication_mesh_id", *predictors]
    df = data[needed].dropna().copy()

    # Re-check mixed outcome after any model-specific missing-value removal.
    mixed = df.groupby("indication_mesh_id")["approval"].nunique()
    df = df[df["indication_mesh_id"].isin(mixed[mixed == 2].index)].copy()

    rank, p = within_rank(df, predictors)
    informative_exposure_strata = int(
        (df.groupby("indication_mesh_id")[exposure].nunique() > 1).sum()
    )

    summary: dict[str, Any] = {
        "estimator": "conditional_logit",
        "model_id": model_id,
        "exposure": exposure,
        "formula": "approval ~ " + " + ".join(predictors) + " | indication_mesh_id",
        "n": int(len(df)),
        "n_A": int(df["approval"].sum()),
        "n_B": int((1 - df["approval"]).sum()),
        "n_indications": int(df["indication_mesh_id"].nunique()),
        "n_genes": int(df["gene"].nunique()),
        "informative_exposure_strata": informative_exposure_strata,
        "status": "pending",
        "error": None,
    }
    diagnostic = {
        "estimator": "conditional_logit",
        "model_id": model_id,
        "exposure": exposure,
        "n": int(len(df)),
        "n_indications": int(df["indication_mesh_id"].nunique()),
        "within_design_rank": rank,
        "n_predictors": p,
        "within_design_full_rank": rank == p,
        "informative_exposure_strata": informative_exposure_strata,
    }

    if len(df) == 0 or df["approval"].nunique() < 2:
        summary.update(status="failed", error="No usable outcome variation")
        return summary, [], diagnostic
    if df[exposure].nunique() < 2 or informative_exposure_strata == 0:
        summary.update(status="failed", error="No within-indication exposure variation")
        return summary, [], diagnostic
    if rank < p:
        summary.update(status="failed", error="Within-indication design matrix is rank deficient")
        return summary, [], diagnostic

    warning_messages: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = ConditionalLogit(
                endog=df["approval"].astype(float),
                exog=df[predictors].astype(float),
                groups=df["indication_mesh_id"],
            )
            fit = model.fit(method="BFGS", maxiter=maxiter, disp=False)
            warning_messages = [
                f"{w.category.__name__}: {w.message}" for w in caught
            ]

        converged = bool(getattr(fit, "mle_retvals", {}).get("converged", True))
        conf = fit.conf_int()
        beta = float(fit.params[exposure])
        ci_low_beta = float(conf.loc[exposure, 0])
        ci_high_beta = float(conf.loc[exposure, 1])

        summary.update(
            status="ok",
            converged=converged,
            beta=beta,
            standard_error=float(fit.bse[exposure]),
            odds_ratio=float(np.exp(beta)),
            ci_low=float(np.exp(ci_low_beta)),
            ci_high=float(np.exp(ci_high_beta)),
            p_value=float(fit.pvalues[exposure]),
            warning_messages=" | ".join(warning_messages) if warning_messages else None,
        )

        coefficients: list[dict[str, Any]] = []
        for term in predictors:
            term_beta = float(fit.params[term])
            lo = float(conf.loc[term, 0])
            hi = float(conf.loc[term, 1])
            coefficients.append(
                {
                    "estimator": "conditional_logit",
                    "model_id": model_id,
                    "exposure_model": exposure,
                    "term": term,
                    "beta": term_beta,
                    "standard_error": float(fit.bse[term]),
                    "odds_ratio": float(np.exp(term_beta)),
                    "ci_low": float(np.exp(lo)),
                    "ci_high": float(np.exp(hi)),
                    "p_value": float(fit.pvalues[term]),
                }
            )
        diagnostic.update(
            converged=converged,
            warning_messages=" | ".join(warning_messages) if warning_messages else None,
        )
        return summary, coefficients, diagnostic
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        summary.update(status="failed", error=message)
        diagnostic.update(converged=False, error=message)
        return summary, [], diagnostic


def fit_fixed_effect_clustered(
    data: pd.DataFrame,
    exposure: str,
    model_id: str,
    maxiter: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    predictors = predictors_for(model_id, exposure)
    needed = [
        "approval",
        "gene",
        "indication_mesh_id",
        *predictors,
    ]
    df = data[needed].dropna().copy()
    mixed = df.groupby("indication_mesh_id")["approval"].nunique()
    df = df[df["indication_mesh_id"].isin(mixed[mixed == 2].index)].copy()

    formula = "approval ~ " + " + ".join(predictors) + " + C(indication_mesh_id)"
    summary: dict[str, Any] = {
        "estimator": "fixed_effect_logit_cluster_gene",
        "model_id": model_id,
        "exposure": exposure,
        "formula": formula,
        "n": int(len(df)),
        "n_A": int(df["approval"].sum()),
        "n_B": int((1 - df["approval"]).sum()),
        "n_indications": int(df["indication_mesh_id"].nunique()),
        "n_genes": int(df["gene"].nunique()),
        "status": "pending",
        "error": None,
    }
    diagnostic = {
        "estimator": "fixed_effect_logit_cluster_gene",
        "model_id": model_id,
        "exposure": exposure,
        "n": int(len(df)),
        "n_indications": int(df["indication_mesh_id"].nunique()),
        "n_gene_clusters": int(df["gene"].nunique()),
    }

    try:
        warning_messages: list[str] = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = smf.logit(formula, data=df).fit(
                disp=False,
                maxiter=maxiter,
                cov_type="cluster",
                cov_kwds={"groups": df["gene"], "use_correction": True},
            )
            warning_messages = [
                f"{w.category.__name__}: {w.message}" for w in caught
            ]

        exog = np.asarray(fit.model.exog, dtype=float)
        rank = int(np.linalg.matrix_rank(exog))
        n_parameters = int(exog.shape[1])
        converged = bool(fit.mle_retvals.get("converged", True))
        separation_warning = any(
            "separation" in message.casefold() for message in warning_messages
        )
        conf = fit.conf_int()
        beta = float(fit.params[exposure])
        lo = float(conf.loc[exposure, 0])
        hi = float(conf.loc[exposure, 1])

        summary.update(
            status="ok",
            converged=converged,
            design_full_rank=rank == n_parameters,
            separation_warning=separation_warning,
            n_parameters=n_parameters,
            events_per_parameter=(float(df["approval"].sum()) / n_parameters),
            condition_number=float(np.linalg.cond(exog)),
            beta=beta,
            cluster_robust_se=float(fit.bse[exposure]),
            odds_ratio=float(np.exp(beta)),
            ci_low=float(np.exp(lo)),
            ci_high=float(np.exp(hi)),
            p_value=float(fit.pvalues[exposure]),
            warning_messages=" | ".join(warning_messages) if warning_messages else None,
        )

        coefficients: list[dict[str, Any]] = []
        # Save substantive predictors only; indication intercepts are nuisance terms.
        for term in ["Intercept", *predictors]:
            if term not in fit.params.index:
                continue
            term_beta = float(fit.params[term])
            term_lo = float(conf.loc[term, 0])
            term_hi = float(conf.loc[term, 1])
            coefficients.append(
                {
                    "estimator": "fixed_effect_logit_cluster_gene",
                    "model_id": model_id,
                    "exposure_model": exposure,
                    "term": term,
                    "beta": term_beta,
                    "standard_error": float(fit.bse[term]),
                    "odds_ratio": float(np.exp(term_beta)),
                    "ci_low": float(np.exp(term_lo)),
                    "ci_high": float(np.exp(term_hi)),
                    "p_value": float(fit.pvalues[term]),
                }
            )

        diagnostic.update(
            n_parameters=n_parameters,
            events_per_parameter=float(df["approval"].sum()) / n_parameters,
            design_rank=rank,
            design_full_rank=rank == n_parameters,
            condition_number=float(np.linalg.cond(exog)),
            converged=converged,
            separation_warning=separation_warning,
            warning_messages=" | ".join(warning_messages) if warning_messages else None,
        )
        return summary, coefficients, diagnostic
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        summary.update(status="failed", error=message)
        diagnostic.update(converged=False, error=message)
        return summary, [], diagnostic


def summarize_differences(differences: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for exposure, group in differences.groupby("exposure"):
        values = pd.to_numeric(group["A_minus_B"], errors="coerce").dropna()
        rows.append(
            {
                "exposure": exposure,
                "n_indications": int(values.size),
                "mean_indication_difference": float(values.mean()),
                "median_indication_difference": float(values.median()),
                "n_positive": int((values > 0).sum()),
                "n_zero": int((values == 0).sum()),
                "n_negative": int((values < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    analysis, indication_counts, qa = build_dataset(args)
    differences = within_indication_differences(analysis)
    difference_summary = summarize_differences(differences)

    regression_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for exposure in EXPOSURES:
        print(f"\nFitting within-indication models for {exposure}")
        for model_id in MODEL_IDS:
            summary, coefficients, diagnostic = fit_conditional(
                analysis, exposure, model_id, args.maxiter
            )
            regression_rows.append(summary)
            coefficient_rows.extend(coefficients)
            diagnostic_rows.append(diagnostic)
            if summary["status"] == "ok":
                print(
                    f"  Conditional {model_id}: OR={summary['odds_ratio']:.4f} "
                    f"({summary['ci_low']:.4f}-{summary['ci_high']:.4f}), "
                    f"p={summary['p_value']:.4g}"
                )
            else:
                print(f"  Conditional {model_id}: {summary['status']} - {summary['error']}")

            fe_summary, fe_coefficients, fe_diagnostic = fit_fixed_effect_clustered(
                analysis, exposure, model_id, args.maxiter
            )
            regression_rows.append(fe_summary)
            coefficient_rows.extend(fe_coefficients)
            diagnostic_rows.append(fe_diagnostic)
            if fe_summary["status"] != "ok":
                print(
                    f"    FE sensitivity: {fe_summary['status']} - "
                    f"{fe_summary['error']}"
                )

    regressions = pd.DataFrame(regression_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)

    analysis.to_parquet(args.output_dir / "10_within_indication_dataset.parquet", index=False)
    indication_counts.to_csv(args.output_dir / "10_indication_counts.csv", index=False)
    differences.to_csv(args.output_dir / "10_within_indication_differences.csv", index=False)
    difference_summary.to_csv(
        args.output_dir / "10_within_indication_difference_summary.csv", index=False
    )
    regressions.to_csv(args.output_dir / "10_within_indication_regression.csv", index=False)
    coefficients.to_csv(args.output_dir / "10_within_indication_coefficients.csv", index=False)
    diagnostics.to_csv(args.output_dir / "10_within_indication_diagnostics.csv", index=False)

    payload = {
        "step": "10_within_indication",
        "qa": qa,
        "difference_summary": difference_summary.to_dict(orient="records"),
        "regressions": regressions.to_dict(orient="records"),
        "diagnostics": diagnostics.to_dict(orient="records"),
    }
    with (args.output_dir / "10_within_indication.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True)

    primary = regressions[
        (regressions["estimator"] == "conditional_logit")
        & (regressions["exposure"] == "n_ancestries_all")
    ]

    print("\n" + "=" * 72)
    print("STEP 10 WITHIN-INDICATION SUMMARY")
    print("=" * 72)
    print(f"All final pairs:             {qa['n_all_pairs']:,}")
    print(f"Exact indications:           {qa['n_all_indications']:,}")
    print(f"Mixed A/B indications:       {qa['n_mixed_indications']:,}")
    print(f"Pairs in mixed indications:  {qa['n_analysis_pairs']:,}")
    print(f"Pool A / Pool B:             {qa['n_analysis_A']:,} / {qa['n_analysis_B']:,}")
    print("\nPrimary conditional-logit results: n_ancestries_all")
    if not primary.empty:
        cols = ["model_id", "n", "n_indications", "odds_ratio", "ci_low", "ci_high", "p_value", "status"]
        print(primary[cols].to_string(index=False))
    print("\nSaved outputs:")
    for filename in [
        "10_within_indication_dataset.parquet",
        "10_indication_counts.csv",
        "10_within_indication_differences.csv",
        "10_within_indication_difference_summary.csv",
        "10_within_indication_regression.csv",
        "10_within_indication_coefficients.csv",
        "10_within_indication_diagnostics.csv",
        "10_within_indication.json",
    ]:
        print(f"  {args.output_dir / filename}")
    print("=" * 72)


if __name__ == "__main__":
    main()
