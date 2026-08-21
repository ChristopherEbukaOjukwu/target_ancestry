#!/usr/bin/env python3
"""
Step 8B: Sequential approval regressions.

Run from target_ancestry/step8 after Step 8A.

Inputs
------
../step4/output/04_trails.parquet
../step2/output/02_pair_studies.parquet
output/08_pair_mesh_categories.parquet
output/08_mesh_category_counts.csv

Outputs
-------
output/08_model_dataset.parquet
output/08_approval_regression.parquet
output/08_approval_regression_coefficients.parquet
output/08_model_diagnostics.csv
output/08_approval_regression.json

Models
------
M1: approval ~ exposure
M2: approval ~ exposure + log1p(n_studies_total)
M3: M2 + earliest_assoc_year
M4: M3 + eligible official MeSH disease-category indicators

The original Minikel A/B outcome is retained. Standard errors are clustered
by gene. Model 4 is a disease-category robustness analysis; Models 1-3 remain
primary.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


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

MIN_BINARY_CELL = 5

REQUIRED_TRAIL_COLUMNS = {
    "ti_uid",
    "gene",
    "pool",
    "all_set",
    "n_ancestries_all",
    "n_ancestries_initial",
    "n_ancestries_replication",
    "n_studies_total",
    "has_replication",
}

REQUIRED_STUDY_COLUMNS = {"ti_uid", "assoc_year"}


# ---------------------------------------------------------------------------
# Arguments and utilities
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Step 8B gene-clustered approval regressions."
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
        "--mesh-pairs",
        type=Path,
        default=Path("output/08_pair_mesh_categories.parquet"),
    )
    parser.add_argument(
        "--mesh-counts",
        type=Path,
        default=Path("output/08_mesh_category_counts.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
    )
    parser.add_argument(
        "--min-binary-cell",
        type=int,
        default=MIN_BINARY_CELL,
        help="Minimum approval-by-exposure cell count for binary models.",
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
    """Normalize a Parquet list/array/set value into a Python set."""
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


def standard_deviation(series: pd.Series) -> float:
    value = float(series.astype(float).std(ddof=1))
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid standard deviation for {series.name}: {value}")
    return value


def binary_cell_counts(df: pd.DataFrame, exposure: str) -> dict[str, int]:
    table = pd.crosstab(df[exposure].astype(int), df["approval"].astype(int))
    return {
        "B_absent": int(table.loc[0, 0]) if 0 in table.index and 0 in table.columns else 0,
        "A_absent": int(table.loc[0, 1]) if 0 in table.index and 1 in table.columns else 0,
        "B_present": int(table.loc[1, 0]) if 1 in table.index and 0 in table.columns else 0,
        "A_present": int(table.loc[1, 1]) if 1 in table.index and 1 in table.columns else 0,
    }


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


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


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------
def build_model_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], dict]:
    for path in [args.trails, args.pair_studies, args.mesh_pairs, args.mesh_counts]:
        if not path.exists():
            fail(f"Required input not found: {path}")

    print(f"Loading Step 4 trails: {args.trails}")
    trails = pd.read_parquet(args.trails)
    require_columns(trails, REQUIRED_TRAIL_COLUMNS, "Step 4 trails")

    if trails["ti_uid"].duplicated().any():
        fail("Step 4 trails must contain one row per ti_uid")
    if set(trails["pool"].dropna().astype(str)) != {"A", "B"}:
        fail(f"Unexpected pool values: {sorted(trails['pool'].dropna().unique())}")

    print(f"Loading pair-study years: {args.pair_studies}")
    pair_studies = pd.read_parquet(args.pair_studies)
    require_columns(pair_studies, REQUIRED_STUDY_COLUMNS, "Step 2 pair studies")
    pair_studies = pair_studies.copy()
    pair_studies["assoc_year"] = pd.to_numeric(
        pair_studies["assoc_year"], errors="coerce"
    )
    pair_year = (
        pair_studies.groupby("ti_uid", as_index=False)
        .agg(
            earliest_assoc_year=("assoc_year", "min"),
            median_assoc_year=("assoc_year", "median"),
            latest_assoc_year=("assoc_year", "max"),
            n_distinct_assoc_years=("assoc_year", "nunique"),
        )
    )

    print(f"Loading Step 8A pair categories: {args.mesh_pairs}")
    mesh_pairs = pd.read_parquet(args.mesh_pairs)
    if "ti_uid" not in mesh_pairs.columns:
        fail("Step 8A pair-category file has no ti_uid column")
    if mesh_pairs["ti_uid"].duplicated().any():
        fail("Step 8A pair-category file must contain one row per ti_uid")

    counts = pd.read_csv(args.mesh_counts)
    required_count_columns = {"mesh_category_code", "eligible_model4"}
    require_columns(counts, required_count_columns, "Step 8A category counts")
    eligible_flag = counts["eligible_model4"]
    if eligible_flag.dtype != bool:
        eligible_flag = (
            eligible_flag.astype(str).str.strip().str.casefold().map(
                {"true": True, "false": False, "1": True, "0": False}
            )
        )
    if eligible_flag.isna().any():
        fail("Could not parse one or more eligible_model4 values as booleans")
    eligible_codes = counts.loc[
        eligible_flag.astype(bool), "mesh_category_code"
    ].astype(str).tolist()
    eligible_cols = [f"mesh_{code}" for code in eligible_codes]
    missing_indicators = sorted(set(eligible_cols) - set(mesh_pairs.columns))
    if missing_indicators:
        fail(f"Eligible MeSH indicators absent from pair-category file: {missing_indicators}")

    # Keep only the Step 8A variables needed for regression to avoid duplicate
    # gene/pool/indication columns during the merge.
    mesh_keep = ["ti_uid", *eligible_cols]
    data = (
        trails.merge(pair_year, on="ti_uid", how="left", validate="one_to_one")
        .merge(mesh_pairs[mesh_keep], on="ti_uid", how="left", validate="one_to_one")
    )

    # Frozen original outcome.
    data["approval"] = (data["pool"] == "A").astype("int8")
    data["log_studies"] = np.log1p(pd.to_numeric(data["n_studies_total"], errors="raise"))

    # Center year for numerical stability. The OR remains per one calendar year.
    year_center = float(data["earliest_assoc_year"].median())
    data["earliest_year_centered"] = data["earliest_assoc_year"] - year_center

    # Ancestry-presence indicators from the existing all-stage sets.
    all_ancestries = data["all_set"].apply(ancestry_set)
    data["has_african"] = all_ancestries.apply(lambda x: "African" in x).astype("int8")
    data["has_east_asian"] = all_ancestries.apply(
        lambda x: "East Asian" in x
    ).astype("int8")
    data["has_south_asian"] = all_ancestries.apply(
        lambda x: "South Asian" in x
    ).astype("int8")
    data["has_replication"] = data["has_replication"].astype("int8")

    # Ensure missing category membership means zero only after a successful
    # one-to-one merge with the Step 8A pair file.
    data[eligible_cols] = data[eligible_cols].fillna(0).astype("int8")
    data["has_eligible_mesh_category"] = data[eligible_cols].sum(axis=1).gt(0)

    # Core QA against the frozen universe.
    qa = {
        "n_pairs": int(len(data)),
        "n_unique_pairs": int(data["ti_uid"].nunique()),
        "n_A": int((data["pool"] == "A").sum()),
        "n_B": int((data["pool"] == "B").sum()),
        "n_genes": int(data["gene"].nunique()),
        "n_missing_earliest_year": int(data["earliest_assoc_year"].isna().sum()),
        "year_min": safe_float(data["earliest_assoc_year"].min()),
        "year_median": safe_float(data["earliest_assoc_year"].median()),
        "year_max": safe_float(data["earliest_assoc_year"].max()),
        "n_multiple_years": int((data["n_distinct_assoc_years"] > 1).sum()),
        "year_center": year_center,
        "eligible_mesh_codes": eligible_codes,
        "n_eligible_mesh_categories": len(eligible_cols),
        "n_pairs_with_eligible_category": int(data["has_eligible_mesh_category"].sum()),
        "n_pairs_reference_category": int((~data["has_eligible_mesh_category"]).sum()),
        "reference_A": int(
            ((~data["has_eligible_mesh_category"]) & (data["pool"] == "A")).sum()
        ),
        "reference_B": int(
            ((~data["has_eligible_mesh_category"]) & (data["pool"] == "B")).sum()
        ),
    }

    expected = {"n_pairs": 790, "n_unique_pairs": 790, "n_A": 157, "n_B": 633}
    for key, expected_value in expected.items():
        if qa[key] != expected_value:
            fail(f"Frozen-universe QA failed for {key}: {qa[key]} != {expected_value}")
    if qa["n_missing_earliest_year"] != 0:
        fail("At least one final pair is missing earliest_assoc_year")

    print(f"  Pairs: {qa['n_pairs']:,} (A={qa['n_A']}, B={qa['n_B']})")
    print(f"  Unique genes: {qa['n_genes']:,}")
    print(
        "  Earliest evidence year: "
        f"{qa['year_min']:.0f}-{qa['year_max']:.0f}; "
        f"median={qa['year_median']:.0f}; missing=0"
    )
    print(
        "  Eligible MeSH categories: "
        f"{qa['n_eligible_mesh_categories']}; "
        f"covered pairs={qa['n_pairs_with_eligible_category']}; "
        f"all-zero reference={qa['n_pairs_reference_category']}"
    )

    return data, eligible_cols, qa


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------
def model_predictors(model_id: str, exposure: str, mesh_cols: list[str]) -> list[str]:
    if model_id == "M1":
        return [exposure]
    if model_id == "M2":
        return [exposure, "log_studies"]
    if model_id == "M3":
        return [exposure, "log_studies", "earliest_year_centered"]
    if model_id == "M4":
        return [exposure, "log_studies", "earliest_year_centered", *mesh_cols]
    raise ValueError(f"Unknown model_id: {model_id}")


def fit_one_model(
    data: pd.DataFrame,
    *,
    exposure: str,
    model_id: str,
    mesh_cols: list[str],
    min_binary_cell: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    predictors = model_predictors(model_id, exposure, mesh_cols)
    needed = ["approval", "pool", "gene", *predictors]
    model_df = data[needed].dropna().copy()

    exposure_type = "count" if exposure in COUNT_EXPOSURES else "binary"
    exposure_sd = standard_deviation(model_df[exposure]) if exposure_type == "count" else None

    summary: dict[str, Any] = {
        "model_id": model_id,
        "exposure": exposure,
        "exposure_type": exposure_type,
        "formula": "approval ~ " + " + ".join(predictors),
        "n": int(len(model_df)),
        "n_A": int(model_df["approval"].sum()),
        "n_B": int((1 - model_df["approval"]).sum()),
        "n_gene_clusters": int(model_df["gene"].nunique()),
        "n_mesh_indicators": len(mesh_cols) if model_id == "M4" else 0,
        "exposure_sd": exposure_sd,
        "status": "pending",
        "error": None,
    }

    diagnostic: dict[str, Any] = {
        "model_id": model_id,
        "exposure": exposure,
        "n": int(len(model_df)),
        "n_events": int(model_df["approval"].sum()),
        "n_gene_clusters": int(model_df["gene"].nunique()),
        "warning_messages": None,
    }

    if len(model_df) == 0 or model_df["approval"].nunique() < 2:
        summary.update(status="failed", error="No usable outcome variation")
        return summary, [], diagnostic
    if model_df[exposure].nunique() < 2:
        summary.update(status="failed", error="Exposure has no variation")
        return summary, [], diagnostic

    if exposure_type == "binary":
        cells = binary_cell_counts(model_df, exposure)
        summary.update({f"cell_{key}": value for key, value in cells.items()})
        if min(cells.values()) < min_binary_cell:
            summary.update(
                status="skipped",
                error=(
                    f"Minimum approval-by-exposure cell is {min(cells.values())}; "
                    f"required >= {min_binary_cell}"
                ),
            )
            return summary, [], diagnostic

    formula = "approval ~ " + " + ".join(predictors)
    caught_messages: list[str] = []

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = smf.logit(formula, data=model_df).fit(
                disp=False,
                maxiter=300,
                cov_type="cluster",
                cov_kwds={
                    "groups": model_df["gene"],
                    "use_correction": True,
                },
            )
            caught_messages = [
                f"{warning.category.__name__}: {warning.message}" for warning in caught
            ]

        exog = np.asarray(fit.model.exog, dtype=float)
        rank = int(np.linalg.matrix_rank(exog))
        n_parameters = int(exog.shape[1])
        condition_number = float(np.linalg.cond(exog))
        converged = bool(fit.mle_retvals.get("converged", True))
        full_rank = rank == n_parameters
        separation_warning = any(
            "separation" in message.casefold() for message in caught_messages
        )

        diagnostic.update(
            n_parameters=n_parameters,
            events_per_parameter=(
                int(model_df["approval"].sum()) / n_parameters if n_parameters else None
            ),
            design_rank=rank,
            design_full_rank=full_rank,
            condition_number=condition_number,
            converged=converged,
            separation_warning=separation_warning,
            warning_messages=" | ".join(caught_messages) if caught_messages else None,
            max_abs_beta=safe_float(np.abs(fit.params).max()),
            max_cluster_se=safe_float(np.abs(fit.bse).max()),
        )

        beta = float(fit.params[exposure])
        ci_low_beta, ci_high_beta = map(float, fit.conf_int().loc[exposure])
        summary.update(
            status="ok",
            converged=converged,
            design_full_rank=full_rank,
            separation_warning=separation_warning,
            n_parameters=n_parameters,
            events_per_parameter=diagnostic["events_per_parameter"],
            condition_number=condition_number,
            beta=beta,
            cluster_robust_se=float(fit.bse[exposure]),
            odds_ratio=float(np.exp(beta)),
            ci_low=float(np.exp(ci_low_beta)),
            ci_high=float(np.exp(ci_high_beta)),
            p_value=float(fit.pvalues[exposure]),
            aic=float(fit.aic),
            bic=float(fit.bic),
            warning_messages=diagnostic["warning_messages"],
        )

        if exposure_type == "count" and exposure_sd is not None:
            summary.update(
                odds_ratio_per_sd=float(np.exp(beta * exposure_sd)),
                ci_low_per_sd=float(np.exp(ci_low_beta * exposure_sd)),
                ci_high_per_sd=float(np.exp(ci_high_beta * exposure_sd)),
            )
        else:
            summary.update(
                odds_ratio_per_sd=None,
                ci_low_per_sd=None,
                ci_high_per_sd=None,
            )

        coefficients: list[dict[str, Any]] = []
        conf = fit.conf_int()
        for term in fit.params.index:
            term_beta = float(fit.params[term])
            term_ci_low, term_ci_high = map(float, conf.loc[term])
            coefficients.append(
                {
                    "model_id": model_id,
                    "exposure_model": exposure,
                    "term": term,
                    "beta": term_beta,
                    "cluster_robust_se": float(fit.bse[term]),
                    "odds_ratio": float(np.exp(term_beta)),
                    "ci_low": float(np.exp(term_ci_low)),
                    "ci_high": float(np.exp(term_ci_high)),
                    "p_value": float(fit.pvalues[term]),
                }
            )

        return summary, coefficients, diagnostic

    except Exception as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        diagnostic.update(
            converged=False,
            warning_messages=" | ".join(caught_messages) if caught_messages else None,
            error=f"{type(exc).__name__}: {exc}",
        )
        return summary, [], diagnostic


def add_sequential_changes(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    results["beta_change_from_previous"] = np.nan
    results["or_ratio_to_previous"] = np.nan

    order = ["M1", "M2", "M3", "M4"]
    for exposure, group_index in results.groupby("exposure").groups.items():
        group = results.loc[group_index].set_index("model_id").reindex(order)
        previous_beta: float | None = None
        previous_or: float | None = None
        for model_id, row in group.iterrows():
            if row.get("status") != "ok" or pd.isna(row.get("beta")):
                continue
            beta = float(row["beta"])
            odds_ratio = float(row["odds_ratio"])
            idx = results.index[
                (results["exposure"] == exposure) & (results["model_id"] == model_id)
            ]
            if previous_beta is not None and len(idx) == 1:
                results.loc[idx, "beta_change_from_previous"] = beta - previous_beta
                results.loc[idx, "or_ratio_to_previous"] = (
                    odds_ratio / previous_or if previous_or else np.nan
                )
            previous_beta = beta
            previous_or = odds_ratio
    return results


def main() -> int:
    args = parse_args()
    if args.min_binary_cell < 1:
        fail("--min-binary-cell must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data, mesh_cols, qa = build_model_dataset(args)

    binary_qa = {
        exposure: binary_cell_counts(data, exposure) for exposure in BINARY_EXPOSURES
    }
    print("\nBinary exposure cell counts (B/A by absent/present):")
    for exposure, counts in binary_qa.items():
        print(f"  {exposure}: {counts}")

    results: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    exposures = [*COUNT_EXPOSURES, *BINARY_EXPOSURES]
    for exposure in exposures:
        print(f"\nFitting models for {exposure}")
        for model_id in ["M1", "M2", "M3", "M4"]:
            summary, coefficient_rows, diagnostic = fit_one_model(
                data,
                exposure=exposure,
                model_id=model_id,
                mesh_cols=mesh_cols,
                min_binary_cell=args.min_binary_cell,
            )
            results.append(summary)
            coefficients.extend(coefficient_rows)
            diagnostics.append(diagnostic)
            if summary["status"] == "ok":
                print(
                    f"  {model_id}: OR={summary['odds_ratio']:.4f} "
                    f"({summary['ci_low']:.4f}-{summary['ci_high']:.4f}), "
                    f"p={summary['p_value']:.4g}"
                )
            else:
                print(f"  {model_id}: {summary['status']} - {summary['error']}")

    result_df = add_sequential_changes(pd.DataFrame(results))
    coefficient_df = pd.DataFrame(coefficients)
    diagnostic_df = pd.DataFrame(diagnostics)

    model_data_out = args.output_dir / "08_model_dataset.parquet"
    results_out = args.output_dir / "08_approval_regression.parquet"
    coefficients_out = args.output_dir / "08_approval_regression_coefficients.parquet"
    diagnostics_out = args.output_dir / "08_model_diagnostics.csv"
    json_out = args.output_dir / "08_approval_regression.json"

    # Remove internal list-like helper columns only if present. Preserve all
    # original Step 4 fields plus year and category variables for auditing.
    data.to_parquet(model_data_out, index=False)
    result_df.to_parquet(results_out, index=False)
    coefficient_df.to_parquet(coefficients_out, index=False)
    diagnostic_df.to_csv(diagnostics_out, index=False)

    primary = result_df.loc[result_df["exposure"] == "n_ancestries_all"].copy()
    payload = {
        "qa": qa,
        "binary_cell_counts": binary_qa,
        "model_definitions": {
            "M1": "approval ~ exposure",
            "M2": "M1 + log1p(n_studies_total)",
            "M3": "M2 + earliest_assoc_year",
            "M4": "M3 + eligible official MeSH disease-category indicators",
        },
        "primary_exposure_results": primary.to_dict("records"),
        "all_results": result_df.to_dict("records"),
    }
    json_out.write_text(
        json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print("STEP 8B PRIMARY RESULTS: n_ancestries_all")
    print("=" * 72)
    display_cols = [
        "model_id",
        "n",
        "odds_ratio",
        "ci_low",
        "ci_high",
        "p_value",
        "odds_ratio_per_sd",
        "events_per_parameter",
        "status",
    ]
    print(primary[display_cols].to_string(index=False))

    model4_primary = primary.loc[primary["model_id"] == "M4"]
    if not model4_primary.empty:
        row = model4_primary.iloc[0]
        print("\nModel 4 diagnostics:")
        print(f"  Parameters:            {row.get('n_parameters')}")
        print(f"  Events per parameter:  {row.get('events_per_parameter')}")
        print(f"  Full-rank design:      {row.get('design_full_rank')}")
        print(f"  Converged:             {row.get('converged')}")
        print(f"  Separation warning:   {row.get('separation_warning')}")
        print(f"  Condition number:      {row.get('condition_number')}")

    print("\nSaved outputs:")
    for path in [
        model_data_out,
        results_out,
        coefficients_out,
        diagnostics_out,
        json_out,
    ]:
        print(f"  {path}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
