#!/usr/bin/env python3
"""
Step 12: Practical significance, power, and descriptive equivalence.

Run from target_ancestry/step12 after Step 8B.

Purpose
-------
Translate the formal approval-regression coefficients into:
  1. adjusted approval-probability differences across meaningful exposure ranges;
  2. cluster-robust delta-method confidence intervals for those probability differences;
  3. analytic minimum detectable effects (MDEs) using the observed robust SE;
  4. approximate power for specified ORs per SD;
  5. descriptive equivalence diagnostics at reference margins.

The script DOES NOT declare formal equivalence by default. The project decision
log states that the equivalence margin remains unresolved and must not be chosen
based on whether the result passes. Reference margins of OR 1.20 and 1.25 per SD
are reported transparently, but are not labeled preregistered.

Inputs
------
../step8/output/08_model_dataset.parquet
../step8/output/08_mesh_category_counts.csv

Outputs
-------
output/12_practical_significance.csv
output/12_predicted_probability_curve.csv
output/12_power_mde.csv
output/12_equivalence_reference.csv
output/12_practical_significance.json

Models reproduce Step 8B:
M1: approval ~ exposure
M2: M1 + log_studies
M3: M2 + earliest_year_centered
M4: M3 + eligible official MeSH category indicators

All fits use gene-clustered sandwich standard errors.
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
from patsy import build_design_matrices
from scipy import stats


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
EXPOSURES = [*COUNT_EXPOSURES, *BINARY_EXPOSURES]
MODEL_IDS = ["M1", "M2", "M3", "M4"]
REFERENCE_EQUIVALENCE_MARGINS = [1.20, 1.25]
POWER_EFFECT_ORS_PER_SD = [1.10, 1.20, 1.25, 1.50]

REQUIRED_MODEL_COLUMNS = {
    "ti_uid",
    "gene",
    "pool",
    "approval",
    "log_studies",
    "earliest_year_centered",
    *EXPOSURES,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute practical significance, MDE, power, and equivalence diagnostics."
    )
    parser.add_argument(
        "--model-data",
        type=Path,
        default=Path("../step8/output/08_model_dataset.parquet"),
    )
    parser.add_argument(
        "--mesh-counts",
        type=Path,
        default=Path("../step8/output/08_mesh_category_counts.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--power",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--formal-equivalence-margin",
        type=float,
        default=None,
        help=(
            "Optional independently justified OR margin per SD. When omitted, "
            "reference margins are descriptive only."
        ),
    )
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        fail(f"{label} is missing required columns: {missing}")


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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def parse_bool_series(series: pd.Series, label: str) -> pd.Series:
    if series.dtype == bool:
        return series
    parsed = (
        series.astype(str)
        .str.strip()
        .str.casefold()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    if parsed.isna().any():
        fail(f"Could not parse {label} as boolean")
    return parsed.astype(bool)


def model_predictors(model_id: str, exposure: str, mesh_cols: list[str]) -> list[str]:
    if model_id == "M1":
        return [exposure]
    if model_id == "M2":
        return [exposure, "log_studies"]
    if model_id == "M3":
        return [exposure, "log_studies", "earliest_year_centered"]
    if model_id == "M4":
        return [exposure, "log_studies", "earliest_year_centered", *mesh_cols]
    raise ValueError(model_id)


def nearest_observed_quantile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="raise").dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError(f"No values for {series.name}")
    try:
        result = np.quantile(values, q, method="nearest")
    except TypeError:  # NumPy < 1.22
        result = np.quantile(values, q, interpolation="nearest")
    return float(result)


def expit_array(x: np.ndarray) -> np.ndarray:
    # Stable logistic transform without importing scipy.special separately.
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def fit_model(
    data: pd.DataFrame,
    exposure: str,
    model_id: str,
    mesh_cols: list[str],
):
    predictors = model_predictors(model_id, exposure, mesh_cols)
    needed = ["approval", "gene", *predictors]
    model_df = data[needed].dropna().copy()
    if model_df["approval"].nunique() != 2:
        raise ValueError("Outcome lacks variation")
    if model_df[exposure].nunique() < 2:
        raise ValueError("Exposure lacks variation")
    formula = "approval ~ " + " + ".join(predictors)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = smf.logit(formula, data=model_df).fit(
            disp=False,
            maxiter=300,
            cov_type="cluster",
            cov_kwds={"groups": model_df["gene"], "use_correction": True},
        )
    warning_messages = [
        f"{item.category.__name__}: {item.message}" for item in caught
    ]
    return fit, model_df, predictors, warning_messages


def design_matrix(fit, new_data: pd.DataFrame) -> np.ndarray:
    design_info = fit.model.data.design_info
    matrix = build_design_matrices(
        [design_info], new_data, return_type="dataframe"
    )[0]
    # Force exact fitted coefficient ordering.
    matrix = matrix.loc[:, fit.params.index]
    return matrix.to_numpy(dtype=float)


def average_prediction_from_beta(x: np.ndarray, beta: np.ndarray) -> float:
    return float(expit_array(x @ beta).mean())


def average_prediction_and_gradient(
    x: np.ndarray, beta: np.ndarray
) -> tuple[float, np.ndarray]:
    probability = expit_array(x @ beta)
    mean_probability = float(probability.mean())
    gradient = np.mean(
        (probability * (1.0 - probability))[:, None] * x, axis=0
    )
    return mean_probability, gradient


def probability_contrast_delta(
    *,
    fit,
    model_df: pd.DataFrame,
    exposure: str,
    low: float,
    high: float,
    alpha: float,
) -> dict[str, Any]:
    low_df = model_df.copy()
    high_df = model_df.copy()
    low_df[exposure] = low
    high_df[exposure] = high

    x_low = design_matrix(fit, low_df)
    x_high = design_matrix(fit, high_df)
    beta = fit.params.to_numpy(dtype=float)
    covariance = fit.cov_params().to_numpy(dtype=float)
    covariance = (covariance + covariance.T) / 2.0

    p_low, grad_low = average_prediction_and_gradient(x_low, beta)
    p_high, grad_high = average_prediction_and_gradient(x_high, beta)
    difference = p_high - p_low
    grad_difference = grad_high - grad_low
    variance = float(grad_difference @ covariance @ grad_difference)
    variance = max(variance, 0.0)
    standard_error = math.sqrt(variance)
    zcrit = stats.norm.ppf(1.0 - alpha / 2.0)

    return {
        "low_value": float(low),
        "high_value": float(high),
        "predicted_approval_low": p_low,
        "predicted_approval_high": p_high,
        "approval_probability_difference": difference,
        "difference_standard_error": standard_error,
        "difference_ci_low": difference - zcrit * standard_error,
        "difference_ci_high": difference + zcrit * standard_error,
    }


def marginal_probability_delta(
    *, fit, model_df: pd.DataFrame, exposure: str, value: float, alpha: float
) -> dict[str, float]:
    new_df = model_df.copy()
    new_df[exposure] = value
    x = design_matrix(fit, new_df)
    beta = fit.params.to_numpy(dtype=float)
    covariance = fit.cov_params().to_numpy(dtype=float)
    covariance = (covariance + covariance.T) / 2.0
    probability, gradient = average_prediction_and_gradient(x, beta)
    variance = float(gradient @ covariance @ gradient)
    standard_error = math.sqrt(max(variance, 0.0))
    zcrit = stats.norm.ppf(1.0 - alpha / 2.0)
    return {
        "predicted_approval_probability": probability,
        "probability_standard_error": standard_error,
        "probability_ci_low": max(0.0, probability - zcrit * standard_error),
        "probability_ci_high": min(1.0, probability + zcrit * standard_error),
    }


def two_sided_power(beta_alt: float, se: float, alpha: float) -> float:
    if se <= 0 or not np.isfinite(se):
        return float("nan")
    zcrit = stats.norm.ppf(1.0 - alpha / 2.0)
    mu = beta_alt / se
    return float(stats.norm.cdf(-zcrit - mu) + 1.0 - stats.norm.cdf(zcrit - mu))


def tost(beta_sd: float, se_sd: float, margin_or: float) -> dict[str, Any]:
    delta = math.log(margin_or)
    lower_bound = -delta
    upper_bound = delta
    z_lower = (beta_sd - lower_bound) / se_sd
    z_upper = (beta_sd - upper_bound) / se_sd
    p_lower = float(1.0 - stats.norm.cdf(z_lower))  # H0 beta <= lower
    p_upper = float(stats.norm.cdf(z_upper))        # H0 beta >= upper
    p_tost = max(p_lower, p_upper)
    z90 = stats.norm.ppf(0.95)
    ci90_low = beta_sd - z90 * se_sd
    ci90_high = beta_sd + z90 * se_sd
    return {
        "margin_or_per_sd": margin_or,
        "lower_or_bound": 1.0 / margin_or,
        "upper_or_bound": margin_or,
        "beta_per_sd": beta_sd,
        "se_per_sd": se_sd,
        "or_per_sd": math.exp(beta_sd),
        "ci90_or_low": math.exp(ci90_low),
        "ci90_or_high": math.exp(ci90_high),
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": p_tost,
        "passes_alpha_0_05": bool(p_tost < 0.05),
    }


def main() -> int:
    args = parse_args()
    if not (0 < args.alpha < 1):
        fail("--alpha must be between 0 and 1")
    if not (0 < args.power < 1):
        fail("--power must be between 0 and 1")
    if args.formal_equivalence_margin is not None and args.formal_equivalence_margin <= 1:
        fail("--formal-equivalence-margin must be greater than 1")
    for path in [args.model_data, args.mesh_counts]:
        if not path.exists():
            fail(f"Required input not found: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Step 8 model data: {args.model_data}")
    data = pd.read_parquet(args.model_data)
    require_columns(data, REQUIRED_MODEL_COLUMNS, "Step 8 model data")
    if data["ti_uid"].duplicated().any():
        fail("Step 8 model data must contain one row per ti_uid")

    counts = pd.read_csv(args.mesh_counts)
    require_columns(counts, {"mesh_category_code", "eligible_model4"}, "MeSH counts")
    eligible = parse_bool_series(counts["eligible_model4"], "eligible_model4")
    eligible_codes = counts.loc[eligible, "mesh_category_code"].astype(str).tolist()
    mesh_cols = [f"mesh_{code}" for code in eligible_codes]
    missing_mesh = sorted(set(mesh_cols) - set(data.columns))
    if missing_mesh:
        fail(f"Model data lacks eligible MeSH indicators: {missing_mesh}")

    qa = {
        "n": int(len(data)),
        "n_A": int(data["approval"].sum()),
        "n_B": int((1 - data["approval"]).sum()),
        "n_genes": int(data["gene"].nunique()),
        "n_mesh_indicators": len(mesh_cols),
        "alpha": args.alpha,
        "target_power": args.power,
        "formal_equivalence_margin": args.formal_equivalence_margin,
        "formal_equivalence_claim_enabled": args.formal_equivalence_margin is not None,
    }
    expected = {"n": 790, "n_A": 157, "n_B": 633}
    for key, expected_value in expected.items():
        if qa[key] != expected_value:
            fail(f"Frozen-universe QA failed for {key}: {qa[key]} != {expected_value}")

    print(
        f"  Pairs: {qa['n']:,} (A={qa['n_A']}, B={qa['n_B']}); "
        f"gene clusters={qa['n_genes']}; MeSH indicators={qa['n_mesh_indicators']}"
    )
    print(
        "  Equivalence mode: "
        + (
            f"formal margin OR {args.formal_equivalence_margin:.3f}/SD supplied"
            if args.formal_equivalence_margin is not None
            else "descriptive reference margins only; no formal claim"
        )
    )

    practical_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    power_rows: list[dict[str, Any]] = []
    equivalence_rows: list[dict[str, Any]] = []
    fit_summaries: list[dict[str, Any]] = []

    for exposure in EXPOSURES:
        print(f"\nFitting effect-size models for {exposure}")
        for model_id in MODEL_IDS:
            try:
                fit, model_df, predictors, warning_messages = fit_model(
                    data, exposure, model_id, mesh_cols
                )
            except Exception as exc:
                print(f"  {model_id}: FAILED - {type(exc).__name__}: {exc}")
                fit_summaries.append(
                    {
                        "exposure": exposure,
                        "model_id": model_id,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            beta = float(fit.params[exposure])
            se = float(fit.bse[exposure])
            exposure_sd = float(model_df[exposure].std(ddof=1))
            beta_sd = beta * exposure_sd
            se_sd = se * exposure_sd
            ci = fit.conf_int().loc[exposure]
            fit_summaries.append(
                {
                    "exposure": exposure,
                    "model_id": model_id,
                    "status": "ok",
                    "formula": "approval ~ " + " + ".join(predictors),
                    "n": int(len(model_df)),
                    "n_A": int(model_df["approval"].sum()),
                    "n_B": int((1 - model_df["approval"]).sum()),
                    "n_gene_clusters": int(model_df["gene"].nunique()),
                    "beta": beta,
                    "cluster_robust_se": se,
                    "odds_ratio": math.exp(beta),
                    "ci_low": math.exp(float(ci.iloc[0])),
                    "ci_high": math.exp(float(ci.iloc[1])),
                    "p_value": float(fit.pvalues[exposure]),
                    "exposure_sd": exposure_sd,
                    "or_per_sd": math.exp(beta_sd),
                    "warning_messages": " | ".join(warning_messages) if warning_messages else None,
                }
            )

            if exposure in COUNT_EXPOSURES:
                p10 = nearest_observed_quantile(model_df[exposure], 0.10)
                p90 = nearest_observed_quantile(model_df[exposure], 0.90)
                observed_min = float(model_df[exposure].min())
                observed_max = float(model_df[exposure].max())
                contrasts = [
                    ("p10_to_p90", p10, p90),
                    ("observed_min_to_max", observed_min, observed_max),
                    ("one_sd_from_mean", float(model_df[exposure].mean()), float(model_df[exposure].mean() + exposure_sd)),
                ]
            else:
                contrasts = [("absent_to_present", 0.0, 1.0)]

            for contrast_name, low, high in contrasts:
                contrast = probability_contrast_delta(
                    fit=fit,
                    model_df=model_df,
                    exposure=exposure,
                    low=low,
                    high=high,
                    alpha=args.alpha,
                )
                practical_rows.append(
                    {
                        "exposure": exposure,
                        "model_id": model_id,
                        "contrast": contrast_name,
                        "n": int(len(model_df)),
                        "n_A": int(model_df["approval"].sum()),
                        "n_B": int((1 - model_df["approval"]).sum()),
                        "exposure_sd": exposure_sd,
                        **contrast,
                    }
                )

            # Predicted probability curve over observed count support, or 0/1.
            if exposure in COUNT_EXPOSURES:
                curve_values = sorted(pd.unique(model_df[exposure].astype(float)))
            else:
                curve_values = [0.0, 1.0]
            for value in curve_values:
                marginal = marginal_probability_delta(
                    fit=fit,
                    model_df=model_df,
                    exposure=exposure,
                    value=float(value),
                    alpha=args.alpha,
                )
                curve_rows.append(
                    {
                        "exposure": exposure,
                        "model_id": model_id,
                        "exposure_value": float(value),
                        **marginal,
                    }
                )

            z_alpha = stats.norm.ppf(1.0 - args.alpha / 2.0)
            z_power = stats.norm.ppf(args.power)
            mde_beta = (z_alpha + z_power) * se
            mde_beta_sd = (z_alpha + z_power) * se_sd
            power_rows.append(
                {
                    "exposure": exposure,
                    "model_id": model_id,
                    "n": int(len(model_df)),
                    "n_A": int(model_df["approval"].sum()),
                    "n_B": int((1 - model_df["approval"]).sum()),
                    "exposure_sd": exposure_sd,
                    "observed_beta": beta,
                    "observed_cluster_se": se,
                    "observed_or_per_unit": math.exp(beta),
                    "observed_or_per_sd": math.exp(beta_sd),
                    "target_power": args.power,
                    "alpha": args.alpha,
                    "mde_beta_per_unit": mde_beta,
                    "mde_or_per_unit": math.exp(mde_beta),
                    "mde_beta_per_sd": mde_beta_sd,
                    "mde_or_per_sd": math.exp(mde_beta_sd),
                }
            )
            for effect_or in POWER_EFFECT_ORS_PER_SD:
                power_rows.append(
                    {
                        "exposure": exposure,
                        "model_id": model_id,
                        "n": int(len(model_df)),
                        "n_A": int(model_df["approval"].sum()),
                        "n_B": int((1 - model_df["approval"]).sum()),
                        "exposure_sd": exposure_sd,
                        "observed_beta": beta,
                        "observed_cluster_se": se,
                        "observed_or_per_unit": math.exp(beta),
                        "observed_or_per_sd": math.exp(beta_sd),
                        "target_power": None,
                        "alpha": args.alpha,
                        "effect_or_per_sd": effect_or,
                        "approximate_power": two_sided_power(
                            math.log(effect_or), se_sd, args.alpha
                        ),
                    }
                )

            margins = list(REFERENCE_EQUIVALENCE_MARGINS)
            if (
                args.formal_equivalence_margin is not None
                and args.formal_equivalence_margin not in margins
            ):
                margins.append(args.formal_equivalence_margin)
            for margin in sorted(margins):
                result = tost(beta_sd, se_sd, margin)
                result.update(
                    {
                        "exposure": exposure,
                        "model_id": model_id,
                        "n": int(len(model_df)),
                        "n_A": int(model_df["approval"].sum()),
                        "n_B": int((1 - model_df["approval"]).sum()),
                        "exposure_sd": exposure_sd,
                        "margin_role": (
                            "formal_user_supplied"
                            if args.formal_equivalence_margin is not None
                            and math.isclose(margin, args.formal_equivalence_margin)
                            else "descriptive_reference"
                        ),
                        "formal_claim_allowed": bool(
                            args.formal_equivalence_margin is not None
                            and math.isclose(margin, args.formal_equivalence_margin)
                        ),
                    }
                )
                equivalence_rows.append(result)

            primary_contrast = next(
                row
                for row in reversed(practical_rows)
                if row["exposure"] == exposure
                and row["model_id"] == model_id
                and row["contrast"]
                == ("p10_to_p90" if exposure in COUNT_EXPOSURES else "absent_to_present")
            )
            print(
                f"  {model_id}: OR/SD={math.exp(beta_sd):.3f}; "
                f"approval difference={100*primary_contrast['approval_probability_difference']:+.2f} pp "
                f"({100*primary_contrast['difference_ci_low']:+.2f}, "
                f"{100*primary_contrast['difference_ci_high']:+.2f})"
            )

    practical_df = pd.DataFrame(practical_rows)
    curve_df = pd.DataFrame(curve_rows)
    power_df = pd.DataFrame(power_rows)
    equivalence_df = pd.DataFrame(equivalence_rows)
    fit_df = pd.DataFrame(fit_summaries)

    practical_out = args.output_dir / "12_practical_significance.csv"
    curve_out = args.output_dir / "12_predicted_probability_curve.csv"
    power_out = args.output_dir / "12_power_mde.csv"
    equivalence_out = args.output_dir / "12_equivalence_reference.csv"
    json_out = args.output_dir / "12_practical_significance.json"

    practical_df.to_csv(practical_out, index=False)
    curve_df.to_csv(curve_out, index=False)
    power_df.to_csv(power_out, index=False)
    equivalence_df.to_csv(equivalence_out, index=False)

    primary = practical_df[
        (practical_df["exposure"] == "n_ancestries_all")
        & (practical_df["contrast"] == "p10_to_p90")
    ].copy()
    primary_power = power_df[
        (power_df["exposure"] == "n_ancestries_all")
        & power_df["mde_or_per_sd"].notna()
    ].copy()
    primary_equivalence = equivalence_df[
        equivalence_df["exposure"] == "n_ancestries_all"
    ].copy()

    payload = {
        "qa": qa,
        "definitions": {
            "primary_probability_contrast": (
                "Average marginal approval probability at the observed 10th versus "
                "90th percentile of ancestry diversity; other covariates retain their "
                "observed values."
            ),
            "probability_ci": (
                "Delta-method interval using the gene-clustered robust coefficient "
                "covariance matrix."
            ),
            "mde": (
                "Analytic normal-approximation minimum detectable effect using the "
                "observed cluster-robust standard error."
            ),
            "equivalence": (
                "Reference TOST calculations are descriptive unless an independently "
                "justified margin is supplied with --formal-equivalence-margin."
            ),
        },
        "primary_practical_significance": primary.to_dict("records"),
        "primary_mde": primary_power.to_dict("records"),
        "primary_equivalence_reference": primary_equivalence.to_dict("records"),
        "model_fits": fit_df.to_dict("records"),
        "all_practical_significance": practical_df.to_dict("records"),
    }
    json_out.write_text(
        json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print("STEP 12 PRACTICAL SIGNIFICANCE: n_ancestries_all")
    print("=" * 72)
    display = primary[
        [
            "model_id",
            "low_value",
            "high_value",
            "predicted_approval_low",
            "predicted_approval_high",
            "approval_probability_difference",
            "difference_ci_low",
            "difference_ci_high",
        ]
    ].copy()
    for col in [
        "predicted_approval_low",
        "predicted_approval_high",
        "approval_probability_difference",
        "difference_ci_low",
        "difference_ci_high",
    ]:
        display[col] = display[col] * 100.0
    print(display.to_string(index=False))

    print("\nMDE (OR per 1 SD) for n_ancestries_all:")
    print(
        primary_power[
            ["model_id", "observed_or_per_sd", "mde_or_per_sd"]
        ].to_string(index=False)
    )

    print("\nEquivalence reference (no formal claim unless margin supplied):")
    print(
        primary_equivalence[
            [
                "model_id",
                "margin_or_per_sd",
                "or_per_sd",
                "ci90_or_low",
                "ci90_or_high",
                "p_tost",
                "passes_alpha_0_05",
                "margin_role",
            ]
        ].to_string(index=False)
    )

    print("\nSaved outputs:")
    for path in [practical_out, curve_out, power_out, equivalence_out, json_out]:
        print(f"  {path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
