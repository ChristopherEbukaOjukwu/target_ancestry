#!/usr/bin/env python3
"""
Step 11: Inverse-probability-of-selection weighted approval regressions.

Run from target_ancestry/step11.

Purpose
-------
Step 9 showed that entry into the 790-pair ancestry cohort varied by clinical
phase and disease category. This script estimates each clinical-stage pair's
probability of entering the final cohort using variables observed for the full
1,354-pair universe, then reweights the 790 analyzed pairs toward that universe.
It reruns the Step 8 model sequence as a selection-bias sensitivity analysis.

Inputs
------
../step9/output/09_cohort_selection_pairs.parquet
../step8/output/08_model_dataset.parquet
../step8/output/08_mesh_category_counts.csv
../step8/input/desc2026.gz

Outputs
-------
output/11_selection_weights.parquet
output/11_selection_model_diagnostics.csv
output/11_selection_balance.csv
output/11_weighted_regression.csv
output/11_weighted_coefficients.csv
output/11_weighted_model_diagnostics.csv
output/11_selection_weighting.json

Selection model
---------------
Five-fold cross-fitted ridge logistic regression predicts complete ancestry
inclusion using approval pool, clinical phase, official MeSH disease categories,
evidence-source presence, support volume, evidence age, and gene multiplicity.
Variables that define downstream linkage (linked_gwas, n_parseable_studies, and
selection_stage) are deliberately excluded.

Outcome models
--------------
SW1: approval ~ exposure
SW2: SW1 + log1p(n_studies_total)
SW3: SW2 + earliest association year
SW4: SW3 + eligible MeSH category indicators

All outcome models use gene-clustered sandwich standard errors. The primary
selection-weighted estimate uses weights truncated at the 1st and 99th
percentiles; untrimmed weights are retained as a sensitivity analysis.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, BinaryIO, Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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

REQUIRED_COHORT_COLUMNS = {
    "ti_uid",
    "gene",
    "pool",
    "approval",
    "included_final",
    "ccat",
    "indication_mesh_id",
    "n_support_rows",
    "n_assoc_sources",
    "earliest_assoc_year",
    "n_assoc_traits",
    "n_pairs_for_gene",
}

REQUIRED_MODEL_COLUMNS = {
    "ti_uid",
    "gene",
    "pool",
    "approval",
    "n_ancestries_all",
    "n_ancestries_initial",
    "n_ancestries_replication",
    "has_replication",
    "has_african",
    "has_east_asian",
    "has_south_asian",
    "log_studies",
    "earliest_year_centered",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run selection-weighted approval regressions."
    )
    parser.add_argument(
        "--cohort",
        type=Path,
        default=Path("../step9/output/09_cohort_selection_pairs.parquet"),
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
        "--mesh",
        type=Path,
        default=Path("../step8/input/desc2026.gz"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ridge-c", type=float, default=1.0)
    parser.add_argument("--trim-lower", type=float, default=0.01)
    parser.add_argument("--trim-upper", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=20260722)
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
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


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



# Official MeSH category mapping (same rule as Steps 8A and 9)

def normalize_mesh_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    text = re.sub(r"^MESH\s*:\s*", "", text)
    return text or None


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def find_child_text(element: ET.Element, path: Iterable[str]) -> str | None:
    current = element
    for wanted in path:
        match = None
        for child in current:
            if local_name(child.tag) == wanted:
                match = child
                break
        if match is None:
            return None
        current = match
    return current.text.strip() if current.text else None


def iter_children_text(
    element: ET.Element, parent_name: str, child_name: str
) -> list[str]:
    parent = None
    for child in element:
        if local_name(child.tag) == parent_name:
            parent = child
            break
    if parent is None:
        return []
    return [
        child.text.strip()
        for child in parent
        if local_name(child.tag) == child_name and child.text
    ]


def open_mesh(path: Path) -> BinaryIO:
    return gzip.open(path, "rb") if path.suffix.lower() == ".gz" else path.open("rb")


def category_code(tree_number: str) -> str | None:
    if re.match(r"^C\d{2}(?:\.|$)", tree_number):
        return tree_number.split(".", 1)[0]
    if tree_number == "F03" or tree_number.startswith("F03."):
        return "F03"
    return None


def parse_mesh_categories(mesh_path: Path) -> dict[str, set[str]]:
    id_to_categories: dict[str, set[str]] = defaultdict(set)
    with open_mesh(mesh_path) as handle:
        for _, element in ET.iterparse(handle, events=("end",)):
            if local_name(element.tag) != "DescriptorRecord":
                continue
            descriptor_ui = find_child_text(element, ["DescriptorUI"])
            trees = iter_children_text(element, "TreeNumberList", "TreeNumber")
            if descriptor_ui:
                for tree in trees:
                    code = category_code(tree)
                    if code:
                        id_to_categories[descriptor_ui].add(code)
            element.clear()
    if not id_to_categories:
        fail(f"No MeSH categories parsed from {mesh_path}")
    return id_to_categories


def add_mesh_indicators(
    cohort: pd.DataFrame, id_to_categories: dict[str, set[str]]
) -> tuple[pd.DataFrame, list[str]]:
    cohort = cohort.copy()
    mesh_ids = cohort["indication_mesh_id"].map(normalize_mesh_id)
    all_codes = sorted(
        {
            code
            for mesh_id in mesh_ids.dropna()
            for code in id_to_categories.get(mesh_id, set())
        }
    )
    mesh_cols = [f"mesh_{code}" for code in all_codes]
    for code, col in zip(all_codes, mesh_cols):
        cohort[col] = mesh_ids.map(
            lambda mesh_id: int(code in id_to_categories.get(mesh_id, set()))
            if mesh_id
            else 0
        ).astype("int8")
    return cohort, mesh_cols



# Selection model and weights

def prepare_selection_features(
    cohort: pd.DataFrame, mesh_cols: list[str]
) -> tuple[pd.DataFrame, list[str], list[str], dict[str, Any]]:
    cohort = cohort.copy()

    cohort["approval"] = pd.to_numeric(cohort["approval"], errors="raise").astype(int)
    cohort["included_final"] = cohort["included_final"].astype(bool)
    cohort["log_support_rows"] = np.log1p(
        pd.to_numeric(cohort["n_support_rows"], errors="coerce").fillna(0)
    )
    cohort["log_assoc_traits"] = np.log1p(
        pd.to_numeric(cohort["n_assoc_traits"], errors="coerce").fillna(0)
    )
    cohort["log_pairs_for_gene"] = np.log1p(
        pd.to_numeric(cohort["n_pairs_for_gene"], errors="coerce").fillna(0)
    )
    cohort["earliest_year_missing"] = cohort["earliest_assoc_year"].isna().astype(int)
    year_median = float(pd.to_numeric(cohort["earliest_assoc_year"], errors="coerce").median())
    cohort["selection_year_centered"] = (
        pd.to_numeric(cohort["earliest_assoc_year"], errors="coerce")
        .fillna(year_median)
        .sub(year_median)
    )

    source_cols = sorted(
        col for col in cohort.columns if col.startswith("has_source_")
    )
    for col in source_cols:
        cohort[col] = pd.to_numeric(cohort[col], errors="coerce").fillna(0).astype(int)

    numeric_cols = [
        "log_support_rows",
        "n_assoc_sources",
        "log_assoc_traits",
        "log_pairs_for_gene",
        "selection_year_centered",
        "earliest_year_missing",
        *mesh_cols,
        *source_cols,
    ]
    categorical_cols = ["ccat"]

    # Explicitly block variables that define or occur after the selection stage.
    forbidden = {
        "linked_gwas",
        "n_parseable_studies",
        "selection_stage",
        "included_final",
    }
    if forbidden.intersection(numeric_cols + categorical_cols):
        fail("A downstream linkage variable was included in the selection model")

    metadata = {
        "year_median": year_median,
        "n_numeric_features": len(numeric_cols),
        "n_categorical_features": len(categorical_cols),
        "mesh_features": mesh_cols,
        "source_features": source_cols,
    }
    return cohort, numeric_cols, categorical_cols, metadata


def cross_fitted_selection_probabilities(
    cohort: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    *,
    folds: int,
    ridge_c: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if folds < 2:
        fail("--folds must be at least 2")
    if ridge_c <= 0:
        fail("--ridge-c must be positive")

    y = cohort["included_final"].astype(int).to_numpy()
    X = cohort[numeric_cols + categorical_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )
    classifier = LogisticRegression(
        C=ridge_c,
        solver="lbfgs",
        max_iter=5000,
        class_weight=None,
        random_state=seed,
    )
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    probabilities = np.full(len(cohort), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []

    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        pipeline.fit(X.iloc[train_idx], y[train_idx])
        fold_prob = pipeline.predict_proba(X.iloc[test_idx])[:, 1]
        probabilities[test_idx] = fold_prob
        fold_rows.append(
            {
                "fold": fold_id,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "test_included": int(y[test_idx].sum()),
                "auc": float(roc_auc_score(y[test_idx], fold_prob)),
                "brier": float(brier_score_loss(y[test_idx], fold_prob)),
                "min_probability": float(fold_prob.min()),
                "max_probability": float(fold_prob.max()),
            }
        )

    if np.isnan(probabilities).any():
        fail("Cross-fitting left one or more selection probabilities missing")

    diagnostics = {
        "folds": fold_rows,
        "overall_auc": float(roc_auc_score(y, probabilities)),
        "overall_brier": float(brier_score_loss(y, probabilities)),
        "probability_min": float(probabilities.min()),
        "probability_median": float(np.median(probabilities)),
        "probability_max": float(probabilities.max()),
    }
    return probabilities, diagnostics


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    return float(weights.sum() ** 2 / np.square(weights).sum())


def construct_weights(
    cohort: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    trim_lower: float,
    trim_upper: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not (0 <= trim_lower < trim_upper <= 1):
        fail("Weight trim quantiles must satisfy 0 <= lower < upper <= 1")

    result = cohort.copy()
    result["selection_probability"] = np.clip(probabilities, 1e-6, 1 - 1e-6)
    inclusion_rate = float(result["included_final"].mean())
    selected = result["included_final"].astype(bool)

    result["selection_weight_untrimmed"] = np.nan
    result.loc[selected, "selection_weight_untrimmed"] = (
        inclusion_rate / result.loc[selected, "selection_probability"]
    )

    selected_weights = result.loc[selected, "selection_weight_untrimmed"].astype(float)
    lower_value = float(selected_weights.quantile(trim_lower))
    upper_value = float(selected_weights.quantile(trim_upper))
    result["selection_weight_trimmed"] = np.nan
    result.loc[selected, "selection_weight_trimmed"] = selected_weights.clip(
        lower=lower_value, upper=upper_value
    )

    # Normalize to mean 1 among selected pairs. This leaves coefficient estimates
    # unchanged but makes diagnostics and effective sample sizes easier to read.
    for col in ["selection_weight_untrimmed", "selection_weight_trimmed"]:
        mean_weight = float(result.loc[selected, col].mean())
        result.loc[selected, col] = result.loc[selected, col] / mean_weight

    untrimmed = result.loc[selected, "selection_weight_untrimmed"].to_numpy(float)
    trimmed = result.loc[selected, "selection_weight_trimmed"].to_numpy(float)
    selected_prob = result.loc[selected, "selection_probability"].to_numpy(float)

    diagnostics = {
        "overall_inclusion_rate": inclusion_rate,
        "selected_probability_min": float(selected_prob.min()),
        "selected_probability_p01": float(np.quantile(selected_prob, 0.01)),
        "selected_probability_median": float(np.median(selected_prob)),
        "selected_probability_p99": float(np.quantile(selected_prob, 0.99)),
        "selected_probability_max": float(selected_prob.max()),
        "trim_quantiles": [trim_lower, trim_upper],
        "trim_values_before_normalization": [lower_value, upper_value],
        "untrimmed_weight_min": float(untrimmed.min()),
        "untrimmed_weight_p01": float(np.quantile(untrimmed, 0.01)),
        "untrimmed_weight_median": float(np.median(untrimmed)),
        "untrimmed_weight_p99": float(np.quantile(untrimmed, 0.99)),
        "untrimmed_weight_max": float(untrimmed.max()),
        "trimmed_weight_min": float(trimmed.min()),
        "trimmed_weight_median": float(np.median(trimmed)),
        "trimmed_weight_max": float(trimmed.max()),
        "effective_n_unweighted": int(selected.sum()),
        "effective_n_untrimmed": effective_sample_size(untrimmed),
        "effective_n_trimmed": effective_sample_size(trimmed),
    }
    return result, diagnostics



# Balance diagnostics

def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights))


def weighted_variance(values: np.ndarray, weights: np.ndarray) -> float:
    mean = weighted_mean(values, weights)
    return float(np.average(np.square(values - mean), weights=weights))


def smd(
    selected_values: np.ndarray,
    target_values: np.ndarray,
    selected_weights: np.ndarray | None = None,
) -> float:
    target_mean = float(np.mean(target_values))
    target_var = float(np.var(target_values, ddof=0))
    if selected_weights is None:
        selected_mean = float(np.mean(selected_values))
        selected_var = float(np.var(selected_values, ddof=0))
    else:
        selected_mean = weighted_mean(selected_values, selected_weights)
        selected_var = weighted_variance(selected_values, selected_weights)
    pooled = math.sqrt((target_var + selected_var) / 2)
    return float((selected_mean - target_mean) / pooled) if pooled > 0 else 0.0


def build_balance_table(
    weighted_cohort: pd.DataFrame,
    numeric_cols: list[str],
    mesh_cols: list[str],
    source_cols: list[str],
) -> pd.DataFrame:
    df = weighted_cohort.copy()
    selected = df["included_final"].astype(bool)
    weights = df.loc[selected, "selection_weight_trimmed"].to_numpy(float)

    phase_dummies = pd.get_dummies(df["ccat"].astype(str), prefix="phase", dtype=float)
    balance_data = pd.concat(
        [
            df[[
                "approval",
                "log_support_rows",
                "n_assoc_sources",
                "log_assoc_traits",
                "log_pairs_for_gene",
                "selection_year_centered",
                "earliest_year_missing",
                *mesh_cols,
                *source_cols,
            ]].astype(float),
            phase_dummies,
        ],
        axis=1,
    )

    rows: list[dict[str, Any]] = []
    for variable in balance_data.columns:
        target_values = balance_data[variable].to_numpy(float)
        selected_values = balance_data.loc[selected, variable].to_numpy(float)
        target_mean = float(np.mean(target_values))
        selected_mean = float(np.mean(selected_values))
        weighted_selected_mean = weighted_mean(selected_values, weights)
        smd_before = smd(selected_values, target_values)
        smd_after = smd(selected_values, target_values, weights)
        rows.append(
            {
                "variable": variable,
                "target_mean": target_mean,
                "selected_unweighted_mean": selected_mean,
                "selected_weighted_mean": weighted_selected_mean,
                "smd_before": smd_before,
                "abs_smd_before": abs(smd_before),
                "smd_after": smd_after,
                "abs_smd_after": abs(smd_after),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_smd_before", ascending=False)



# Weighted outcome models with manual gene-clustered sandwich covariance

def model_predictors(model_id: str, exposure: str, mesh_cols: list[str]) -> list[str]:
    if model_id == "SW1":
        return [exposure]
    if model_id == "SW2":
        return [exposure, "log_studies"]
    if model_id == "SW3":
        return [exposure, "log_studies", "earliest_year_centered"]
    if model_id == "SW4":
        return [exposure, "log_studies", "earliest_year_centered", *mesh_cols]
    raise ValueError(f"Unknown model_id: {model_id}")


def cluster_sandwich_covariance(
    fit: Any,
    groups: pd.Series,
    weights: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    X = np.asarray(fit.model.exog, dtype=float)
    y = np.asarray(fit.model.endog, dtype=float).reshape(-1)
    mu = np.asarray(fit.fittedvalues, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)

    hessian_weight = weights * mu * (1 - mu)
    bread = X.T @ (X * hessian_weight[:, None])
    rank = int(np.linalg.matrix_rank(bread))
    bread_inv = np.linalg.pinv(bread)

    score_rows = X * (weights * (y - mu))[:, None]
    group_codes, unique_groups = pd.factorize(groups.astype(str), sort=False)
    cluster_scores = np.zeros((len(unique_groups), X.shape[1]), dtype=float)
    np.add.at(cluster_scores, group_codes, score_rows)
    meat = cluster_scores.T @ cluster_scores

    n = X.shape[0]
    k = X.shape[1]
    g = len(unique_groups)
    correction = 1.0
    if g > 1 and n > k:
        correction = (g / (g - 1)) * ((n - 1) / (n - k))
    covariance = correction * bread_inv @ meat @ bread_inv

    diagnostics = {
        "n": int(n),
        "n_parameters": int(k),
        "n_gene_clusters": int(g),
        "bread_rank": rank,
        "design_full_rank": bool(rank == k),
        "finite_sample_correction": float(correction),
        "condition_number": float(np.linalg.cond(X)),
    }
    return covariance, diagnostics


def fit_weighted_model(
    data: pd.DataFrame,
    *,
    exposure: str,
    model_id: str,
    mesh_cols: list[str],
    weight_scheme: str,
    weight_col: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    predictors = model_predictors(model_id, exposure, mesh_cols)
    needed = ["approval", "gene", *predictors]
    if weight_col:
        needed.append(weight_col)
    model_df = data[needed].dropna().copy()

    if weight_col is None:
        weights = np.ones(len(model_df), dtype=float)
    else:
        weights = model_df[weight_col].to_numpy(float)

    summary: dict[str, Any] = {
        "weight_scheme": weight_scheme,
        "model_id": model_id,
        "exposure": exposure,
        "formula": "approval ~ " + " + ".join(predictors),
        "n": int(len(model_df)),
        "n_A": int(model_df["approval"].sum()),
        "n_B": int((1 - model_df["approval"]).sum()),
        "n_gene_clusters": int(model_df["gene"].nunique()),
        "effective_sample_size": effective_sample_size(weights),
        "status": "pending",
        "error": None,
    }
    diagnostic = dict(summary)

    if model_df["approval"].nunique() < 2:
        summary.update(status="failed", error="No outcome variation")
        return summary, [], diagnostic
    if model_df[exposure].nunique() < 2:
        summary.update(status="failed", error="Exposure has no variation")
        return summary, [], diagnostic

    formula = summary["formula"]
    caught_messages: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = smf.glm(
                formula,
                data=model_df,
                family=sm.families.Binomial(),
                freq_weights=weights,
            ).fit(maxiter=300, disp=False)
            caught_messages = [
                f"{warning.category.__name__}: {warning.message}" for warning in caught
            ]

        covariance, cov_diag = cluster_sandwich_covariance(
            fit, model_df["gene"], weights
        )
        standard_errors = np.sqrt(np.clip(np.diag(covariance), 0, np.inf))
        terms = list(fit.params.index)
        se_series = pd.Series(standard_errors, index=terms)
        beta_series = fit.params.astype(float)
        z_series = beta_series / se_series
        p_series = pd.Series(
            2 * stats.norm.sf(np.abs(z_series)), index=terms, dtype=float
        )
        ci_low_beta = beta_series - 1.959963984540054 * se_series
        ci_high_beta = beta_series + 1.959963984540054 * se_series

        exposure_beta = float(beta_series[exposure])
        exposure_se = float(se_series[exposure])
        summary.update(
            status="ok",
            converged=bool(getattr(fit, "converged", True)),
            beta=exposure_beta,
            cluster_robust_se=exposure_se,
            odds_ratio=float(np.exp(exposure_beta)),
            ci_low=float(np.exp(ci_low_beta[exposure])),
            ci_high=float(np.exp(ci_high_beta[exposure])),
            p_value=float(p_series[exposure]),
            warning_messages=" | ".join(caught_messages) if caught_messages else None,
            **cov_diag,
        )

        coefficients: list[dict[str, Any]] = []
        for term in terms:
            coefficients.append(
                {
                    "weight_scheme": weight_scheme,
                    "model_id": model_id,
                    "exposure_model": exposure,
                    "term": term,
                    "beta": float(beta_series[term]),
                    "cluster_robust_se": float(se_series[term]),
                    "odds_ratio": float(np.exp(beta_series[term])),
                    "ci_low": float(np.exp(ci_low_beta[term])),
                    "ci_high": float(np.exp(ci_high_beta[term])),
                    "p_value": float(p_series[term]),
                }
            )
        diagnostic.update(
            status="ok",
            converged=summary["converged"],
            warning_messages=summary["warning_messages"],
            **cov_diag,
        )
        return summary, coefficients, diagnostic

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        summary.update(status="failed", error=error)
        diagnostic.update(status="failed", error=error)
        return summary, [], diagnostic


def parse_eligible_mesh_columns(
    model_data: pd.DataFrame, counts_path: Path
) -> list[str]:
    counts = pd.read_csv(counts_path)
    require_columns(counts, {"mesh_category_code", "eligible_model4"}, "MeSH counts")
    eligible = counts["eligible_model4"]
    if eligible.dtype != bool:
        eligible = (
            eligible.astype(str)
            .str.strip()
            .str.casefold()
            .map({"true": True, "false": False, "1": True, "0": False})
        )
    if eligible.isna().any():
        fail("Could not parse eligible_model4")
    cols = [
        f"mesh_{code}"
        for code in counts.loc[eligible.astype(bool), "mesh_category_code"].astype(str)
    ]
    missing = sorted(set(cols) - set(model_data.columns))
    if missing:
        fail(f"Eligible MeSH columns missing from Step 8 model data: {missing}")
    return cols


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for path in [args.cohort, args.model_data, args.mesh_counts, args.mesh]:
        if not path.exists():
            fail(f"Required input not found: {path}")

    print(f"Loading Step 9 cohort: {args.cohort}")
    cohort = pd.read_parquet(args.cohort)
    require_columns(cohort, REQUIRED_COHORT_COLUMNS, "Step 9 cohort")
    if cohort["ti_uid"].duplicated().any():
        fail("Step 9 cohort must contain one row per ti_uid")
    if len(cohort) != 1354:
        fail(f"Expected 1,354 clinical-stage pairs, observed {len(cohort):,}")
    if int(cohort["included_final"].sum()) != 790:
        fail("Expected 790 selected pairs")

    print(f"Parsing official MeSH categories: {args.mesh}")
    id_to_categories = parse_mesh_categories(args.mesh)
    cohort, selection_mesh_cols = add_mesh_indicators(cohort, id_to_categories)

    cohort, numeric_cols, categorical_cols, feature_meta = prepare_selection_features(
        cohort, selection_mesh_cols
    )

    print(
        "Fitting cross-fitted selection model: "
        f"{args.folds} folds; ridge C={args.ridge_c:g}"
    )
    probabilities, selection_fit_diag = cross_fitted_selection_probabilities(
        cohort,
        numeric_cols,
        categorical_cols,
        folds=args.folds,
        ridge_c=args.ridge_c,
        seed=args.seed,
    )
    weighted_cohort, weight_diag = construct_weights(
        cohort,
        probabilities,
        trim_lower=args.trim_lower,
        trim_upper=args.trim_upper,
    )

    source_cols = feature_meta["source_features"]
    balance = build_balance_table(
        weighted_cohort,
        numeric_cols,
        selection_mesh_cols,
        source_cols,
    )

    print(f"Loading Step 8 model data: {args.model_data}")
    model_data = pd.read_parquet(args.model_data)
    require_columns(model_data, REQUIRED_MODEL_COLUMNS, "Step 8 model data")
    if model_data["ti_uid"].duplicated().any():
        fail("Step 8 model data must contain one row per ti_uid")
    if len(model_data) != 790:
        fail(f"Expected 790 Step 8 model rows, observed {len(model_data):,}")

    eligible_mesh_cols = parse_eligible_mesh_columns(model_data, args.mesh_counts)
    weight_merge = weighted_cohort.loc[
        weighted_cohort["included_final"],
        [
            "ti_uid",
            "selection_probability",
            "selection_weight_untrimmed",
            "selection_weight_trimmed",
        ],
    ]
    model_data = model_data.merge(
        weight_merge, on="ti_uid", how="left", validate="one_to_one"
    )
    if model_data["selection_weight_trimmed"].isna().any():
        fail("At least one Step 8 pair did not receive a selection weight")

    weight_schemes = {
        "unweighted": None,
        "ipw_untrimmed": "selection_weight_untrimmed",
        "ipw_trimmed": "selection_weight_trimmed",
    }

    results: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []
    model_diagnostics: list[dict[str, Any]] = []
    exposures = [*COUNT_EXPOSURES, *BINARY_EXPOSURES]

    for exposure in exposures:
        print(f"\nFitting selection-weighted models for {exposure}")
        for weight_scheme, weight_col in weight_schemes.items():
            for model_id in ["SW1", "SW2", "SW3", "SW4"]:
                summary, coefficient_rows, diagnostic = fit_weighted_model(
                    model_data,
                    exposure=exposure,
                    model_id=model_id,
                    mesh_cols=eligible_mesh_cols,
                    weight_scheme=weight_scheme,
                    weight_col=weight_col,
                )
                results.append(summary)
                coefficients.extend(coefficient_rows)
                model_diagnostics.append(diagnostic)

                if (
                    exposure == "n_ancestries_all"
                    and weight_scheme in {"unweighted", "ipw_trimmed"}
                    and summary["status"] == "ok"
                ):
                    print(
                        f"  {weight_scheme:13s} {model_id}: "
                        f"OR={summary['odds_ratio']:.4f} "
                        f"({summary['ci_low']:.4f}-{summary['ci_high']:.4f}), "
                        f"p={summary['p_value']:.4g}"
                    )

    result_df = pd.DataFrame(results)
    coefficient_df = pd.DataFrame(coefficients)
    model_diag_df = pd.DataFrame(model_diagnostics)

    selection_diag_rows = [
        {
            "diagnostic": "overall_auc",
            "value": selection_fit_diag["overall_auc"],
        },
        {
            "diagnostic": "overall_brier",
            "value": selection_fit_diag["overall_brier"],
        },
    ]
    for key, value in weight_diag.items():
        if isinstance(value, (int, float, np.number)):
            selection_diag_rows.append({"diagnostic": key, "value": value})
    selection_diag_df = pd.DataFrame(selection_diag_rows)

    weights_out = args.output_dir / "11_selection_weights.parquet"
    selection_diag_out = args.output_dir / "11_selection_model_diagnostics.csv"
    balance_out = args.output_dir / "11_selection_balance.csv"
    regression_out = args.output_dir / "11_weighted_regression.csv"
    coefficients_out = args.output_dir / "11_weighted_coefficients.csv"
    model_diag_out = args.output_dir / "11_weighted_model_diagnostics.csv"
    json_out = args.output_dir / "11_selection_weighting.json"

    weighted_cohort.to_parquet(weights_out, index=False)
    selection_diag_df.to_csv(selection_diag_out, index=False)
    balance.to_csv(balance_out, index=False)
    result_df.to_csv(regression_out, index=False)
    coefficient_df.to_csv(coefficients_out, index=False)
    model_diag_df.to_csv(model_diag_out, index=False)

    primary = result_df.loc[
        (result_df["exposure"] == "n_ancestries_all")
        & (result_df["weight_scheme"].isin(["unweighted", "ipw_trimmed"]))
    ]
    payload = {
        "step": "11",
        "selection_model": {
            "method": "5-fold cross-fitted ridge logistic regression",
            "folds": args.folds,
            "ridge_c": args.ridge_c,
            "features": feature_meta,
            "fit_diagnostics": selection_fit_diag,
        },
        "weights": weight_diag,
        "balance": {
            "max_abs_smd_before": safe_float(balance["abs_smd_before"].max()),
            "max_abs_smd_after": safe_float(balance["abs_smd_after"].max()),
            "n_abs_smd_over_0_1_before": int((balance["abs_smd_before"] > 0.1).sum()),
            "n_abs_smd_over_0_1_after": int((balance["abs_smd_after"] > 0.1).sum()),
        },
        "model_definitions": {
            "SW1": "approval ~ exposure",
            "SW2": "SW1 + log1p(n_studies_total)",
            "SW3": "SW2 + earliest association year",
            "SW4": "SW3 + eligible MeSH disease-category indicators",
        },
        "primary_results": primary.to_dict("records"),
        "all_results": result_df.to_dict("records"),
    }
    json_out.write_text(
        json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print("STEP 11 SELECTION-WEIGHTING SUMMARY")
    print("=" * 72)
    print(f"Clinical-stage universe:       {len(weighted_cohort):,}")
    print(f"Analyzed ancestry pairs:       {weighted_cohort['included_final'].sum():,}")
    print(f"Cross-fitted selection AUC:    {selection_fit_diag['overall_auc']:.4f}")
    print(f"Cross-fitted selection Brier:  {selection_fit_diag['overall_brier']:.4f}")
    print(
        "Trimmed weight range:        "
        f"{weight_diag['trimmed_weight_min']:.3f}-"
        f"{weight_diag['trimmed_weight_max']:.3f}"
    )
    print(
        "Effective sample size:       "
        f"{weight_diag['effective_n_trimmed']:.1f} / 790"
    )
    print(
        "Max |SMD| before/after:      "
        f"{balance['abs_smd_before'].max():.3f} / "
        f"{balance['abs_smd_after'].max():.3f}"
    )
    print("\nPrimary all-stage diversity comparison:")
    display = result_df.loc[
        (result_df["exposure"] == "n_ancestries_all")
        & (result_df["weight_scheme"].isin(["unweighted", "ipw_trimmed"]))
        & (result_df["status"] == "ok"),
        [
            "weight_scheme",
            "model_id",
            "n",
            "effective_sample_size",
            "odds_ratio",
            "ci_low",
            "ci_high",
            "p_value",
        ],
    ]
    print(display.to_string(index=False))
    print("\nSaved outputs:")
    for path in [
        weights_out,
        selection_diag_out,
        balance_out,
        regression_out,
        coefficients_out,
        model_diag_out,
        json_out,
    ]:
        print(f"  {path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
