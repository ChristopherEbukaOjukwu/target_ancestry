#!/usr/bin/env python3
"""
Step 15F — Integrate finalized mechanistic results.

This step combines the frozen portability and colocalization branches
into analysis-ready unit- and comparison-level datasets and produces
final descriptive tables.

Scientific rules
----------------
1. The universe is the 213 PRIMARY, nonmixed gene-trait-source units
   frozen in Step 15B5.
2. Portability uses the Step 15D primary r2 < 0.10 estimates. Naive,
   FIQT, and Deming slopes are reported symmetrically.
3. Portability pool contrasts are exploratory, unit-level,
   nonparametric descriptions:
      - median and bootstrap interval within each pool;
      - A-minus-B median difference with a stratified bootstrap interval;
      - two-sided label-permutation p-value;
      - Cliff's delta, positive when Pool A is higher.
4. Colocalization uses the final Step 15E2 stability classification.
   Because only 14 units are powered, results are counts/proportions by
   pool only. No approval hypothesis test is performed.
5. Colocalization labels are ancestry-aware:
      ROBUST_SHARED
      ROBUST_DISTINCT
      ANCESTRY_DISCORDANT
      MIXED_WITH_PRIOR_SENSITIVITY
      PRIOR_SENSITIVE
6. Unpowered colocalization posterior probabilities are not promoted
   into the combined headline dataset.

Pool definitions
----------------
A = launched/approved
B = Phase I-III/not launched

Inputs
------
output/15b5_unit_analysis_universe.parquet
output/15_coloc_comparisons_primary_locked.parquet
output/15d_primary_unit_portability.parquet
output/15d_primary_comparison_portability.parquet
output/15e2_coloc_unit_stability_final.parquet
output/15e2_coloc_comparison_stability.parquet

Outputs
-------
output/15f_mechanistic_units.parquet
output/15f_mechanistic_units.csv
output/15f_mechanistic_comparisons.parquet
output/15f_mechanistic_comparisons.csv
output/15f_portability_pool_summary.parquet
output/15f_portability_pool_summary.csv
output/15f_portability_pool_contrasts.parquet
output/15f_portability_pool_contrasts.csv
output/15f_coloc_pool_summary.parquet
output/15f_coloc_pool_summary.csv
output/15f_mechanistic_coverage.parquet
output/15f_mechanistic_coverage.csv
output/15f_mechanistic_summary.json
output/15f_results_notes.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "15F.1"
ESTIMATORS = ("naive", "fiqt", "deming")
POOL_LABELS = {
    "A": "Launched/approved",
    "B": "Phase I-III/not launched",
}
COLOC_CLASSES = (
    "ROBUST_SHARED",
    "ROBUST_DISTINCT",
    "ANCESTRY_DISCORDANT",
    "MIXED_WITH_PRIOR_SENSITIVITY",
    "PRIOR_SENSITIVE",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unit-universe",
        type=Path,
        default=Path(
            "output/15b5_unit_analysis_universe.parquet"
        ),
    )
    parser.add_argument(
        "--comparison-universe",
        type=Path,
        default=Path(
            "output/15_coloc_comparisons_primary_locked.parquet"
        ),
    )
    parser.add_argument(
        "--portability-units",
        type=Path,
        default=Path(
            "output/15d_primary_unit_portability.parquet"
        ),
    )
    parser.add_argument(
        "--portability-comparisons",
        type=Path,
        default=Path(
            "output/15d_primary_comparison_portability.parquet"
        ),
    )
    parser.add_argument(
        "--coloc-units",
        type=Path,
        default=Path(
            "output/15e2_coloc_unit_stability_final.parquet"
        ),
    )
    parser.add_argument(
        "--coloc-comparisons",
        type=Path,
        default=Path(
            "output/15e2_coloc_comparison_stability.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
    )
    parser.add_argument(
        "--primary-r2-threshold",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=20000,
    )
    parser.add_argument(
        "--permutation-replicates",
        type=int,
        default=100000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--expected-primary-units",
        type=int,
        default=213,
    )
    parser.add_argument(
        "--expected-primary-comparisons",
        type=int,
        default=538,
    )
    parser.add_argument(
        "--expected-portability-units",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--expected-portability-comparisons",
        type=int,
        default=104,
    )
    parser.add_argument(
        "--expected-powered-coloc-units",
        type=int,
        default=14,
    )
    parser.add_argument(
        "--expected-powered-coloc-comparisons",
        type=int,
        default=17,
    )
    return parser.parse_args()


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return clean(value).casefold() in {
        "true",
        "1",
        "yes",
        "y",
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_seed(base_seed: int, text: str) -> int:
    digest = hashlib.sha256(
        f"{base_seed}|{text}".encode("utf-8")
    ).digest()
    return int.from_bytes(
        digest[:8],
        "little",
        signed=False,
    ) % (2**32 - 1)


def json_default(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        return (
            None
            if not math.isfinite(numeric)
            else numeric
        )
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(
        f"Cannot serialize {type(value).__name__}"
    )


def atomic_json_write(
    data: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_suffix(
        path.suffix + ".part"
    )
    temporary.write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = sorted(
        required - set(frame.columns)
    )
    if missing:
        raise SystemExit(
            f"{label} missing required columns: {missing}"
        )


def require_unique(
    frame: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    duplicates = frame.duplicated(
        columns,
        keep=False,
    )
    if duplicates.any():
        raise SystemExit(
            f"{label} has duplicate keys {columns}. Examples:\n"
            + frame.loc[
                duplicates,
                columns,
            ]
            .head(20)
            .to_string(index=False)
        )


def normalize_pool(series: pd.Series) -> pd.Series:
    result = series.map(
        lambda value: clean(value).upper()
    )
    invalid = sorted(
        set(result)
        - {"A", "B"}
    )
    if invalid:
        raise SystemExit(
            "Unexpected pool assignments: "
            f"{invalid}"
        )
    return result


def assert_expected(
    observed: int,
    expected: int,
    label: str,
) -> None:
    if expected > 0 and observed != expected:
        raise SystemExit(
            f"{label}: expected {expected}, observed {observed}."
        )


def primary_unit_universe(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        frame,
        {
            "gene_trait_uid",
            "gene",
            "candidate_source",
            "candidate_trait_key",
            "candidate_trait_name",
            "unit_analysis_role",
            "mixed_pool",
            "primary_approval_eligible",
            "primary_mechanistic_universe",
            "portability_primary_locked",
            "coloc_primary_locked",
            "coloc_powered_primary_locked",
            "sensitivity_pool_assignment",
        },
        "Step 15B5 unit universe",
    )

    primary = frame[
        frame[
            "primary_mechanistic_universe"
        ].map(as_bool)
    ].copy()

    logic = (
        primary[
            "unit_analysis_role"
        ].astype(str).eq("PRIMARY")
        & ~primary[
            "mixed_pool"
        ].map(as_bool)
        & primary[
            "primary_approval_eligible"
        ].map(as_bool)
    )
    if not logic.all():
        raise SystemExit(
            "Primary unit universe no longer equals "
            "PRIMARY, nonmixed, approval-eligible units."
        )

    require_unique(
        primary,
        ["gene_trait_uid"],
        "Primary unit universe",
    )
    primary[
        "sensitivity_pool_assignment"
    ] = normalize_pool(
        primary[
            "sensitivity_pool_assignment"
        ]
    )
    primary["approval_pool"] = primary[
        "sensitivity_pool_assignment"
    ]
    primary["approval_group"] = primary[
        "approval_pool"
    ].map(POOL_LABELS)
    return primary


def primary_comparison_universe(
    frame: pd.DataFrame,
    primary_units: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        frame,
        {
            "comparison_uid",
            "gene_trait_uid",
            "gene",
            "candidate_source",
            "candidate_trait_name",
            "comparison_population",
            "coloc_primary_locked",
            "primary_mechanistic_universe",
        },
        "Primary comparison universe",
    )
    base = frame[
        frame["coloc_primary_locked"].map(
            as_bool
        )
        & frame[
            "primary_mechanistic_universe"
        ].map(as_bool)
    ].copy()
    require_unique(
        base,
        ["comparison_uid"],
        "Primary comparison universe",
    )

    pool_metadata = primary_units[
        [
            "gene_trait_uid",
            "approval_pool",
            "approval_group",
        ]
    ]
    base = base.drop(
        columns=[
            "sensitivity_pool_assignment",
            "approval_pool",
            "approval_group",
        ],
        errors="ignore",
    ).merge(
        pool_metadata,
        on="gene_trait_uid",
        how="left",
        validate="many_to_one",
    )
    if base["approval_pool"].isna().any():
        raise SystemExit(
            "Primary comparisons could not all be assigned to a pool."
        )
    return base


def common_metadata(
    frame: pd.DataFrame,
    key: str,
    columns: list[str],
) -> pd.DataFrame:
    available = [
        column
        for column in columns
        if column in frame.columns
    ]
    rows = []
    for key_value, group in frame.groupby(
        key,
        sort=False,
        dropna=False,
    ):
        row = {key: key_value}
        for column in available:
            values = group[
                column
            ].drop_duplicates()
            if len(values) != 1:
                raise SystemExit(
                    f"{column} varies within {key}={key_value}: "
                    f"{values.tolist()}"
                )
            row[column] = values.iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def wide_portability_units(
    frame: pd.DataFrame,
    primary_threshold: float,
) -> pd.DataFrame:
    require_columns(
        frame,
        {
            "gene_trait_uid",
            "estimator",
            "r2_threshold",
            "slope",
            "n_comparison_ancestries",
            "comparison_ancestries",
            "n_variants_total",
            "n_variants_minimum",
            "n_variants_median",
            "sign_concordance",
            "pearson_r",
            "any_deming_near_boundary",
        },
        "Step 15D primary unit portability",
    )

    selected = frame[
        np.isclose(
            frame[
                "r2_threshold"
            ].astype(float),
            primary_threshold,
            rtol=0.0,
            atol=1e-12,
        )
        & frame[
            "estimator"
        ].astype(str).isin(
            ESTIMATORS
        )
    ].copy()

    require_unique(
        selected,
        [
            "gene_trait_uid",
            "estimator",
        ],
        "Unit portability estimates",
    )

    estimator_sets = (
        selected.groupby(
            "gene_trait_uid"
        )["estimator"]
        .apply(
            lambda values: set(
                values.astype(str)
            )
        )
    )
    incomplete = estimator_sets[
        estimator_sets.map(
            lambda values: values
            != set(ESTIMATORS)
        )
    ]
    if len(incomplete):
        raise SystemExit(
            "Portability units lack all three estimators: "
            + ", ".join(
                incomplete.index.astype(str).tolist()[:20]
            )
        )

    slopes = selected.pivot(
        index="gene_trait_uid",
        columns="estimator",
        values="slope",
    ).rename(
        columns={
            estimator: (
                f"portability_{estimator}_slope"
            )
            for estimator in ESTIMATORS
        }
    ).reset_index()

    metadata = common_metadata(
        selected,
        "gene_trait_uid",
        [
            "n_comparison_ancestries",
            "comparison_ancestries",
            "n_variants_total",
            "n_variants_minimum",
            "n_variants_median",
            "sign_concordance",
            "pearson_r",
            "any_deming_near_boundary",
        ],
    ).rename(
        columns={
            "n_comparison_ancestries":
                "portability_n_comparison_ancestries",
            "comparison_ancestries":
                "portability_comparison_ancestries",
            "n_variants_total":
                "portability_n_variants_total",
            "n_variants_minimum":
                "portability_n_variants_minimum",
            "n_variants_median":
                "portability_n_variants_median",
            "sign_concordance":
                "portability_sign_concordance",
            "pearson_r":
                "portability_pearson_r",
            "any_deming_near_boundary":
                "portability_any_deming_near_boundary",
        }
    )

    result = slopes.merge(
        metadata,
        on="gene_trait_uid",
        how="left",
        validate="one_to_one",
    )
    result[
        "portability_primary_r2_threshold"
    ] = float(primary_threshold)
    return result


def wide_portability_comparisons(
    frame: pd.DataFrame,
    primary_threshold: float,
) -> pd.DataFrame:
    require_columns(
        frame,
        {
            "comparison_uid",
            "estimator",
            "r2_threshold",
            "slope",
            "ci_lower",
            "ci_upper",
            "n_variants",
            "sign_concordance",
            "pearson_r",
            "lead_sign_concordant",
            "median_abs_eur_z",
            "deming_near_boundary",
        },
        "Step 15D primary comparison portability",
    )

    selected = frame[
        np.isclose(
            frame[
                "r2_threshold"
            ].astype(float),
            primary_threshold,
            rtol=0.0,
            atol=1e-12,
        )
        & frame[
            "estimator"
        ].astype(str).isin(
            ESTIMATORS
        )
    ].copy()

    require_unique(
        selected,
        [
            "comparison_uid",
            "estimator",
        ],
        "Comparison portability estimates",
    )

    wide_parts = []
    for value_column in [
        "slope",
        "ci_lower",
        "ci_upper",
    ]:
        wide = selected.pivot(
            index="comparison_uid",
            columns="estimator",
            values=value_column,
        ).rename(
            columns={
                estimator: (
                    f"portability_{estimator}_{value_column}"
                )
                for estimator in ESTIMATORS
            }
        )
        wide_parts.append(wide)

    wide_values = pd.concat(
        wide_parts,
        axis=1,
    ).reset_index()

    metadata = common_metadata(
        selected,
        "comparison_uid",
        [
            "n_variants",
            "sign_concordance",
            "pearson_r",
            "lead_sign_concordant",
            "median_abs_eur_z",
            "deming_near_boundary",
        ],
    ).rename(
        columns={
            "n_variants":
                "portability_n_variants",
            "sign_concordance":
                "portability_sign_concordance",
            "pearson_r":
                "portability_pearson_r",
            "lead_sign_concordant":
                "portability_lead_sign_concordant",
            "median_abs_eur_z":
                "portability_median_abs_eur_z",
            "deming_near_boundary":
                "portability_deming_near_boundary",
        }
    )

    result = wide_values.merge(
        metadata,
        on="comparison_uid",
        how="left",
        validate="one_to_one",
    )
    result[
        "portability_primary_r2_threshold"
    ] = float(primary_threshold)
    return result


def bootstrap_median_ci(
    values: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(
        values,
        dtype=float,
    )
    values = values[
        np.isfinite(values)
    ]
    if len(values) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(values),
        size=(
            replicates,
            len(values),
        ),
    )
    medians = np.median(
        values[indices],
        axis=1,
    )
    lower, upper = np.percentile(
        medians,
        [2.5, 97.5],
    )
    return float(lower), float(upper)


def bootstrap_median_difference_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    rng = np.random.default_rng(seed)
    indices_a = rng.integers(
        0,
        len(a),
        size=(
            replicates,
            len(a),
        ),
    )
    indices_b = rng.integers(
        0,
        len(b),
        size=(
            replicates,
            len(b),
        ),
    )
    differences = (
        np.median(
            a[indices_a],
            axis=1,
        )
        - np.median(
            b[indices_b],
            axis=1,
        )
    )
    lower, upper = np.percentile(
        differences,
        [2.5, 97.5],
    )
    return float(lower), float(upper)


def permutation_median_p(
    values_a: np.ndarray,
    values_b: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[float, int]:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    observed = float(
        np.median(a)
        - np.median(b)
    )
    pooled = np.concatenate([a, b])
    n_a = len(a)
    rng = np.random.default_rng(seed)

    extreme = 0
    for _ in range(replicates):
        permuted = rng.permutation(pooled)
        difference = float(
            np.median(
                permuted[:n_a]
            )
            - np.median(
                permuted[n_a:]
            )
        )
        if abs(difference) >= abs(observed) - 1e-15:
            extreme += 1

    p_value = (
        extreme + 1
    ) / (
        replicates + 1
    )
    return float(p_value), int(extreme)


def cliffs_delta(
    values_a: np.ndarray,
    values_b: np.ndarray,
) -> float:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    differences = (
        a[:, None]
        - b[None, :]
    )
    return float(
        (
            np.sum(differences > 0)
            - np.sum(differences < 0)
        )
        / differences.size
    )


def cliff_magnitude(delta: float) -> str:
    absolute = abs(delta)
    if absolute < 0.147:
        return "NEGLIGIBLE"
    if absolute < 0.33:
        return "SMALL"
    if absolute < 0.474:
        return "MEDIUM"
    return "LARGE"


def portability_descriptions(
    units: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    permutation_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    contrast_rows = []

    for estimator in ESTIMATORS:
        column = (
            f"portability_{estimator}_slope"
        )
        available = units[
            units[
                "portability_final_available"
            ]
            & pd.to_numeric(
                units[column],
                errors="coerce",
            ).notna()
        ].copy()
        available[column] = pd.to_numeric(
            available[column],
            errors="coerce",
        )

        for pool in ("A", "B"):
            group = available[
                available[
                    "approval_pool"
                ].eq(pool)
            ]
            values = group[
                column
            ].to_numpy(dtype=float)
            lower, upper = bootstrap_median_ci(
                values,
                bootstrap_replicates,
                stable_seed(
                    seed,
                    f"{estimator}|{pool}|median",
                ),
            )
            q1, q3 = np.percentile(
                values,
                [25, 75],
            )
            summary_rows.append(
                {
                    "estimator": estimator,
                    "approval_pool": pool,
                    "approval_group": (
                        POOL_LABELS[pool]
                    ),
                    "n_units": int(
                        len(group)
                    ),
                    "n_unique_genes": int(
                        group["gene"].nunique()
                    ),
                    "median_slope": float(
                        np.median(values)
                    ),
                    "median_ci_lower": lower,
                    "median_ci_upper": upper,
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(q3 - q1),
                    "mean_slope": float(
                        np.mean(values)
                    ),
                    "standard_deviation": float(
                        np.std(
                            values,
                            ddof=1,
                        )
                    )
                    if len(values) > 1
                    else math.nan,
                    "minimum": float(
                        np.min(values)
                    ),
                    "maximum": float(
                        np.max(values)
                    ),
                    "bootstrap_replicates": int(
                        bootstrap_replicates
                    ),
                }
            )

        values_a = available.loc[
            available[
                "approval_pool"
            ].eq("A"),
            column,
        ].to_numpy(dtype=float)
        values_b = available.loc[
            available[
                "approval_pool"
            ].eq("B"),
            column,
        ].to_numpy(dtype=float)

        if len(values_a) == 0 or len(values_b) == 0:
            raise SystemExit(
                f"Estimator {estimator} lacks one approval pool."
            )

        difference = float(
            np.median(values_a)
            - np.median(values_b)
        )
        ci_lower, ci_upper = (
            bootstrap_median_difference_ci(
                values_a,
                values_b,
                bootstrap_replicates,
                stable_seed(
                    seed,
                    f"{estimator}|difference",
                ),
            )
        )
        p_value, extreme = (
            permutation_median_p(
                values_a,
                values_b,
                permutation_replicates,
                stable_seed(
                    seed,
                    f"{estimator}|permutation",
                ),
            )
        )
        delta = cliffs_delta(
            values_a,
            values_b,
        )

        contrast_rows.append(
            {
                "estimator": estimator,
                "n_pool_A": int(
                    len(values_a)
                ),
                "n_pool_B": int(
                    len(values_b)
                ),
                "median_pool_A": float(
                    np.median(values_a)
                ),
                "median_pool_B": float(
                    np.median(values_b)
                ),
                "median_difference_A_minus_B": (
                    difference
                ),
                "difference_ci_lower": (
                    ci_lower
                ),
                "difference_ci_upper": (
                    ci_upper
                ),
                "cliffs_delta_A_vs_B": delta,
                "cliffs_delta_magnitude": (
                    cliff_magnitude(delta)
                ),
                "exploratory_permutation_p_two_sided": (
                    p_value
                ),
                "permutation_extreme_count": (
                    extreme
                ),
                "permutation_replicates": int(
                    permutation_replicates
                ),
                "bootstrap_replicates": int(
                    bootstrap_replicates
                ),
                "direction_definition": (
                    "Positive values mean Pool A "
                    "(launched/approved) is higher."
                ),
                "inference_status": (
                    "EXPLORATORY_UNADJUSTED_UNIT_LEVEL"
                ),
            }
        )

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(contrast_rows),
    )


def coloc_descriptions(
    units: pd.DataFrame,
) -> pd.DataFrame:
    powered = units[
        units[
            "coloc_powered_final_available"
        ]
    ].copy()

    rows = []
    for pool in ("A", "B"):
        pool_all = units[
            units[
                "approval_pool"
            ].eq(pool)
        ]
        pool_powered = powered[
            powered[
                "approval_pool"
            ].eq(pool)
        ]
        denominator = len(
            pool_powered
        )

        rows.append(
            {
                "approval_pool": pool,
                "approval_group": (
                    POOL_LABELS[pool]
                ),
                "classification": (
                    "ALL_POWERED"
                ),
                "n_units": int(
                    denominator
                ),
                "proportion_among_powered": (
                    1.0
                    if denominator
                    else math.nan
                ),
                "primary_pool_units": int(
                    len(pool_all)
                ),
                "powered_fraction_of_primary_pool": (
                    float(
                        denominator
                        / len(pool_all)
                    )
                    if len(pool_all)
                    else math.nan
                ),
                "inference_status": (
                    "DESCRIPTIVE_ONLY_NO_POOL_TEST"
                ),
            }
        )

        for classification in COLOC_CLASSES:
            count = int(
                pool_powered[
                    "unit_stability_final"
                ].eq(
                    classification
                ).sum()
            )
            rows.append(
                {
                    "approval_pool": pool,
                    "approval_group": (
                        POOL_LABELS[pool]
                    ),
                    "classification": (
                        classification
                    ),
                    "n_units": count,
                    "proportion_among_powered": (
                        float(
                            count
                            / denominator
                        )
                        if denominator
                        else math.nan
                    ),
                    "primary_pool_units": int(
                        len(pool_all)
                    ),
                    "powered_fraction_of_primary_pool": (
                        float(
                            denominator
                            / len(pool_all)
                        )
                        if len(pool_all)
                        else math.nan
                    ),
                    "inference_status": (
                        "DESCRIPTIVE_ONLY_NO_POOL_TEST"
                    ),
                }
            )

    return pd.DataFrame(rows)


def coverage_table(
    units: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> pd.DataFrame:
    unit_rows = []
    comparison_rows = []

    unit_definitions = [
        (
            "PRIMARY_MECHANISTIC_UNIVERSE",
            np.ones(
                len(units),
                dtype=bool,
            ),
        ),
        (
            "PORTABILITY_PRE_LD_LOCKED",
            units[
                "portability_primary_locked"
            ].map(as_bool).to_numpy(),
        ),
        (
            "PORTABILITY_FINAL",
            units[
                "portability_final_available"
            ].to_numpy(),
        ),
        (
            "COLOC_INPUT_LOCKED",
            units[
                "coloc_primary_locked"
            ].map(as_bool).to_numpy(),
        ),
        (
            "COLOC_POWERED_FROZEN",
            units[
                "coloc_powered_primary_locked"
            ].map(as_bool).to_numpy(),
        ),
        (
            "COLOC_POWERED_FINAL",
            units[
                "coloc_powered_final_available"
            ].to_numpy(),
        ),
        (
            "BOTH_FINAL_MECHANISMS",
            units[
                "both_final_mechanisms_available"
            ].to_numpy(),
        ),
    ]

    for stage, mask in unit_definitions:
        for pool in ("ALL", "A", "B"):
            pool_mask = (
                np.ones(
                    len(units),
                    dtype=bool,
                )
                if pool == "ALL"
                else units[
                    "approval_pool"
                ].eq(pool).to_numpy()
            )
            count = int(
                np.sum(
                    mask
                    & pool_mask
                )
            )
            denominator = int(
                np.sum(pool_mask)
            )
            unit_rows.append(
                {
                    "level": "UNIT",
                    "stage": stage,
                    "approval_pool": pool,
                    "n": count,
                    "denominator_primary": (
                        denominator
                    ),
                    "fraction_of_primary": (
                        float(
                            count
                            / denominator
                        )
                        if denominator
                        else math.nan
                    ),
                }
            )

    comparison_definitions = [
        (
            "PRIMARY_MECHANISTIC_COMPARISONS",
            np.ones(
                len(comparisons),
                dtype=bool,
            ),
        ),
        (
            "PORTABILITY_FINAL",
            comparisons[
                "portability_final_available"
            ].to_numpy(),
        ),
        (
            "COLOC_POWERED_FINAL",
            comparisons[
                "coloc_powered_final_available"
            ].to_numpy(),
        ),
        (
            "BOTH_FINAL_MECHANISMS",
            comparisons[
                "both_final_mechanisms_available"
            ].to_numpy(),
        ),
    ]

    for stage, mask in comparison_definitions:
        for pool in ("ALL", "A", "B"):
            pool_mask = (
                np.ones(
                    len(comparisons),
                    dtype=bool,
                )
                if pool == "ALL"
                else comparisons[
                    "approval_pool"
                ].eq(pool).to_numpy()
            )
            count = int(
                np.sum(
                    mask
                    & pool_mask
                )
            )
            denominator = int(
                np.sum(pool_mask)
            )
            comparison_rows.append(
                {
                    "level": "COMPARISON",
                    "stage": stage,
                    "approval_pool": pool,
                    "n": count,
                    "denominator_primary": (
                        denominator
                    ),
                    "fraction_of_primary": (
                        float(
                            count
                            / denominator
                        )
                        if denominator
                        else math.nan
                    ),
                }
            )

    return pd.DataFrame(
        unit_rows
        + comparison_rows
    )


def write_notes(
    path: Path,
    *,
    unit_count: int,
    portability_unit_count: int,
    coloc_unit_count: int,
) -> None:
    text = f"""# Step 15F mechanistic results

## Frozen analysis universe

- Primary nonmixed gene-trait-source units: {unit_count}
- Units with final primary portability estimates: {portability_unit_count}
- Units powered for final colocalization classification: {coloc_unit_count}

## Portability

The primary portability universe uses ancestry-paired pruning at
$r^2 < 0.10$. Naive, FIQT, and Deming through-origin slopes are carried
forward symmetrically. Pool A versus Pool B summaries are exploratory,
unadjusted, unit-level descriptions. A positive median difference or
Cliff's delta means larger portability in Pool A.

The permutation p-values do not account for repeated genes, traits,
sources, or other covariates and are not confirmatory approval models.

## Colocalization

Only units powered in both ancestries receive a final colocalization
classification. Pool tables are descriptive counts and proportions;
no approval comparison test is performed.

The authoritative field is `unit_stability_final`. It preserves
ancestry disagreement and prior sensitivity instead of reducing
multiple ancestries to a maximum or a misleading median.

## Interpretation boundary

Portability measures the relative magnitude of marginal variant effects
within measurable gene-trait neighborhoods. Colocalization measures
support for shared versus distinct regional causal signals under the
single-causal-variant ABF model and tested priors. Neither quantity is a
direct measure of disease portability or clinical efficacy.
"""
    path.write_text(
        text,
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()

    if args.bootstrap_replicates < 100:
        raise SystemExit(
            "--bootstrap-replicates must be at least 100."
        )
    if args.permutation_replicates < 100:
        raise SystemExit(
            "--permutation-replicates must be at least 100."
        )

    input_paths = [
        args.unit_universe,
        args.comparison_universe,
        args.portability_units,
        args.portability_comparisons,
        args.coloc_units,
        args.coloc_comparisons,
    ]
    for path in input_paths:
        if not path.exists():
            raise SystemExit(
                f"Missing input: {path}"
            )

    unit_universe_raw = pd.read_parquet(
        args.unit_universe
    )
    comparison_universe_raw = (
        pd.read_parquet(
            args.comparison_universe
        )
    )
    portability_units_raw = (
        pd.read_parquet(
            args.portability_units
        )
    )
    portability_comparisons_raw = (
        pd.read_parquet(
            args.portability_comparisons
        )
    )
    coloc_units_raw = pd.read_parquet(
        args.coloc_units
    )
    coloc_comparisons_raw = (
        pd.read_parquet(
            args.coloc_comparisons
        )
    )

    units = primary_unit_universe(
        unit_universe_raw
    )
    comparisons = (
        primary_comparison_universe(
            comparison_universe_raw,
            units,
        )
    )

    assert_expected(
        len(units),
        args.expected_primary_units,
        "Primary unit count",
    )
    assert_expected(
        len(comparisons),
        args.expected_primary_comparisons,
        "Primary comparison count",
    )

    portability_units = (
        wide_portability_units(
            portability_units_raw,
            args.primary_r2_threshold,
        )
    )
    portability_comparisons = (
        wide_portability_comparisons(
            portability_comparisons_raw,
            args.primary_r2_threshold,
        )
    )

    assert_expected(
        len(portability_units),
        args.expected_portability_units,
        "Final portability unit count",
    )
    assert_expected(
        len(portability_comparisons),
        args.expected_portability_comparisons,
        "Final portability comparison count",
    )

    require_columns(
        coloc_units_raw,
        {
            "gene_trait_uid",
            "unit_stability_final",
            "primary_unit_category",
            "n_powered_comparisons",
            "comparison_ancestries",
            "primary_pp_h4_minimum",
            "primary_pp_h4_median",
            "primary_pp_h4_maximum",
        },
        "Step 15E2 unit stability",
    )
    require_unique(
        coloc_units_raw,
        ["gene_trait_uid"],
        "Step 15E2 unit stability",
    )
    assert_expected(
        len(coloc_units_raw),
        args.expected_powered_coloc_units,
        "Powered colocalization unit count",
    )

    require_columns(
        coloc_comparisons_raw,
        {
            "comparison_uid",
            "comparison_prior_stability",
            "primary_category",
            "primary_pp_h3",
            "primary_pp_h4",
            "minimum_pp_h3",
            "maximum_pp_h3",
            "minimum_pp_h4",
            "maximum_pp_h4",
        },
        "Step 15E2 comparison stability",
    )
    require_unique(
        coloc_comparisons_raw,
        ["comparison_uid"],
        "Step 15E2 comparison stability",
    )
    assert_expected(
        len(coloc_comparisons_raw),
        args.expected_powered_coloc_comparisons,
        "Powered colocalization comparison count",
    )

    unit_coloc_columns = [
        column
        for column in [
            "gene_trait_uid",
            "unit_stability_final",
            "primary_unit_category",
            "n_powered_comparisons",
            "comparison_ancestries",
            "comparison_stability_classes",
            "primary_pp_h4_minimum",
            "primary_pp_h4_median",
            "primary_pp_h4_maximum",
        ]
        if column in coloc_units_raw.columns
    ]
    unit_coloc = coloc_units_raw[
        unit_coloc_columns
    ].rename(
        columns={
            "n_powered_comparisons":
                "coloc_n_powered_comparisons",
            "comparison_ancestries":
                "coloc_powered_comparison_ancestries",
            "comparison_stability_classes":
                "coloc_comparison_stability_classes",
            "primary_pp_h4_minimum":
                "coloc_primary_pp_h4_minimum",
            "primary_pp_h4_median":
                "coloc_primary_pp_h4_median",
            "primary_pp_h4_maximum":
                "coloc_primary_pp_h4_maximum",
        }
    )

    units = units.merge(
        portability_units,
        on="gene_trait_uid",
        how="left",
        validate="one_to_one",
    ).merge(
        unit_coloc,
        on="gene_trait_uid",
        how="left",
        validate="one_to_one",
    )

    units[
        "portability_final_available"
    ] = units[
        "portability_naive_slope"
    ].notna()
    units[
        "coloc_powered_final_available"
    ] = units[
        "unit_stability_final"
    ].notna()
    units[
        "both_final_mechanisms_available"
    ] = (
        units[
            "portability_final_available"
        ]
        & units[
            "coloc_powered_final_available"
        ]
    )
    units["mechanistic_availability"] = np.select(
        [
            units[
                "both_final_mechanisms_available"
            ],
            units[
                "portability_final_available"
            ],
            units[
                "coloc_powered_final_available"
            ],
        ],
        [
            "BOTH",
            "PORTABILITY_ONLY",
            "COLOC_POWERED_ONLY",
        ],
        default="NEITHER",
    )

    coloc_comparison_columns = [
        column
        for column in [
            "comparison_uid",
            "comparison_prior_stability",
            "primary_category",
            "primary_pp_h3",
            "primary_pp_h4",
            "primary_conditional_h4",
            "minimum_pp_h3",
            "maximum_pp_h3",
            "minimum_pp_h4",
            "maximum_pp_h4",
        ]
        if column in coloc_comparisons_raw.columns
    ]
    coloc_comparison = (
        coloc_comparisons_raw[
            coloc_comparison_columns
        ].rename(
            columns={
                "comparison_prior_stability":
                    "coloc_comparison_prior_stability",
                "primary_category":
                    "coloc_primary_category",
                "primary_pp_h3":
                    "coloc_primary_pp_h3",
                "primary_pp_h4":
                    "coloc_primary_pp_h4",
                "primary_conditional_h4":
                    "coloc_primary_conditional_h4",
                "minimum_pp_h3":
                    "coloc_minimum_pp_h3",
                "maximum_pp_h3":
                    "coloc_maximum_pp_h3",
                "minimum_pp_h4":
                    "coloc_minimum_pp_h4",
                "maximum_pp_h4":
                    "coloc_maximum_pp_h4",
            }
        )
    )

    comparisons = comparisons.merge(
        portability_comparisons,
        on="comparison_uid",
        how="left",
        validate="one_to_one",
    ).merge(
        coloc_comparison,
        on="comparison_uid",
        how="left",
        validate="one_to_one",
    )
    comparisons[
        "portability_final_available"
    ] = comparisons[
        "portability_naive_slope"
    ].notna()
    comparisons[
        "coloc_powered_final_available"
    ] = comparisons[
        "coloc_comparison_prior_stability"
    ].notna()
    comparisons[
        "both_final_mechanisms_available"
    ] = (
        comparisons[
            "portability_final_available"
        ]
        & comparisons[
            "coloc_powered_final_available"
        ]
    )
    comparisons[
        "mechanistic_availability"
    ] = np.select(
        [
            comparisons[
                "both_final_mechanisms_available"
            ],
            comparisons[
                "portability_final_available"
            ],
            comparisons[
                "coloc_powered_final_available"
            ],
        ],
        [
            "BOTH",
            "PORTABILITY_ONLY",
            "COLOC_POWERED_ONLY",
        ],
        default="NEITHER",
    )

    if int(
        units[
            "portability_final_available"
        ].sum()
    ) != len(portability_units):
        raise SystemExit(
            "Not all portability units merged into the primary universe."
        )
    if int(
        units[
            "coloc_powered_final_available"
        ].sum()
    ) != len(coloc_units_raw):
        raise SystemExit(
            "Not all powered colocalization units merged "
            "into the primary universe."
        )
    if int(
        comparisons[
            "portability_final_available"
        ].sum()
    ) != len(portability_comparisons):
        raise SystemExit(
            "Not all portability comparisons merged into "
            "the primary comparison universe."
        )
    if int(
        comparisons[
            "coloc_powered_final_available"
        ].sum()
    ) != len(coloc_comparisons_raw):
        raise SystemExit(
            "Not all powered colocalization comparisons merged "
            "into the primary comparison universe."
        )

    portability_summary, portability_contrasts = (
        portability_descriptions(
            units,
            bootstrap_replicates=(
                args.bootstrap_replicates
            ),
            permutation_replicates=(
                args.permutation_replicates
            ),
            seed=args.seed,
        )
    )
    coloc_summary = coloc_descriptions(
        units
    )
    coverage = coverage_table(
        units,
        comparisons,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "mechanistic_units_parquet":
            args.output_dir
            / "15f_mechanistic_units.parquet",
        "mechanistic_units_csv":
            args.output_dir
            / "15f_mechanistic_units.csv",
        "mechanistic_comparisons_parquet":
            args.output_dir
            / "15f_mechanistic_comparisons.parquet",
        "mechanistic_comparisons_csv":
            args.output_dir
            / "15f_mechanistic_comparisons.csv",
        "portability_pool_summary_parquet":
            args.output_dir
            / "15f_portability_pool_summary.parquet",
        "portability_pool_summary_csv":
            args.output_dir
            / "15f_portability_pool_summary.csv",
        "portability_pool_contrasts_parquet":
            args.output_dir
            / "15f_portability_pool_contrasts.parquet",
        "portability_pool_contrasts_csv":
            args.output_dir
            / "15f_portability_pool_contrasts.csv",
        "coloc_pool_summary_parquet":
            args.output_dir
            / "15f_coloc_pool_summary.parquet",
        "coloc_pool_summary_csv":
            args.output_dir
            / "15f_coloc_pool_summary.csv",
        "coverage_parquet":
            args.output_dir
            / "15f_mechanistic_coverage.parquet",
        "coverage_csv":
            args.output_dir
            / "15f_mechanistic_coverage.csv",
        "summary_json":
            args.output_dir
            / "15f_mechanistic_summary.json",
        "notes_markdown":
            args.output_dir
            / "15f_results_notes.md",
    }

    units.to_parquet(
        paths[
            "mechanistic_units_parquet"
        ],
        index=False,
    )
    units.to_csv(
        paths[
            "mechanistic_units_csv"
        ],
        index=False,
    )
    comparisons.to_parquet(
        paths[
            "mechanistic_comparisons_parquet"
        ],
        index=False,
    )
    comparisons.to_csv(
        paths[
            "mechanistic_comparisons_csv"
        ],
        index=False,
    )
    portability_summary.to_parquet(
        paths[
            "portability_pool_summary_parquet"
        ],
        index=False,
    )
    portability_summary.to_csv(
        paths[
            "portability_pool_summary_csv"
        ],
        index=False,
    )
    portability_contrasts.to_parquet(
        paths[
            "portability_pool_contrasts_parquet"
        ],
        index=False,
    )
    portability_contrasts.to_csv(
        paths[
            "portability_pool_contrasts_csv"
        ],
        index=False,
    )
    coloc_summary.to_parquet(
        paths[
            "coloc_pool_summary_parquet"
        ],
        index=False,
    )
    coloc_summary.to_csv(
        paths[
            "coloc_pool_summary_csv"
        ],
        index=False,
    )
    coverage.to_parquet(
        paths["coverage_parquet"],
        index=False,
    )
    coverage.to_csv(
        paths["coverage_csv"],
        index=False,
    )

    write_notes(
        paths["notes_markdown"],
        unit_count=len(units),
        portability_unit_count=int(
            units[
                "portability_final_available"
            ].sum()
        ),
        coloc_unit_count=int(
            units[
                "coloc_powered_final_available"
            ].sum()
        ),
    )

    unit_availability_counts = {
        clean(key): int(value)
        for key, value in (
            units[
                "mechanistic_availability"
            ]
            .value_counts()
            .items()
        )
    }
    comparison_availability_counts = {
        clean(key): int(value)
        for key, value in (
            comparisons[
                "mechanistic_availability"
            ]
            .value_counts()
            .items()
        )
    }
    coloc_class_counts = {
        clean(key): int(value)
        for key, value in (
            units.loc[
                units[
                    "coloc_powered_final_available"
                ],
                "unit_stability_final",
            ]
            .value_counts()
            .items()
        )
    }

    summary = {
        "step": "15F",
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "design": {
            "primary_unit_universe": (
                "Step 15B5 PRIMARY, nonmixed, "
                "approval-eligible gene-trait-source units."
            ),
            "portability": (
                "Step 15D primary r2 < 0.10 unit estimates; "
                "naive, FIQT, and Deming reported symmetrically."
            ),
            "portability_pool_comparison": (
                "Exploratory unit-level median contrast, "
                "stratified bootstrap interval, label-permutation "
                "p-value, and Cliff's delta."
            ),
            "colocalization": (
                "Step 15E2 powered-unit final stability "
                "classification; pool summaries are descriptive only."
            ),
        },
        "counts": {
            "primary_units": int(
                len(units)
            ),
            "primary_comparisons": int(
                len(comparisons)
            ),
            "portability_final_units": int(
                units[
                    "portability_final_available"
                ].sum()
            ),
            "portability_final_comparisons": int(
                comparisons[
                    "portability_final_available"
                ].sum()
            ),
            "coloc_powered_final_units": int(
                units[
                    "coloc_powered_final_available"
                ].sum()
            ),
            "coloc_powered_final_comparisons": int(
                comparisons[
                    "coloc_powered_final_available"
                ].sum()
            ),
            "both_final_units": int(
                units[
                    "both_final_mechanisms_available"
                ].sum()
            ),
            "both_final_comparisons": int(
                comparisons[
                    "both_final_mechanisms_available"
                ].sum()
            ),
            "primary_units_by_pool": {
                pool: int(
                    units[
                        "approval_pool"
                    ].eq(pool).sum()
                )
                for pool in ("A", "B")
            },
            "portability_units_by_pool": {
                pool: int(
                    (
                        units[
                            "approval_pool"
                        ].eq(pool)
                        & units[
                            "portability_final_available"
                        ]
                    ).sum()
                )
                for pool in ("A", "B")
            },
            "coloc_powered_units_by_pool": {
                pool: int(
                    (
                        units[
                            "approval_pool"
                        ].eq(pool)
                        & units[
                            "coloc_powered_final_available"
                        ]
                    ).sum()
                )
                for pool in ("A", "B")
            },
        },
        "unit_mechanistic_availability_counts": (
            unit_availability_counts
        ),
        "comparison_mechanistic_availability_counts": (
            comparison_availability_counts
        ),
        "portability_pool_contrasts": (
            portability_contrasts.to_dict(
                "records"
            )
        ),
        "coloc_final_unit_class_counts": (
            coloc_class_counts
        ),
        "coloc_inference_status": (
            "DESCRIPTIVE_ONLY; no approval hypothesis test "
            "because only 14 units were powered."
        ),
        "outputs": {
            key: str(value)
            for key, value in paths.items()
        },
        "completed_at_utc": utc_now(),
    }
    atomic_json_write(
        summary,
        paths["summary_json"],
    )

    print("=" * 78)
    print(
        "STEP 15F — INTEGRATED MECHANISTIC RESULTS"
    )
    print("=" * 78)
    print("Status: PASS")
    print(
        "Primary units:",
        len(units),
    )
    print(
        "Primary comparisons:",
        len(comparisons),
    )
    print(
        "Portability final:",
        int(
            units[
                "portability_final_available"
            ].sum()
        ),
        "units |",
        int(
            comparisons[
                "portability_final_available"
            ].sum()
        ),
        "comparisons",
    )
    print(
        "Colocalization powered final:",
        int(
            units[
                "coloc_powered_final_available"
            ].sum()
        ),
        "units |",
        int(
            comparisons[
                "coloc_powered_final_available"
            ].sum()
        ),
        "comparisons",
    )
    print(
        "Both final mechanisms:",
        int(
            units[
                "both_final_mechanisms_available"
            ].sum()
        ),
        "units",
    )
    print()
    print("Portability Pool A minus Pool B:")
    for row in portability_contrasts.itertuples(
        index=False
    ):
        print(
            f"  {row.estimator}: "
            f"median difference="
            f"{row.median_difference_A_minus_B:.4f} "
            f"(95% bootstrap interval "
            f"{row.difference_ci_lower:.4f}, "
            f"{row.difference_ci_upper:.4f}); "
            f"Cliff's delta="
            f"{row.cliffs_delta_A_vs_B:.3f}; "
            f"permutation p="
            f"{row.exploratory_permutation_p_two_sided:.4g}"
        )
    print()
    print(
        "Final powered colocalization unit classes:"
    )
    for classification in COLOC_CLASSES:
        print(
            f"  {classification}: "
            f"{coloc_class_counts.get(classification, 0)}"
        )
    print()
    print(
        "Summary:",
        paths["summary_json"],
    )
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
