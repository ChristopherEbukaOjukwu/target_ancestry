#!/usr/bin/env python3
"""
Step 15D — Estimate cross-ancestry effect portability.

For every comparison that passed Step 15C3 ancestry-paired LD pruning,
this script estimates the through-origin slope of the comparison-
ancestry marginal effects on the EUR marginal effects.

Three estimators are reported without assigning one a privileged
interpretation:

1. naive
   Through-origin ordinary least-squares slope:
       sum(beta_EUR * beta_comparison) / sum(beta_EUR^2)

2. fiqt
   The same slope after FDR Inverse Quantile Transformation (FIQT)
   winner's-curse correction of the EUR effects. EUR alone is corrected
   because variants were selected on EUR genome-wide significance.

3. deming
   Weighted through-origin errors-in-variables fit minimizing:
       sum[(beta_comparison - b * beta_EUR)^2
           / (se_comparison^2 + b^2 * se_EUR^2)]

Comparison-level 95% confidence intervals for the primary r2 < 0.10
universe use a seeded percentile bootstrap over the variants retained
after ancestry-paired LD pruning. This treats the pruning design as the
LD control; it does not claim to model the complete signed LD
covariance matrix.

For each gene-trait-source unit, comparison-ancestry slopes are
aggregated by their median. Headline confidence intervals bootstrap
gene-trait-source units, not target-indication pairs.

The script is resumable. One JSON cache is written per comparison under
intermediate/15d/comparisons. Re-running with the same settings reuses
valid caches.

Inputs
------
output/15c3_ld_pruning_summary.json
output/15c3_variant_pruning_decisions.parquet
output/15c3_comparison_pruning_results.parquet
output/15c1_portability_comparison_manifest.parquet
output/15b5_unit_analysis_universe.parquet
intermediate/15b4/comparisons/<comparison_uid>.parquet

Outputs
-------
intermediate/15d/comparisons/<comparison_uid>.json
output/15d_variant_effects_primary.parquet
output/15d_comparison_portability_estimates.parquet
output/15d_unit_portability_estimates.parquet
output/15d_primary_comparison_portability.parquet
output/15d_primary_unit_portability.parquet
output/15d_portability_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm


SCRIPT_VERSION = "15D.2"
ESTIMATORS = ("naive", "fiqt", "deming")
DEFAULT_THRESHOLDS = (0.01, 0.10, 0.20)
PRIMARY_THRESHOLD = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pruning-summary",
        type=Path,
        default=Path(
            "output/15c3_ld_pruning_summary.json"
        ),
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=Path(
            "output/15c3_variant_pruning_decisions.parquet"
        ),
    )
    parser.add_argument(
        "--comparison-results",
        type=Path,
        default=Path(
            "output/15c3_comparison_pruning_results.parquet"
        ),
    )
    parser.add_argument(
        "--comparison-manifest",
        type=Path,
        default=Path(
            "output/"
            "15c1_portability_comparison_manifest.parquet"
        ),
    )
    parser.add_argument(
        "--unit-universe",
        type=Path,
        default=Path(
            "output/15b5_unit_analysis_universe.parquet"
        ),
    )
    parser.add_argument(
        "--harmonized-dir",
        type=Path,
        default=Path(
            "intermediate/15b4/comparisons"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            "intermediate/15d/comparisons"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
    )
    parser.add_argument(
        "--thresholds",
        default="0.01,0.10,0.20",
    )
    parser.add_argument(
        "--primary-threshold",
        type=float,
        default=PRIMARY_THRESHOLD,
    )
    parser.add_argument(
        "--minimum-variants",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--comparison-bootstrap",
        type=int,
        default=1000,
        help=(
            "Variant-bootstrap replicates for primary comparison "
            "confidence intervals."
        ),
    )
    parser.add_argument(
        "--unit-bootstrap",
        type=int,
        default=5000,
        help=(
            "Unit-bootstrap replicates for headline median "
            "confidence intervals."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--max-comparisons",
        type=int,
        default=0,
        help=(
            "Process at most this many uncached comparisons. "
            "Zero processes all pending comparisons."
        ),
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
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


def parse_thresholds(text: str) -> tuple[float, ...]:
    values = tuple(
        sorted(
            {
                float(token.strip())
                for token in text.split(",")
                if token.strip()
            }
        )
    )
    if not values:
        raise SystemExit(
            "At least one LD threshold is required."
        )
    for value in values:
        if not 0.0 < value < 1.0:
            raise SystemExit(
                f"Invalid LD threshold: {value}"
            )
    return values


def threshold_key(value: float) -> str:
    return f"{value:.8g}"


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(
            f"{label} missing required columns: {missing}"
        )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def json_default(value: Any) -> Any:
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
        f"Cannot JSON-serialize {type(value).__name__}"
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


def stable_seed(
    base_seed: int,
    text: str,
) -> int:
    digest = hashlib.sha256(
        f"{base_seed}|{text}".encode("utf-8")
    ).digest()
    return int.from_bytes(
        digest[:8],
        "little",
        signed=False,
    ) % (2**32 - 1)


def finite_numeric(
    series: pd.Series,
) -> np.ndarray:
    return pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(dtype=float)


def naive_slope(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    denominator = float(
        np.dot(x, x)
    )
    if (
        not math.isfinite(denominator)
        or denominator <= 0.0
    ):
        return math.nan
    numerator = float(
        np.dot(x, y)
    )
    return numerator / denominator


def fiqt_correct(
    beta: np.ndarray,
    standard_error: np.ndarray,
) -> np.ndarray:
    """
    FDR Inverse Quantile Transformation.

    Applies Benjamini-Hochberg adjusted two-sided p-values to the
    observed z scores and transforms the adjusted probabilities back
    to signed normal quantiles.
    """
    beta = np.asarray(
        beta,
        dtype=float,
    )
    standard_error = np.asarray(
        standard_error,
        dtype=float,
    )
    if len(beta) == 0:
        return beta.copy()

    z = beta / standard_error
    p = 2.0 * norm.sf(
        np.abs(z)
    )
    p = np.clip(
        p,
        np.nextafter(0.0, 1.0),
        1.0,
    )

    count = len(p)
    order = np.argsort(
        p,
        kind="mergesort",
    )
    ranked_p = p[order]
    raw_adjusted = (
        ranked_p
        * count
        / np.arange(
            1,
            count + 1,
            dtype=float,
        )
    )
    monotone_adjusted = np.minimum.accumulate(
        raw_adjusted[::-1]
    )[::-1]
    monotone_adjusted = np.clip(
        monotone_adjusted,
        np.nextafter(0.0, 1.0),
        1.0,
    )

    adjusted_p = np.empty(
        count,
        dtype=float,
    )
    adjusted_p[order] = monotone_adjusted

    corrected_z = norm.isf(
        adjusted_p / 2.0
    ) * np.sign(z)

    invalid = ~np.isfinite(
        corrected_z
    )
    corrected_z[invalid] = z[invalid]

    return corrected_z * standard_error


def deming_objective(
    slope: float,
    x: np.ndarray,
    y: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
) -> float:
    denominator = (
        np.square(sy)
        + np.square(slope)
        * np.square(sx)
    )
    if (
        not np.isfinite(
            denominator
        ).all()
        or np.any(
            denominator <= 0.0
        )
    ):
        return math.inf

    residual = y - slope * x
    value = float(
        np.sum(
            np.square(residual)
            / denominator
        )
    )
    return (
        value
        if math.isfinite(value)
        else math.inf
    )


def deming_slope(
    x: np.ndarray,
    y: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
) -> dict[str, Any]:
    ratios = np.divide(
        y,
        x,
        out=np.full_like(
            y,
            np.nan,
            dtype=float,
        ),
        where=np.abs(x) > 1e-15,
    )
    finite_ratios = np.abs(
        ratios[
            np.isfinite(ratios)
        ]
    )
    naive = naive_slope(x, y)

    scale_candidates = [2.0]
    if math.isfinite(naive):
        scale_candidates.append(
            5.0 * abs(naive)
        )
    if finite_ratios.size:
        scale_candidates.append(
            2.0
            * float(
                np.quantile(
                    finite_ratios,
                    0.95,
                )
            )
        )

    bound = min(
        100.0,
        max(scale_candidates),
    )
    result = None

    for _ in range(4):
        result = minimize_scalar(
            deming_objective,
            args=(x, y, sx, sy),
            bounds=(-bound, bound),
            method="bounded",
            options={
                "xatol": 1e-9,
                "maxiter": 500,
            },
        )
        slope = float(result.x)
        near_boundary = (
            abs(slope) >= 0.995 * bound
        )
        if (
            result.success
            and math.isfinite(slope)
            and not near_boundary
        ):
            break
        if bound >= 100.0:
            break
        bound = min(
            100.0,
            bound * 2.0,
        )

    assert result is not None
    slope = float(result.x)
    near_boundary = bool(
        abs(slope) >= 0.995 * bound
    )

    return {
        "slope": (
            slope
            if math.isfinite(slope)
            else math.nan
        ),
        "success": bool(
            result.success
            and math.isfinite(slope)
        ),
        "near_boundary": (
            near_boundary
        ),
        "bound": float(bound),
        "objective": (
            float(result.fun)
            if math.isfinite(
                float(result.fun)
            )
            else None
        ),
        "message": clean(
            result.message
        ),
    }


def estimate_all(
    x: np.ndarray,
    y: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
) -> dict[str, Any]:
    x_fiqt = fiqt_correct(
        x,
        sx,
    )
    deming = deming_slope(
        x,
        y,
        sx,
        sy,
    )
    return {
        "naive": naive_slope(
            x,
            y,
        ),
        "fiqt": naive_slope(
            x_fiqt,
            y,
        ),
        "deming": deming["slope"],
        "deming_success": (
            deming["success"]
        ),
        "deming_near_boundary": (
            deming["near_boundary"]
        ),
        "deming_bound": (
            deming["bound"]
        ),
        "deming_objective": (
            deming["objective"]
        ),
        "eur_beta_fiqt": x_fiqt,
    }


def percentile_ci(
    values: list[float],
) -> tuple[
    float | None,
    float | None,
    int,
]:
    finite = np.asarray(
        [
            value
            for value in values
            if math.isfinite(value)
        ],
        dtype=float,
    )
    if finite.size < 20:
        return None, None, int(
            finite.size
        )
    lower, upper = np.percentile(
        finite,
        [2.5, 97.5],
    )
    return (
        float(lower),
        float(upper),
        int(finite.size),
    )


def bootstrap_comparison(
    *,
    x: np.ndarray,
    y: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if replicates <= 0:
        return {
            estimator: {
                "ci_lower": None,
                "ci_upper": None,
                "valid_replicates": 0,
                "requested_replicates": 0,
            }
            for estimator in ESTIMATORS
        }

    rng = np.random.default_rng(seed)
    count = len(x)
    estimates = {
        estimator: []
        for estimator in ESTIMATORS
    }

    for _ in range(replicates):
        indices = rng.integers(
            0,
            count,
            size=count,
        )
        xb = x[indices]
        yb = y[indices]
        sxb = sx[indices]
        syb = sy[indices]

        naive = naive_slope(
            xb,
            yb,
        )
        fiqt = naive_slope(
            fiqt_correct(
                xb,
                sxb,
            ),
            yb,
        )
        deming = deming_slope(
            xb,
            yb,
            sxb,
            syb,
        )["slope"]

        estimates["naive"].append(
            naive
        )
        estimates["fiqt"].append(
            fiqt
        )
        estimates["deming"].append(
            deming
        )

    output = {}
    for estimator in ESTIMATORS:
        lower, upper, valid = (
            percentile_ci(
                estimates[estimator]
            )
        )
        output[estimator] = {
            "ci_lower": lower,
            "ci_upper": upper,
            "valid_replicates": valid,
            "requested_replicates": int(
                replicates
            ),
        }
    return output


def descriptive_metrics(
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    if len(x) >= 2:
        x_sd = float(
            np.std(x, ddof=1)
        )
        y_sd = float(
            np.std(y, ddof=1)
        )
        pearson = (
            float(
                np.corrcoef(
                    x,
                    y,
                )[0, 1]
            )
            if x_sd > 0.0
            and y_sd > 0.0
            else math.nan
        )
    else:
        pearson = math.nan

    sign_concordance = float(
        np.mean(
            np.sign(x)
            == np.sign(y)
        )
    )
    lead_index = int(
        np.argmax(
            np.abs(x)
        )
    )
    lead_sign_concordant = bool(
        np.sign(x[lead_index])
        == np.sign(y[lead_index])
    )

    return {
        "pearson_r": pearson,
        "sign_concordance": (
            sign_concordance
        ),
        "lead_variant_index": (
            lead_index
        ),
        "lead_sign_concordant": (
            lead_sign_concordant
        ),
        "median_abs_eur_z": float(
            np.median(
                np.abs(x / np.ones_like(x))
            )
        ),
    }


def cache_path(
    cache_dir: Path,
    comparison_uid: str,
) -> Path:
    safe = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        comparison_uid,
    )
    return cache_dir / f"{safe}.json"


def expected_variant_map(
    retained: pd.DataFrame,
    comparison_uid: str,
    thresholds: tuple[float, ...],
) -> dict[str, list[str]]:
    subset = retained[
        retained[
            "comparison_uid"
        ].eq(comparison_uid)
    ]
    result = {}
    for threshold in thresholds:
        selected = subset[
            np.isclose(
                subset[
                    "r2_threshold"
                ].astype(float),
                threshold,
                rtol=0.0,
                atol=1e-12,
            )
        ].sort_values(
            [
                "eur_p_rank",
                "request_order",
                "variant_id",
            ],
            kind="mergesort",
        )
        if not selected.empty:
            result[
                threshold_key(threshold)
            ] = selected[
                "variant_id"
            ].astype(str).tolist()
    return result


def cache_valid(
    path: Path,
    *,
    comparison_uid: str,
    variant_map: dict[
        str,
        list[str],
    ],
    comparison_bootstrap: int,
    seed: int,
) -> bool:
    if not path.exists():
        return False
    try:
        data = read_json(path)
    except Exception:
        return False

    return (
        clean(
            data.get("script_version")
        )
        == SCRIPT_VERSION
        and clean(
            data.get("comparison_uid")
        )
        == comparison_uid
        and data.get(
            "variant_ids_by_threshold"
        )
        == variant_map
        and int(
            data.get(
                "comparison_bootstrap",
                -1,
            )
        )
        == comparison_bootstrap
        and int(
            data.get(
                "seed",
                -1,
            )
        )
        == seed
        and clean(
            data.get("status")
        )
        == "PASS"
    )


def metadata_value(
    row: pd.Series,
    column: str,
    default: Any = None,
) -> Any:
    return (
        row[column]
        if column in row.index
        else default
    )


def load_effect_rows(
    harmonized_path: Path,
    retained_rows: pd.DataFrame,
) -> pd.DataFrame:
    harmonized = pd.read_parquet(
        harmonized_path
    )
    require_columns(
        harmonized,
        {
            "variant_id",
            "eur_beta",
            "eur_se",
            "comparison_beta",
            "comparison_se",
        },
        str(harmonized_path),
    )

    if harmonized[
        "variant_id"
    ].duplicated().any():
        duplicates = harmonized.loc[
            harmonized[
                "variant_id"
            ].duplicated(
                keep=False
            ),
            "variant_id",
        ].astype(str).tolist()
        raise RuntimeError(
            "Harmonized comparison contains "
            "duplicate variant IDs: "
            f"{duplicates[:10]}"
        )

    columns = [
        "variant_id",
        "eur_beta",
        "eur_se",
        "comparison_beta",
        "comparison_se",
    ]
    optional = [
        "eur_p",
        "comparison_p",
        "eur_af",
        "comparison_af",
        "eur_maf",
        "comparison_maf",
        "qc_qualified",
        "eur_gws",
        "allele_relation",
    ]
    columns.extend(
        [
            column
            for column in optional
            if column in harmonized.columns
        ]
    )

    merged = retained_rows.merge(
        harmonized[columns],
        on="variant_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing = merged[
        ~merged["_merge"].eq("both")
    ]
    if len(missing):
        raise RuntimeError(
            "Retained variants absent from "
            "harmonized comparison: "
            + ", ".join(
                missing[
                    "variant_id"
                ].astype(str).tolist()[:20]
            )
        )
    return merged.drop(
        columns="_merge"
    )


def validate_effect_arrays(
    frame: pd.DataFrame,
    minimum_variants: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    x = finite_numeric(
        frame["eur_beta"]
    )
    sx = finite_numeric(
        frame["eur_se"]
    )
    y = finite_numeric(
        frame["comparison_beta"]
    )
    sy = finite_numeric(
        frame["comparison_se"]
    )

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(sx)
        & np.isfinite(sy)
        & (sx > 0.0)
        & (sy > 0.0)
    )
    if not valid.all():
        invalid_ids = frame.loc[
            ~valid,
            "variant_id",
        ].astype(str).tolist()
        raise RuntimeError(
            "Retained variants have invalid beta/SE: "
            f"{invalid_ids[:20]}"
        )
    if len(x) < minimum_variants:
        raise RuntimeError(
            f"Only {len(x)} retained variants; "
            f"minimum={minimum_variants}."
        )
    return x, y, sx, sy


def process_comparison(
    *,
    comparison_row: pd.Series,
    retained: pd.DataFrame,
    thresholds: tuple[float, ...],
    primary_threshold: float,
    minimum_variants: int,
    comparison_bootstrap: int,
    base_seed: int,
    harmonized_dir: Path,
) -> dict[str, Any]:
    comparison_uid = clean(
        comparison_row[
            "comparison_uid"
        ]
    )
    harmonized_path = (
        harmonized_dir
        / f"{comparison_uid}.parquet"
    )
    if not harmonized_path.exists():
        raise RuntimeError(
            "Missing harmonized comparison: "
            f"{harmonized_path}"
        )

    variant_map = expected_variant_map(
        retained,
        comparison_uid,
        thresholds,
    )
    estimates = []
    primary_variant_effects = []

    for threshold in thresholds:
        key = threshold_key(
            threshold
        )
        variant_ids = variant_map.get(
            key,
            [],
        )
        if not variant_ids:
            continue

        retained_rows = retained[
            retained[
                "comparison_uid"
            ].eq(comparison_uid)
            & np.isclose(
                retained[
                    "r2_threshold"
                ].astype(float),
                threshold,
                rtol=0.0,
                atol=1e-12,
            )
        ].sort_values(
            [
                "eur_p_rank",
                "request_order",
                "variant_id",
            ],
            kind="mergesort",
        ).copy()

        effects = load_effect_rows(
            harmonized_path,
            retained_rows,
        )
        x, y, sx, sy = (
            validate_effect_arrays(
                effects,
                minimum_variants,
            )
        )

        point = estimate_all(
            x,
            y,
            sx,
            sy,
        )
        descriptive = (
            descriptive_metrics(
                x,
                y,
            )
        )
        median_abs_z = float(
            np.median(
                np.abs(x / sx)
            )
        )

        is_primary = math.isclose(
            threshold,
            primary_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        bootstrap = (
            bootstrap_comparison(
                x=x,
                y=y,
                sx=sx,
                sy=sy,
                replicates=(
                    comparison_bootstrap
                ),
                seed=stable_seed(
                    base_seed,
                    (
                        f"{comparison_uid}|"
                        f"{threshold_key(threshold)}"
                    ),
                ),
            )
            if is_primary
            else {
                estimator: {
                    "ci_lower": None,
                    "ci_upper": None,
                    "valid_replicates": 0,
                    "requested_replicates": 0,
                }
                for estimator in ESTIMATORS
            }
        )

        base = {
            "comparison_uid": (
                comparison_uid
            ),
            "gene_trait_uid": clean(
                comparison_row[
                    "gene_trait_uid"
                ]
            ),
            "gene": clean(
                comparison_row["gene"]
            ),
            "candidate_source": clean(
                comparison_row[
                    "candidate_source"
                ]
            ),
            "candidate_trait_key": clean(
                metadata_value(
                    comparison_row,
                    "candidate_trait_key",
                    "",
                )
            ),
            "candidate_trait_name": clean(
                comparison_row[
                    "candidate_trait_name"
                ]
            ),
            "comparison_population": clean(
                comparison_row[
                    "comparison_population"
                ]
            ),
            "comparison_reference_population": clean(
                metadata_value(
                    comparison_row,
                    (
                        "comparison_reference_"
                        "population"
                    ),
                    comparison_row[
                        "comparison_population"
                    ],
                )
            ),
            "genome_build": clean(
                metadata_value(
                    comparison_row,
                    "genome_build",
                    "",
                )
            ),
            "unit_analysis_role": clean(
                metadata_value(
                    comparison_row,
                    "unit_analysis_role",
                    "",
                )
            ),
            "mixed_pool": as_bool(
                metadata_value(
                    comparison_row,
                    "mixed_pool",
                    False,
                )
            ),
            "primary_approval_eligible": (
                as_bool(
                    metadata_value(
                        comparison_row,
                        (
                            "primary_approval_"
                            "eligible"
                        ),
                        True,
                    )
                )
            ),
            "r2_threshold": float(
                threshold
            ),
            "is_primary_threshold": (
                is_primary
            ),
            "n_variants": int(
                len(effects)
            ),
            "median_abs_eur_z": (
                median_abs_z
            ),
            "pearson_r": (
                descriptive[
                    "pearson_r"
                ]
            ),
            "sign_concordance": (
                descriptive[
                    "sign_concordance"
                ]
            ),
            "lead_sign_concordant": (
                descriptive[
                    "lead_sign_concordant"
                ]
            ),
            "deming_success": (
                point[
                    "deming_success"
                ]
            ),
            "deming_near_boundary": (
                point[
                    "deming_near_boundary"
                ]
            ),
            "deming_bound": (
                point[
                    "deming_bound"
                ]
            ),
            "deming_objective": (
                point[
                    "deming_objective"
                ]
            ),
            "ci_method": (
                "percentile bootstrap over "
                "ancestry-paired LD-pruned "
                "variants"
                if is_primary
                else "not calculated"
            ),
        }

        for estimator in ESTIMATORS:
            estimates.append(
                {
                    **base,
                    "estimator": estimator,
                    "slope": float(
                        point[estimator]
                    ),
                    "ci_lower": (
                        bootstrap[
                            estimator
                        ]["ci_lower"]
                    ),
                    "ci_upper": (
                        bootstrap[
                            estimator
                        ]["ci_upper"]
                    ),
                    "bootstrap_valid_replicates": int(
                        bootstrap[
                            estimator
                        ][
                            "valid_replicates"
                        ]
                    ),
                    "bootstrap_requested_replicates": int(
                        bootstrap[
                            estimator
                        ][
                            "requested_replicates"
                        ]
                    ),
                }
            )

        if is_primary:
            effects = effects.copy()
            effects[
                "eur_beta_fiqt"
            ] = point[
                "eur_beta_fiqt"
            ]
            effects[
                "comparison_uid"
            ] = comparison_uid
            effects[
                "gene_trait_uid"
            ] = clean(
                comparison_row[
                    "gene_trait_uid"
                ]
            )
            effects["gene"] = clean(
                comparison_row["gene"]
            )
            effects[
                "candidate_source"
            ] = clean(
                comparison_row[
                    "candidate_source"
                ]
            )
            effects[
                "candidate_trait_name"
            ] = clean(
                comparison_row[
                    "candidate_trait_name"
                ]
            )
            effects[
                "comparison_population"
            ] = clean(
                comparison_row[
                    "comparison_population"
                ]
            )
            effects[
                "r2_threshold"
            ] = float(threshold)
            primary_variant_effects.extend(
                effects.to_dict(
                    "records"
                )
            )

    return {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "comparison_uid": comparison_uid,
        "comparison_bootstrap": int(
            comparison_bootstrap
        ),
        "seed": int(base_seed),
        "variant_ids_by_threshold": (
            variant_map
        ),
        "estimates": estimates,
        "primary_variant_effects": (
            primary_variant_effects
        ),
        "completed_at_utc": utc_now(),
    }


def median_ci_units(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(
        values,
        dtype=float,
    )
    values = values[
        np.isfinite(values)
    ]
    if values.size == 0:
        return {
            "median": None,
            "ci_lower": None,
            "ci_upper": None,
            "n_units": 0,
            "bootstrap_replicates": int(
                replicates
            ),
        }

    median = float(
        np.median(values)
    )
    if replicates <= 0:
        return {
            "median": median,
            "ci_lower": None,
            "ci_upper": None,
            "n_units": int(
                values.size
            ),
            "bootstrap_replicates": 0,
        }

    rng = np.random.default_rng(
        seed
    )
    indices = rng.integers(
        0,
        values.size,
        size=(
            replicates,
            values.size,
        ),
    )
    bootstrap_medians = np.median(
        values[indices],
        axis=1,
    )
    lower, upper = np.percentile(
        bootstrap_medians,
        [2.5, 97.5],
    )
    return {
        "median": median,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "n_units": int(
            values.size
        ),
        "bootstrap_replicates": int(
            replicates
        ),
    }


def records_by_group(
    frame: pd.DataFrame,
    group_column: str,
    *,
    value_column: str,
) -> dict[str, Any]:
    output = {}
    for group_value, group in frame.groupby(
        group_column,
        dropna=False,
        sort=True,
    ):
        values = pd.to_numeric(
            group[value_column],
            errors="coerce",
        )
        values = values[
            np.isfinite(values)
        ]
        output[clean(group_value)] = {
            "n": int(len(values)),
            "median": (
                float(
                    np.median(values)
                )
                if len(values)
                else None
            ),
            "mean": (
                float(
                    np.mean(values)
                )
                if len(values)
                else None
            ),
        }
    return output


def aggregate_unit_estimates(
    comparison_estimates: pd.DataFrame,
    unit_universe: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    group_columns = [
        "gene_trait_uid",
        "r2_threshold",
        "estimator",
    ]
    for keys, group in comparison_estimates.groupby(
        group_columns,
        sort=True,
        dropna=False,
    ):
        (
            gene_trait_uid,
            threshold,
            estimator,
        ) = keys

        invariant_columns = [
            "gene",
            "candidate_source",
            "candidate_trait_key",
            "candidate_trait_name",
            "unit_analysis_role",
            "mixed_pool",
            "primary_approval_eligible",
        ]
        row = {
            "gene_trait_uid": (
                gene_trait_uid
            ),
            "r2_threshold": float(
                threshold
            ),
            "estimator": estimator,
            "n_comparison_ancestries": int(
                len(group)
            ),
            "comparison_ancestries": (
                " || ".join(
                    sorted(
                        set(
                            group[
                                "comparison_population"
                            ].astype(str)
                        )
                    )
                )
            ),
            "n_variants_total": int(
                group["n_variants"].sum()
            ),
            "n_variants_minimum": int(
                group["n_variants"].min()
            ),
            "n_variants_median": float(
                group["n_variants"].median()
            ),
            "slope": float(
                group["slope"].median()
            ),
            "slope_mean_across_ancestries": float(
                group["slope"].mean()
            ),
            "sign_concordance": float(
                group[
                    "sign_concordance"
                ].median()
            ),
            "pearson_r": float(
                group["pearson_r"].median()
            )
            if group[
                "pearson_r"
            ].notna().any()
            else math.nan,
            "any_deming_near_boundary": bool(
                group[
                    "deming_near_boundary"
                ].fillna(False).any()
            ),
        }
        for column in invariant_columns:
            values = group[
                column
            ].drop_duplicates()
            if len(values) != 1:
                raise RuntimeError(
                    f"{column} varies within "
                    f"{gene_trait_uid}: "
                    f"{values.tolist()}"
                )
            row[column] = values.iloc[0]
        rows.append(row)

    unit_estimates = pd.DataFrame(
        rows
    )

    optional_metadata = [
        "gene_trait_uid",
        "genome_build",
        "canonical_group",
        "locked_mapping_tier",
        "mapping_tiers_contributing",
        "sensitivity_pool_assignment",
        "pools",
        "n_A_pairs",
        "n_B_pairs",
        "n_target_indication_pairs",
        "n_indications",
        "indications",
        "non_eur_pops_available",
    ]
    available = [
        column
        for column in optional_metadata
        if column in unit_universe.columns
    ]
    if "gene_trait_uid" in available:
        metadata = unit_universe[
            available
        ].drop_duplicates(
            "gene_trait_uid"
        )
        unit_estimates = (
            unit_estimates.merge(
                metadata,
                on="gene_trait_uid",
                how="left",
                validate="many_to_one",
            )
        )

    return unit_estimates.sort_values(
        [
            "r2_threshold",
            "estimator",
            "candidate_source",
            "candidate_trait_name",
            "gene",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def build_summary(
    *,
    comparison_estimates: pd.DataFrame,
    unit_estimates: pd.DataFrame,
    thresholds: tuple[float, ...],
    primary_threshold: float,
    unit_bootstrap: int,
    seed: int,
    processing_errors: list[
        dict[str, Any]
    ],
    expected_comparisons: int,
) -> dict[str, Any]:
    threshold_results = {}

    for threshold in thresholds:
        unit_threshold = unit_estimates[
            np.isclose(
                unit_estimates[
                    "r2_threshold"
                ].astype(float),
                threshold,
                rtol=0.0,
                atol=1e-12,
            )
        ]
        comparison_threshold = (
            comparison_estimates[
                np.isclose(
                    comparison_estimates[
                        "r2_threshold"
                    ].astype(float),
                    threshold,
                    rtol=0.0,
                    atol=1e-12,
                )
            ]
        )

        estimator_results = {}
        for estimator in ESTIMATORS:
            units = unit_threshold[
                unit_threshold[
                    "estimator"
                ].eq(estimator)
            ]
            comparisons = (
                comparison_threshold[
                    comparison_threshold[
                        "estimator"
                    ].eq(estimator)
                ]
            )
            values = pd.to_numeric(
                units["slope"],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )
            headline = median_ci_units(
                values,
                replicates=unit_bootstrap,
                seed=stable_seed(
                    seed,
                    (
                        f"headline|{threshold}|"
                        f"{estimator}"
                    ),
                ),
            )
            estimator_results[
                estimator
            ] = {
                **headline,
                "n_comparisons": int(
                    len(comparisons)
                ),
                "by_comparison_ancestry": (
                    records_by_group(
                        comparisons,
                        "comparison_population",
                        value_column="slope",
                    )
                ),
                "by_source": (
                    records_by_group(
                        units,
                        "candidate_source",
                        value_column="slope",
                    )
                ),
                "by_pool": (
                    records_by_group(
                        units,
                        (
                            "sensitivity_pool_"
                            "assignment"
                        ),
                        value_column="slope",
                    )
                    if (
                        "sensitivity_pool_assignment"
                        in units.columns
                    )
                    else {}
                ),
            }

        threshold_results[
            threshold_key(threshold)
        ] = {
            "n_comparisons": int(
                comparison_threshold[
                    "comparison_uid"
                ].nunique()
            ),
            "n_units": int(
                unit_threshold[
                    "gene_trait_uid"
                ].nunique()
            ),
            "n_retained_variants": int(
                comparison_threshold[
                    comparison_threshold[
                        "estimator"
                    ].eq("naive")
                ][
                    "n_variants"
                ].sum()
            ),
            "estimators": (
                estimator_results
            ),
        }

    primary_units = unit_estimates[
        np.isclose(
            unit_estimates[
                "r2_threshold"
            ].astype(float),
            primary_threshold,
            rtol=0.0,
            atol=1e-12,
        )
    ]
    estimator_wide = (
        primary_units.pivot(
            index="gene_trait_uid",
            columns="estimator",
            values="slope",
        )
    )
    agreement = {}
    for first, second in [
        ("naive", "fiqt"),
        ("naive", "deming"),
        ("fiqt", "deming"),
    ]:
        if (
            first in estimator_wide.columns
            and second
            in estimator_wide.columns
        ):
            subset = estimator_wide[
                [first, second]
            ].dropna()
            agreement[
                f"{first}_vs_{second}"
            ] = {
                "n_units": int(
                    len(subset)
                ),
                "pearson_r": (
                    float(
                        subset.corr(
                            method="pearson"
                        ).iloc[0, 1]
                    )
                    if len(subset) >= 2
                    else None
                ),
                "median_difference": (
                    float(
                        np.median(
                            subset[second]
                            - subset[first]
                        )
                    )
                    if len(subset)
                    else None
                ),
            }

    completed_comparisons = int(
        comparison_estimates[
            "comparison_uid"
        ].nunique()
    )
    status = (
        "PASS"
        if (
            not processing_errors
            and completed_comparisons
            == expected_comparisons
        )
        else (
            "INCOMPLETE"
            if completed_comparisons
            < expected_comparisons
            else "FAIL"
        )
    )

    return {
        "step": "15D",
        "script_version": SCRIPT_VERSION,
        "status": status,
        "estimator_definitions": {
            "naive": (
                "Through-origin OLS slope of "
                "comparison beta on EUR beta."
            ),
            "fiqt": (
                "Through-origin slope after FIQT "
                "correction of EUR beta only; "
                "selection was on EUR significance."
            ),
            "deming": (
                "Weighted through-origin "
                "errors-in-variables fit using "
                "both EUR and comparison standard "
                "errors."
            ),
        },
        "uncertainty": {
            "comparison_level": (
                "Seeded percentile bootstrap over "
                "variants retained after ancestry-"
                "paired LD pruning; primary "
                "r2 < 0.10 comparisons only."
            ),
            "headline_level": (
                "Seeded percentile bootstrap over "
                "gene-trait-source units."
            ),
            "ld_limitation": (
                "Signed LD covariance is not modeled "
                "after pruning; uncertainty is LD-"
                "controlled by the ancestry-paired "
                "pruning design rather than by a "
                "full covariance likelihood."
            ),
        },
        "primary_threshold": float(
            primary_threshold
        ),
        "expected_comparisons_any_threshold": int(
            expected_comparisons
        ),
        "completed_comparisons_any_threshold": int(
            completed_comparisons
        ),
        "processing_errors": (
            processing_errors
        ),
        "threshold_results": (
            threshold_results
        ),
        "primary_estimator_agreement": (
            agreement
        ),
        "completed_at_utc": utc_now(),
    }


def main() -> int:
    args = parse_args()
    thresholds = parse_thresholds(
        args.thresholds
    )
    if not any(
        math.isclose(
            threshold,
            args.primary_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for threshold in thresholds
    ):
        raise SystemExit(
            "--primary-threshold must be "
            "included in --thresholds."
        )
    if args.minimum_variants < 3:
        raise SystemExit(
            "--minimum-variants must be at least 3."
        )
    if args.comparison_bootstrap < 0:
        raise SystemExit(
            "--comparison-bootstrap cannot be negative."
        )
    if args.unit_bootstrap < 0:
        raise SystemExit(
            "--unit-bootstrap cannot be negative."
        )

    required_paths = [
        args.pruning_summary,
        args.decisions,
        args.comparison_results,
        args.comparison_manifest,
        args.unit_universe,
    ]
    for path in required_paths:
        if not path.exists():
            raise SystemExit(
                f"Missing required input: {path}"
            )

    pruning_summary = read_json(
        args.pruning_summary
    )
    if clean(
        pruning_summary.get("status")
    ) not in {
        "PASS",
        "PASS_WITH_LD_EXCLUSIONS",
    }:
        raise SystemExit(
            "Step 15C3 summary is not valid: "
            f"{pruning_summary.get('status')}"
        )
    if pruning_summary.get(
        "processing_errors"
    ):
        raise SystemExit(
            "Step 15C3 still contains processing errors."
        )
    if pruning_summary.get(
        "missing_or_stale_caches"
    ):
        raise SystemExit(
            "Step 15C3 still contains missing/stale caches."
        )

    decisions = pd.read_parquet(
        args.decisions
    )
    comparison_results = (
        pd.read_parquet(
            args.comparison_results
        )
    )
    comparison_manifest = (
        pd.read_parquet(
            args.comparison_manifest
        )
    )
    unit_universe = pd.read_parquet(
        args.unit_universe
    )

    require_columns(
        decisions,
        {
            "comparison_uid",
            "gene_trait_uid",
            "gene",
            "candidate_source",
            "candidate_trait_name",
            "comparison_population",
            "variant_id",
            "eur_p",
            "eur_p_rank",
            "request_order",
            "r2_threshold",
            "retained",
        },
        "Step 15C3 pruning decisions",
    )
    require_columns(
        comparison_results,
        {
            "comparison_uid",
            "r2_threshold",
            "portability_eligible",
            "n_retained",
        },
        "Step 15C3 comparison results",
    )
    require_columns(
        comparison_manifest,
        {
            "comparison_uid",
            "gene_trait_uid",
            "gene",
            "candidate_source",
            "candidate_trait_name",
            "comparison_population",
        },
        "Step 15C1 comparison manifest",
    )
    require_columns(
        unit_universe,
        {
            "gene_trait_uid",
            "gene",
            "candidate_source",
            "candidate_trait_name",
        },
        "Step 15B5 unit universe",
    )

    requested_threshold = comparison_results[
        "r2_threshold"
    ].astype(float).map(
        lambda value: any(
            math.isclose(
                value,
                threshold,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for threshold in thresholds
        )
    )
    eligible = comparison_results[
        comparison_results[
            "portability_eligible"
        ].map(as_bool)
        & requested_threshold
    ].copy()

    # Eligibility belongs to a comparison-threshold pair. A comparison
    # can retain only one or two variants at r2 < 0.01, fail that
    # universe, and still pass r2 < 0.10 or r2 < 0.20. Therefore,
    # retained variants must be restricted to exact eligible pairs,
    # not to every threshold for any comparison that passes somewhere.
    eligible["_threshold_key"] = (
        eligible["r2_threshold"]
        .astype(float)
        .round(12)
    )
    eligible_pairs = eligible[
        [
            "comparison_uid",
            "_threshold_key",
            "n_retained",
        ]
    ].drop_duplicates(
        [
            "comparison_uid",
            "_threshold_key",
        ]
    )

    retained_candidates = decisions[
        decisions["retained"].map(
            as_bool
        )
        & decisions[
            "r2_threshold"
        ].astype(float).map(
            lambda value: any(
                math.isclose(
                    value,
                    threshold,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for threshold in thresholds
            )
        )
    ].copy()
    retained_candidates["_threshold_key"] = (
        retained_candidates[
            "r2_threshold"
        ].astype(float).round(12)
    )
    retained = retained_candidates.merge(
        eligible_pairs[
            [
                "comparison_uid",
                "_threshold_key",
            ]
        ],
        on=[
            "comparison_uid",
            "_threshold_key",
        ],
        how="inner",
        validate="many_to_one",
    ).drop(
        columns="_threshold_key"
    )

    eligible_uids = sorted(
        set(
            eligible[
                "comparison_uid"
            ].astype(str)
        )
    )

    eligible_counts = (
        eligible_pairs.set_index(
            [
                "comparison_uid",
                "_threshold_key",
            ]
        )["n_retained"]
        .rename("expected")
    )
    observed_counts = (
        retained.assign(
            _threshold_key=retained[
                "r2_threshold"
            ].astype(float).round(12)
        )
        .groupby(
            [
                "comparison_uid",
                "_threshold_key",
            ]
        )
        .size()
        .rename("observed")
    )
    joined_counts = pd.concat(
        [
            eligible_counts,
            observed_counts,
        ],
        axis=1,
    ).fillna(0)
    count_mismatch = joined_counts[
        joined_counts[
            "expected"
        ].astype(int)
        .ne(
            joined_counts[
                "observed"
            ].astype(int)
        )
    ]
    if len(count_mismatch):
        raise SystemExit(
            "Retained variant counts do not match "
            "Step 15C3 eligible comparison-threshold pairs:\n"
            + count_mismatch.head(
                20
            ).to_string()
        )

    comparison_metadata = (
        comparison_manifest.set_index(
            "comparison_uid",
            drop=False,
        )
    )
    missing_metadata = sorted(
        set(eligible_uids)
        - set(
            comparison_metadata.index
        )
    )
    if missing_metadata:
        raise SystemExit(
            "Eligible comparisons absent from "
            "the Step 15C1 comparison manifest: "
            f"{missing_metadata[:20]}"
        )

    args.cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pending = []
    for comparison_uid in eligible_uids:
        variant_map = expected_variant_map(
            retained,
            comparison_uid,
            thresholds,
        )
        path = cache_path(
            args.cache_dir,
            comparison_uid,
        )
        if (
            args.rebuild_cache
            or not cache_valid(
                path,
                comparison_uid=(
                    comparison_uid
                ),
                variant_map=variant_map,
                comparison_bootstrap=(
                    args.comparison_bootstrap
                ),
                seed=args.seed,
            )
        ):
            pending.append(
                comparison_uid
            )

    print("=" * 78)
    print(
        "STEP 15D — CROSS-ANCESTRY "
        "EFFECT PORTABILITY"
    )
    print("=" * 78)
    print(
        "Eligible comparisons at any threshold:",
        len(eligible_uids),
    )
    print(
        "Pending comparison caches:",
        len(pending),
    )
    print(
        "Primary comparison bootstrap:",
        args.comparison_bootstrap,
    )
    print(
        "Headline unit bootstrap:",
        args.unit_bootstrap,
    )
    print(
        "Thresholds:",
        ", ".join(
            f"{threshold:.2f}"
            for threshold in thresholds
        ),
    )
    print("=" * 78)

    processing_errors = []
    processed_now = 0
    limit = (
        len(pending)
        if args.max_comparisons <= 0
        else min(
            len(pending),
            args.max_comparisons,
        )
    )

    for ordinal, comparison_uid in enumerate(
        pending[:limit],
        start=1,
    ):
        row = comparison_metadata.loc[
            comparison_uid
        ]
        print(
            f"[{ordinal}/{limit}] "
            f"{comparison_uid} | "
            f"{clean(row['gene'])} | "
            f"{clean(row['comparison_population'])}",
            flush=True,
        )
        path = cache_path(
            args.cache_dir,
            comparison_uid,
        )
        try:
            result = process_comparison(
                comparison_row=row,
                retained=retained,
                thresholds=thresholds,
                primary_threshold=(
                    args.primary_threshold
                ),
                minimum_variants=(
                    args.minimum_variants
                ),
                comparison_bootstrap=(
                    args.comparison_bootstrap
                ),
                base_seed=args.seed,
                harmonized_dir=(
                    args.harmonized_dir
                ),
            )
            atomic_json_write(
                result,
                path,
            )
        except Exception as exc:
            error = {
                "comparison_uid": (
                    comparison_uid
                ),
                "gene": clean(
                    row["gene"]
                ),
                "comparison_population": clean(
                    row[
                        "comparison_population"
                    ]
                ),
                "error_type": (
                    type(exc).__name__
                ),
                "error_message": str(exc),
                "traceback": (
                    traceback.format_exc()
                ),
            }
            processing_errors.append(
                error
            )
            atomic_json_write(
                {
                    "script_version": (
                        SCRIPT_VERSION
                    ),
                    "status": "FAIL",
                    **error,
                    "completed_at_utc": (
                        utc_now()
                    ),
                },
                path,
            )
            print(
                "  ERROR:",
                type(exc).__name__,
                str(exc),
                flush=True,
            )
        processed_now += 1

    all_estimates = []
    all_primary_variant_effects = []
    valid_cache_uids = []
    invalid_or_missing = []

    for comparison_uid in eligible_uids:
        path = cache_path(
            args.cache_dir,
            comparison_uid,
        )
        variant_map = expected_variant_map(
            retained,
            comparison_uid,
            thresholds,
        )
        if not cache_valid(
            path,
            comparison_uid=(
                comparison_uid
            ),
            variant_map=variant_map,
            comparison_bootstrap=(
                args.comparison_bootstrap
            ),
            seed=args.seed,
        ):
            invalid_or_missing.append(
                comparison_uid
            )
            continue
        data = read_json(path)
        valid_cache_uids.append(
            comparison_uid
        )
        all_estimates.extend(
            data["estimates"]
        )
        all_primary_variant_effects.extend(
            data[
                "primary_variant_effects"
            ]
        )

    if not all_estimates:
        print()
        print(
            "No valid Step 15D comparison "
            "caches are available yet."
        )
        return 1

    comparison_estimates = pd.DataFrame(
        all_estimates
    )
    primary_variant_effects = (
        pd.DataFrame(
            all_primary_variant_effects
        )
    )
    unit_estimates = (
        aggregate_unit_estimates(
            comparison_estimates,
            unit_universe,
        )
    )

    comparison_path = (
        args.output_dir
        / (
            "15d_comparison_portability_"
            "estimates.parquet"
        )
    )
    unit_path = (
        args.output_dir
        / (
            "15d_unit_portability_"
            "estimates.parquet"
        )
    )
    primary_comparison_path = (
        args.output_dir
        / (
            "15d_primary_comparison_"
            "portability.parquet"
        )
    )
    primary_unit_path = (
        args.output_dir
        / (
            "15d_primary_unit_"
            "portability.parquet"
        )
    )
    variant_path = (
        args.output_dir
        / (
            "15d_variant_effects_"
            "primary.parquet"
        )
    )

    comparison_estimates.to_parquet(
        comparison_path,
        index=False,
    )
    unit_estimates.to_parquet(
        unit_path,
        index=False,
    )
    comparison_estimates[
        comparison_estimates[
            "is_primary_threshold"
        ].map(as_bool)
    ].to_parquet(
        primary_comparison_path,
        index=False,
    )
    unit_estimates[
        np.isclose(
            unit_estimates[
                "r2_threshold"
            ].astype(float),
            args.primary_threshold,
            rtol=0.0,
            atol=1e-12,
        )
    ].to_parquet(
        primary_unit_path,
        index=False,
    )
    primary_variant_effects.to_parquet(
        variant_path,
        index=False,
    )

    summary = build_summary(
        comparison_estimates=(
            comparison_estimates
        ),
        unit_estimates=unit_estimates,
        thresholds=thresholds,
        primary_threshold=(
            args.primary_threshold
        ),
        unit_bootstrap=(
            args.unit_bootstrap
        ),
        seed=args.seed,
        processing_errors=(
            processing_errors
        ),
        expected_comparisons=(
            len(eligible_uids)
        ),
    )
    summary[
        "comparison_bootstrap_replicates"
    ] = int(
        args.comparison_bootstrap
    )
    summary[
        "unit_bootstrap_replicates"
    ] = int(
        args.unit_bootstrap
    )
    summary[
        "valid_comparison_caches"
    ] = int(
        len(valid_cache_uids)
    )
    summary[
        "invalid_or_missing_comparison_caches"
    ] = invalid_or_missing
    summary["outputs"] = {
        "variant_effects_primary": str(
            variant_path
        ),
        "comparison_estimates": str(
            comparison_path
        ),
        "unit_estimates": str(
            unit_path
        ),
        "primary_comparison_estimates": str(
            primary_comparison_path
        ),
        "primary_unit_estimates": str(
            primary_unit_path
        ),
    }

    summary_path = (
        args.output_dir
        / "15d_portability_summary.json"
    )
    atomic_json_write(
        summary,
        summary_path,
    )

    primary_key = threshold_key(
        args.primary_threshold
    )
    primary_summary = summary[
        "threshold_results"
    ].get(
        primary_key,
        {},
    )

    print()
    print("=" * 78)
    print(
        "STEP 15D — CROSS-ANCESTRY "
        "EFFECT PORTABILITY"
    )
    print("=" * 78)
    print("Status:", summary["status"])
    print(
        "Valid comparison caches:",
        len(valid_cache_uids),
        "/",
        len(eligible_uids),
    )
    print(
        "Invalid or missing caches:",
        len(invalid_or_missing),
    )
    if primary_summary:
        print(
            "Primary comparisons:",
            primary_summary[
                "n_comparisons"
            ],
        )
        print(
            "Primary units:",
            primary_summary[
                "n_units"
            ],
        )
        for estimator in ESTIMATORS:
            result = primary_summary[
                "estimators"
            ][estimator]
            print(
                f"{estimator}: median "
                f"{result['median']:.4f} "
                f"(95% CI "
                f"{result['ci_lower']:.4f}, "
                f"{result['ci_upper']:.4f})"
            )
    print(
        "Summary:",
        summary_path,
    )
    print("=" * 78)

    return (
        0
        if summary["status"] == "PASS"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
