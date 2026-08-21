#!/usr/bin/env python3
"""
Step 15E — Cross-ancestry colocalization with coloc-style ABFs.

For each frozen primary comparison from Step 15B5, apply the
single-causal-variant approximate Bayes-factor enumeration used by
coloc.abf to harmonized EUR and comparison-ancestry summary statistics.

Primary design
--------------
- Use regression coefficients and standard errors directly.
- Use primary exact/QC-qualified variants.
- Reuse the identical locus/variant set for every prior sensitivity.
- p1 = p2 = 1e-4.
- Primary p12 = 1e-5.
- p12 sensitivity = 1e-6, 1e-5, 1e-4.
- Effect-prior SD:
    quantitative = 0.15 * estimated sdY
    case-control = 0.20
- Estimate quantitative sdY separately by ancestry from coefficient
  variance, ancestry-specific MAF, and sample size.
- Calculate all primary comparisons.
- Headline aggregation uses only comparisons marked
  coloc_powered_both_ancestries.
- Within each gene-trait-source unit, aggregate PP.H4 by the median
  across powered comparison ancestries. Never use the maximum ancestry.

This is ABF colocalization under a single-causal-variant-per-trait
assumption. It does not run SuSiE.

Inputs
------
output/15_coloc_comparisons_primary_locked.parquet
output/15b5_unit_analysis_universe.parquet
intermediate/15b4/comparisons/<comparison_uid>.parquet

Outputs
-------
intermediate/15e/comparisons/<comparison_uid>.json
intermediate/15e/comparisons/<comparison_uid>.parquet
output/15e_coloc_comparison_results.parquet
output/15e_coloc_unit_results.parquet
output/15e_coloc_variant_posteriors_primary.parquet
output/15e_coloc_headline_powered_units.parquet
output/15e_coloc_summary.json
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
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logsumexp


SCRIPT_VERSION = "15E.3"
DEFAULT_P1 = 1e-4
DEFAULT_P2 = 1e-4
PRIMARY_P12 = 1e-5
DEFAULT_P12_VALUES = (1e-6, 1e-5, 1e-4)
QUANT_EFFECT_PRIOR_SD = 0.15
CC_EFFECT_PRIOR_SD = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "output/15_coloc_comparisons_primary_locked.parquet"
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
            "intermediate/15e/comparisons"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
    )
    parser.add_argument("--p1", type=float, default=DEFAULT_P1)
    parser.add_argument("--p2", type=float, default=DEFAULT_P2)
    parser.add_argument(
        "--p12-values",
        default="1e-6,1e-5,1e-4",
    )
    parser.add_argument(
        "--primary-p12",
        type=float,
        default=PRIMARY_P12,
    )
    parser.add_argument(
        "--quant-effect-prior-sd",
        type=float,
        default=QUANT_EFFECT_PRIOR_SD,
    )
    parser.add_argument(
        "--cc-effect-prior-sd",
        type=float,
        default=CC_EFFECT_PRIOR_SD,
    )
    parser.add_argument(
        "--qc-column",
        default="qc_qualified",
    )
    parser.add_argument(
        "--minimum-variants",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--headline-bootstrap",
        type=int,
        default=5000,
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
            "Process at most this many pending comparisons. "
            "Zero processes all."
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


def json_default(value: Any) -> Any:
    # pandas/parquet scalar values often arrive as NumPy scalar types.
    # np.bool_ is not handled by Python's standard JSON encoder even
    # though its displayed type name is "bool".
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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
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


def parse_positive_values(text: str) -> tuple[float, ...]:
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
            "At least one p12 value is required."
        )
    if any(
        value <= 0.0
        or value >= 1.0
        for value in values
    ):
        raise SystemExit(
            "Every p12 value must lie between 0 and 1."
        )
    return values


def prior_key(value: float) -> str:
    return f"{value:.12g}"


def stable_seed(base_seed: int, text: str) -> int:
    digest = hashlib.sha256(
        f"{base_seed}|{text}".encode("utf-8")
    ).digest()
    return int.from_bytes(
        digest[:8],
        "little",
        signed=False,
    ) % (2**32 - 1)


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


def numeric_array(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(dtype=float)


def scalar_from_row(
    row: pd.Series,
    candidates: tuple[str, ...],
) -> float:
    for column in candidates:
        if column not in row.index:
            continue
        value = pd.to_numeric(
            pd.Series([row[column]]),
            errors="coerce",
        ).iloc[0]
        if (
            pd.notna(value)
            and math.isfinite(float(value))
            and float(value) > 0.0
        ):
            return float(value)
    raise RuntimeError(
        "No positive sample-size value found among: "
        + ", ".join(candidates)
    )


def resolve_maf_column(
    frame: pd.DataFrame,
    prefix: str,
) -> np.ndarray:
    maf_column = f"{prefix}_maf"
    af_column = f"{prefix}_af"

    if maf_column in frame.columns:
        maf = numeric_array(frame[maf_column])
    elif af_column in frame.columns:
        af = numeric_array(frame[af_column])
        maf = np.minimum(af, 1.0 - af)
    else:
        raise RuntimeError(
            f"No {prefix} MAF or AF column."
        )

    if (
        not np.isfinite(maf).all()
        or np.any(maf <= 0.0)
        or np.any(maf > 0.5 + 1e-12)
    ):
        raise RuntimeError(
            f"Invalid {prefix} MAF values."
        )
    return maf


def estimate_sdy(
    variance_beta: np.ndarray,
    maf: np.ndarray,
    sample_size: float,
) -> float:
    """
    Match coloc::sdY.est:
      oneover = 1/varbeta
      nvx = 2*N*maf*(1-maf)
      lm(nvx ~ oneover - 1)
    """
    one_over = 1.0 / variance_beta
    nvx = (
        2.0
        * sample_size
        * maf
        * (1.0 - maf)
    )
    valid = (
        np.isfinite(one_over)
        & np.isfinite(nvx)
        & (one_over > 0.0)
        & (nvx > 0.0)
    )
    if int(valid.sum()) < 3:
        raise RuntimeError(
            "Too few variants to estimate sdY."
        )

    x = one_over[valid]
    y = nvx[valid]
    denominator = float(np.dot(x, x))
    if denominator <= 0.0:
        raise RuntimeError(
            "Invalid sdY regression denominator."
        )

    coefficient = float(
        np.dot(x, y)
        / denominator
    )
    if (
        not math.isfinite(coefficient)
        or coefficient <= 0.0
    ):
        raise RuntimeError(
            "Estimated trait variance is nonpositive."
        )
    return float(math.sqrt(coefficient))


def coloc_type(value: Any) -> str:
    rendered = clean(value).casefold()
    if rendered in {
        "quant",
        "quantitative",
        "continuous",
        "biomarkers",
        "biomarker",
    }:
        return "quant"
    if rendered in {
        "cc",
        "case-control",
        "case_control",
        "binary",
    }:
        return "cc"
    raise RuntimeError(
        f"Unsupported coloc trait type: {value}"
    )


def approximate_log_abf(
    beta: np.ndarray,
    standard_error: np.ndarray,
    *,
    trait_type: str,
    sd_y: float | None,
    quant_effect_prior_sd: float,
    cc_effect_prior_sd: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    variance = np.square(standard_error)
    z = beta / standard_error

    if trait_type == "quant":
        if (
            sd_y is None
            or not math.isfinite(sd_y)
            or sd_y <= 0.0
        ):
            raise RuntimeError(
                "Quantitative trait requires positive sdY."
            )
        prior_sd = quant_effect_prior_sd * sd_y
    elif trait_type == "cc":
        prior_sd = cc_effect_prior_sd
    else:
        raise RuntimeError(
            f"Unknown coloc type: {trait_type}"
        )

    prior_variance = prior_sd**2
    shrinkage = (
        prior_variance
        / (
            prior_variance
            + variance
        )
    )
    log_abf = 0.5 * (
        np.log1p(-shrinkage)
        + shrinkage * np.square(z)
    )

    if not np.isfinite(log_abf).all():
        raise RuntimeError(
            "Nonfinite approximate Bayes factors."
        )

    return log_abf, z, shrinkage


def adjust_prior(
    prior: float,
    n_variants: int,
) -> tuple[float, bool]:
    if n_variants * prior >= 1.0:
        return (
            1.0 / (n_variants + 1.0),
            True,
        )
    return prior, False


def log_difference(
    first: float,
    second: float,
) -> float:
    """
    Stable log(exp(first) - exp(second)).
    """
    if not math.isfinite(first):
        return -math.inf
    if second == -math.inf:
        return first
    if second > first:
        if second - first < 1e-10:
            return -math.inf
        raise RuntimeError(
            "Negative H3 Bayes-factor difference."
        )
    difference = second - first
    if difference >= 0.0:
        return -math.inf
    return float(
        first
        + math.log1p(
            -math.exp(difference)
        )
    )


def combine_coloc_abf(
    log_abf_1: np.ndarray,
    log_abf_2: np.ndarray,
    *,
    p1: float,
    p2: float,
    p12: float,
) -> dict[str, Any]:
    if len(log_abf_1) != len(log_abf_2):
        raise RuntimeError(
            "ABF vectors have unequal lengths."
        )

    n_variants = len(log_abf_1)
    p1_used, p1_adjusted = adjust_prior(
        p1,
        n_variants,
    )
    p2_used, p2_adjusted = adjust_prior(
        p2,
        n_variants,
    )
    p12_used, p12_adjusted = adjust_prior(
        p12,
        n_variants,
    )

    combined = log_abf_1 + log_abf_2
    sum_1 = float(logsumexp(log_abf_1))
    sum_2 = float(logsumexp(log_abf_2))
    sum_shared = float(logsumexp(combined))

    log_hypotheses = np.asarray(
        [
            0.0,
            math.log(p1_used) + sum_1,
            math.log(p2_used) + sum_2,
            (
                math.log(p1_used)
                + math.log(p2_used)
                + log_difference(
                    sum_1 + sum_2,
                    sum_shared,
                )
            ),
            math.log(p12_used) + sum_shared,
        ],
        dtype=float,
    )

    denominator = float(
        logsumexp(log_hypotheses)
    )
    posterior = np.exp(
        log_hypotheses
        - denominator
    )

    if (
        not np.isfinite(posterior).all()
        or not math.isclose(
            float(posterior.sum()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-10,
        )
    ):
        raise RuntimeError(
            "Invalid coloc hypothesis posterior."
        )

    h3_h4 = float(
        posterior[3]
        + posterior[4]
    )
    conditional_h4 = (
        float(
            posterior[4]
            / h3_h4
        )
        if h3_h4 > 0.0
        else math.nan
    )

    return {
        "pp_h0": float(posterior[0]),
        "pp_h1": float(posterior[1]),
        "pp_h2": float(posterior[2]),
        "pp_h3": float(posterior[3]),
        "pp_h4": float(posterior[4]),
        "pp_h4_given_h3_h4": conditional_h4,
        "p1_used": float(p1_used),
        "p2_used": float(p2_used),
        "p12_used": float(p12_used),
        "p1_adjusted": p1_adjusted,
        "p2_adjusted": p2_adjusted,
        "p12_adjusted": p12_adjusted,
    }


def shared_variant_posterior(
    log_abf_1: np.ndarray,
    log_abf_2: np.ndarray,
) -> np.ndarray:
    combined = log_abf_1 + log_abf_2
    posterior = np.exp(
        combined
        - logsumexp(combined)
    )
    if not math.isclose(
        float(posterior.sum()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise RuntimeError(
            "Conditional shared-variant posterior "
            "does not sum to one."
        )
    return posterior


def single_trait_variant_posterior(
    log_abf: np.ndarray,
) -> np.ndarray:
    return np.exp(
        log_abf
        - logsumexp(log_abf)
    )


def credible_set(
    variant_ids: list[str],
    posterior: np.ndarray,
    probability: float = 0.95,
) -> dict[str, Any]:
    order = np.argsort(
        -posterior,
        kind="mergesort",
    )
    cumulative = np.cumsum(
        posterior[order]
    )
    end = int(
        np.searchsorted(
            cumulative,
            probability,
            side="left",
        )
    )
    end = min(
        end,
        len(order) - 1,
    )
    chosen = order[: end + 1]
    return {
        "credible_probability": float(
            probability
        ),
        "credible_set_size": int(
            len(chosen)
        ),
        "credible_set_variants": [
            variant_ids[int(index)]
            for index in chosen
        ],
        "credible_set_posterior_mass": float(
            posterior[chosen].sum()
        ),
    }


def evidence_category(
    pp_h3: float,
    pp_h4: float,
) -> str:
    if pp_h4 >= 0.8:
        return "SHARED_H4_GE_0P8"
    if pp_h3 >= 0.8:
        return "DISTINCT_H3_GE_0P8"
    if pp_h4 >= 0.5:
        return "MODERATE_SHARED_H4_GE_0P5"
    return "INCONCLUSIVE"


def cache_paths(
    cache_dir: Path,
    comparison_uid: str,
) -> tuple[Path, Path]:
    safe = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        comparison_uid,
    )
    return (
        cache_dir / f"{safe}.json",
        cache_dir / f"{safe}.parquet",
    )


def input_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def cache_valid(
    metadata_path: Path,
    variant_path: Path,
    *,
    comparison_uid: str,
    harmonized_path: Path,
    p1: float,
    p2: float,
    p12_values: tuple[float, ...],
    primary_p12: float,
    qc_column: str,
    quant_effect_prior_sd: float,
    cc_effect_prior_sd: float,
) -> bool:
    if (
        not metadata_path.exists()
        or not variant_path.exists()
    ):
        return False

    try:
        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return False

    return (
        clean(metadata.get("script_version"))
        == SCRIPT_VERSION
        and clean(metadata.get("comparison_uid"))
        == comparison_uid
        and metadata.get("input_signature")
        == input_signature(harmonized_path)
        and math.isclose(
            float(metadata.get("p1", -1.0)),
            p1,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and math.isclose(
            float(metadata.get("p2", -1.0)),
            p2,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and metadata.get("p12_values")
        == [
            float(value)
            for value in p12_values
        ]
        and math.isclose(
            float(
                metadata.get(
                    "primary_p12",
                    -1.0,
                )
            ),
            primary_p12,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and clean(metadata.get("qc_column"))
        == qc_column
        and math.isclose(
            float(
                metadata.get(
                    "quant_effect_prior_sd",
                    -1.0,
                )
            ),
            quant_effect_prior_sd,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and math.isclose(
            float(
                metadata.get(
                    "cc_effect_prior_sd",
                    -1.0,
                )
            ),
            cc_effect_prior_sd,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and clean(metadata.get("status"))
        == "PASS"
    )


def process_comparison(
    row: pd.Series,
    *,
    harmonized_path: Path,
    qc_column: str,
    minimum_variants: int,
    p1: float,
    p2: float,
    p12_values: tuple[float, ...],
    primary_p12: float,
    quant_effect_prior_sd: float,
    cc_effect_prior_sd: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = pd.read_parquet(
        harmonized_path
    )
    require_columns(
        frame,
        {
            "variant_id",
            "eur_beta",
            "eur_se",
            "comparison_beta",
            "comparison_se",
            qc_column,
        },
        str(harmonized_path),
    )

    primary = frame[
        frame[qc_column].map(as_bool)
    ].copy()

    # n_harmonized_primary counts exact harmonized rows before the
    # full Step 15B4 QC gate. Because this analysis filters on
    # qc_qualified, validate against n_qc_qualified instead.
    if qc_column == "qc_qualified":
        expected_count_column = "n_qc_qualified"
    elif qc_column == "qc_qualified_sensitivity":
        expected_count_column = (
            "n_qc_qualified_sensitivity"
        )
    else:
        expected_count_column = None

    if (
        expected_count_column is not None
        and expected_count_column in row.index
    ):
        expected = int(
            row[expected_count_column]
        )
        if len(primary) != expected:
            raise RuntimeError(
                f"{qc_column} row count mismatch: "
                f"manifest {expected_count_column}={expected}, "
                f"file={len(primary)}"
            )

    minimum_manifest = int(
        row["min_coloc_variants"]
    )
    minimum_required = max(
        minimum_variants,
        minimum_manifest,
    )
    if len(primary) < minimum_required:
        raise RuntimeError(
            f"Only {len(primary)} variants; "
            f"minimum={minimum_required}."
        )

    if primary[
        "variant_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate variant IDs in primary locus."
        )

    beta_eur = numeric_array(
        primary["eur_beta"]
    )
    se_eur = numeric_array(
        primary["eur_se"]
    )
    beta_comparison = numeric_array(
        primary["comparison_beta"]
    )
    se_comparison = numeric_array(
        primary["comparison_se"]
    )

    valid = (
        np.isfinite(beta_eur)
        & np.isfinite(se_eur)
        & (se_eur > 0.0)
        & np.isfinite(beta_comparison)
        & np.isfinite(se_comparison)
        & (se_comparison > 0.0)
    )
    if not valid.all():
        invalid = primary.loc[
            ~valid,
            "variant_id",
        ].astype(str).tolist()
        raise RuntimeError(
            "Invalid beta/SE for variants: "
            + ", ".join(invalid[:20])
        )

    eur_type = coloc_type(
        row["eur_coloc_type"]
    )
    comparison_type = coloc_type(
        row["comparison_coloc_type"]
    )

    variance_eur = np.square(se_eur)
    variance_comparison = np.square(
        se_comparison
    )

    eur_sdy = None
    comparison_sdy = None
    eur_n = None
    comparison_n = None

    if eur_type == "quant":
        eur_n = scalar_from_row(
            row,
            (
                "eur_n_total",
                "eur_n_total_median",
                "eur_n",
                "eur_sample_size",
            ),
        )
        eur_maf = resolve_maf_column(
            primary,
            "eur",
        )
        eur_sdy = estimate_sdy(
            variance_eur,
            eur_maf,
            eur_n,
        )

    if comparison_type == "quant":
        comparison_n = scalar_from_row(
            row,
            (
                "comparison_n_total",
                "comparison_n_total_median",
                "comparison_n",
                "comparison_sample_size",
            ),
        )
        comparison_maf = resolve_maf_column(
            primary,
            "comparison",
        )
        comparison_sdy = estimate_sdy(
            variance_comparison,
            comparison_maf,
            comparison_n,
        )

    log_abf_eur, z_eur, r_eur = (
        approximate_log_abf(
            beta_eur,
            se_eur,
            trait_type=eur_type,
            sd_y=eur_sdy,
            quant_effect_prior_sd=(
                quant_effect_prior_sd
            ),
            cc_effect_prior_sd=(
                cc_effect_prior_sd
            ),
        )
    )
    (
        log_abf_comparison,
        z_comparison,
        r_comparison,
    ) = approximate_log_abf(
        beta_comparison,
        se_comparison,
        trait_type=comparison_type,
        sd_y=comparison_sdy,
        quant_effect_prior_sd=(
            quant_effect_prior_sd
        ),
        cc_effect_prior_sd=(
            cc_effect_prior_sd
        ),
    )

    shared_pp = shared_variant_posterior(
        log_abf_eur,
        log_abf_comparison,
    )
    eur_variant_pp = (
        single_trait_variant_posterior(
            log_abf_eur
        )
    )
    comparison_variant_pp = (
        single_trait_variant_posterior(
            log_abf_comparison
        )
    )

    variant_ids = primary[
        "variant_id"
    ].astype(str).tolist()
    lead_index = int(
        np.argmax(shared_pp)
    )
    cs95 = credible_set(
        variant_ids,
        shared_pp,
        probability=0.95,
    )

    prior_results = []
    for p12 in p12_values:
        result = combine_coloc_abf(
            log_abf_eur,
            log_abf_comparison,
            p1=p1,
            p2=p2,
            p12=p12,
        )
        prior_results.append(
            {
                "p12_requested": float(
                    p12
                ),
                "is_primary_prior": bool(
                    math.isclose(
                        p12,
                        primary_p12,
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    )
                ),
                **result,
                "evidence_category": (
                    evidence_category(
                        result["pp_h3"],
                        result["pp_h4"],
                    )
                ),
            }
        )

    metadata_columns = [
        "comparison_uid",
        "gene_trait_uid",
        "gene",
        "candidate_source",
        "candidate_trait_key",
        "candidate_trait_name",
        "comparison_population",
        "comparison_reference_population",
        "genome_build",
        "unit_analysis_role",
        "mixed_pool",
        "primary_approval_eligible",
        "coloc_powered_both_ancestries",
        "approval_analysis_eligible",
    ]
    comparison_metadata = {
        column: (
            row[column]
            if column in row.index
            else None
        )
        for column in metadata_columns
    }

    metadata = {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        **comparison_metadata,
        "comparison_uid": clean(
            row["comparison_uid"]
        ),
        "input_signature": input_signature(
            harmonized_path
        ),
        "harmonized_path": str(
            harmonized_path
        ),
        "qc_column": qc_column,
        "n_variants": int(len(primary)),
        "eur_coloc_type": eur_type,
        "comparison_coloc_type": (
            comparison_type
        ),
        "eur_n": eur_n,
        "comparison_n": comparison_n,
        "eur_sd_y": eur_sdy,
        "comparison_sd_y": comparison_sdy,
        "p1": float(p1),
        "p2": float(p2),
        "p12_values": [
            float(value)
            for value in p12_values
        ],
        "primary_p12": float(
            primary_p12
        ),
        "quant_effect_prior_sd": float(
            quant_effect_prior_sd
        ),
        "cc_effect_prior_sd": float(
            cc_effect_prior_sd
        ),
        "prior_results": prior_results,
        "lead_shared_variant": (
            variant_ids[lead_index]
        ),
        "lead_shared_variant_pp_h4": float(
            shared_pp[lead_index]
        ),
        **cs95,
        "completed_at_utc": utc_now(),
    }

    keep_columns = [
        column
        for column in [
            "variant_id",
            "chrom",
            "pos",
            "ref",
            "alt",
            "eur_beta",
            "eur_se",
            "comparison_beta",
            "comparison_se",
            "eur_p",
            "comparison_p",
            "eur_af",
            "comparison_af",
            "eur_maf",
            "comparison_maf",
        ]
        if column in primary.columns
    ]
    variant_output = primary[
        keep_columns
    ].copy()
    variant_output["comparison_uid"] = clean(
        row["comparison_uid"]
    )
    variant_output["gene_trait_uid"] = clean(
        row["gene_trait_uid"]
    )
    variant_output["gene"] = clean(
        row["gene"]
    )
    variant_output["candidate_source"] = clean(
        row["candidate_source"]
    )
    variant_output[
        "candidate_trait_name"
    ] = clean(
        row["candidate_trait_name"]
    )
    variant_output[
        "comparison_population"
    ] = clean(
        row["comparison_population"]
    )
    variant_output[
        "coloc_powered_both_ancestries"
    ] = as_bool(
        row[
            "coloc_powered_both_ancestries"
        ]
    )
    variant_output["eur_z"] = z_eur
    variant_output[
        "comparison_z"
    ] = z_comparison
    variant_output[
        "eur_shrinkage_r"
    ] = r_eur
    variant_output[
        "comparison_shrinkage_r"
    ] = r_comparison
    variant_output[
        "eur_log_abf"
    ] = log_abf_eur
    variant_output[
        "comparison_log_abf"
    ] = log_abf_comparison
    variant_output[
        "eur_variant_pp_conditional_association"
    ] = eur_variant_pp
    variant_output[
        "comparison_variant_pp_conditional_association"
    ] = comparison_variant_pp
    variant_output[
        "snp_pp_h4_conditional_shared"
    ] = shared_pp
    variant_output[
        "in_h4_credible_set_95"
    ] = variant_output[
        "variant_id"
    ].astype(str).isin(
        cs95["credible_set_variants"]
    )

    return metadata, variant_output


def comparison_rows_from_cache(
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    base_columns = [
        "comparison_uid",
        "gene_trait_uid",
        "gene",
        "candidate_source",
        "candidate_trait_key",
        "candidate_trait_name",
        "comparison_population",
        "comparison_reference_population",
        "genome_build",
        "unit_analysis_role",
        "mixed_pool",
        "primary_approval_eligible",
        "coloc_powered_both_ancestries",
        "approval_analysis_eligible",
        "n_variants",
        "eur_coloc_type",
        "comparison_coloc_type",
        "eur_n",
        "comparison_n",
        "eur_sd_y",
        "comparison_sd_y",
        "lead_shared_variant",
        "lead_shared_variant_pp_h4",
        "credible_set_size",
        "credible_set_posterior_mass",
    ]
    base = {
        column: metadata.get(column)
        for column in base_columns
    }
    return [
        {
            **base,
            **prior_result,
        }
        for prior_result in metadata[
            "prior_results"
        ]
    ]


def aggregate_units(
    comparison_results: pd.DataFrame,
    unit_universe: pd.DataFrame,
) -> pd.DataFrame:
    all_rows = comparison_results.assign(
        aggregation_scope="ALL_ELIGIBLE"
    )
    powered_rows = comparison_results[
        comparison_results[
            "coloc_powered_both_ancestries"
        ].map(as_bool)
    ].assign(
        aggregation_scope=(
            "POWERED_BOTH_ONLY"
        )
    )
    expanded = pd.concat(
        [
            all_rows,
            powered_rows,
        ],
        ignore_index=True,
    )

    rows = []
    for (
        gene_trait_uid,
        p12_requested,
        scope,
    ), group in expanded.groupby(
        [
            "gene_trait_uid",
            "p12_requested",
            "aggregation_scope",
        ],
        sort=True,
        dropna=False,
    ):
        invariant_columns = [
            "gene",
            "candidate_source",
            "candidate_trait_key",
            "candidate_trait_name",
            "unit_analysis_role",
            "mixed_pool",
            "primary_approval_eligible",
            "approval_analysis_eligible",
        ]
        output = {
            "gene_trait_uid": gene_trait_uid,
            "p12_requested": float(
                p12_requested
            ),
            "aggregation_scope": scope,
            "is_primary_prior": bool(
                group[
                    "is_primary_prior"
                ].map(as_bool).all()
            ),
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
            "pp_h0_median": float(
                group["pp_h0"].median()
            ),
            "pp_h1_median": float(
                group["pp_h1"].median()
            ),
            "pp_h2_median": float(
                group["pp_h2"].median()
            ),
            "pp_h3_median": float(
                group["pp_h3"].median()
            ),
            "pp_h4_median": float(
                group["pp_h4"].median()
            ),
            "pp_h4_minimum": float(
                group["pp_h4"].min()
            ),
            "pp_h4_maximum": float(
                group["pp_h4"].max()
            ),
            "pp_h4_given_h3_h4_median": float(
                group[
                    "pp_h4_given_h3_h4"
                ].median()
            ),
            "n_shared_h4_ge_0p8": int(
                (
                    group["pp_h4"]
                    >= 0.8
                ).sum()
            ),
            "all_ancestries_h4_ge_0p8": bool(
                (
                    group["pp_h4"]
                    >= 0.8
                ).all()
            ),
            "any_ancestry_h4_ge_0p8": bool(
                (
                    group["pp_h4"]
                    >= 0.8
                ).any()
            ),
            "unit_evidence_category": (
                evidence_category(
                    float(
                        group[
                            "pp_h3"
                        ].median()
                    ),
                    float(
                        group[
                            "pp_h4"
                        ].median()
                    ),
                )
            ),
        }

        for column in invariant_columns:
            if column not in group.columns:
                output[column] = None
                continue
            values = group[
                column
            ].drop_duplicates()
            if len(values) != 1:
                raise RuntimeError(
                    f"{column} varies within "
                    f"{gene_trait_uid}: "
                    f"{values.tolist()}"
                )
            output[column] = values.iloc[0]

        rows.append(output)

    unit_results = pd.DataFrame(rows)

    if "gene_trait_uid" in unit_universe.columns:
        metadata_columns = [
            column
            for column in [
                "gene_trait_uid",
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
            if column
            in unit_universe.columns
        ]
        metadata = unit_universe[
            metadata_columns
        ].drop_duplicates(
            "gene_trait_uid"
        )
        unit_results = unit_results.merge(
            metadata,
            on="gene_trait_uid",
            how="left",
            validate="many_to_one",
        )

    return unit_results.sort_values(
        [
            "p12_requested",
            "aggregation_scope",
            "candidate_source",
            "candidate_trait_name",
            "gene",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def bootstrap_median(
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
            "bootstrap_replicates": (
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

    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        values.size,
        size=(
            replicates,
            values.size,
        ),
    )
    boot = np.median(
        values[indices],
        axis=1,
    )
    lower, upper = np.percentile(
        boot,
        [2.5, 97.5],
    )
    return {
        "median": median,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "n_units": int(values.size),
        "bootstrap_replicates": int(
            replicates
        ),
    }


def prior_summary(
    unit_results: pd.DataFrame,
    *,
    p12: float,
    scope: str,
    headline_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    subset = unit_results[
        np.isclose(
            unit_results[
                "p12_requested"
            ].astype(float),
            p12,
            rtol=0.0,
            atol=1e-15,
        )
        & unit_results[
            "aggregation_scope"
        ].eq(scope)
    ].copy()

    headline = bootstrap_median(
        subset[
            "pp_h4_median"
        ].to_numpy(dtype=float),
        replicates=headline_bootstrap,
        seed=stable_seed(
            seed,
            f"{p12}|{scope}|pp_h4",
        ),
    )
    conditional = bootstrap_median(
        subset[
            "pp_h4_given_h3_h4_median"
        ].to_numpy(dtype=float),
        replicates=headline_bootstrap,
        seed=stable_seed(
            seed,
            (
                f"{p12}|{scope}|"
                "conditional_h4"
            ),
        ),
    )

    return {
        "p12": float(p12),
        "aggregation_scope": scope,
        "n_units": int(len(subset)),
        "n_comparisons": int(
            subset[
                "n_comparison_ancestries"
            ].sum()
        ),
        "pp_h4_unit_median": headline,
        "pp_h4_given_h3_h4_unit_median": (
            conditional
        ),
        "n_units_h4_ge_0p8": int(
            (
                subset[
                    "pp_h4_median"
                ]
                >= 0.8
            ).sum()
        ),
        "n_units_h3_ge_0p8": int(
            (
                subset[
                    "pp_h3_median"
                ]
                >= 0.8
            ).sum()
        ),
        "evidence_categories": {
            clean(category): int(count)
            for category, count in (
                subset[
                    "unit_evidence_category"
                ]
                .value_counts(
                    dropna=False
                )
                .items()
            )
        },
        "by_source": {
            clean(source): {
                "n_units": int(
                    len(group)
                ),
                "median_pp_h4": float(
                    group[
                        "pp_h4_median"
                    ].median()
                ),
            }
            for source, group in subset.groupby(
                "candidate_source",
                dropna=False,
            )
        },
    }


def main() -> int:
    args = parse_args()
    p12_values = parse_positive_values(
        args.p12_values
    )

    if not any(
        math.isclose(
            value,
            args.primary_p12,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        for value in p12_values
    ):
        raise SystemExit(
            "--primary-p12 must be included "
            "in --p12-values."
        )
    if args.p1 <= 0.0 or args.p2 <= 0.0:
        raise SystemExit(
            "p1 and p2 must be positive."
        )
    if args.minimum_variants < 1:
        raise SystemExit(
            "--minimum-variants must be positive."
        )

    for path in [
        args.manifest,
        args.unit_universe,
    ]:
        if not path.exists():
            raise SystemExit(
                f"Missing input: {path}"
            )

    manifest = pd.read_parquet(
        args.manifest
    )
    unit_universe = pd.read_parquet(
        args.unit_universe
    )

    require_columns(
        manifest,
        {
            "comparison_uid",
            "gene_trait_uid",
            "gene",
            "candidate_source",
            "candidate_trait_name",
            "comparison_population",
            "n_harmonized_primary",
            "n_qc_qualified",
            "min_coloc_variants",
            "eur_coloc_type",
            "comparison_coloc_type",
            "coloc_input_eligible",
            "coloc_powered_both_ancestries",
        },
        "colocalization manifest",
    )

    manifest = manifest[
        manifest[
            "coloc_input_eligible"
        ].map(as_bool)
    ].copy()

    if manifest[
        "comparison_uid"
    ].duplicated().any():
        raise SystemExit(
            "Duplicate comparison_uid values "
            "in colocalization manifest."
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
    for row in manifest.itertuples(
        index=False,
    ):
        comparison_uid = clean(
            row.comparison_uid
        )
        harmonized_path = (
            args.harmonized_dir
            / f"{comparison_uid}.parquet"
        )
        if not harmonized_path.exists():
            raise SystemExit(
                "Missing harmonized comparison: "
                f"{harmonized_path}"
            )

        metadata_path, variant_path = (
            cache_paths(
                args.cache_dir,
                comparison_uid,
            )
        )
        if (
            args.rebuild_cache
            or not cache_valid(
                metadata_path,
                variant_path,
                comparison_uid=(
                    comparison_uid
                ),
                harmonized_path=(
                    harmonized_path
                ),
                p1=args.p1,
                p2=args.p2,
                p12_values=p12_values,
                primary_p12=(
                    args.primary_p12
                ),
                qc_column=args.qc_column,
                quant_effect_prior_sd=(
                    args.quant_effect_prior_sd
                ),
                cc_effect_prior_sd=(
                    args.cc_effect_prior_sd
                ),
            )
        ):
            pending.append(
                comparison_uid
            )

    print("=" * 78)
    print(
        "STEP 15E — CROSS-ANCESTRY "
        "COLOCALIZATION"
    )
    print("=" * 78)
    print(
        "Primary comparisons:",
        len(manifest),
    )
    print(
        "Powered in both ancestries:",
        int(
            manifest[
                "coloc_powered_both_ancestries"
            ].map(as_bool).sum()
        ),
    )
    print(
        "Pending caches:",
        len(pending),
    )
    print(
        "Priors:",
        f"p1={args.p1:g}, "
        f"p2={args.p2:g}, "
        "p12="
        + ", ".join(
            f"{value:g}"
            for value in p12_values
        ),
    )
    print(
        "Primary p12:",
        f"{args.primary_p12:g}",
    )
    print("=" * 78)

    manifest_lookup = manifest.set_index(
        "comparison_uid",
        drop=False,
    )
    processing_errors = []
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
        row = manifest_lookup.loc[
            comparison_uid
        ]
        print(
            f"[{ordinal}/{limit}] "
            f"{comparison_uid} | "
            f"{clean(row['gene'])} | "
            f"{clean(row['comparison_population'])}",
            flush=True,
        )

        harmonized_path = (
            args.harmonized_dir
            / f"{comparison_uid}.parquet"
        )
        metadata_path, variant_path = (
            cache_paths(
                args.cache_dir,
                comparison_uid,
            )
        )

        try:
            metadata, variant_output = (
                process_comparison(
                    row,
                    harmonized_path=(
                        harmonized_path
                    ),
                    qc_column=args.qc_column,
                    minimum_variants=(
                        args.minimum_variants
                    ),
                    p1=args.p1,
                    p2=args.p2,
                    p12_values=p12_values,
                    primary_p12=(
                        args.primary_p12
                    ),
                    quant_effect_prior_sd=(
                        args.quant_effect_prior_sd
                    ),
                    cc_effect_prior_sd=(
                        args.cc_effect_prior_sd
                    ),
                )
            )
            variant_output.to_parquet(
                variant_path,
                index=False,
            )
            atomic_json_write(
                metadata,
                metadata_path,
            )
        except Exception as exc:
            error = {
                "comparison_uid": comparison_uid,
                "gene": clean(row["gene"]),
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
            variant_path.unlink(
                missing_ok=True
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
                metadata_path,
            )
            print(
                "  ERROR:",
                type(exc).__name__,
                str(exc),
                flush=True,
            )

    valid_uids = []
    invalid_or_missing = []
    comparison_rows = []
    variant_frames = []

    for comparison_uid in manifest[
        "comparison_uid"
    ].astype(str):
        harmonized_path = (
            args.harmonized_dir
            / f"{comparison_uid}.parquet"
        )
        metadata_path, variant_path = (
            cache_paths(
                args.cache_dir,
                comparison_uid,
            )
        )
        if not cache_valid(
            metadata_path,
            variant_path,
            comparison_uid=(
                comparison_uid
            ),
            harmonized_path=(
                harmonized_path
            ),
            p1=args.p1,
            p2=args.p2,
            p12_values=p12_values,
            primary_p12=(
                args.primary_p12
            ),
            qc_column=args.qc_column,
            quant_effect_prior_sd=(
                args.quant_effect_prior_sd
            ),
            cc_effect_prior_sd=(
                args.cc_effect_prior_sd
            ),
        ):
            invalid_or_missing.append(
                comparison_uid
            )
            continue

        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )
        comparison_rows.extend(
            comparison_rows_from_cache(
                metadata
            )
        )
        variant_frames.append(
            pd.read_parquet(
                variant_path
            )
        )
        valid_uids.append(
            comparison_uid
        )

    if not comparison_rows:
        print(
            "No valid Step 15E caches "
            "are available."
        )
        return 1

    comparison_results = pd.DataFrame(
        comparison_rows
    )
    variant_results = pd.concat(
        variant_frames,
        ignore_index=True,
    )
    unit_results = aggregate_units(
        comparison_results,
        unit_universe,
    )

    comparison_path = (
        args.output_dir
        / (
            "15e_coloc_comparison_"
            "results.parquet"
        )
    )
    unit_path = (
        args.output_dir
        / "15e_coloc_unit_results.parquet"
    )
    variant_path = (
        args.output_dir
        / (
            "15e_coloc_variant_posteriors_"
            "primary.parquet"
        )
    )
    headline_path = (
        args.output_dir
        / (
            "15e_coloc_headline_powered_"
            "units.parquet"
        )
    )

    comparison_results.to_parquet(
        comparison_path,
        index=False,
    )
    unit_results.to_parquet(
        unit_path,
        index=False,
    )
    variant_results.to_parquet(
        variant_path,
        index=False,
    )

    headline_units = unit_results[
        unit_results[
            "aggregation_scope"
        ].eq(
            "POWERED_BOTH_ONLY"
        )
        & np.isclose(
            unit_results[
                "p12_requested"
            ].astype(float),
            args.primary_p12,
            rtol=0.0,
            atol=1e-15,
        )
    ].copy()
    headline_units.to_parquet(
        headline_path,
        index=False,
    )

    summaries = {}
    for p12 in p12_values:
        summaries[
            prior_key(p12)
        ] = {
            "powered_both_only": (
                prior_summary(
                    unit_results,
                    p12=p12,
                    scope=(
                        "POWERED_BOTH_ONLY"
                    ),
                    headline_bootstrap=(
                        args.headline_bootstrap
                    ),
                    seed=args.seed,
                )
            ),
            "all_eligible": (
                prior_summary(
                    unit_results,
                    p12=p12,
                    scope=(
                        "ALL_ELIGIBLE"
                    ),
                    headline_bootstrap=(
                        args.headline_bootstrap
                    ),
                    seed=args.seed,
                )
            ),
        }

    powered_comparison_results = (
        comparison_results[
            comparison_results[
                "coloc_powered_both_ancestries"
            ].map(as_bool)
            & comparison_results[
                "is_primary_prior"
            ].map(as_bool)
        ]
    )

    status = (
        "PASS"
        if (
            not processing_errors
            and len(valid_uids)
            == len(manifest)
        )
        else "INCOMPLETE"
    )

    summary = {
        "step": "15E",
        "script_version": SCRIPT_VERSION,
        "status": status,
        "method": (
            "coloc-style approximate Bayes-factor "
            "enumeration with one causal variant "
            "per trait."
        ),
        "hypotheses": {
            "H0": (
                "Neither ancestry has an "
                "association in the region."
            ),
            "H1": "EUR association only.",
            "H2": (
                "Comparison-ancestry association "
                "only."
            ),
            "H3": (
                "Both associated with distinct "
                "causal variants."
            ),
            "H4": (
                "Both associated with one shared "
                "causal variant."
            ),
        },
        "priors": {
            "p1": float(args.p1),
            "p2": float(args.p2),
            "p12_values": [
                float(value)
                for value in p12_values
            ],
            "primary_p12": float(
                args.primary_p12
            ),
            "quant_effect_prior_sd_multiplier": float(
                args.quant_effect_prior_sd
            ),
            "case_control_effect_prior_sd": float(
                args.cc_effect_prior_sd
            ),
        },
        "locus_policy": (
            "The same primary exact/QC-qualified "
            "variant set is reused for every p12 "
            "sensitivity within a comparison."
        ),
        "headline_policy": (
            "Only comparisons powered in both "
            "ancestries contribute. Within each "
            "gene-trait-source unit, PP.H4 is the "
            "median across powered comparison "
            "ancestries; the maximum is not used."
        ),
        "counts": {
            "manifest_comparisons": int(
                len(manifest)
            ),
            "valid_comparison_caches": int(
                len(valid_uids)
            ),
            "invalid_or_missing_caches": int(
                len(invalid_or_missing)
            ),
            "powered_comparisons_primary_prior": int(
                powered_comparison_results[
                    "comparison_uid"
                ].nunique()
            ),
            "headline_powered_units_primary_prior": int(
                len(headline_units)
            ),
            "primary_variant_rows": int(
                len(variant_results)
            ),
        },
        "prior_results": summaries,
        "processing_errors": (
            processing_errors
        ),
        "invalid_or_missing_comparison_caches": (
            invalid_or_missing
        ),
        "outputs": {
            "comparison_results": str(
                comparison_path
            ),
            "unit_results": str(
                unit_path
            ),
            "variant_posteriors_primary": str(
                variant_path
            ),
            "headline_powered_units": str(
                headline_path
            ),
        },
        "limitations": [
            (
                "ABF enumeration assumes at most "
                "one causal variant per trait in "
                "the analyzed region."
            ),
            (
                "Unpowered comparisons are retained "
                "for transparency and sensitivity "
                "but excluded from the headline."
            ),
            (
                "PP.H4 depends on the shared-causal "
                "prior p12; prior sensitivity is "
                "reported on an unchanged locus "
                "universe."
            ),
        ],
        "completed_at_utc": utc_now(),
    }

    summary_path = (
        args.output_dir
        / "15e_coloc_summary.json"
    )
    atomic_json_write(
        summary,
        summary_path,
    )

    primary_summary = summaries[
        prior_key(
            args.primary_p12
        )
    ]["powered_both_only"]

    print()
    print("=" * 78)
    print(
        "STEP 15E — CROSS-ANCESTRY "
        "COLOCALIZATION"
    )
    print("=" * 78)
    print("Status:", status)
    print(
        "Valid comparison caches:",
        len(valid_uids),
        "/",
        len(manifest),
    )
    print(
        "Invalid or missing caches:",
        len(invalid_or_missing),
    )
    print(
        "Powered comparisons:",
        primary_summary[
            "n_comparisons"
        ],
    )
    print(
        "Headline powered units:",
        primary_summary[
            "n_units"
        ],
    )
    pp_h4 = primary_summary[
        "pp_h4_unit_median"
    ]
    if pp_h4["median"] is not None:
        print(
            "Headline median PP.H4:",
            f"{pp_h4['median']:.4f}",
            (
                f"(95% bootstrap interval "
                f"{pp_h4['ci_lower']:.4f}, "
                f"{pp_h4['ci_upper']:.4f})"
            ),
        )
    print(
        "Headline units with median "
        "PP.H4 >= 0.8:",
        primary_summary[
            "n_units_h4_ge_0p8"
        ],
    )
    print(
        "Summary:",
        summary_path,
    )
    print("=" * 78)

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
