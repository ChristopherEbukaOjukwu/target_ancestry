#!/usr/bin/env python3
"""
Step 15B4 — Harmonize variants, apply QC, and build feasibility tables.

This step standardizes each of the 1,062 locked EUR-versus-non-EUR
comparisons. It does not calculate portability slopes, LD-pruned estimates,
colocalization probabilities, or approval models.

Primary rules
-------------
Variant:
    autosomal or sex-chromosome biallelic SNP represented by one of A/C/G/T;
    duplicated CHR/POS/REF/ALT keys are excluded.

Statistics:
    finite beta and standard error in both ancestries;
    standard error > 0;
    valid association p-value representation.

Frequency:
    valid alternate-allele frequency in both ancestries;
    MAF >= 1% in both ancestries.

Pan-UKB quality:
    low_confidence_{population} must be explicitly false in EUR and the
    comparison ancestry.

GBMI harmonization:
    primary: exact normalized CHR/POS/REF/ALT match;
    sensitivity: exact plus explicit REF/ALT swaps;
    swapped non-EUR beta is multiplied by -1 and AF becomes 1-AF;
    complement/strand-only matching is not performed;
    is_diff_AF_gnomAD must be explicitly false in both source files.

Feasibility gates
-----------------
Portability pre-LD:
    at least 3 shared QC-qualified EUR genome-wide-significant variants.
    This is provisional. Final eligibility is determined after LD processing.

Colocalization input:
    at least 50 shared QC-qualified variants plus usable sample-size metadata.

Powered colocalization flag:
    at least one QC-qualified genome-wide-significant variant in each ancestry.

Inputs
------
output/15b3_comparison_extraction_map.parquet
output/15b3_extraction_summary.json
output/15a_trait_catalog.csv
intermediate/15b3/Pan-UKB/*.parquet
intermediate/15b3/GBMI/*.parquet

Outputs
-------
intermediate/15b4/comparisons/*.parquet
intermediate/15b4/comparisons/*.json
output/15b4_qc_feasibility.csv
output/15b4_qc_feasibility.parquet
output/15b4_variant_funnel.csv
output/15b4_variant_funnel.parquet
output/15b4_failures.csv
output/15b4_qc_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "15B4.1"
GWS_P = 5e-8
GWS_NEGLOG10 = -math.log10(GWS_P)
VALID_ALLELES = {"A", "C", "G", "T"}

PANUKB = "Pan-UKB"
GBMI = "GBMI"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comparison-map",
        type=Path,
        default=Path(
            "output/15b3_comparison_extraction_map.parquet"
        ),
    )
    parser.add_argument(
        "--extraction-summary",
        type=Path,
        default=Path("output/15b3_extraction_summary.json"),
    )
    parser.add_argument(
        "--trait-catalog",
        type=Path,
        default=Path("output/15a_trait_catalog.csv"),
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=Path("intermediate/15b4/comparisons"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
    )
    parser.add_argument(
        "--maf-threshold",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--min-portability-variants",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--min-coloc-variants",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )
    parser.add_argument(
        "--max-comparisons",
        type=int,
        default=0,
        help="Debugging limit; zero processes all comparisons.",
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
    return clean(value).casefold() in {"true", "1", "yes", "y"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_column(value: Any) -> str:
    text = clean(value).casefold().lstrip("#")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def normalize_chrom(value: Any) -> str:
    chrom = clean(value)
    if chrom.casefold().startswith("chr"):
        chrom = chrom[3:]
    if chrom == "M":
        chrom = "MT"
    return chrom


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_parquet_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(
        temp,
        index=False,
        compression="zstd",
    )
    temp.replace(path)


def atomic_json_write(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def column_lookup(frame: pd.DataFrame) -> dict[str, str]:
    return {
        normalize_column(column): column
        for column in frame.columns
    }


def require_column(
    lookup: dict[str, str],
    name: str,
) -> str:
    normalized = normalize_column(name)
    if normalized not in lookup:
        raise RuntimeError(f"Required column missing: {name}")
    return lookup[normalized]


def optional_column(
    lookup: dict[str, str],
    *names: str,
) -> str:
    for name in names:
        normalized = normalize_column(name)
        if normalized in lookup:
            return lookup[normalized]
    return ""


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_boolean_series(series: pd.Series) -> pd.Series:
    true_values = {
        "true",
        "t",
        "1",
        "yes",
        "y",
    }
    false_values = {
        "false",
        "f",
        "0",
        "no",
        "n",
    }

    def convert(value: Any):
        text = clean(value).casefold()
        if text in true_values:
            return True
        if text in false_values:
            return False
        return pd.NA

    return series.map(convert).astype("boolean")


def valid_snp_mask(
    chrom: pd.Series,
    pos: pd.Series,
    ref: pd.Series,
    alt: pd.Series,
) -> pd.Series:
    normalized_chrom = chrom.map(normalize_chrom)
    numeric_pos = numeric(pos)
    ref_upper = ref.astype("string").str.upper()
    alt_upper = alt.astype("string").str.upper()

    return (
        normalized_chrom.ne("")
        & numeric_pos.notna()
        & numeric_pos.gt(0)
        & ref_upper.isin(VALID_ALLELES)
        & alt_upper.isin(VALID_ALLELES)
        & ref_upper.ne(alt_upper)
    )


def valid_p_mask(series: pd.Series) -> pd.Series:
    values = numeric(series)
    return values.notna() & values.ge(0) & values.le(1)


def valid_neglog10_p_mask(series: pd.Series) -> pd.Series:
    values = numeric(series)
    return values.notna() & values.ge(0)


def valid_beta_se_mask(
    beta: pd.Series,
    se: pd.Series,
) -> pd.Series:
    beta_num = numeric(beta)
    se_num = numeric(se)
    return (
        beta_num.notna()
        & se_num.notna()
        & np.isfinite(beta_num)
        & np.isfinite(se_num)
        & se_num.gt(0)
    )


def valid_af_mask(series: pd.Series) -> pd.Series:
    values = numeric(series)
    return (
        values.notna()
        & np.isfinite(values)
        & values.gt(0)
        & values.lt(1)
    )


def maf(series: pd.Series) -> pd.Series:
    values = numeric(series)
    return pd.concat(
        [values, 1.0 - values],
        axis=1,
    ).min(axis=1)


def safe_p_from_neglog10(series: pd.Series) -> pd.Series:
    values = numeric(series)
    clipped = values.clip(lower=0, upper=323)
    result = pd.Series(
        np.power(10.0, -clipped),
        index=series.index,
        dtype="float64",
    )
    result[values.isna()] = np.nan
    return result


def variant_key_columns(
    frame: pd.DataFrame,
    *,
    chrom: str,
    pos: str,
    ref: str,
    alt: str,
) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    result["chrom"] = frame[chrom].map(normalize_chrom)
    result["pos"] = numeric(frame[pos]).astype("Int64")
    result["ref"] = (
        frame[ref].astype("string").str.upper()
    )
    result["alt"] = (
        frame[alt].astype("string").str.upper()
    )
    result["variant_id"] = (
        result["chrom"].astype("string")
        + ":"
        + result["pos"].astype("string")
        + ":"
        + result["ref"]
        + ":"
        + result["alt"]
    )
    return result


def remove_duplicate_keys(
    frame: pd.DataFrame,
    key_columns: list[str],
) -> tuple[pd.DataFrame, int]:
    duplicate = frame.duplicated(
        key_columns,
        keep=False,
    )
    return frame.loc[~duplicate].copy(), int(duplicate.sum())


def comparison_paths(
    directory: Path,
    comparison_uid: str,
) -> tuple[Path, Path]:
    return (
        directory / f"{comparison_uid}.parquet",
        directory / f"{comparison_uid}.json",
    )


def cache_valid(
    parquet_path: Path,
    sidecar_path: Path,
    comparison_uid: str,
) -> bool:
    if not parquet_path.exists() or not sidecar_path.exists():
        return False
    try:
        sidecar = read_json(sidecar_path)
    except Exception:
        return False

    return (
        clean(sidecar.get("comparison_uid"))
        == comparison_uid
        and clean(sidecar.get("status")) == "PASS"
        and int(sidecar.get("output_size_bytes", -1))
        == parquet_path.stat().st_size
        and clean(sidecar.get("script_version"))
        == SCRIPT_VERSION
    )


def get_trait_row(
    catalog: pd.DataFrame,
    trait_key: str,
) -> pd.Series | None:
    matches = catalog[
        catalog["trait_key"].astype(str).eq(trait_key)
    ]
    if len(matches) == 1:
        return matches.iloc[0]
    return None


def trait_value(
    row: pd.Series | None,
    *candidate_names: str,
) -> float | None:
    if row is None:
        return None

    normalized = {
        normalize_column(column): column
        for column in row.index
    }
    for name in candidate_names:
        column = normalized.get(normalize_column(name))
        if not column:
            continue
        value = pd.to_numeric(
            pd.Series([row[column]]),
            errors="coerce",
        ).iloc[0]
        if pd.notna(value):
            return float(value)
    return None


def panukb_sample_metadata(
    trait_row: pd.Series | None,
    population: str,
) -> dict[str, Any]:
    population = population.upper()
    n_cases = trait_value(
        trait_row,
        f"n_cases_{population}",
    )
    n_controls = trait_value(
        trait_row,
        f"n_controls_{population}",
    )

    trait_type = ""
    if trait_row is not None:
        for candidate in ["trait_type", "phenotype_type"]:
            if candidate in trait_row.index:
                trait_type = clean(trait_row[candidate])
                break

    quantitative = (
        trait_type.casefold()
        in {"continuous", "biomarkers", "biomarker"}
        or (
            n_cases is not None
            and (n_controls is None or n_controls == 0)
        )
    )

    if quantitative:
        n_total = n_cases
        case_fraction = None
        usable = n_total is not None and n_total > 0
        coloc_type = "quant"
    else:
        n_total = (
            n_cases + n_controls
            if n_cases is not None
            and n_controls is not None
            else None
        )
        case_fraction = (
            n_cases / n_total
            if n_total is not None and n_total > 0
            else None
        )
        usable = (
            n_total is not None
            and n_total > 0
            and case_fraction is not None
            and 0 < case_fraction < 1
        )
        coloc_type = "cc"

    return {
        "trait_type": trait_type,
        "coloc_type": coloc_type,
        "n_cases": n_cases,
        "n_controls": n_controls,
        "n_total": n_total,
        "case_fraction": case_fraction,
        "usable": bool(usable),
    }


def summarize_per_variant_sample_metadata(
    frame: pd.DataFrame,
    cases_column: str,
    controls_column: str,
    mask: pd.Series,
) -> dict[str, Any]:
    cases = numeric(frame[cases_column])
    controls = numeric(frame[controls_column])
    total = cases + controls
    selected = mask & cases.gt(0) & controls.gt(0)

    if not selected.any():
        return {
            "n_cases_median": None,
            "n_controls_median": None,
            "n_total_median": None,
            "case_fraction_median": None,
            "usable": False,
        }

    selected_cases = cases[selected]
    selected_controls = controls[selected]
    selected_total = total[selected]
    fractions = selected_cases / selected_total

    return {
        "n_cases_median": float(selected_cases.median()),
        "n_controls_median": float(
            selected_controls.median()
        ),
        "n_total_median": float(selected_total.median()),
        "case_fraction_median": float(fractions.median()),
        "usable": bool(
            selected_total.gt(0).all()
            and fractions.gt(0).all()
            and fractions.lt(1).all()
        ),
    }


def process_panukb(
    comparison: pd.Series,
    source_frame: pd.DataFrame,
    trait_catalog: pd.DataFrame,
    *,
    maf_threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    population = clean(
        comparison["comparison_population"]
    ).upper()
    lookup = column_lookup(source_frame)

    chrom_col = require_column(lookup, "chr")
    pos_col = require_column(lookup, "pos")
    ref_col = require_column(lookup, "ref")
    alt_col = require_column(lookup, "alt")

    fields = {}
    for pop in ["EUR", population]:
        fields[pop] = {
            "af": require_column(lookup, f"af_{pop}"),
            "beta": require_column(
                lookup,
                f"beta_{pop}",
            ),
            "se": require_column(lookup, f"se_{pop}"),
            "neglog10_p": require_column(
                lookup,
                f"neglog10_pval_{pop}",
            ),
            "low_confidence": require_column(
                lookup,
                f"low_confidence_{pop}",
            ),
        }

    keys = variant_key_columns(
        source_frame,
        chrom=chrom_col,
        pos=pos_col,
        ref=ref_col,
        alt=alt_col,
    )

    frame = keys.copy()
    frame["eur_beta"] = numeric(
        source_frame[fields["EUR"]["beta"]]
    )
    frame["eur_se"] = numeric(
        source_frame[fields["EUR"]["se"]]
    )
    frame["eur_af"] = numeric(
        source_frame[fields["EUR"]["af"]]
    )
    frame["eur_neglog10_p"] = numeric(
        source_frame[fields["EUR"]["neglog10_p"]]
    )
    frame["eur_p"] = safe_p_from_neglog10(
        source_frame[fields["EUR"]["neglog10_p"]]
    )
    frame["eur_low_confidence"] = parse_boolean_series(
        source_frame[fields["EUR"]["low_confidence"]]
    )

    frame["comparison_beta"] = numeric(
        source_frame[fields[population]["beta"]]
    )
    frame["comparison_se"] = numeric(
        source_frame[fields[population]["se"]]
    )
    frame["comparison_af"] = numeric(
        source_frame[fields[population]["af"]]
    )
    frame["comparison_neglog10_p"] = numeric(
        source_frame[
            fields[population]["neglog10_p"]
        ]
    )
    frame["comparison_p"] = safe_p_from_neglog10(
        source_frame[
            fields[population]["neglog10_p"]
        ]
    )
    frame["comparison_low_confidence"] = (
        parse_boolean_series(
            source_frame[
                fields[population]["low_confidence"]
            ]
        )
    )

    frame["allele_relation"] = "exact"
    frame["primary_harmonized"] = True
    frame["sensitivity_harmonized"] = True

    frame["valid_biallelic_snp"] = valid_snp_mask(
        frame["chrom"],
        frame["pos"],
        frame["ref"],
        frame["alt"],
    )

    frame, n_duplicate_rows = remove_duplicate_keys(
        frame,
        ["chrom", "pos", "ref", "alt"],
    )

    frame["valid_beta_se"] = (
        valid_beta_se_mask(
            frame["eur_beta"],
            frame["eur_se"],
        )
        & valid_beta_se_mask(
            frame["comparison_beta"],
            frame["comparison_se"],
        )
    )
    frame["valid_p"] = (
        valid_neglog10_p_mask(
            frame["eur_neglog10_p"]
        )
        & valid_neglog10_p_mask(
            frame["comparison_neglog10_p"]
        )
    )
    frame["valid_af"] = (
        valid_af_mask(frame["eur_af"])
        & valid_af_mask(frame["comparison_af"])
    )
    frame["eur_maf"] = maf(frame["eur_af"])
    frame["comparison_maf"] = maf(
        frame["comparison_af"]
    )
    frame["maf_pass"] = (
        frame["valid_af"]
        & frame["eur_maf"].ge(maf_threshold)
        & frame["comparison_maf"].ge(maf_threshold)
    )
    frame["quality_pass"] = (
        frame["eur_low_confidence"].eq(False)
        & frame["comparison_low_confidence"].eq(False)
    ).fillna(False)

    frame["qc_qualified"] = (
        frame["valid_biallelic_snp"]
        & frame["primary_harmonized"]
        & frame["valid_beta_se"]
        & frame["valid_p"]
        & frame["maf_pass"]
        & frame["quality_pass"]
    )

    frame["eur_gws"] = (
        frame["qc_qualified"]
        & frame["eur_neglog10_p"].ge(GWS_NEGLOG10)
    )
    frame["comparison_gws"] = (
        frame["qc_qualified"]
        & frame["comparison_neglog10_p"].ge(
            GWS_NEGLOG10
        )
    )

    trait_row = get_trait_row(
        trait_catalog,
        clean(comparison["candidate_trait_key"]),
    )
    eur_sample = panukb_sample_metadata(
        trait_row,
        "EUR",
    )
    comparison_sample = panukb_sample_metadata(
        trait_row,
        population,
    )

    metrics = {
        "n_raw_eur_variants": int(len(source_frame)),
        "n_raw_comparison_variants": int(
            len(source_frame)
        ),
        "n_raw_shared_rows": int(len(source_frame)),
        "n_duplicate_key_rows_excluded": (
            n_duplicate_rows
        ),
        "n_harmonized_exact": int(len(frame)),
        "n_harmonized_swapped": 0,
        "n_harmonized_primary": int(
            frame["primary_harmonized"].sum()
        ),
        "n_harmonized_sensitivity": int(
            frame["sensitivity_harmonized"].sum()
        ),
        "n_valid_biallelic_snp": int(
            frame["valid_biallelic_snp"].sum()
        ),
        "n_valid_beta_se": int(
            frame["valid_beta_se"].sum()
        ),
        "n_valid_p": int(frame["valid_p"].sum()),
        "n_valid_af": int(frame["valid_af"].sum()),
        "n_maf_pass": int(frame["maf_pass"].sum()),
        "n_quality_pass": int(
            frame["quality_pass"].sum()
        ),
        "n_qc_qualified": int(
            frame["qc_qualified"].sum()
        ),
        "n_eur_gws_qc": int(frame["eur_gws"].sum()),
        "n_comparison_gws_qc": int(
            frame["comparison_gws"].sum()
        ),
        "eur_sample_metadata": eur_sample,
        "comparison_sample_metadata": (
            comparison_sample
        ),
        "sample_metadata_usable": bool(
            eur_sample["usable"]
            and comparison_sample["usable"]
        ),
        "gbmi_swapped_sensitivity_only": False,
    }
    return frame, metrics


def prepare_gbmi_source(
    source_frame: pd.DataFrame,
    *,
    prefix: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    lookup = column_lookup(source_frame)

    chrom_col = require_column(lookup, "CHR")
    pos_col = require_column(lookup, "POS")
    ref_col = require_column(lookup, "REF")
    alt_col = require_column(lookup, "ALT")
    beta_col = require_column(
        lookup,
        "inv_var_meta_beta",
    )
    se_col = require_column(
        lookup,
        "inv_var_meta_sebeta",
    )
    p_col = require_column(lookup, "inv_var_meta_p")
    af_col = require_column(lookup, "all_meta_AF")
    cases_col = require_column(lookup, "N_case")
    controls_col = require_column(lookup, "N_ctrl")
    diff_af_col = require_column(
        lookup,
        "is_diff_AF_gnomAD",
    )
    strand_flip_col = require_column(
        lookup,
        "is_strand_flip",
    )

    keys = variant_key_columns(
        source_frame,
        chrom=chrom_col,
        pos=pos_col,
        ref=ref_col,
        alt=alt_col,
    )
    frame = keys.copy()
    frame[f"{prefix}_beta_source"] = numeric(
        source_frame[beta_col]
    )
    frame[f"{prefix}_se"] = numeric(source_frame[se_col])
    frame[f"{prefix}_p"] = numeric(source_frame[p_col])
    frame[f"{prefix}_af_source"] = numeric(
        source_frame[af_col]
    )
    frame[f"{prefix}_n_cases"] = numeric(
        source_frame[cases_col]
    )
    frame[f"{prefix}_n_controls"] = numeric(
        source_frame[controls_col]
    )
    frame[f"{prefix}_diff_af_gnomad"] = (
        parse_boolean_series(source_frame[diff_af_col])
    )
    frame[f"{prefix}_strand_flip_flag"] = (
        parse_boolean_series(
            source_frame[strand_flip_col]
        )
    )

    frame["valid_biallelic_snp_source"] = valid_snp_mask(
        frame["chrom"],
        frame["pos"],
        frame["ref"],
        frame["alt"],
    )

    frame, duplicate_rows = remove_duplicate_keys(
        frame,
        ["chrom", "pos", "ref", "alt"],
    )

    metadata = {
        "n_raw_rows": int(len(source_frame)),
        "n_duplicate_key_rows_excluded": duplicate_rows,
    }
    return frame, metadata


def process_gbmi(
    comparison: pd.Series,
    eur_source: pd.DataFrame,
    comparison_source: pd.DataFrame,
    *,
    maf_threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    eur, eur_meta = prepare_gbmi_source(
        eur_source,
        prefix="eur",
    )
    other, other_meta = prepare_gbmi_source(
        comparison_source,
        prefix="comparison",
    )

    eur_columns = [
        "chrom",
        "pos",
        "ref",
        "alt",
        "variant_id",
        "valid_biallelic_snp_source",
        "eur_beta_source",
        "eur_se",
        "eur_p",
        "eur_af_source",
        "eur_n_cases",
        "eur_n_controls",
        "eur_diff_af_gnomad",
        "eur_strand_flip_flag",
    ]
    other_columns = [
        "chrom",
        "pos",
        "ref",
        "alt",
        "variant_id",
        "valid_biallelic_snp_source",
        "comparison_beta_source",
        "comparison_se",
        "comparison_p",
        "comparison_af_source",
        "comparison_n_cases",
        "comparison_n_controls",
        "comparison_diff_af_gnomad",
        "comparison_strand_flip_flag",
    ]

    exact = eur[eur_columns].merge(
        other[other_columns],
        on=["chrom", "pos", "ref", "alt"],
        how="inner",
        suffixes=("_eur_key", "_comparison_key"),
        validate="one_to_one",
    )
    exact["allele_relation"] = "exact"

    exact_eur_keys = set(
        zip(
            exact["chrom"],
            exact["pos"],
            exact["ref"],
            exact["alt"],
        )
    )
    exact_other_keys = exact_eur_keys.copy()

    eur_remaining = eur[
        ~eur.apply(
            lambda row: (
                row["chrom"],
                row["pos"],
                row["ref"],
                row["alt"],
            )
            in exact_eur_keys,
            axis=1,
        )
    ].copy()
    other_remaining = other[
        ~other.apply(
            lambda row: (
                row["chrom"],
                row["pos"],
                row["ref"],
                row["alt"],
            )
            in exact_other_keys,
            axis=1,
        )
    ].copy()

    swapped_other = other_remaining.rename(
        columns={
            "ref": "alt",
            "alt": "ref",
        }
    )
    swapped = eur_remaining[eur_columns].merge(
        swapped_other[other_columns],
        on=["chrom", "pos", "ref", "alt"],
        how="inner",
        suffixes=("_eur_key", "_comparison_key"),
        validate="one_to_one",
    )
    swapped["allele_relation"] = "swapped"

    frame = pd.concat(
        [exact, swapped],
        ignore_index=True,
        sort=False,
    )

    if frame.empty:
        # Return an empty, schema-stable result.
        frame = pd.DataFrame(
            columns=[
                "chrom",
                "pos",
                "ref",
                "alt",
                "variant_id",
                "allele_relation",
            ]
        )

    frame["variant_id"] = (
        frame["chrom"].astype("string")
        + ":"
        + frame["pos"].astype("string")
        + ":"
        + frame["ref"].astype("string")
        + ":"
        + frame["alt"].astype("string")
    )
    frame["primary_harmonized"] = frame[
        "allele_relation"
    ].eq("exact")
    frame["sensitivity_harmonized"] = frame[
        "allele_relation"
    ].isin(["exact", "swapped"])

    frame["eur_beta"] = frame["eur_beta_source"]
    frame["eur_af"] = frame["eur_af_source"]
    frame["comparison_beta"] = np.where(
        frame["allele_relation"].eq("swapped"),
        -numeric(frame["comparison_beta_source"]),
        numeric(frame["comparison_beta_source"]),
    )
    frame["comparison_af"] = np.where(
        frame["allele_relation"].eq("swapped"),
        1.0 - numeric(frame["comparison_af_source"]),
        numeric(frame["comparison_af_source"]),
    )

    frame["valid_biallelic_snp"] = (
        valid_snp_mask(
            frame["chrom"],
            frame["pos"],
            frame["ref"],
            frame["alt"],
        )
        & frame["valid_biallelic_snp_source_eur_key"]
        .fillna(False)
        .astype(bool)
        & frame[
            "valid_biallelic_snp_source_comparison_key"
        ]
        .fillna(False)
        .astype(bool)
    )
    frame["valid_beta_se"] = (
        valid_beta_se_mask(
            frame["eur_beta"],
            frame["eur_se"],
        )
        & valid_beta_se_mask(
            frame["comparison_beta"],
            frame["comparison_se"],
        )
    )
    frame["valid_p"] = (
        valid_p_mask(frame["eur_p"])
        & valid_p_mask(frame["comparison_p"])
    )
    frame["valid_af"] = (
        valid_af_mask(frame["eur_af"])
        & valid_af_mask(frame["comparison_af"])
    )
    frame["eur_maf"] = maf(frame["eur_af"])
    frame["comparison_maf"] = maf(
        frame["comparison_af"]
    )
    frame["maf_pass"] = (
        frame["valid_af"]
        & frame["eur_maf"].ge(maf_threshold)
        & frame["comparison_maf"].ge(maf_threshold)
    )

    frame["quality_pass"] = (
        frame["eur_diff_af_gnomad"].eq(False)
        & frame["comparison_diff_af_gnomad"].eq(False)
    ).fillna(False)

    frame["qc_qualified"] = (
        frame["valid_biallelic_snp"]
        & frame["primary_harmonized"]
        & frame["valid_beta_se"]
        & frame["valid_p"]
        & frame["maf_pass"]
        & frame["quality_pass"]
    )
    frame["qc_qualified_sensitivity"] = (
        frame["valid_biallelic_snp"]
        & frame["sensitivity_harmonized"]
        & frame["valid_beta_se"]
        & frame["valid_p"]
        & frame["maf_pass"]
        & frame["quality_pass"]
    )

    frame["eur_gws"] = (
        frame["qc_qualified"]
        & frame["eur_p"].le(GWS_P)
    )
    frame["comparison_gws"] = (
        frame["qc_qualified"]
        & frame["comparison_p"].le(GWS_P)
    )
    frame["eur_gws_sensitivity"] = (
        frame["qc_qualified_sensitivity"]
        & frame["eur_p"].le(GWS_P)
    )
    frame["comparison_gws_sensitivity"] = (
        frame["qc_qualified_sensitivity"]
        & frame["comparison_p"].le(GWS_P)
    )

    eur_sample = summarize_per_variant_sample_metadata(
        frame,
        "eur_n_cases",
        "eur_n_controls",
        frame["qc_qualified"],
    )
    comparison_sample = (
        summarize_per_variant_sample_metadata(
            frame,
            "comparison_n_cases",
            "comparison_n_controls",
            frame["qc_qualified"],
        )
    )

    metrics = {
        "n_raw_eur_variants": eur_meta["n_raw_rows"],
        "n_raw_comparison_variants": (
            other_meta["n_raw_rows"]
        ),
        "n_raw_shared_rows": None,
        "n_duplicate_key_rows_excluded": (
            eur_meta["n_duplicate_key_rows_excluded"]
            + other_meta[
                "n_duplicate_key_rows_excluded"
            ]
        ),
        "n_harmonized_exact": int(
            frame["allele_relation"].eq("exact").sum()
        ),
        "n_harmonized_swapped": int(
            frame["allele_relation"].eq("swapped").sum()
        ),
        "n_harmonized_primary": int(
            frame["primary_harmonized"].sum()
        ),
        "n_harmonized_sensitivity": int(
            frame["sensitivity_harmonized"].sum()
        ),
        "n_valid_biallelic_snp": int(
            frame["valid_biallelic_snp"].sum()
        ),
        "n_valid_beta_se": int(
            frame["valid_beta_se"].sum()
        ),
        "n_valid_p": int(frame["valid_p"].sum()),
        "n_valid_af": int(frame["valid_af"].sum()),
        "n_maf_pass": int(frame["maf_pass"].sum()),
        "n_quality_pass": int(
            frame["quality_pass"].sum()
        ),
        "n_qc_qualified": int(
            frame["qc_qualified"].sum()
        ),
        "n_qc_qualified_sensitivity": int(
            frame["qc_qualified_sensitivity"].sum()
        ),
        "n_eur_gws_qc": int(frame["eur_gws"].sum()),
        "n_comparison_gws_qc": int(
            frame["comparison_gws"].sum()
        ),
        "n_eur_gws_qc_sensitivity": int(
            frame["eur_gws_sensitivity"].sum()
        ),
        "n_comparison_gws_qc_sensitivity": int(
            frame["comparison_gws_sensitivity"].sum()
        ),
        "eur_sample_metadata": {
            "trait_type": "binary",
            "coloc_type": "cc",
            **eur_sample,
        },
        "comparison_sample_metadata": {
            "trait_type": "binary",
            "coloc_type": "cc",
            **comparison_sample,
        },
        "sample_metadata_usable": bool(
            eur_sample["usable"]
            and comparison_sample["usable"]
        ),
        "gbmi_swapped_sensitivity_only": True,
        "n_eur_strand_flip_flagged": int(
            frame["eur_strand_flip_flag"].eq(True).sum()
        ),
        "n_comparison_strand_flip_flagged": int(
            frame[
                "comparison_strand_flip_flag"
            ].eq(True).sum()
        ),
        "n_eur_diff_af_gnomad_flagged": int(
            frame["eur_diff_af_gnomad"].eq(True).sum()
        ),
        "n_comparison_diff_af_gnomad_flagged": int(
            frame[
                "comparison_diff_af_gnomad"
            ].eq(True).sum()
        ),
    }
    return frame, metrics


def portability_failure_reason(
    metrics: dict[str, Any],
    *,
    min_variants: int,
) -> str:
    reasons = []
    if metrics["n_harmonized_primary"] == 0:
        reasons.append("no_primary_harmonized_variants")
    if metrics["n_qc_qualified"] == 0:
        reasons.append("no_qc_qualified_variants")
    if metrics["n_eur_gws_qc"] < min_variants:
        reasons.append(
            "fewer_than_"
            f"{min_variants}_qualified_EUR_GWS_variants"
        )
    return " | ".join(reasons)


def coloc_failure_reason(
    metrics: dict[str, Any],
    *,
    min_variants: int,
) -> str:
    reasons = []
    if metrics["n_qc_qualified"] < min_variants:
        reasons.append(
            "fewer_than_"
            f"{min_variants}_qualified_shared_variants"
        )
    if not metrics["sample_metadata_usable"]:
        reasons.append("sample_metadata_unavailable")
    return " | ".join(reasons)


def flatten_sample_metadata(
    prefix: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        f"{prefix}_{key}": value
        for key, value in metadata.items()
    }


def feasibility_row(
    comparison: pd.Series,
    metrics: dict[str, Any],
    *,
    maf_threshold: float,
    min_portability_variants: int,
    min_coloc_variants: int,
    output_parquet: Path,
    output_sidecar: Path,
    cache_action: str,
) -> dict[str, Any]:
    portability_eligible = (
        metrics["n_eur_gws_qc"]
        >= min_portability_variants
    )
    coloc_eligible = (
        metrics["n_qc_qualified"]
        >= min_coloc_variants
        and metrics["sample_metadata_usable"]
    )
    powered_both = (
        metrics["n_eur_gws_qc"] >= 1
        and metrics["n_comparison_gws_qc"] >= 1
    )

    row = {
        "comparison_uid": clean(
            comparison["comparison_uid"]
        ),
        "gene_trait_uid": clean(
            comparison["gene_trait_uid"]
        ),
        "gene": clean(comparison["gene"]),
        "candidate_source": clean(
            comparison["candidate_source"]
        ),
        "candidate_trait_key": clean(
            comparison["candidate_trait_key"]
        ),
        "candidate_trait_name": clean(
            comparison["candidate_trait_name"]
        ),
        "comparison_population": clean(
            comparison["comparison_population"]
        ),
        "unit_analysis_role": clean(
            comparison["unit_analysis_role"]
        ),
        "mixed_pool": as_bool(comparison["mixed_pool"]),
        "primary_approval_eligible": as_bool(
            comparison["primary_approval_eligible"]
        ),
        "maf_threshold": maf_threshold,
        "gws_p_threshold": GWS_P,
        "min_portability_variants_pre_ld": (
            min_portability_variants
        ),
        "min_coloc_variants": min_coloc_variants,
        **{
            key: value
            for key, value in metrics.items()
            if key
            not in {
                "eur_sample_metadata",
                "comparison_sample_metadata",
            }
        },
        **flatten_sample_metadata(
            "eur",
            metrics["eur_sample_metadata"],
        ),
        **flatten_sample_metadata(
            "comparison",
            metrics["comparison_sample_metadata"],
        ),
        "portability_pre_ld_eligible": (
            portability_eligible
        ),
        "portability_failure_reason": (
            portability_failure_reason(
                metrics,
                min_variants=min_portability_variants,
            )
        ),
        "coloc_input_eligible": coloc_eligible,
        "coloc_failure_reason": coloc_failure_reason(
            metrics,
            min_variants=min_coloc_variants,
        ),
        "coloc_powered_both_ancestries": powered_both,
        "harmonized_parquet": str(output_parquet),
        "harmonized_sidecar": str(output_sidecar),
        "cache_action": cache_action,
    }
    return row


def sidecar_from_feasibility(
    row: dict[str, Any],
    output_parquet: Path,
) -> dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        **row,
        "output_size_bytes": output_parquet.stat().st_size,
        "output_sha256": sha256_file(output_parquet),
        "completed_at_utc": utc_now(),
    }


def process_comparison(
    comparison: pd.Series,
    trait_catalog: pd.DataFrame,
    *,
    comparison_dir: Path,
    maf_threshold: float,
    min_portability_variants: int,
    min_coloc_variants: int,
    force: bool,
) -> dict[str, Any]:
    comparison_uid = clean(comparison["comparison_uid"])
    output_parquet, output_sidecar = comparison_paths(
        comparison_dir,
        comparison_uid,
    )

    if (
        not force
        and cache_valid(
            output_parquet,
            output_sidecar,
            comparison_uid,
        )
    ):
        sidecar = read_json(output_sidecar)
        sidecar["cache_action"] = "REUSED"
        return {
            key: value
            for key, value in sidecar.items()
            if key
            not in {
                "script_version",
                "status",
                "output_size_bytes",
                "output_sha256",
                "completed_at_utc",
            }
        }

    eur_path = Path(
        clean(comparison["eur_extraction_parquet"])
    )
    comparison_path = Path(
        clean(
            comparison["comparison_extraction_parquet"]
        )
    )
    if not eur_path.exists():
        raise RuntimeError(
            f"EUR extraction missing: {eur_path}"
        )
    if not comparison_path.exists():
        raise RuntimeError(
            "Comparison extraction missing: "
            f"{comparison_path}"
        )

    source = clean(comparison["candidate_source"])
    if source == PANUKB:
        source_frame = pd.read_parquet(eur_path)
        harmonized, metrics = process_panukb(
            comparison,
            source_frame,
            trait_catalog,
            maf_threshold=maf_threshold,
        )
    elif source == GBMI:
        eur_source = pd.read_parquet(eur_path)
        comparison_source = pd.read_parquet(
            comparison_path
        )
        harmonized, metrics = process_gbmi(
            comparison,
            eur_source,
            comparison_source,
            maf_threshold=maf_threshold,
        )
    else:
        raise RuntimeError(
            f"Unsupported source: {source}"
        )

    metadata_columns = {
        "comparison_uid": comparison_uid,
        "gene_trait_uid": clean(
            comparison["gene_trait_uid"]
        ),
        "gene": clean(comparison["gene"]),
        "candidate_source": source,
        "candidate_trait_key": clean(
            comparison["candidate_trait_key"]
        ),
        "candidate_trait_name": clean(
            comparison["candidate_trait_name"]
        ),
        "comparison_population": clean(
            comparison["comparison_population"]
        ),
        "unit_analysis_role": clean(
            comparison["unit_analysis_role"]
        ),
        "mixed_pool": as_bool(comparison["mixed_pool"]),
        "primary_approval_eligible": as_bool(
            comparison["primary_approval_eligible"]
        ),
    }
    for column, value in reversed(
        list(metadata_columns.items())
    ):
        harmonized.insert(0, column, value)

    atomic_parquet_write(
        harmonized,
        output_parquet,
    )

    row = feasibility_row(
        comparison,
        metrics,
        maf_threshold=maf_threshold,
        min_portability_variants=(
            min_portability_variants
        ),
        min_coloc_variants=min_coloc_variants,
        output_parquet=output_parquet,
        output_sidecar=output_sidecar,
        cache_action="PROCESSED_NOW",
    )
    sidecar = sidecar_from_feasibility(
        row,
        output_parquet,
    )
    atomic_json_write(sidecar, output_sidecar)
    return row


def funnel_table(feasibility: pd.DataFrame) -> pd.DataFrame:
    stages = [
        (
            "raw_eur_variants",
            "n_raw_eur_variants",
        ),
        (
            "raw_comparison_variants",
            "n_raw_comparison_variants",
        ),
        (
            "primary_harmonized",
            "n_harmonized_primary",
        ),
        (
            "valid_biallelic_snp",
            "n_valid_biallelic_snp",
        ),
        (
            "valid_beta_se",
            "n_valid_beta_se",
        ),
        (
            "maf_pass",
            "n_maf_pass",
        ),
        (
            "quality_pass",
            "n_quality_pass",
        ),
        (
            "qc_qualified",
            "n_qc_qualified",
        ),
        (
            "eur_gws_qc",
            "n_eur_gws_qc",
        ),
        (
            "comparison_gws_qc",
            "n_comparison_gws_qc",
        ),
    ]

    rows = []
    grouping_sets = [
        [],
        ["candidate_source"],
        ["candidate_source", "unit_analysis_role"],
        ["candidate_source", "comparison_population"],
    ]

    for grouping in grouping_sets:
        if grouping:
            grouped = feasibility.groupby(
                grouping,
                dropna=False,
            )
        else:
            grouped = [("ALL", feasibility)]

        for key, group in grouped:
            if not isinstance(key, tuple):
                key = (key,)
            group_values = {
                column: value
                for column, value in zip(grouping, key)
            }
            for stage_order, (
                stage,
                column,
            ) in enumerate(stages, start=1):
                values = pd.to_numeric(
                    group[column],
                    errors="coerce",
                )
                rows.append(
                    {
                        "grouping": (
                            "ALL"
                            if not grouping
                            else " + ".join(grouping)
                        ),
                        **{
                            column_name: group_values.get(
                                column_name,
                                "ALL",
                            )
                            for column_name in [
                                "candidate_source",
                                "unit_analysis_role",
                                "comparison_population",
                            ]
                        },
                        "stage_order": stage_order,
                        "stage": stage,
                        "metric_column": column,
                        "n_comparisons": int(len(group)),
                        "total_variants": int(
                            values.fillna(0).sum()
                        ),
                        "median_variants": float(
                            values.median()
                        ),
                        "minimum_variants": int(
                            values.min()
                        ),
                        "maximum_variants": int(
                            values.max()
                        ),
                        "n_comparisons_with_at_least_one": (
                            int(values.gt(0).sum())
                        ),
                    }
                )

    return pd.DataFrame(rows)


def write_outputs(
    feasibility: pd.DataFrame,
    failures: pd.DataFrame,
    *,
    output_dir: Path,
    n_expected: int,
    n_processed_in_run: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    feasibility = feasibility.sort_values(
        [
            "candidate_source",
            "candidate_trait_name",
            "gene",
            "comparison_population",
        ]
    )
    feasibility_csv = (
        output_dir / "15b4_qc_feasibility.csv"
    )
    feasibility_parquet = (
        output_dir / "15b4_qc_feasibility.parquet"
    )
    feasibility.to_csv(feasibility_csv, index=False)
    feasibility.to_parquet(
        feasibility_parquet,
        index=False,
        compression="zstd",
    )

    funnel = funnel_table(feasibility)
    funnel_csv = output_dir / "15b4_variant_funnel.csv"
    funnel_parquet = (
        output_dir / "15b4_variant_funnel.parquet"
    )
    funnel.to_csv(funnel_csv, index=False)
    funnel.to_parquet(
        funnel_parquet,
        index=False,
        compression="zstd",
    )

    failures_csv = output_dir / "15b4_failures.csv"
    failures.to_csv(failures_csv, index=False)

    source_summary = {}
    for source, group in feasibility.groupby(
        "candidate_source"
    ):
        source_summary[str(source)] = {
            "n_comparisons": int(len(group)),
            "n_portability_pre_ld_eligible": int(
                group[
                    "portability_pre_ld_eligible"
                ].sum()
            ),
            "n_coloc_input_eligible": int(
                group["coloc_input_eligible"].sum()
            ),
            "n_coloc_powered_both": int(
                group[
                    "coloc_powered_both_ancestries"
                ].sum()
            ),
            "median_qc_qualified_variants": float(
                group["n_qc_qualified"].median()
            ),
            "median_eur_gws_variants": float(
                group["n_eur_gws_qc"].median()
            ),
        }

    primary = feasibility[
        feasibility["unit_analysis_role"].eq("PRIMARY")
    ]
    primary_nonmixed = primary[
        ~primary["mixed_pool"].map(as_bool)
    ]

    summary = {
        "step": "15B4",
        "script_version": SCRIPT_VERSION,
        "status": (
            "PASS"
            if len(failures) == 0
            and len(feasibility) == n_expected
            else (
                "PARTIAL"
                if len(failures) == 0
                else "FAIL"
            )
        ),
        "n_expected_comparisons": n_expected,
        "n_feasibility_rows": int(len(feasibility)),
        "n_processed_in_this_run": n_processed_in_run,
        "n_failures": int(len(failures)),
        "thresholds": {
            "maf": args.maf_threshold,
            "gws_p": GWS_P,
            "portability_pre_ld_min_eur_gws": (
                args.min_portability_variants
            ),
            "coloc_min_shared_qc_variants": (
                args.min_coloc_variants
            ),
        },
        "primary_rules": {
            "variant_type": "biallelic SNP",
            "duplicate_variant_keys": "exclude all duplicates",
            "panukb_quality": (
                "low_confidence false in both ancestries"
            ),
            "gbmi_primary_match": (
                "exact normalized CHR/POS/REF/ALT"
            ),
            "gbmi_swapped_match": (
                "sensitivity only; non-EUR beta sign reversed "
                "and AF replaced by 1-AF"
            ),
            "gbmi_complement_matching": "not performed",
            "gbmi_quality": (
                "is_diff_AF_gnomAD false in both files"
            ),
        },
        "all_comparisons": {
            "n_portability_pre_ld_eligible": int(
                feasibility[
                    "portability_pre_ld_eligible"
                ].sum()
            ),
            "n_coloc_input_eligible": int(
                feasibility["coloc_input_eligible"].sum()
            ),
            "n_coloc_powered_both": int(
                feasibility[
                    "coloc_powered_both_ancestries"
                ].sum()
            ),
        },
        "primary_comparisons": {
            "n": int(len(primary)),
            "n_portability_pre_ld_eligible": int(
                primary[
                    "portability_pre_ld_eligible"
                ].sum()
            ),
            "n_coloc_input_eligible": int(
                primary["coloc_input_eligible"].sum()
            ),
            "n_coloc_powered_both": int(
                primary[
                    "coloc_powered_both_ancestries"
                ].sum()
            ),
        },
        "primary_nonmixed_comparisons": {
            "n": int(len(primary_nonmixed)),
            "n_portability_pre_ld_eligible": int(
                primary_nonmixed[
                    "portability_pre_ld_eligible"
                ].sum()
            ),
            "n_coloc_input_eligible": int(
                primary_nonmixed[
                    "coloc_input_eligible"
                ].sum()
            ),
            "n_coloc_powered_both": int(
                primary_nonmixed[
                    "coloc_powered_both_ancestries"
                ].sum()
            ),
        },
        "by_source": source_summary,
        "cache_actions": {
            str(key): int(value)
            for key, value in feasibility[
                "cache_action"
            ].value_counts().items()
        },
        "outputs": {
            "feasibility_csv": str(feasibility_csv),
            "feasibility_parquet": str(
                feasibility_parquet
            ),
            "variant_funnel_csv": str(funnel_csv),
            "variant_funnel_parquet": str(
                funnel_parquet
            ),
            "failures_csv": str(failures_csv),
        },
        "completed_at_utc": utc_now(),
    }

    summary_path = output_dir / "15b4_qc_summary.json"
    atomic_json_write(summary, summary_path)
    return summary


def main() -> int:
    args = parse_args()

    if not 0 < args.maf_threshold < 0.5:
        raise SystemExit(
            "--maf-threshold must be between 0 and 0.5."
        )
    if args.min_portability_variants < 1:
        raise SystemExit(
            "--min-portability-variants must be positive."
        )
    if args.min_coloc_variants < 1:
        raise SystemExit(
            "--min-coloc-variants must be positive."
        )

    required = [
        args.comparison_map,
        args.extraction_summary,
        args.trait_catalog,
    ]
    for path in required:
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")

    extraction_summary = read_json(
        args.extraction_summary
    )
    if clean(extraction_summary.get("status")) != "PASS":
        raise SystemExit(
            "Step 15B3 extraction summary is not PASS."
        )

    comparisons = pd.read_parquet(args.comparison_map)
    trait_catalog = pd.read_csv(
        args.trait_catalog,
        low_memory=False,
    )

    if len(comparisons) != 1062:
        print(
            "WARNING: expected 1,062 locked comparisons, "
            f"observed {len(comparisons):,}.",
            file=sys.stderr,
        )

    if not comparisons[
        "both_extractions_available"
    ].map(as_bool).all():
        raise SystemExit(
            "Not every comparison has both extractions."
        )

    comparisons = comparisons.sort_values(
        [
            "candidate_source",
            "candidate_trait_name",
            "gene",
            "comparison_population",
        ]
    )
    n_expected = len(comparisons)

    if args.max_comparisons > 0:
        run_comparisons = comparisons.head(
            args.max_comparisons
        )
    else:
        run_comparisons = comparisons

    args.comparison_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("STEP 15B4 — HARMONIZATION AND QC")
    print("=" * 78)
    print(f"Locked comparisons:                    {n_expected:,}")
    print(
        f"Comparisons in this run:               "
        f"{len(run_comparisons):,}"
    )
    print(
        f"MAF threshold:                         "
        f"{args.maf_threshold:.3f}"
    )
    print(
        f"Portability pre-LD EUR-GWS minimum:    "
        f"{args.min_portability_variants:,}"
    )
    print(
        f"Coloc shared-QC minimum:               "
        f"{args.min_coloc_variants:,}"
    )
    print(f"Resume mode:                           {not args.force}")
    print("=" * 78)

    feasibility_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for index, (_, comparison) in enumerate(
        run_comparisons.iterrows(),
        start=1,
    ):
        if index == 1 or index % 25 == 0:
            print(
                f"[{index:>4}/{len(run_comparisons):>4}] "
                f"{comparison['candidate_source']:<8} | "
                f"{comparison['gene']} | "
                f"{comparison['candidate_trait_name']} | "
                f"{comparison['comparison_population']}"
            )

        try:
            row = process_comparison(
                comparison,
                trait_catalog,
                comparison_dir=args.comparison_dir,
                maf_threshold=args.maf_threshold,
                min_portability_variants=(
                    args.min_portability_variants
                ),
                min_coloc_variants=(
                    args.min_coloc_variants
                ),
                force=args.force,
            )
            feasibility_rows.append(row)
        except Exception as exc:
            failure_rows.append(
                {
                    "comparison_uid": clean(
                        comparison["comparison_uid"]
                    ),
                    "gene_trait_uid": clean(
                        comparison["gene_trait_uid"]
                    ),
                    "gene": clean(comparison["gene"]),
                    "candidate_source": clean(
                        comparison["candidate_source"]
                    ),
                    "candidate_trait_name": clean(
                        comparison[
                            "candidate_trait_name"
                        ]
                    ),
                    "comparison_population": clean(
                        comparison[
                            "comparison_population"
                        ]
                    ),
                    "failure_type": type(exc).__name__,
                    "failure_reason": str(exc),
                    "failed_at_utc": utc_now(),
                }
            )
            print(
                "    FAILURE: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    # Include valid cached comparisons outside a debugging run limit.
    processed = {
        row["comparison_uid"]
        for row in feasibility_rows
    }
    failed = {
        row["comparison_uid"]
        for row in failure_rows
    }

    for _, comparison in comparisons.iterrows():
        uid = clean(comparison["comparison_uid"])
        if uid in processed or uid in failed:
            continue
        parquet_path, sidecar_path = comparison_paths(
            args.comparison_dir,
            uid,
        )
        if cache_valid(parquet_path, sidecar_path, uid):
            sidecar = read_json(sidecar_path)
            sidecar["cache_action"] = (
                "REUSED_OUTSIDE_RUN_LIMIT"
            )
            feasibility_rows.append(
                {
                    key: value
                    for key, value in sidecar.items()
                    if key
                    not in {
                        "script_version",
                        "status",
                        "output_size_bytes",
                        "output_sha256",
                        "completed_at_utc",
                    }
                }
            )

    feasibility = pd.DataFrame(feasibility_rows)
    failures = pd.DataFrame(failure_rows)

    if (
        len(feasibility)
        and feasibility["comparison_uid"].duplicated().any()
    ):
        raise SystemExit(
            "Duplicate comparison_uid in feasibility output."
        )

    summary = write_outputs(
        feasibility,
        failures,
        output_dir=args.output_dir,
        n_expected=n_expected,
        n_processed_in_run=len(run_comparisons),
        args=args,
    )

    print("=" * 78)
    print("STEP 15B4 — QC FEASIBILITY SUMMARY")
    print("=" * 78)
    print(
        f"Overall status:                        "
        f"{summary['status']}"
    )
    print(
        f"Feasibility rows:                      "
        f"{summary['n_feasibility_rows']:,}/"
        f"{summary['n_expected_comparisons']:,}"
    )
    print(
        f"Portability pre-LD eligible:           "
        f"{summary['all_comparisons']['n_portability_pre_ld_eligible']:,}"
    )
    print(
        f"Coloc input eligible:                  "
        f"{summary['all_comparisons']['n_coloc_input_eligible']:,}"
    )
    print(
        f"Coloc powered in both ancestries:      "
        f"{summary['all_comparisons']['n_coloc_powered_both']:,}"
    )
    print(
        f"Primary nonmixed comparisons:          "
        f"{summary['primary_nonmixed_comparisons']['n']:,}"
    )
    print(
        "Primary nonmixed portability eligible:"
        f" {summary['primary_nonmixed_comparisons']['n_portability_pre_ld_eligible']:,}"
    )
    print(
        "Primary nonmixed coloc eligible:      "
        f"{summary['primary_nonmixed_comparisons']['n_coloc_input_eligible']:,}"
    )
    print(f"Failures:                              {summary['n_failures']:,}")
    print()
    print("No slopes, LD pruning, PP4, or approval models were calculated.")
    print("=" * 78)

    if summary["status"] == "PASS":
        return 0
    if summary["status"] == "PARTIAL":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
