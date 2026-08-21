#!/usr/bin/env python3
"""
Step 15C1 — Freeze LD-reference policy and build portability variant requests.

This step does not access LD matrices and does not calculate slopes.

Primary portability universe
----------------------------
The 235 Step 15B5 primary, nonmixed, pre-LD-eligible comparisons.

Candidate variants
------------------
For each comparison, select variants from the Step 15B4 harmonized file where:
    qc_qualified == True
    eur_gws == True

These are the exact primary-harmonized variants. GBMI REF/ALT swaps remain
sensitivity-only and are not used in the primary portability request.

LD-reference policy
-------------------
Pan-UKB, GRCh37:
    Use the public Pan-UKB covariate-adjusted in-sample LD BlockMatrix and
    keyed variant-index Hail Table for EUR and the comparison ancestry.
    All six Pan-UKB groups are directly supported:
    AFR, AMR, CSA, EAS, EUR, MID.

GBMI, GRCh38:
    Use the phased 1000 Genomes GRCh38 unrelated-sample call set.
    Subset individuals by the original source population label:
    AFR, AMR, EAS, EUR, or SAS.
    The analysis label CSA is mapped to SAS only when the GBMI source label
    is SAS. No MID proxy is permitted.

Frozen pruning rule for Step 15C2/15C3
--------------------------------------
1. Rank candidate variants by ascending EUR p-value, then position/alleles.
2. A variant must be represented in both ancestry LD references.
3. Greedily retain variants only when r^2 < 0.10 with every retained variant
   in both the EUR and comparison-ancestry LD matrices.
4. Final portability eligibility requires at least 3 retained variants.
5. Sensitivities use r^2 < 0.01 and r^2 < 0.20 on the same candidate set.

Inputs
------
output/15_portability_comparisons_primary_locked.parquet
output/15_portability_comparisons_sensitivity_locked.parquet
output/15b1_extraction_manifest.parquet
output/15b5_analysis_universe_summary.json

Outputs
-------
output/15c1_ld_panel_policy.csv
output/15c1_ld_panel_policy.parquet
output/15c1_portability_comparison_manifest.csv
output/15c1_portability_comparison_manifest.parquet
output/15c1_portability_variant_requests_primary.csv
output/15c1_portability_variant_requests_primary.parquet
output/15c1_portability_variant_requests_sensitivity.csv
output/15c1_portability_variant_requests_sensitivity.parquet
output/15c1_unique_ld_variant_requests.csv
output/15c1_unique_ld_variant_requests.parquet
output/15c1_portability_ld_summary.json
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

import pandas as pd


SCRIPT_VERSION = "15C1.2"
PANUKB = "Pan-UKB"
GBMI = "GBMI"

PANUKB_POPS = {"AFR", "AMR", "CSA", "EAS", "EUR", "MID"}
KGP_POPS = {"AFR", "AMR", "EAS", "EUR", "SAS"}

KGP_VCF_BASE = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/"
    "data_collections/1000_genomes_project/release/"
    "20190312_biallelic_SNV_and_INDEL"
)
KGP_PANEL_URL = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/"
    "20130502/integrated_call_samples_v3.20130502.ALL.panel"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--primary-comparisons",
        type=Path,
        default=Path(
            "output/"
            "15_portability_comparisons_primary_locked.parquet"
        ),
    )
    p.add_argument(
        "--sensitivity-comparisons",
        type=Path,
        default=Path(
            "output/"
            "15_portability_comparisons_sensitivity_locked.parquet"
        ),
    )
    p.add_argument(
        "--step15b1-manifest",
        type=Path,
        default=Path(
            "output/15b1_extraction_manifest.parquet"
        ),
    )
    p.add_argument(
        "--universe-summary",
        type=Path,
        default=Path(
            "output/15b5_analysis_universe_summary.json"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
    )
    return p.parse_args()


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
        "true", "1", "yes", "y"
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json_write(
    data: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


def write_table(
    frame: pd.DataFrame,
    stem: str,
    output_dir: Path,
) -> dict[str, Any]:
    csv_path = output_dir / f"{stem}.csv"
    parquet_path = output_dir / f"{stem}.parquet"
    frame.to_csv(csv_path, index=False)
    frame.to_parquet(
        parquet_path,
        index=False,
        compression="zstd",
    )
    return {
        "csv": str(csv_path),
        "csv_sha256": sha256_file(csv_path),
        "parquet": str(parquet_path),
        "parquet_sha256": sha256_file(
            parquet_path
        ),
        "n_rows": int(len(frame)),
    }


def require_columns(
    frame: pd.DataFrame,
    columns: set[str],
    label: str,
) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise SystemExit(
            f"{label} missing required columns: {missing}"
        )


def kgp_vcf_url(chrom: str) -> str:
    chrom = clean(chrom)
    return (
        f"{KGP_VCF_BASE}/"
        f"ALL.chr{chrom}."
        "shapeit2_integrated_snvindels_v2a_27022019."
        "GRCh38.phased.vcf.gz"
    )


def map_gbmi_reference_population(
    analysis_population: str,
    source_population_label: str,
) -> tuple[str, str, bool]:
    analysis = clean(analysis_population).upper()
    source = clean(source_population_label).upper()

    # Preserve the source's actual SAS label even though the analysis
    # harmonization label is CSA.
    if source in KGP_POPS:
        reference = source
        method = "exact_source_population_label"
        return reference, method, True

    if analysis in KGP_POPS:
        return (
            analysis,
            "exact_analysis_population_label",
            True,
        )

    if analysis == "CSA" and source == "SAS":
        return "SAS", "exact_source_SAS_for_analysis_CSA", True

    return "", "no_supported_population_reference", False


def panel_records(
    source: str,
    build: str,
    analysis_population: str,
    source_population_label: str,
    chrom: str,
) -> dict[str, Any]:
    source = clean(source)
    build = clean(build)
    analysis_population = (
        clean(analysis_population).upper()
    )
    source_population_label = (
        clean(source_population_label).upper()
    )

    if source == PANUKB:
        if build != "GRCh37":
            return {
                "ld_reference_ready": False,
                "failure_reason": (
                    "Pan-UKB build is not GRCh37"
                ),
            }
        if analysis_population not in PANUKB_POPS:
            return {
                "ld_reference_ready": False,
                "failure_reason": (
                    "Pan-UKB comparison ancestry lacks "
                    "a released LD matrix"
                ),
            }

        return {
            "ld_reference_ready": True,
            "failure_reason": "",
            "eur_reference_family": "Pan-UKB in-sample LD",
            "eur_reference_population": "EUR",
            "eur_population_match": "exact",
            "eur_ld_matrix": (
                "s3://pan-ukb-us-east-1/ld_release/"
                "UKBB.EUR.ldadj.bm"
            ),
            "eur_variant_index": (
                "s3://pan-ukb-us-east-1/ld_release/"
                "UKBB.EUR.ldadj.variant.ht"
            ),
            "comparison_reference_family": (
                "Pan-UKB in-sample LD"
            ),
            "comparison_reference_population": (
                analysis_population
            ),
            "comparison_population_match": "exact",
            "comparison_ld_matrix": (
                "s3://pan-ukb-us-east-1/ld_release/"
                f"UKBB.{analysis_population}.ldadj.bm"
            ),
            "comparison_variant_index": (
                "s3://pan-ukb-us-east-1/ld_release/"
                f"UKBB.{analysis_population}."
                "ldadj.variant.ht"
            ),
            "kgp_vcf_url": "",
            "kgp_vcf_index_url": "",
            "kgp_sample_panel_url": "",
        }

    if source == GBMI:
        if build != "GRCh38":
            return {
                "ld_reference_ready": False,
                "failure_reason": (
                    "GBMI build is not GRCh38"
                ),
            }

        comparison_ref, match_method, ready = (
            map_gbmi_reference_population(
                analysis_population,
                source_population_label,
            )
        )
        if not ready:
            return {
                "ld_reference_ready": False,
                "failure_reason": (
                    "No exact 1000 Genomes source-population "
                    "reference; no proxy allowed"
                ),
            }

        vcf = kgp_vcf_url(chrom)
        return {
            "ld_reference_ready": True,
            "failure_reason": "",
            "eur_reference_family": (
                "1000 Genomes phased GRCh38"
            ),
            "eur_reference_population": "EUR",
            "eur_population_match": "exact_superpopulation",
            "eur_ld_matrix": "",
            "eur_variant_index": "",
            "comparison_reference_family": (
                "1000 Genomes phased GRCh38"
            ),
            "comparison_reference_population": (
                comparison_ref
            ),
            "comparison_population_match": match_method,
            "comparison_ld_matrix": "",
            "comparison_variant_index": "",
            "kgp_vcf_url": vcf,
            "kgp_vcf_index_url": vcf + ".tbi",
            "kgp_sample_panel_url": KGP_PANEL_URL,
        }

    return {
        "ld_reference_ready": False,
        "failure_reason": f"Unsupported source: {source}",
    }


def load_source_labels(
    manifest_path: Path,
) -> pd.DataFrame:
    manifest = pd.read_parquet(manifest_path)
    require_columns(
        manifest,
        {
            "comparison_uid",
            "source_population_label",
            "chrom",
            "genome_build",
        },
        "Step 15B1 extraction manifest",
    )

    selected = manifest[
        [
            "comparison_uid",
            "source_population_label",
            "chrom",
            "genome_build",
        ]
    ].copy()

    if selected["comparison_uid"].duplicated().any():
        raise SystemExit(
            "Step 15B1 comparison_uid is not unique."
        )
    return selected


def enrich_comparisons(
    comparisons: pd.DataFrame,
    source_labels: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        comparisons,
        {
            "comparison_uid",
            "gene_trait_uid",
            "gene",
            "candidate_source",
            "candidate_trait_key",
            "candidate_trait_name",
            "comparison_population",
            "unit_analysis_role",
            "mixed_pool",
            "n_eur_gws_qc",
            "portability_pre_ld_eligible",
            "harmonized_parquet",
        },
        "Locked portability comparison table",
    )

    result = comparisons.merge(
        source_labels,
        on="comparison_uid",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not result["_merge"].eq("both").all():
        missing = result.loc[
            ~result["_merge"].eq("both"),
            "comparison_uid",
        ].tolist()
        raise SystemExit(
            "Comparisons missing from Step 15B1 manifest: "
            f"{missing[:20]}"
        )
    result = result.drop(columns="_merge")

    panel_rows = []
    for _, row in result.iterrows():
        panel_rows.append(
            panel_records(
                source=clean(row["candidate_source"]),
                build=clean(row["genome_build"]),
                analysis_population=clean(
                    row["comparison_population"]
                ),
                source_population_label=clean(
                    row["source_population_label"]
                ),
                chrom=clean(row["chrom"]),
            )
        )

    panel_frame = pd.DataFrame(panel_rows)
    result = pd.concat(
        [
            result.reset_index(drop=True),
            panel_frame.reset_index(drop=True),
        ],
        axis=1,
    )
    result["portability_ld_request_ready"] = (
        result["portability_pre_ld_eligible"].map(
            as_bool
        )
        & result["ld_reference_ready"].map(as_bool)
    )
    return result


def extract_variant_requests(
    comparisons: pd.DataFrame,
    *,
    universe_label: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for _, comparison in comparisons.iterrows():
        path = Path(
            clean(comparison["harmonized_parquet"])
        )
        if not path.exists():
            raise SystemExit(
                f"Harmonized file missing: {path}"
            )

        frame = pd.read_parquet(path)
        require_columns(
            frame,
            {
                "chrom",
                "pos",
                "ref",
                "alt",
                "variant_id",
                "eur_beta",
                "eur_se",
                "eur_p",
                "eur_af",
                "comparison_beta",
                "comparison_se",
                "comparison_p",
                "comparison_af",
                "qc_qualified",
                "eur_gws",
                "allele_relation",
                "primary_harmonized",
            },
            f"Harmonized comparison {comparison['comparison_uid']}",
        )

        selected = frame[
            frame["qc_qualified"].map(as_bool)
            & frame["eur_gws"].map(as_bool)
            & frame["primary_harmonized"].map(as_bool)
            & frame["allele_relation"].eq("exact")
        ].copy()

        if len(selected) != int(
            comparison["n_eur_gws_qc"]
        ):
            raise SystemExit(
                "EUR-GWS count mismatch for "
                f"{comparison['comparison_uid']}: "
                f"table={len(selected)}, "
                f"manifest={comparison['n_eur_gws_qc']}"
            )

        selected = selected.sort_values(
            [
                "eur_p",
                "pos",
                "ref",
                "alt",
            ],
            kind="mergesort",
        )
        selected["eur_p_rank"] = range(
            1,
            len(selected) + 1,
        )

        metadata = {
            "analysis_universe": universe_label,
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
            "source_population_label": clean(
                comparison["source_population_label"]
            ),
            "genome_build": clean(
                comparison["genome_build"]
            ),
            "unit_analysis_role": clean(
                comparison["unit_analysis_role"]
            ),
            "mixed_pool": as_bool(
                comparison["mixed_pool"]
            ),
            "eur_reference_family": clean(
                comparison["eur_reference_family"]
            ),
            "eur_reference_population": clean(
                comparison["eur_reference_population"]
            ),
            "comparison_reference_family": clean(
                comparison[
                    "comparison_reference_family"
                ]
            ),
            "comparison_reference_population": clean(
                comparison[
                    "comparison_reference_population"
                ]
            ),
            "eur_ld_matrix": clean(
                comparison["eur_ld_matrix"]
            ),
            "eur_variant_index": clean(
                comparison["eur_variant_index"]
            ),
            "comparison_ld_matrix": clean(
                comparison["comparison_ld_matrix"]
            ),
            "comparison_variant_index": clean(
                comparison[
                    "comparison_variant_index"
                ]
            ),
            "kgp_vcf_url": clean(
                comparison["kgp_vcf_url"]
            ),
            "kgp_vcf_index_url": clean(
                comparison["kgp_vcf_index_url"]
            ),
            "kgp_sample_panel_url": clean(
                comparison["kgp_sample_panel_url"]
            ),
        }
        for key, value in reversed(
            list(metadata.items())
        ):
            if key in selected.columns:
                existing = selected[key]

                # The Step 15B4 harmonized files already carry several
                # locked comparison metadata fields. Validate that those
                # values agree with Step 15B5 rather than inserting duplicate
                # columns or silently overwriting them.
                if pd.api.types.is_bool_dtype(existing.dtype):
                    expected = as_bool(value)
                    observed = existing.map(as_bool)
                    mismatch = observed.ne(expected)
                else:
                    expected = clean(value)
                    observed = existing.map(clean)
                    mismatch = observed.ne(expected)

                if mismatch.any():
                    examples = (
                        existing.loc[mismatch]
                        .astype(str)
                        .drop_duplicates()
                        .head(10)
                        .tolist()
                    )
                    raise SystemExit(
                        "Locked metadata mismatch for "
                        f"{comparison['comparison_uid']} column "
                        f"{key}: expected={value!r}, "
                        f"observed_examples={examples}"
                    )
                continue

            selected.insert(0, key, value)

        rows.append(selected)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(
        rows,
        ignore_index=True,
        sort=False,
    )

    if result.duplicated(
        ["comparison_uid", "variant_id"]
    ).any():
        duplicate = result.loc[
            result.duplicated(
                ["comparison_uid", "variant_id"],
                keep=False,
            ),
            [
                "comparison_uid",
                "variant_id",
            ],
        ]
        raise SystemExit(
            "Duplicate comparison-variant requests:\n"
            + duplicate.head(30).to_string(index=False)
        )

    return result


def build_panel_policy(
    comparisons: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "candidate_source",
        "genome_build",
        "comparison_population",
        "source_population_label",
        "eur_reference_family",
        "eur_reference_population",
        "eur_population_match",
        "eur_ld_matrix",
        "eur_variant_index",
        "comparison_reference_family",
        "comparison_reference_population",
        "comparison_population_match",
        "comparison_ld_matrix",
        "comparison_variant_index",
        "kgp_vcf_url",
        "kgp_vcf_index_url",
        "kgp_sample_panel_url",
        "ld_reference_ready",
        "failure_reason",
    ]
    policy = comparisons[columns].drop_duplicates()
    policy = policy.sort_values(
        [
            "candidate_source",
            "comparison_population",
            "source_population_label",
        ]
    )
    return policy


def build_unique_ld_requests(
    requests: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, row in requests.iterrows():
        base = {
            "candidate_source": row[
                "candidate_source"
            ],
            "genome_build": row["genome_build"],
            "chrom": row["chrom"],
            "pos": row["pos"],
            "ref": row["ref"],
            "alt": row["alt"],
            "variant_id": row["variant_id"],
        }

        rows.append(
            {
                **base,
                "ld_role": "EUR",
                "reference_family": row[
                    "eur_reference_family"
                ],
                "reference_population": row[
                    "eur_reference_population"
                ],
                "ld_matrix": row["eur_ld_matrix"],
                "variant_index": row[
                    "eur_variant_index"
                ],
                "kgp_vcf_url": row["kgp_vcf_url"],
                "kgp_vcf_index_url": row[
                    "kgp_vcf_index_url"
                ],
                "kgp_sample_panel_url": row[
                    "kgp_sample_panel_url"
                ],
            }
        )
        rows.append(
            {
                **base,
                "ld_role": "COMPARISON",
                "reference_family": row[
                    "comparison_reference_family"
                ],
                "reference_population": row[
                    "comparison_reference_population"
                ],
                "ld_matrix": row[
                    "comparison_ld_matrix"
                ],
                "variant_index": row[
                    "comparison_variant_index"
                ],
                "kgp_vcf_url": row["kgp_vcf_url"],
                "kgp_vcf_index_url": row[
                    "kgp_vcf_index_url"
                ],
                "kgp_sample_panel_url": row[
                    "kgp_sample_panel_url"
                ],
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .drop_duplicates(
            [
                "candidate_source",
                "genome_build",
                "reference_family",
                "reference_population",
                "variant_id",
            ]
        )
        .sort_values(
            [
                "candidate_source",
                "reference_population",
                "chrom",
                "pos",
                "ref",
                "alt",
            ]
        )
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    required_paths = [
        args.primary_comparisons,
        args.sensitivity_comparisons,
        args.step15b1_manifest,
        args.universe_summary,
    ]
    for path in required_paths:
        if not path.exists():
            raise SystemExit(
                f"Missing required input: {path}"
            )

    universe = read_json(args.universe_summary)
    if clean(universe.get("status")) != "PASS":
        raise SystemExit(
            "Step 15B5 universe summary is not PASS."
        )

    primary = pd.read_parquet(
        args.primary_comparisons
    )
    sensitivity = pd.read_parquet(
        args.sensitivity_comparisons
    )
    labels = load_source_labels(
        args.step15b1_manifest
    )

    expected_primary = int(
        universe["comparison_counts"][
            "portability_primary_pre_ld"
        ]
    )
    expected_sensitivity = int(
        universe["comparison_counts"][
            "portability_sensitivity_pre_ld"
        ]
    )

    if len(primary) != expected_primary:
        raise SystemExit(
            "Primary portability count mismatch: "
            f"{len(primary)} != {expected_primary}"
        )
    if len(sensitivity) != expected_sensitivity:
        raise SystemExit(
            "Sensitivity portability count mismatch: "
            f"{len(sensitivity)} != {expected_sensitivity}"
        )

    primary_manifest = enrich_comparisons(
        primary,
        labels,
    )
    sensitivity_manifest = enrich_comparisons(
        sensitivity,
        labels,
    )

    unsupported_primary = primary_manifest[
        ~primary_manifest[
            "ld_reference_ready"
        ].map(as_bool)
    ]
    unsupported_sensitivity = sensitivity_manifest[
        ~sensitivity_manifest[
            "ld_reference_ready"
        ].map(as_bool)
    ]

    primary_ready = primary_manifest[
        primary_manifest[
            "ld_reference_ready"
        ].map(as_bool)
    ].copy()
    sensitivity_ready = sensitivity_manifest[
        sensitivity_manifest[
            "ld_reference_ready"
        ].map(as_bool)
    ].copy()

    primary_requests = extract_variant_requests(
        primary_ready,
        universe_label="PRIMARY_NONMIXED",
    )
    sensitivity_requests = extract_variant_requests(
        sensitivity_ready,
        universe_label="SENSITIVITY_ALL",
    )

    panel_policy = build_panel_policy(
        pd.concat(
            [
                primary_manifest,
                sensitivity_manifest,
            ],
            ignore_index=True,
        )
    )
    unique_requests = build_unique_ld_requests(
        primary_requests
    )

    outputs = {}
    outputs["panel_policy"] = write_table(
        panel_policy,
        "15c1_ld_panel_policy",
        args.output_dir,
    )
    outputs["comparison_manifest"] = write_table(
        primary_manifest,
        "15c1_portability_comparison_manifest",
        args.output_dir,
    )
    outputs["primary_variant_requests"] = (
        write_table(
            primary_requests,
            "15c1_portability_variant_requests_primary",
            args.output_dir,
        )
    )
    outputs["sensitivity_variant_requests"] = (
        write_table(
            sensitivity_requests,
            "15c1_portability_variant_requests_sensitivity",
            args.output_dir,
        )
    )
    outputs["unique_ld_variant_requests"] = (
        write_table(
            unique_requests,
            "15c1_unique_ld_variant_requests",
            args.output_dir,
        )
    )

    primary_source_pop = {
        f"{source}:{population}": int(count)
        for (source, population), count
        in primary_ready.groupby(
            [
                "candidate_source",
                "comparison_population",
            ]
        ).size().items()
    }

    summary = {
        "step": "15C1",
        "script_version": SCRIPT_VERSION,
        "status": (
            "PASS"
            if len(unsupported_primary) == 0
            else "PASS_WITH_PRIMARY_LD_EXCLUSIONS"
        ),
        "frozen_at_utc": utc_now(),
        "input_counts": {
            "primary_pre_ld_comparisons": int(
                len(primary)
            ),
            "sensitivity_pre_ld_comparisons": int(
                len(sensitivity)
            ),
        },
        "ld_ready_counts": {
            "primary": int(len(primary_ready)),
            "primary_unsupported": int(
                len(unsupported_primary)
            ),
            "sensitivity": int(
                len(sensitivity_ready)
            ),
            "sensitivity_unsupported": int(
                len(unsupported_sensitivity)
            ),
        },
        "variant_request_counts": {
            "primary_comparison_variant_rows": int(
                len(primary_requests)
            ),
            "primary_unique_variants": int(
                primary_requests[
                    "variant_id"
                ].nunique()
                if len(primary_requests)
                else 0
            ),
            "sensitivity_comparison_variant_rows": int(
                len(sensitivity_requests)
            ),
            "primary_unique_panel_variant_requests": int(
                len(unique_requests)
            ),
        },
        "primary_ready_by_source_population": (
            primary_source_pop
        ),
        "unsupported_primary_comparisons": (
            unsupported_primary[
                [
                    "comparison_uid",
                    "gene_trait_uid",
                    "gene",
                    "candidate_source",
                    "candidate_trait_name",
                    "comparison_population",
                    "source_population_label",
                    "failure_reason",
                ]
            ].to_dict("records")
        ),
        "frozen_pruning_policy": {
            "candidate_set": (
                "Step 15B4 qc_qualified and EUR GWS "
                "primary-exact variants"
            ),
            "ranking": (
                "ascending EUR p-value, then position, REF, ALT"
            ),
            "presence_requirement": (
                "variant present in both EUR and comparison "
                "LD references"
            ),
            "primary_r2_threshold": 0.10,
            "primary_rule": (
                "reject candidate when r^2 >= 0.10 with any "
                "retained variant in either ancestry"
            ),
            "sensitivity_r2_thresholds": [
                0.01,
                0.20,
            ],
            "minimum_retained_variants": 3,
            "no_proxy_rule": (
                "unsupported source populations are excluded; "
                "no continental proxy is substituted"
            ),
        },
        "reference_policy": {
            "Pan-UKB": (
                "source-matched Pan-UKB covariate-adjusted "
                "in-sample LD BlockMatrix, GRCh37"
            ),
            "GBMI": (
                "1000 Genomes phased unrelated-sample "
                "GRCh38 call set, subset by original source "
                "population label"
            ),
        },
        "outputs": outputs,
    }

    summary_path = (
        args.output_dir
        / "15c1_portability_ld_summary.json"
    )
    atomic_json_write(summary, summary_path)

    print("=" * 78)
    print("STEP 15C1 — FROZEN LD REQUESTS")
    print("=" * 78)
    print(
        f"Primary pre-LD comparisons:            "
        f"{len(primary):,}"
    )
    print(
        f"Primary LD-reference ready:            "
        f"{len(primary_ready):,}"
    )
    print(
        f"Primary LD-reference unsupported:      "
        f"{len(unsupported_primary):,}"
    )
    print(
        f"Primary comparison-variant requests:   "
        f"{len(primary_requests):,}"
    )
    print(
        f"Primary unique candidate variants:     "
        f"{primary_requests['variant_id'].nunique() if len(primary_requests) else 0:,}"
    )
    print(
        f"Unique panel-variant requests:          "
        f"{len(unique_requests):,}"
    )
    print(
        f"Sensitivity LD-reference ready:        "
        f"{len(sensitivity_ready):,}/"
        f"{len(sensitivity):,}"
    )
    print()
    print(
        "No LD values, pruning, slopes, or "
        "colocalization probabilities were calculated."
    )
    print(f"Summary: {summary_path}")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
