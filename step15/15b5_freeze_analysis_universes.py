#!/usr/bin/env python3
"""
Step 15B5 — Freeze portability and colocalization analysis universes.

This is the final Step 15B handoff. It converts the comparison-level QC
results into immutable comparison- and gene-trait-unit universes for the
substantive analyses.

No LD processing, slope estimation, colocalization, prior sensitivity, or
approval modeling is performed.

Primary mechanistic universe
----------------------------
- unit_analysis_role == PRIMARY
- mixed_pool == False

Portability primary universe
----------------------------
Primary mechanistic comparisons that passed the Step 15B4 pre-LD gate:
at least three shared QC-qualified EUR genome-wide-significant variants.
This remains provisional until Step 15C applies LD processing.

Colocalization primary universe
-------------------------------
Primary mechanistic comparisons with at least 50 shared QC-qualified variants
and usable sample-size metadata.

Powered colocalization subset
-----------------------------
Primary colocalization comparisons with at least one QC-qualified
genome-wide-significant variant in each ancestry.

Sensitivity universes
---------------------
All Step 15B4-eligible comparisons, including sensitivity mappings and
mixed-pool units. These are stored separately and cannot replace the primary
universes.

Inputs
------
output/15b4_qc_feasibility.parquet
output/15b4_qc_summary.json
output/15_gene_trait_units_locked.parquet

Outputs
-------
output/15b5_comparison_analysis_universe.csv
output/15b5_comparison_analysis_universe.parquet
output/15b5_unit_analysis_universe.csv
output/15b5_unit_analysis_universe.parquet

output/15_portability_comparisons_primary_locked.csv
output/15_portability_comparisons_primary_locked.parquet
output/15_portability_comparisons_sensitivity_locked.csv
output/15_portability_comparisons_sensitivity_locked.parquet
output/15_portability_units_primary_locked.csv
output/15_portability_units_primary_locked.parquet

output/15_coloc_comparisons_primary_locked.csv
output/15_coloc_comparisons_primary_locked.parquet
output/15_coloc_comparisons_powered_locked.csv
output/15_coloc_comparisons_powered_locked.parquet
output/15_coloc_comparisons_sensitivity_locked.csv
output/15_coloc_comparisons_sensitivity_locked.parquet
output/15_coloc_units_primary_locked.csv
output/15_coloc_units_primary_locked.parquet

output/15b5_analysis_universe_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_VERSION = "15B5.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feasibility",
        type=Path,
        default=Path("output/15b4_qc_feasibility.parquet"),
    )
    parser.add_argument(
        "--qc-summary",
        type=Path,
        default=Path("output/15b4_qc_summary.json"),
    )
    parser.add_argument(
        "--locked-units",
        type=Path,
        default=Path("output/15_gene_trait_units_locked.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json_write(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_table(
    frame: pd.DataFrame,
    csv_path: Path,
    parquet_path: Path,
) -> dict[str, Any]:
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
        "parquet_sha256": sha256_file(parquet_path),
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
            f"{label} is missing required columns: {missing}"
        )


def invariant_value(
    group: pd.DataFrame,
    column: str,
) -> Any:
    values = group[column].drop_duplicates()
    if len(values) != 1:
        rendered = values.astype(str).tolist()
        raise RuntimeError(
            f"{column} is not invariant within "
            f"{group['gene_trait_uid'].iloc[0]}: {rendered}"
        )
    return values.iloc[0]


def build_comparison_universe(
    feasibility: pd.DataFrame,
) -> pd.DataFrame:
    result = feasibility.copy()

    bool_columns = [
        "mixed_pool",
        "primary_approval_eligible",
        "portability_pre_ld_eligible",
        "coloc_input_eligible",
        "coloc_powered_both_ancestries",
        "sample_metadata_usable",
    ]
    for column in bool_columns:
        result[column] = result[column].map(as_bool)

    result["is_primary_mapping"] = result[
        "unit_analysis_role"
    ].eq("PRIMARY")
    result["is_nonmixed"] = ~result["mixed_pool"]
    result["primary_mechanistic_universe"] = (
        result["is_primary_mapping"]
        & result["is_nonmixed"]
    )

    result["portability_primary_locked"] = (
        result["primary_mechanistic_universe"]
        & result["portability_pre_ld_eligible"]
    )
    result["portability_sensitivity_locked"] = result[
        "portability_pre_ld_eligible"
    ]

    result["coloc_primary_locked"] = (
        result["primary_mechanistic_universe"]
        & result["coloc_input_eligible"]
    )
    result["coloc_powered_primary_locked"] = (
        result["coloc_primary_locked"]
        & result["coloc_powered_both_ancestries"]
    )
    result["coloc_sensitivity_locked"] = result[
        "coloc_input_eligible"
    ]
    result["coloc_powered_sensitivity_locked"] = (
        result["coloc_input_eligible"]
        & result["coloc_powered_both_ancestries"]
    )

    result["approval_analysis_eligible"] = result[
        "primary_approval_eligible"
    ]

    # The locked mapping design defined approval eligibility as primary and
    # nonmixed. Fail rather than allow the two definitions to drift.
    mismatch = result[
        result["approval_analysis_eligible"]
        .ne(result["primary_mechanistic_universe"])
    ]
    if len(mismatch):
        raise SystemExit(
            "primary_approval_eligible no longer equals "
            "PRIMARY & nonmixed. Example rows:\n"
            + mismatch[
                [
                    "comparison_uid",
                    "gene_trait_uid",
                    "unit_analysis_role",
                    "mixed_pool",
                    "primary_approval_eligible",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    return result.sort_values(
        [
            "candidate_source",
            "candidate_trait_name",
            "gene",
            "comparison_population",
            "comparison_uid",
        ]
    )


def build_unit_universe(
    comparisons: pd.DataFrame,
    locked_units: pd.DataFrame,
) -> pd.DataFrame:
    invariant_columns = [
        "gene",
        "candidate_source",
        "candidate_trait_key",
        "candidate_trait_name",
        "unit_analysis_role",
        "mixed_pool",
        "primary_approval_eligible",
    ]

    rows: list[dict[str, Any]] = []
    for gene_trait_uid, group in comparisons.groupby(
        "gene_trait_uid",
        sort=False,
        dropna=False,
    ):
        row = {
            "gene_trait_uid": gene_trait_uid,
        }
        for column in invariant_columns:
            row[column] = invariant_value(group, column)

        populations = sorted(
            set(
                group["comparison_population"]
                .dropna()
                .astype(str)
            )
        )

        row.update(
            {
                "n_comparison_ancestries": int(len(group)),
                "comparison_ancestries": " || ".join(populations),
                "n_portability_pre_ld_eligible_ancestries": int(
                    group[
                        "portability_pre_ld_eligible"
                    ].sum()
                ),
                "n_coloc_input_eligible_ancestries": int(
                    group["coloc_input_eligible"].sum()
                ),
                "n_coloc_powered_both_ancestries": int(
                    group[
                        "coloc_powered_both_ancestries"
                    ].sum()
                ),
                "minimum_qc_qualified_variants": int(
                    pd.to_numeric(
                        group["n_qc_qualified"],
                        errors="coerce",
                    ).min()
                ),
                "median_qc_qualified_variants": float(
                    pd.to_numeric(
                        group["n_qc_qualified"],
                        errors="coerce",
                    ).median()
                ),
                "maximum_qc_qualified_variants": int(
                    pd.to_numeric(
                        group["n_qc_qualified"],
                        errors="coerce",
                    ).max()
                ),
                "maximum_eur_gws_qc_variants": int(
                    pd.to_numeric(
                        group["n_eur_gws_qc"],
                        errors="coerce",
                    ).max()
                ),
                "maximum_comparison_gws_qc_variants": int(
                    pd.to_numeric(
                        group["n_comparison_gws_qc"],
                        errors="coerce",
                    ).max()
                ),
                "primary_mechanistic_universe": bool(
                    group[
                        "primary_mechanistic_universe"
                    ].all()
                ),
                "portability_primary_locked": bool(
                    group[
                        "portability_primary_locked"
                    ].any()
                ),
                "portability_sensitivity_locked": bool(
                    group[
                        "portability_sensitivity_locked"
                    ].any()
                ),
                "coloc_primary_locked": bool(
                    group["coloc_primary_locked"].any()
                ),
                "coloc_powered_primary_locked": bool(
                    group[
                        "coloc_powered_primary_locked"
                    ].any()
                ),
                "coloc_sensitivity_locked": bool(
                    group["coloc_sensitivity_locked"].any()
                ),
                "coloc_powered_sensitivity_locked": bool(
                    group[
                        "coloc_powered_sensitivity_locked"
                    ].any()
                ),
                "all_ancestries_coloc_input_eligible": bool(
                    group["coloc_input_eligible"].all()
                ),
                "all_ancestries_portability_pre_ld_eligible": bool(
                    group[
                        "portability_pre_ld_eligible"
                    ].all()
                ),
            }
        )
        rows.append(row)

    unit_summary = pd.DataFrame(rows)

    locked_required = {
        "gene_trait_uid",
        "gene",
        "candidate_source",
        "candidate_trait_key",
        "candidate_trait_name",
        "genome_build",
        "canonical_group",
        "unit_analysis_role",
        "locked_mapping_tier",
        "mapping_tiers_contributing",
        "mixed_pool",
        "primary_approval_eligible",
        "sensitivity_pool_assignment",
        "pools",
        "n_A_pairs",
        "n_B_pairs",
        "n_target_indication_pairs",
        "n_indications",
        "indications",
        "non_eur_pops_available",
    }
    require_columns(
        locked_units,
        locked_required,
        "Locked unit table",
    )

    if locked_units["gene_trait_uid"].duplicated().any():
        raise SystemExit(
            "Locked unit table has duplicate gene_trait_uid."
        )

    additional_columns = [
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
    unit_summary = unit_summary.merge(
        locked_units[additional_columns],
        on="gene_trait_uid",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not unit_summary["_merge"].eq("both").all():
        missing = unit_summary.loc[
            ~unit_summary["_merge"].eq("both"),
            "gene_trait_uid",
        ].tolist()
        raise SystemExit(
            "QC units not found in locked unit table: "
            f"{missing[:20]}"
        )
    unit_summary = unit_summary.drop(columns="_merge")

    if len(unit_summary) != len(locked_units):
        locked_missing = sorted(
            set(locked_units["gene_trait_uid"])
            - set(unit_summary["gene_trait_uid"])
        )
        raise SystemExit(
            "Not every locked unit is represented in QC output. "
            f"Missing examples: {locked_missing[:20]}"
        )

    return unit_summary.sort_values(
        [
            "candidate_source",
            "candidate_trait_name",
            "gene",
            "gene_trait_uid",
        ]
    )


def reason_counts(
    frame: pd.DataFrame,
    column: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in frame[column].fillna(""):
        text = clean(value)
        if not text:
            continue
        for reason in text.split(" | "):
            reason = clean(reason)
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
    return dict(
        sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for path in [
        args.feasibility,
        args.qc_summary,
        args.locked_units,
    ]:
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")

    qc_summary = read_json(args.qc_summary)
    if clean(qc_summary.get("status")) != "PASS":
        raise SystemExit(
            "Step 15B4 QC summary is not PASS."
        )

    feasibility = pd.read_parquet(args.feasibility)
    locked_units = pd.read_parquet(args.locked_units)

    required_feasibility = {
        "comparison_uid",
        "gene_trait_uid",
        "gene",
        "candidate_source",
        "candidate_trait_key",
        "candidate_trait_name",
        "comparison_population",
        "unit_analysis_role",
        "mixed_pool",
        "primary_approval_eligible",
        "n_qc_qualified",
        "n_eur_gws_qc",
        "n_comparison_gws_qc",
        "portability_pre_ld_eligible",
        "portability_failure_reason",
        "coloc_input_eligible",
        "coloc_failure_reason",
        "coloc_powered_both_ancestries",
        "sample_metadata_usable",
        "harmonized_parquet",
        "harmonized_sidecar",
    }
    require_columns(
        feasibility,
        required_feasibility,
        "Step 15B4 feasibility table",
    )

    if feasibility["comparison_uid"].duplicated().any():
        raise SystemExit(
            "Step 15B4 comparison_uid is not unique."
        )
    if len(feasibility) != int(
        qc_summary["n_expected_comparisons"]
    ):
        raise SystemExit(
            "Step 15B4 row count differs from its summary."
        )

    comparisons = build_comparison_universe(feasibility)
    units = build_unit_universe(
        comparisons,
        locked_units,
    )

    portability_primary = comparisons[
        comparisons["portability_primary_locked"]
    ].copy()
    portability_sensitivity = comparisons[
        comparisons["portability_sensitivity_locked"]
    ].copy()
    portability_units_primary = units[
        units["portability_primary_locked"]
    ].copy()

    coloc_primary = comparisons[
        comparisons["coloc_primary_locked"]
    ].copy()
    coloc_powered = comparisons[
        comparisons["coloc_powered_primary_locked"]
    ].copy()
    coloc_sensitivity = comparisons[
        comparisons["coloc_sensitivity_locked"]
    ].copy()
    coloc_units_primary = units[
        units["coloc_primary_locked"]
    ].copy()

    outputs: dict[str, Any] = {}

    output_specs = [
        (
            "comparison_universe",
            comparisons,
            "15b5_comparison_analysis_universe",
        ),
        (
            "unit_universe",
            units,
            "15b5_unit_analysis_universe",
        ),
        (
            "portability_primary_comparisons",
            portability_primary,
            "15_portability_comparisons_primary_locked",
        ),
        (
            "portability_sensitivity_comparisons",
            portability_sensitivity,
            "15_portability_comparisons_sensitivity_locked",
        ),
        (
            "portability_primary_units",
            portability_units_primary,
            "15_portability_units_primary_locked",
        ),
        (
            "coloc_primary_comparisons",
            coloc_primary,
            "15_coloc_comparisons_primary_locked",
        ),
        (
            "coloc_powered_comparisons",
            coloc_powered,
            "15_coloc_comparisons_powered_locked",
        ),
        (
            "coloc_sensitivity_comparisons",
            coloc_sensitivity,
            "15_coloc_comparisons_sensitivity_locked",
        ),
        (
            "coloc_primary_units",
            coloc_units_primary,
            "15_coloc_units_primary_locked",
        ),
    ]

    for key, frame, stem in output_specs:
        outputs[key] = write_table(
            frame,
            args.output_dir / f"{stem}.csv",
            args.output_dir / f"{stem}.parquet",
        )

    primary_comparisons = comparisons[
        comparisons["primary_mechanistic_universe"]
    ]
    primary_units = units[
        units["primary_mechanistic_universe"]
    ]

    summary = {
        "step": "15B5",
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "frozen_at_utc": utc_now(),
        "input_checks": {
            "step15b4_status": qc_summary["status"],
            "n_step15b4_comparisons": int(
                len(feasibility)
            ),
            "n_locked_gene_trait_units": int(
                len(locked_units)
            ),
            "approval_flag_matches_primary_nonmixed": True,
        },
        "definitions": {
            "primary_mechanistic_universe": (
                "unit_analysis_role == PRIMARY and mixed_pool == false"
            ),
            "portability_primary": (
                "primary mechanistic comparison and Step 15B4 "
                "portability_pre_ld_eligible == true"
            ),
            "portability_sensitivity": (
                "all Step 15B4 portability-pre-LD-eligible comparisons, "
                "including sensitivity mappings and mixed-pool units"
            ),
            "coloc_primary": (
                "primary mechanistic comparison and Step 15B4 "
                "coloc_input_eligible == true"
            ),
            "coloc_powered_primary": (
                "primary coloc comparison with at least one "
                "QC-qualified genome-wide-significant variant in both "
                "ancestries"
            ),
            "coloc_sensitivity": (
                "all Step 15B4 coloc-input-eligible comparisons, "
                "including sensitivity mappings and mixed-pool units"
            ),
            "prior_sensitivity_rule": (
                "All Step 15D prior settings must be applied to the same "
                "locked primary colocalization comparison universe."
            ),
            "mixed_pool_rule": (
                "Mixed-pool units are excluded from primary analyses and "
                "retained only in sensitivity analyses."
            ),
        },
        "comparison_counts": {
            "all": int(len(comparisons)),
            "primary_nonmixed": int(
                len(primary_comparisons)
            ),
            "portability_primary_pre_ld": int(
                len(portability_primary)
            ),
            "portability_sensitivity_pre_ld": int(
                len(portability_sensitivity)
            ),
            "coloc_primary": int(len(coloc_primary)),
            "coloc_powered_primary": int(
                len(coloc_powered)
            ),
            "coloc_sensitivity": int(
                len(coloc_sensitivity)
            ),
            "coloc_powered_sensitivity": int(
                comparisons[
                    "coloc_powered_sensitivity_locked"
                ].sum()
            ),
        },
        "unit_counts": {
            "all": int(len(units)),
            "primary_nonmixed": int(len(primary_units)),
            "portability_primary_pre_ld": int(
                len(portability_units_primary)
            ),
            "coloc_primary": int(
                len(coloc_units_primary)
            ),
            "coloc_powered_primary": int(
                units[
                    "coloc_powered_primary_locked"
                ].sum()
            ),
        },
        "primary_portability_by_source": {
            str(key): int(value)
            for key, value in portability_primary[
                "candidate_source"
            ].value_counts().items()
        },
        "primary_coloc_by_source": {
            str(key): int(value)
            for key, value in coloc_primary[
                "candidate_source"
            ].value_counts().items()
        },
        "primary_powered_coloc_by_source": {
            str(key): int(value)
            for key, value in coloc_powered[
                "candidate_source"
            ].value_counts().items()
        },
        "portability_failure_reasons_all_comparisons": (
            reason_counts(
                comparisons,
                "portability_failure_reason",
            )
        ),
        "coloc_failure_reasons_all_comparisons": (
            reason_counts(
                comparisons,
                "coloc_failure_reason",
            )
        ),
        "outputs": outputs,
    }

    summary_path = (
        args.output_dir
        / "15b5_analysis_universe_summary.json"
    )
    atomic_json_write(summary, summary_path)

    print("=" * 78)
    print("STEP 15B5 — FROZEN ANALYSIS UNIVERSES")
    print("=" * 78)
    print(
        f"All comparisons:                       "
        f"{summary['comparison_counts']['all']:,}"
    )
    print(
        f"Primary nonmixed comparisons:          "
        f"{summary['comparison_counts']['primary_nonmixed']:,}"
    )
    print(
        f"Primary portability pre-LD:            "
        f"{summary['comparison_counts']['portability_primary_pre_ld']:,}"
    )
    print(
        f"Sensitivity portability pre-LD:        "
        f"{summary['comparison_counts']['portability_sensitivity_pre_ld']:,}"
    )
    print(
        f"Primary coloc comparisons:             "
        f"{summary['comparison_counts']['coloc_primary']:,}"
    )
    print(
        f"Primary powered coloc comparisons:     "
        f"{summary['comparison_counts']['coloc_powered_primary']:,}"
    )
    print(
        f"All locked gene-trait units:           "
        f"{summary['unit_counts']['all']:,}"
    )
    print(
        f"Primary nonmixed units:                "
        f"{summary['unit_counts']['primary_nonmixed']:,}"
    )
    print(
        f"Primary portability units pre-LD:      "
        f"{summary['unit_counts']['portability_primary_pre_ld']:,}"
    )
    print(
        f"Primary coloc units:                   "
        f"{summary['unit_counts']['coloc_primary']:,}"
    )
    print(
        f"Primary powered coloc units:           "
        f"{summary['unit_counts']['coloc_powered_primary']:,}"
    )
    print()
    print("No LD processing, slopes, PP4, or approval models were calculated.")
    print(f"Summary: {summary_path}")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
