#!/usr/bin/env python3
"""
Step 15E0 — Colocalization input preflight.

This script does not calculate colocalization. It inventories the frozen
Step 15B5 colocalization universe and the Step 15B4 harmonized regional
files so that Step 15E uses the actual stored schema rather than inferred
column names.

It reports:
- the located primary colocalization universe;
- comparison and gene-trait-source unit counts;
- all columns related to eligibility, power, trait type, sample size,
  case/control status, priors, and ancestry;
- counts for Boolean-like eligibility/power fields;
- one representative harmonized comparison per source;
- availability and numeric quality of beta, SE, p-value, AF/MAF, and
  sample-size fields;
- candidate trait-type fields needed to choose coloc effect priors.

Outputs
-------
output/15e0_coloc_input_preflight.json
output/15e0_coloc_manifest_schema.csv
output/15e0_coloc_harmonized_schema.csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "15E0.1"

PREFERRED_MANIFESTS = (
    Path("output/15_coloc_comparisons_primary_locked.parquet"),
    Path("output/15b5_coloc_comparisons_primary_locked.parquet"),
    Path("output/15b5_colocalization_comparisons_primary_locked.parquet"),
)

PREFERRED_UNIT_UNIVERSES = (
    Path("output/15b5_unit_analysis_universe.parquet"),
    Path("output/15_coloc_units_primary_locked.parquet"),
)

FIELD_PATTERNS = {
    "identity": (
        r"comparison_uid|gene_trait_uid|gene$|trait|source|population|"
        r"ancestry|genome_build|chrom|start|end"
    ),
    "eligibility_power": (
        r"eligible|powered|power|coloc|primary|mixed|analysis_role|locked"
    ),
    "trait_type": (
        r"trait_type|phenotype_type|outcome_type|data_type|binary|"
        r"case_control|casecontrol|quantitative|continuous|dichotom"
    ),
    "sample_size": (
        r"(^|_)n($|_)|sample|case|control|effective|neff"
    ),
    "prior": r"prior|sdY|variance|var_y",
}

HARMONIZED_CANDIDATES = {
    "variant_id": (
        "variant_id",
        "snp",
        "rsid",
    ),
    "eur_beta": (
        "eur_beta",
        "beta_eur",
    ),
    "eur_se": (
        "eur_se",
        "se_eur",
        "eur_standard_error",
    ),
    "comparison_beta": (
        "comparison_beta",
        "beta_comparison",
        "non_eur_beta",
    ),
    "comparison_se": (
        "comparison_se",
        "se_comparison",
        "non_eur_se",
    ),
    "eur_p": (
        "eur_p",
        "p_eur",
        "eur_pvalue",
        "eur_p_value",
    ),
    "comparison_p": (
        "comparison_p",
        "p_comparison",
        "non_eur_p",
        "comparison_pvalue",
    ),
    "eur_af": (
        "eur_af",
        "af_eur",
        "eur_effect_allele_frequency",
    ),
    "comparison_af": (
        "comparison_af",
        "af_comparison",
        "non_eur_af",
    ),
    "eur_maf": (
        "eur_maf",
        "maf_eur",
    ),
    "comparison_maf": (
        "comparison_maf",
        "maf_comparison",
        "non_eur_maf",
    ),
    "eur_n": (
        "eur_n",
        "n_eur",
        "eur_sample_size",
        "eur_neff",
    ),
    "comparison_n": (
        "comparison_n",
        "n_comparison",
        "comparison_sample_size",
        "comparison_neff",
        "non_eur_n",
    ),
    "eur_n_case": (
        "eur_n_case",
        "eur_n_cases",
        "eur_cases",
        "eur_ncase",
    ),
    "eur_n_control": (
        "eur_n_control",
        "eur_n_controls",
        "eur_controls",
        "eur_nctrl",
    ),
    "comparison_n_case": (
        "comparison_n_case",
        "comparison_n_cases",
        "comparison_cases",
        "comparison_ncase",
    ),
    "comparison_n_control": (
        "comparison_n_control",
        "comparison_n_controls",
        "comparison_controls",
        "comparison_nctrl",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional explicit primary colocalization manifest.",
    )
    parser.add_argument(
        "--unit-universe",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--harmonized-dir",
        type=Path,
        default=Path("intermediate/15b4/comparisons"),
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        return None if not math.isfinite(numeric) else numeric
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def atomic_json_write(data: dict[str, Any], path: Path) -> None:
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


def locate_file(
    explicit: Path | None,
    preferred: tuple[Path, ...],
    globs: tuple[str, ...],
    label: str,
) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise SystemExit(f"{label} does not exist: {explicit}")
        return explicit

    for path in preferred:
        if path.exists():
            return path

    matches: list[Path] = []
    for pattern in globs:
        matches.extend(Path(".").glob(pattern))
    matches = sorted(
        {
            path
            for path in matches
            if path.is_file()
        },
        key=lambda path: str(path),
    )

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SystemExit(
            f"Could not locate {label}. Tried preferred paths and "
            f"patterns: {list(globs)}"
        )
    raise SystemExit(
        f"Multiple possible {label} files found; pass the path explicitly:\n"
        + "\n".join(f"  {path}" for path in matches)
    )


def field_groups(columns: list[str]) -> dict[str, list[str]]:
    output = {}
    for group, pattern in FIELD_PATTERNS.items():
        output[group] = [
            column
            for column in columns
            if re.search(
                pattern,
                column,
                flags=re.IGNORECASE,
            )
        ]
    return output


def boolean_like_counts(series: pd.Series) -> dict[str, int]:
    rendered = (
        series.map(clean)
        .replace("", "<MISSING>")
        .value_counts(dropna=False)
    )
    return {
        str(key): int(value)
        for key, value in rendered.items()
    }


def resolve_candidate(
    columns: list[str],
    candidates: tuple[str, ...],
) -> str | None:
    lookup = {
        column.casefold(): column
        for column in columns
    }
    for candidate in candidates:
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    return None


def numeric_summary(
    frame: pd.DataFrame,
    column: str,
) -> dict[str, Any]:
    numeric = pd.to_numeric(
        frame[column],
        errors="coerce",
    )
    finite = numeric[
        np.isfinite(numeric)
    ]
    return {
        "column": column,
        "n_rows": int(len(frame)),
        "n_nonmissing": int(numeric.notna().sum()),
        "n_finite": int(len(finite)),
        "minimum": (
            float(finite.min())
            if len(finite)
            else None
        ),
        "median": (
            float(finite.median())
            if len(finite)
            else None
        ),
        "maximum": (
            float(finite.max())
            if len(finite)
            else None
        ),
        "n_nonpositive": int(
            (finite <= 0).sum()
        ),
    }


def manifest_schema_frame(
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    groups = field_groups(
        manifest.columns.tolist()
    )
    reverse_groups: dict[str, list[str]] = {
        column: []
        for column in manifest.columns
    }
    for group, columns in groups.items():
        for column in columns:
            reverse_groups[column].append(group)

    rows = []
    for column in manifest.columns:
        example_values = (
            manifest[column]
            .dropna()
            .map(clean)
            .loc[lambda values: values.ne("")]
            .drop_duplicates()
            .head(5)
            .tolist()
        )
        rows.append(
            {
                "column": column,
                "dtype": str(
                    manifest[column].dtype
                ),
                "n_nonmissing": int(
                    manifest[column].notna().sum()
                ),
                "n_unique_nonmissing": int(
                    manifest[column].nunique(
                        dropna=True
                    )
                ),
                "groups": " | ".join(
                    reverse_groups[column]
                ),
                "example_values": " || ".join(
                    example_values
                ),
            }
        )
    return pd.DataFrame(rows)


def representative_rows(
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    source_column = (
        "candidate_source"
        if "candidate_source" in manifest.columns
        else None
    )
    sort_columns = [
        column
        for column in [
            "candidate_source",
            "candidate_trait_name",
            "gene",
            "comparison_population",
            "comparison_uid",
        ]
        if column in manifest.columns
    ]
    ordered = manifest.sort_values(
        sort_columns,
        kind="mergesort",
    ) if sort_columns else manifest.copy()

    if source_column:
        return (
            ordered.groupby(
                source_column,
                sort=True,
                dropna=False,
            )
            .head(1)
            .reset_index(drop=True)
        )
    return ordered.head(2).reset_index(drop=True)


def inspect_harmonized_file(
    path: Path,
    source: str,
    comparison_uid: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frame = pd.read_parquet(path)
    columns = frame.columns.tolist()

    resolved_fields = {
        semantic: resolve_candidate(
            columns,
            candidates,
        )
        for semantic, candidates in HARMONIZED_CANDIDATES.items()
    }

    numeric_fields = {}
    for semantic, column in resolved_fields.items():
        if (
            column is not None
            and semantic != "variant_id"
        ):
            numeric_fields[semantic] = (
                numeric_summary(
                    frame,
                    column,
                )
            )

    schema_rows = []
    for column in columns:
        example_values = (
            frame[column]
            .dropna()
            .map(clean)
            .loc[lambda values: values.ne("")]
            .drop_duplicates()
            .head(5)
            .tolist()
        )
        schema_rows.append(
            {
                "comparison_uid": comparison_uid,
                "candidate_source": source,
                "path": str(path),
                "column": column,
                "dtype": str(
                    frame[column].dtype
                ),
                "n_nonmissing": int(
                    frame[column].notna().sum()
                ),
                "n_unique_nonmissing": int(
                    frame[column].nunique(
                        dropna=True
                    )
                ),
                "example_values": " || ".join(
                    example_values
                ),
            }
        )

    return {
        "comparison_uid": comparison_uid,
        "candidate_source": source,
        "path": str(path),
        "n_rows": int(len(frame)),
        "n_columns": int(len(columns)),
        "columns": columns,
        "resolved_fields": resolved_fields,
        "numeric_fields": numeric_fields,
    }, schema_rows


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = locate_file(
        args.manifest,
        PREFERRED_MANIFESTS,
        (
            "output/*coloc*comparison*primary*locked*.parquet",
            "output/*colocalization*comparison*primary*locked*.parquet",
            "output/*coloc*primary*.parquet",
        ),
        "primary colocalization comparison manifest",
    )
    manifest = pd.read_parquet(
        manifest_path
    )

    if "comparison_uid" not in manifest.columns:
        raise SystemExit(
            f"{manifest_path} has no comparison_uid column."
        )
    if manifest["comparison_uid"].duplicated().any():
        raise SystemExit(
            "Primary colocalization manifest has duplicate comparison_uid."
        )

    unit_path = None
    unit_universe = None
    try:
        unit_path = locate_file(
            args.unit_universe,
            PREFERRED_UNIT_UNIVERSES,
            (
                "output/*unit*analysis*universe*.parquet",
                "output/*coloc*unit*.parquet",
            ),
            "unit universe",
        )
        unit_universe = pd.read_parquet(
            unit_path
        )
    except SystemExit:
        if args.unit_universe is not None:
            raise

    manifest_schema = manifest_schema_frame(
        manifest
    )
    manifest_schema_path = (
        args.output_dir
        / "15e0_coloc_manifest_schema.csv"
    )
    manifest_schema.to_csv(
        manifest_schema_path,
        index=False,
    )

    groups = field_groups(
        manifest.columns.tolist()
    )

    boolean_fields = {}
    for column in groups[
        "eligibility_power"
    ]:
        unique_count = manifest[
            column
        ].nunique(
            dropna=True
        )
        if unique_count <= 20:
            boolean_fields[column] = (
                boolean_like_counts(
                    manifest[column]
                )
            )

    representatives = representative_rows(
        manifest
    )
    harmonized_inspections = []
    harmonized_schema_rows = []
    missing_harmonized = []

    for row in representatives.itertuples(
        index=False
    ):
        row_dict = row._asdict()
        comparison_uid = clean(
            row_dict["comparison_uid"]
        )
        source = clean(
            row_dict.get(
                "candidate_source",
                "",
            )
        )
        path = (
            args.harmonized_dir
            / f"{comparison_uid}.parquet"
        )
        if not path.exists():
            missing_harmonized.append(
                str(path)
            )
            continue

        inspection, schema_rows = (
            inspect_harmonized_file(
                path,
                source,
                comparison_uid,
            )
        )
        harmonized_inspections.append(
            inspection
        )
        harmonized_schema_rows.extend(
            schema_rows
        )

    harmonized_schema_path = (
        args.output_dir
        / "15e0_coloc_harmonized_schema.csv"
    )
    pd.DataFrame(
        harmonized_schema_rows
    ).to_csv(
        harmonized_schema_path,
        index=False,
    )

    count_summary: dict[str, Any] = {
        "n_comparisons": int(
            len(manifest)
        ),
        "n_gene_trait_units": (
            int(
                manifest[
                    "gene_trait_uid"
                ].nunique()
            )
            if "gene_trait_uid"
            in manifest.columns
            else None
        ),
        "comparisons_by_source": (
            {
                clean(key): int(value)
                for key, value in (
                    manifest.groupby(
                        "candidate_source",
                        dropna=False,
                    )
                    .size()
                    .items()
                )
            }
            if "candidate_source"
            in manifest.columns
            else {}
        ),
        "comparisons_by_population": (
            {
                clean(key): int(value)
                for key, value in (
                    manifest.groupby(
                        "comparison_population",
                        dropna=False,
                    )
                    .size()
                    .items()
                )
            }
            if "comparison_population"
            in manifest.columns
            else {}
        ),
    }

    if unit_universe is not None:
        count_summary[
            "unit_universe_path"
        ] = str(unit_path)
        count_summary[
            "unit_universe_rows"
        ] = int(
            len(unit_universe)
        )
        count_summary[
            "unit_universe_columns"
        ] = unit_universe.columns.tolist()

    summary = {
        "step": "15E0",
        "script_version": SCRIPT_VERSION,
        "status": (
            "PASS"
            if harmonized_inspections
            and not missing_harmonized
            else "PASS_WITH_WARNINGS"
        ),
        "manifest_path": str(
            manifest_path
        ),
        "harmonized_dir": str(
            args.harmonized_dir
        ),
        "counts": count_summary,
        "manifest_columns": (
            manifest.columns.tolist()
        ),
        "manifest_field_groups": (
            groups
        ),
        "eligibility_power_value_counts": (
            boolean_fields
        ),
        "representative_harmonized_files": (
            harmonized_inspections
        ),
        "missing_representative_harmonized_files": (
            missing_harmonized
        ),
        "outputs": {
            "manifest_schema": str(
                manifest_schema_path
            ),
            "harmonized_schema": str(
                harmonized_schema_path
            ),
        },
        "completed_at_utc": utc_now(),
    }
    summary_path = (
        args.output_dir
        / "15e0_coloc_input_preflight.json"
    )
    atomic_json_write(
        summary,
        summary_path,
    )

    print("=" * 78)
    print(
        "STEP 15E0 — COLOCALIZATION INPUT PREFLIGHT"
    )
    print("=" * 78)
    print("Status:", summary["status"])
    print("Manifest:", manifest_path)
    print(
        "Comparisons:",
        count_summary[
            "n_comparisons"
        ],
    )
    print(
        "Gene-trait-source units:",
        count_summary[
            "n_gene_trait_units"
        ],
    )
    print(
        "Sources:",
        count_summary[
            "comparisons_by_source"
        ],
    )
    print(
        "Populations:",
        count_summary[
            "comparisons_by_population"
        ],
    )
    print()
    print("Eligibility/power-related fields:")
    for column in groups[
        "eligibility_power"
    ]:
        print(
            " ",
            column,
            boolean_fields.get(
                column,
                "<many values>",
            ),
        )
    print()
    print("Trait-type candidates:")
    for column in groups[
        "trait_type"
    ]:
        values = (
            manifest[column]
            .dropna()
            .map(clean)
            .loc[
                lambda items: items.ne("")
            ]
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        print(
            " ",
            column,
            "=>",
            values,
        )
    print()
    print("Sample-size candidates:")
    for column in groups[
        "sample_size"
    ]:
        print(" ", column)
    print()
    for inspection in harmonized_inspections:
        print(
            "Representative harmonized file:",
            inspection[
                "candidate_source"
            ],
            "|",
            inspection[
                "comparison_uid"
            ],
        )
        print(
            "  rows:",
            inspection["n_rows"],
        )
        print(
            "  resolved fields:",
            inspection[
                "resolved_fields"
            ],
        )
    if missing_harmonized:
        print()
        print(
            "Missing representative harmonized files:"
        )
        for path in missing_harmonized:
            print(" ", path)
    print()
    print("Summary:", summary_path)
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
