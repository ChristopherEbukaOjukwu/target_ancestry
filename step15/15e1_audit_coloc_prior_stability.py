#!/usr/bin/env python3
"""
Step 15E1 — Audit colocalization prior stability.

Reads Step 15E unit-level results and classifies each powered
gene-trait-source unit across all tested p12 values.

A unit is:
- ROBUST_SHARED if PP.H4 >= 0.8 at every tested p12;
- ROBUST_DISTINCT if PP.H3 >= 0.8 at every tested p12;
- STABLE_MODERATE_SHARED if PP.H4 >= 0.5 at every tested p12 but the
  ROBUST_SHARED threshold is not met;
- PRIOR_SENSITIVE otherwise.

The primary-prior category is also retained separately.

Outputs
-------
output/15e1_coloc_prior_stability_units.parquet
output/15e1_coloc_prior_stability_units.csv
output/15e1_coloc_prior_stability_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "15E1.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unit-results",
        type=Path,
        default=Path(
            "output/15e_coloc_unit_results.parquet"
        ),
    )
    parser.add_argument(
        "--primary-p12",
        type=float,
        default=1e-5,
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
    return str(value).strip()


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
    raise TypeError(type(value).__name__)


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


def category(
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


def main() -> int:
    args = parse_args()
    if not args.unit_results.exists():
        raise SystemExit(
            f"Missing input: {args.unit_results}"
        )

    frame = pd.read_parquet(
        args.unit_results
    )
    required = {
        "gene_trait_uid",
        "gene",
        "candidate_trait_name",
        "candidate_source",
        "comparison_ancestries",
        "n_comparison_ancestries",
        "p12_requested",
        "aggregation_scope",
        "pp_h3_median",
        "pp_h4_median",
        "pp_h4_given_h3_h4_median",
        "sensitivity_pool_assignment",
    }
    missing = sorted(
        required - set(frame.columns)
    )
    if missing:
        raise SystemExit(
            f"Missing required columns: {missing}"
        )

    powered = frame[
        frame[
            "aggregation_scope"
        ].eq("POWERED_BOTH_ONLY")
    ].copy()

    priors = sorted(
        powered[
            "p12_requested"
        ].astype(float)
        .drop_duplicates()
        .tolist()
    )
    if not priors:
        raise SystemExit(
            "No powered-unit results found."
        )

    rows = []
    metadata_columns = [
        "gene_trait_uid",
        "gene",
        "candidate_trait_name",
        "candidate_source",
        "comparison_ancestries",
        "n_comparison_ancestries",
        "sensitivity_pool_assignment",
    ]

    for gene_trait_uid, group in powered.groupby(
        "gene_trait_uid",
        sort=True,
    ):
        group = group.sort_values(
            "p12_requested",
            kind="mergesort",
        )
        observed_priors = sorted(
            group[
                "p12_requested"
            ].astype(float)
            .tolist()
        )
        if observed_priors != priors:
            raise RuntimeError(
                f"{gene_trait_uid} does not contain "
                "the complete prior set."
            )

        metadata = {}
        for column in metadata_columns:
            values = group[
                column
            ].drop_duplicates()
            if len(values) != 1:
                raise RuntimeError(
                    f"{column} varies within "
                    f"{gene_trait_uid}: "
                    f"{values.tolist()}"
                )
            metadata[column] = values.iloc[0]

        h3 = group[
            "pp_h3_median"
        ].astype(float).to_numpy()
        h4 = group[
            "pp_h4_median"
        ].astype(float).to_numpy()
        conditional = group[
            "pp_h4_given_h3_h4_median"
        ].astype(float).to_numpy()

        categories = [
            category(
                float(pp_h3),
                float(pp_h4),
            )
            for pp_h3, pp_h4 in zip(
                h3,
                h4,
            )
        ]

        if np.all(h4 >= 0.8):
            stability = "ROBUST_SHARED"
        elif np.all(h3 >= 0.8):
            stability = "ROBUST_DISTINCT"
        elif np.all(h4 >= 0.5):
            stability = (
                "STABLE_MODERATE_SHARED"
            )
        else:
            stability = "PRIOR_SENSITIVE"

        primary_match = np.isclose(
            group[
                "p12_requested"
            ].astype(float),
            args.primary_p12,
            rtol=0.0,
            atol=1e-15,
        )
        if int(primary_match.sum()) != 1:
            raise RuntimeError(
                f"{gene_trait_uid} lacks exactly one "
                "primary-prior result."
            )
        primary = group.loc[
            primary_match
        ].iloc[0]

        output = {
            **metadata,
            "n_priors": int(
                len(priors)
            ),
            "p12_values": " || ".join(
                f"{value:g}"
                for value in priors
            ),
            "prior_stability_class": (
                stability
            ),
            "primary_p12": float(
                args.primary_p12
            ),
            "primary_pp_h3": float(
                primary[
                    "pp_h3_median"
                ]
            ),
            "primary_pp_h4": float(
                primary[
                    "pp_h4_median"
                ]
            ),
            "primary_pp_h4_given_h3_h4": float(
                primary[
                    "pp_h4_given_h3_h4_median"
                ]
            ),
            "primary_category": category(
                float(
                    primary[
                        "pp_h3_median"
                    ]
                ),
                float(
                    primary[
                        "pp_h4_median"
                    ]
                ),
            ),
            "minimum_pp_h4": float(
                np.min(h4)
            ),
            "maximum_pp_h4": float(
                np.max(h4)
            ),
            "minimum_pp_h3": float(
                np.min(h3)
            ),
            "maximum_pp_h3": float(
                np.max(h3)
            ),
            "minimum_conditional_h4": float(
                np.min(conditional)
            ),
            "maximum_conditional_h4": float(
                np.max(conditional)
            ),
            "categories_across_priors": (
                " || ".join(categories)
            ),
        }

        for prior, pp_h3, pp_h4, cond, cat in zip(
            priors,
            h3,
            h4,
            conditional,
            categories,
        ):
            suffix = (
                f"{prior:.0e}"
                .replace("-", "m")
                .replace("+", "p")
            )
            output[
                f"pp_h3_p12_{suffix}"
            ] = float(pp_h3)
            output[
                f"pp_h4_p12_{suffix}"
            ] = float(pp_h4)
            output[
                f"conditional_h4_p12_{suffix}"
            ] = float(cond)
            output[
                f"category_p12_{suffix}"
            ] = cat

        rows.append(output)

    result = pd.DataFrame(rows).sort_values(
        [
            "prior_stability_class",
            "primary_pp_h4",
            "gene",
        ],
        ascending=[
            True,
            False,
            True,
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    parquet_path = (
        args.output_dir
        / (
            "15e1_coloc_prior_stability_"
            "units.parquet"
        )
    )
    csv_path = (
        args.output_dir
        / (
            "15e1_coloc_prior_stability_"
            "units.csv"
        )
    )
    summary_path = (
        args.output_dir
        / (
            "15e1_coloc_prior_stability_"
            "summary.json"
        )
    )

    result.to_parquet(
        parquet_path,
        index=False,
    )
    result.to_csv(
        csv_path,
        index=False,
    )

    summary = {
        "step": "15E1",
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "n_powered_units": int(
            len(result)
        ),
        "p12_values": [
            float(value)
            for value in priors
        ],
        "primary_p12": float(
            args.primary_p12
        ),
        "prior_stability_counts": {
            clean(key): int(value)
            for key, value in (
                result[
                    "prior_stability_class"
                ]
                .value_counts()
                .items()
            )
        },
        "primary_category_counts": {
            clean(key): int(value)
            for key, value in (
                result[
                    "primary_category"
                ]
                .value_counts()
                .items()
            )
        },
        "by_pool": {
            clean(pool): {
                "n_units": int(
                    len(group)
                ),
                "prior_stability_counts": {
                    clean(key): int(value)
                    for key, value in (
                        group[
                            "prior_stability_class"
                        ]
                        .value_counts()
                        .items()
                    )
                },
                "primary_category_counts": {
                    clean(key): int(value)
                    for key, value in (
                        group[
                            "primary_category"
                        ]
                        .value_counts()
                        .items()
                    )
                },
            }
            for pool, group in result.groupby(
                "sensitivity_pool_assignment",
                dropna=False,
            )
        },
        "outputs": {
            "parquet": str(
                parquet_path
            ),
            "csv": str(csv_path),
        },
        "completed_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    }
    atomic_json_write(
        summary,
        summary_path,
    )

    print("=" * 78)
    print(
        "STEP 15E1 — COLOCALIZATION PRIOR STABILITY"
    )
    print("=" * 78)
    print("Status: PASS")
    print(
        "Powered units:",
        len(result),
    )
    print(
        "Prior-stability counts:"
    )
    for key, value in summary[
        "prior_stability_counts"
    ].items():
        print(
            f"  {key}: {value}"
        )
    print(
        "Primary-prior categories:"
    )
    for key, value in summary[
        "primary_category_counts"
    ].items():
        print(
            f"  {key}: {value}"
        )
    print()
    print(
        result[
            [
                "gene",
                "candidate_trait_name",
                "candidate_source",
                "comparison_ancestries",
                "sensitivity_pool_assignment",
                "prior_stability_class",
                "primary_category",
                "minimum_pp_h4",
                "primary_pp_h4",
                "maximum_pp_h4",
            ]
        ].to_string(index=False)
    )
    print()
    print(
        "Summary:",
        summary_path,
    )
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
