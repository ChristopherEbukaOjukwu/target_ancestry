#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_VERSION = "15E2.2"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--comparison-results",
        type=Path,
        default=Path(
            "output/15e_coloc_comparison_results.parquet"
        ),
    )
    p.add_argument(
        "--primary-p12",
        type=float,
        default=1e-5,
    )
    p.add_argument(
        "--unit-universe",
        type=Path,
        default=Path(
            "output/15b5_unit_analysis_universe.parquet"
        ),
        help=(
            "Frozen Step 15B5 unit metadata used to attach "
            "sensitivity_pool_assignment by gene_trait_uid."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
    )
    return p.parse_args()


def as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {
        "true", "1", "yes", "y"
    }


def evidence_category(h3, h4):
    if h4 >= 0.8:
        return "SHARED_H4_GE_0P8"
    if h3 >= 0.8:
        return "DISTINCT_H3_GE_0P8"
    if h4 >= 0.5:
        return "MODERATE_SHARED_H4_GE_0P5"
    return "INCONCLUSIVE"


def comparison_class(group):
    h3 = group["pp_h3"].astype(float).to_numpy()
    h4 = group["pp_h4"].astype(float).to_numpy()
    if np.all(h4 >= 0.8):
        return "ROBUST_SHARED"
    if np.all(h3 >= 0.8):
        return "ROBUST_DISTINCT"
    return "PRIOR_SENSITIVE"


def unit_class(classes):
    classes = set(classes)
    if classes == {"ROBUST_SHARED"}:
        return "ROBUST_SHARED"
    if classes == {"ROBUST_DISTINCT"}:
        return "ROBUST_DISTINCT"
    if {
        "ROBUST_SHARED",
        "ROBUST_DISTINCT",
    }.issubset(classes):
        return "ANCESTRY_DISCORDANT"
    if len(classes) > 1 and "PRIOR_SENSITIVE" in classes:
        return "MIXED_WITH_PRIOR_SENSITIVITY"
    return "PRIOR_SENSITIVE"


def main():
    a = parse_args()
    if not a.comparison_results.exists():
        raise SystemExit(
            f"Missing input: {a.comparison_results}"
        )

    df = pd.read_parquet(a.comparison_results)
    required = {
        "comparison_uid",
        "gene_trait_uid",
        "gene",
        "candidate_trait_name",
        "candidate_source",
        "comparison_population",
        "coloc_powered_both_ancestries",
        "p12_requested",
        "pp_h3",
        "pp_h4",
        "pp_h4_given_h3_h4",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(
            f"Missing columns: {missing}"
        )

    # Step 15E comparison results do not necessarily carry the
    # unit-level pool assignment. Attach it from the frozen Step 15B5
    # unit universe before constructing comparison and unit summaries.
    if "sensitivity_pool_assignment" not in df.columns:
        if not a.unit_universe.exists():
            raise SystemExit(
                "comparison results lack "
                "sensitivity_pool_assignment and the unit universe "
                f"was not found: {a.unit_universe}"
            )

        unit_meta = pd.read_parquet(
            a.unit_universe
        )
        unit_required = {
            "gene_trait_uid",
            "sensitivity_pool_assignment",
        }
        unit_missing = sorted(
            unit_required
            - set(unit_meta.columns)
        )
        if unit_missing:
            raise SystemExit(
                "Unit universe missing columns: "
                f"{unit_missing}"
            )

        unit_meta = unit_meta[
            [
                "gene_trait_uid",
                "sensitivity_pool_assignment",
            ]
        ].drop_duplicates()

        duplicate_units = unit_meta[
            "gene_trait_uid"
        ].duplicated(
            keep=False
        )
        if duplicate_units.any():
            conflicts = (
                unit_meta.loc[
                    duplicate_units
                ]
                .groupby(
                    "gene_trait_uid"
                )[
                    "sensitivity_pool_assignment"
                ]
                .nunique(
                    dropna=False
                )
            )
            conflicts = conflicts[
                conflicts > 1
            ]
            if len(conflicts):
                raise SystemExit(
                    "Conflicting pool assignments in unit universe: "
                    + ", ".join(
                        conflicts.index.astype(str).tolist()[:20]
                    )
                )
            unit_meta = unit_meta.drop_duplicates(
                "gene_trait_uid"
            )

        df = df.merge(
            unit_meta,
            on="gene_trait_uid",
            how="left",
            validate="many_to_one",
        )

    missing_pool = df[
        "sensitivity_pool_assignment"
    ].isna() | df[
        "sensitivity_pool_assignment"
    ].astype(str).str.strip().eq("")
    if missing_pool.any():
        missing_units = sorted(
            df.loc[
                missing_pool,
                "gene_trait_uid",
            ].astype(str).unique().tolist()
        )
        raise SystemExit(
            "Missing sensitivity_pool_assignment after metadata merge "
            "for units: "
            + ", ".join(
                missing_units[:20]
            )
        )

    powered = df[
        df[
            "coloc_powered_both_ancestries"
        ].map(as_bool)
    ].copy()

    priors = sorted(
        powered["p12_requested"]
        .astype(float)
        .unique()
        .tolist()
    )

    comp_rows = []
    comp_group = [
        "comparison_uid",
        "gene_trait_uid",
        "gene",
        "candidate_trait_name",
        "candidate_source",
        "comparison_population",
        "sensitivity_pool_assignment",
    ]

    for keys, group in powered.groupby(
        comp_group,
        sort=True,
        dropna=False,
    ):
        meta = dict(zip(comp_group, keys))
        group = group.sort_values(
            "p12_requested",
            kind="mergesort",
        )
        observed = sorted(
            group["p12_requested"]
            .astype(float)
            .tolist()
        )
        if observed != priors:
            raise RuntimeError(
                f"Incomplete prior set: "
                f"{meta['comparison_uid']}"
            )

        primary_mask = np.isclose(
            group["p12_requested"].astype(float),
            a.primary_p12,
            rtol=0.0,
            atol=1e-15,
        )
        if int(primary_mask.sum()) != 1:
            raise RuntimeError(
                f"Missing primary prior: "
                f"{meta['comparison_uid']}"
            )
        primary = group.loc[
            primary_mask
        ].iloc[0]

        comp_rows.append(
            {
                **meta,
                "comparison_prior_stability":
                    comparison_class(group),
                "primary_p12": float(
                    a.primary_p12
                ),
                "primary_pp_h3": float(
                    primary["pp_h3"]
                ),
                "primary_pp_h4": float(
                    primary["pp_h4"]
                ),
                "primary_conditional_h4": float(
                    primary[
                        "pp_h4_given_h3_h4"
                    ]
                ),
                "primary_category":
                    evidence_category(
                        float(primary["pp_h3"]),
                        float(primary["pp_h4"]),
                    ),
                "minimum_pp_h3": float(
                    group["pp_h3"].min()
                ),
                "maximum_pp_h3": float(
                    group["pp_h3"].max()
                ),
                "minimum_pp_h4": float(
                    group["pp_h4"].min()
                ),
                "maximum_pp_h4": float(
                    group["pp_h4"].max()
                ),
            }
        )

    comp = pd.DataFrame(comp_rows).sort_values(
        ["gene", "comparison_population"],
        kind="mergesort",
    ).reset_index(drop=True)

    unit_rows = []
    unit_group = [
        "gene_trait_uid",
        "gene",
        "candidate_trait_name",
        "candidate_source",
        "sensitivity_pool_assignment",
    ]

    for keys, group in comp.groupby(
        unit_group,
        sort=True,
        dropna=False,
    ):
        meta = dict(zip(unit_group, keys))
        classes = set(
            group[
                "comparison_prior_stability"
            ].astype(str)
        )
        primary_categories = set(
            group["primary_category"].astype(str)
        )

        if primary_categories == {
            "SHARED_H4_GE_0P8"
        }:
            primary_unit = "CONSISTENT_SHARED"
        elif primary_categories == {
            "DISTINCT_H3_GE_0P8"
        }:
            primary_unit = "CONSISTENT_DISTINCT"
        elif {
            "SHARED_H4_GE_0P8",
            "DISTINCT_H3_GE_0P8",
        }.issubset(primary_categories):
            primary_unit = "ANCESTRY_DISCORDANT"
        else:
            primary_unit = "MIXED_OR_INCONCLUSIVE"

        unit_rows.append(
            {
                **meta,
                "n_powered_comparisons": int(
                    len(group)
                ),
                "comparison_ancestries":
                    " || ".join(
                        sorted(
                            group[
                                "comparison_population"
                            ].astype(str).unique()
                        )
                    ),
                "comparison_stability_classes":
                    " || ".join(
                        sorted(classes)
                    ),
                "unit_stability_final":
                    unit_class(classes),
                "primary_unit_category":
                    primary_unit,
                "primary_pp_h4_minimum": float(
                    group["primary_pp_h4"].min()
                ),
                "primary_pp_h4_median": float(
                    group["primary_pp_h4"].median()
                ),
                "primary_pp_h4_maximum": float(
                    group["primary_pp_h4"].max()
                ),
            }
        )

    units = pd.DataFrame(unit_rows).sort_values(
        ["unit_stability_final", "gene"],
        kind="mergesort",
    ).reset_index(drop=True)

    a.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    comp_pq = a.output_dir / (
        "15e2_coloc_comparison_stability.parquet"
    )
    comp_csv = a.output_dir / (
        "15e2_coloc_comparison_stability.csv"
    )
    unit_pq = a.output_dir / (
        "15e2_coloc_unit_stability_final.parquet"
    )
    unit_csv = a.output_dir / (
        "15e2_coloc_unit_stability_final.csv"
    )
    summary_path = a.output_dir / (
        "15e2_coloc_stability_summary.json"
    )

    comp.to_parquet(comp_pq, index=False)
    comp.to_csv(comp_csv, index=False)
    units.to_parquet(unit_pq, index=False)
    units.to_csv(unit_csv, index=False)

    summary = {
        "step": "15E2",
        "script_version": SCRIPT_VERSION,
        "status": "PASS",
        "n_powered_comparisons": int(len(comp)),
        "n_powered_units": int(len(units)),
        "comparison_stability_counts":
            comp[
                "comparison_prior_stability"
            ].value_counts().to_dict(),
        "unit_stability_counts":
            units[
                "unit_stability_final"
            ].value_counts().to_dict(),
        "primary_unit_category_counts":
            units[
                "primary_unit_category"
            ].value_counts().to_dict(),
        "completed_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print("=" * 78)
    print(
        "STEP 15E2 — FINAL COLOCALIZATION STABILITY"
    )
    print("=" * 78)
    print("Status: PASS")
    print("Powered comparisons:", len(comp))
    print("Powered units:", len(units))
    print("Comparison stability:")
    for k, v in summary[
        "comparison_stability_counts"
    ].items():
        print(f"  {k}: {v}")
    print("Final unit stability:")
    for k, v in summary[
        "unit_stability_counts"
    ].items():
        print(f"  {k}: {v}")
    print()
    print(
        units[
            [
                "gene",
                "candidate_trait_name",
                "candidate_source",
                "comparison_ancestries",
                "sensitivity_pool_assignment",
                "unit_stability_final",
                "primary_unit_category",
                "primary_pp_h4_minimum",
                "primary_pp_h4_median",
                "primary_pp_h4_maximum",
            ]
        ].to_string(index=False)
    )
    print()
    print("Summary:", summary_path)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
