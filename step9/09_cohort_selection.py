#!/usr/bin/env python3
"""
Step 9: Cohort-selection analysis for the ancestry-trail sample.

Run from target_ancestry/step9.

Inputs:
  ../step1/output/01_pairs.parquet
  ../step2/output/02_pair_studies.parquet
  ../step4/output/04_trails.parquet
  ../genetic_support/data/merge2.tsv.gz
  ../step8/input/desc2026.gz

Outputs:
  output/09_cohort_selection_pairs.parquet
  output/09_cohort_selection_summary.csv
  output/09_retention_by_pool.csv
  output/09_retention_by_phase.csv
  output/09_retention_by_mesh_category.csv
  output/09_retention_by_indication.csv
  output/09_retention_by_evidence_source.csv
  output/09_cohort_selection_tests.csv
  output/09_cohort_selection.json

The analysis distinguishes:
  1. 1,354 supported clinical-stage pairs (Step 1)
  2. 797 pairs linked to at least one parseable OTG/GWAS study (Step 2)
  3. 790 pairs with a complete ancestry trail (Step 4)

MeSH category retention uses official Cxx disease branches plus F03 Mental
Disorders. Categories are multi-label and are used descriptively only.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np
import pandas as pd
from scipy import stats


REQUIRED_PAIR_COLUMNS = {
    "ti_uid",
    "gene",
    "indication_mesh_id",
    "indication_mesh_term",
    "ccat",
    "pool",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze selection into the final ancestry-trail cohort.")
    parser.add_argument("--pairs", type=Path, default=Path("../step1/output/01_pairs.parquet"))
    parser.add_argument("--pair-studies", type=Path, default=Path("../step2/output/02_pair_studies.parquet"))
    parser.add_argument("--trails", type=Path, default=Path("../step4/output/04_trails.parquet"))
    parser.add_argument("--merge", type=Path, default=Path("../genetic_support/data/merge2.tsv.gz"))
    parser.add_argument("--mesh", type=Path, default=Path("../step8/input/desc2026.gz"))
    parser.add_argument("--mesh-release", default="2026")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--support-threshold", type=float, default=0.8)
    return parser.parse_args()


def normalize_mesh_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    value = str(value).strip().upper()
    value = re.sub(r"^MESH\s*:\s*", "", value)
    return value or None


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


def iter_children_text(element: ET.Element, parent_name: str, child_name: str) -> list[str]:
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


def parse_mesh(mesh_path: Path) -> tuple[dict[str, set[str]], dict[str, str]]:
    id_to_categories: dict[str, set[str]] = defaultdict(set)
    tree_to_name: dict[str, str] = {}

    with open_mesh(mesh_path) as handle:
        for _, element in ET.iterparse(handle, events=("end",)):
            if local_name(element.tag) != "DescriptorRecord":
                continue
            descriptor_ui = find_child_text(element, ["DescriptorUI"])
            descriptor_name = find_child_text(element, ["DescriptorName", "String"])
            trees = iter_children_text(element, "TreeNumberList", "TreeNumber")
            if descriptor_ui:
                for tree in trees:
                    code = category_code(tree)
                    if code:
                        id_to_categories[descriptor_ui].add(code)
                    if descriptor_name:
                        tree_to_name.setdefault(tree, descriptor_name)
            element.clear()

    if not id_to_categories:
        raise RuntimeError(f"No MeSH category mappings parsed from {mesh_path}")
    return id_to_categories, tree_to_name


def category_names(tree_to_name: dict[str, str]) -> dict[str, str]:
    names: dict[str, str] = {}
    for code in sorted({*([f"C{i:02d}" for i in range(1, 27)]), "F03"}):
        if code in tree_to_name:
            names[code] = tree_to_name[code]
    return names


def clean_source_name(value: object) -> str:
    text = "missing" if pd.isna(value) else str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "missing"


def safe_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if not values.empty else math.nan


def safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else math.nan


def standardized_mean_difference(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return math.nan
    pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def binary_smd(a: pd.Series, b: pd.Series) -> float:
    pa = pd.to_numeric(a, errors="coerce").dropna().mean()
    pb = pd.to_numeric(b, errors="coerce").dropna().mean()
    denom = math.sqrt((pa * (1 - pa) + pb * (1 - pb)) / 2)
    return float((pa - pb) / denom) if denom > 0 else 0.0


def compare_numeric(df: pd.DataFrame, variable: str) -> dict:
    inc = pd.to_numeric(df.loc[df["included_final"], variable], errors="coerce").dropna()
    exc = pd.to_numeric(df.loc[~df["included_final"], variable], errors="coerce").dropna()
    if inc.empty or exc.empty:
        p_value = math.nan
    else:
        p_value = float(stats.mannwhitneyu(inc, exc, alternative="two-sided").pvalue)
    return {
        "variable": variable,
        "type": "numeric",
        "included_n": int(inc.size),
        "excluded_n": int(exc.size),
        "included_mean": float(inc.mean()) if not inc.empty else math.nan,
        "excluded_mean": float(exc.mean()) if not exc.empty else math.nan,
        "included_median": float(inc.median()) if not inc.empty else math.nan,
        "excluded_median": float(exc.median()) if not exc.empty else math.nan,
        "standardized_difference": standardized_mean_difference(inc, exc),
        "test": "Mann-Whitney U",
        "p_value": p_value,
    }


def compare_binary(df: pd.DataFrame, variable: str) -> dict:
    table = pd.crosstab(df["included_final"], df[variable].astype(bool)).reindex(
        index=[False, True], columns=[False, True], fill_value=0
    )
    odds_ratio, p_value = stats.fisher_exact(table.to_numpy())
    inc = df.loc[df["included_final"], variable].astype(float)
    exc = df.loc[~df["included_final"], variable].astype(float)
    return {
        "variable": variable,
        "type": "binary",
        "included_n": int(len(inc)),
        "excluded_n": int(len(exc)),
        "included_proportion": float(inc.mean()),
        "excluded_proportion": float(exc.mean()),
        "standardized_difference": binary_smd(inc, exc),
        "test": "Fisher exact",
        "odds_ratio_inclusion": float(odds_ratio),
        "p_value": float(p_value),
    }


def retention_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_total=("ti_uid", "size"),
            n_included=("included_final", "sum"),
            n_linked=("linked_gwas", "sum"),
            n_A=("approval", "sum"),
            n_B=("approval", lambda x: int((x == 0).sum())),
            included_A=("included_final", lambda x: 0),  # filled below
        )
        .reset_index()
    )
    # Pool-specific retained counts require the original rows.
    inc_a = (
        df[df["included_final"] & (df["pool"] == "A")]
        .groupby(group_cols, dropna=False)["ti_uid"].size().rename("included_A")
    )
    inc_b = (
        df[df["included_final"] & (df["pool"] == "B")]
        .groupby(group_cols, dropna=False)["ti_uid"].size().rename("included_B")
    )
    grouped = grouped.drop(columns=["included_A"]).merge(
        inc_a.reset_index(), on=group_cols, how="left"
    ).merge(inc_b.reset_index(), on=group_cols, how="left")
    grouped[["included_A", "included_B"]] = grouped[["included_A", "included_B"]].fillna(0).astype(int)
    grouped["retention_rate"] = grouped["n_included"] / grouped["n_total"]
    grouped["linkage_rate"] = grouped["n_linked"] / grouped["n_total"]
    grouped["retention_A"] = np.where(grouped["n_A"] > 0, grouped["included_A"] / grouped["n_A"], np.nan)
    grouped["retention_B"] = np.where(grouped["n_B"] > 0, grouped["included_B"] / grouped["n_B"], np.nan)
    return grouped.sort_values(["n_total"], ascending=False).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading Step 1 pairs: {args.pairs}")
    pairs = pd.read_parquet(args.pairs)
    missing = REQUIRED_PAIR_COLUMNS - set(pairs.columns)
    if missing:
        raise ValueError(f"Step 1 pairs missing columns: {sorted(missing)}")
    if pairs["ti_uid"].duplicated().any():
        raise ValueError("Step 1 pairs must contain one row per ti_uid")
    if len(pairs) != 1354:
        print(f"WARNING: expected 1,354 Step 1 pairs, observed {len(pairs):,}")

    print(f"Loading Step 2 pair-study rows: {args.pair_studies}")
    pair_studies = pd.read_parquet(args.pair_studies)
    linked_uids = set(pair_studies["ti_uid"].dropna())

    print(f"Loading Step 4 trails: {args.trails}")
    trails = pd.read_parquet(args.trails)
    included_uids = set(trails["ti_uid"].dropna())

    print(f"Loading Minikel associations: {args.merge}")
    merge = pd.read_csv(args.merge, sep="\t", compression="infer", low_memory=False)
    supported_pair_total = int(merge.loc[pd.to_numeric(merge["comb_norm"], errors="coerce") >= args.support_threshold, "ti_uid"].nunique())

    support = merge.loc[
        (pd.to_numeric(merge["comb_norm"], errors="coerce") >= args.support_threshold)
        & merge["ti_uid"].isin(set(pairs["ti_uid"])),
        ["ti_uid", "assoc_source", "assoc_year", "assoc_mesh_id", "arow"],
    ].copy()
    support["assoc_year"] = pd.to_numeric(support["assoc_year"], errors="coerce")
    support["source_clean"] = support["assoc_source"].map(clean_source_name)

    print(f"Parsing MeSH release {args.mesh_release}: {args.mesh}")
    id_to_categories, tree_to_name = parse_mesh(args.mesh)
    code_names = category_names(tree_to_name)

    cohort = pairs.copy()
    cohort["indication_mesh_id"] = cohort["indication_mesh_id"].map(normalize_mesh_id)
    cohort["approval"] = (cohort["pool"] == "A").astype(int)
    cohort["linked_gwas"] = cohort["ti_uid"].isin(linked_uids)
    cohort["included_final"] = cohort["ti_uid"].isin(included_uids)
    cohort["selection_stage"] = np.select(
        [cohort["included_final"], cohort["linked_gwas"]],
        ["complete_ancestry", "linked_no_ancestry"],
        default="not_linked_to_parseable_gwas",
    )

    support_summary = (
        support.groupby("ti_uid", as_index=False)
        .agg(
            n_support_rows=("arow", "size"),
            n_assoc_sources=("source_clean", "nunique"),
            earliest_assoc_year=("assoc_year", "min"),
            median_assoc_year=("assoc_year", "median"),
            latest_assoc_year=("assoc_year", "max"),
            n_distinct_assoc_years=("assoc_year", "nunique"),
            n_assoc_traits=("assoc_mesh_id", "nunique"),
        )
    )
    source_counts = (
        support.groupby(["ti_uid", "source_clean"]).size().unstack(fill_value=0)
    )
    source_counts.columns = [f"n_support_rows_{c}" for c in source_counts.columns]
    source_presence = (source_counts > 0).astype(int)
    source_presence.columns = [c.replace("n_support_rows_", "has_source_") for c in source_presence.columns]
    source_counts = source_counts.reset_index()
    source_presence = source_presence.reset_index()

    study_counts = (
        pair_studies.groupby("ti_uid", as_index=False)
        .agg(n_parseable_studies=("study_id", "nunique"))
    )

    cohort = cohort.merge(support_summary, on="ti_uid", how="left", validate="one_to_one")
    cohort = cohort.merge(source_counts, on="ti_uid", how="left", validate="one_to_one")
    cohort = cohort.merge(source_presence, on="ti_uid", how="left", validate="one_to_one")
    cohort = cohort.merge(study_counts, on="ti_uid", how="left", validate="one_to_one")

    count_cols = [c for c in cohort.columns if c.startswith("n_support_rows_")]
    presence_cols = [c for c in cohort.columns if c.startswith("has_source_")]
    cohort[count_cols + presence_cols + ["n_parseable_studies"]] = cohort[
        count_cols + presence_cols + ["n_parseable_studies"]
    ].fillna(0)

    gene_sizes = cohort.groupby("gene")["ti_uid"].transform("size")
    cohort["n_pairs_for_gene"] = gene_sizes
    cohort["multi_indication_gene"] = gene_sizes > 1

    # Official MeSH multi-label categories for all Step 1 indications.
    category_rows: list[dict] = []
    for row in cohort[["ti_uid", "indication_mesh_id"]].itertuples(index=False):
        for code in sorted(id_to_categories.get(row.indication_mesh_id, set())):
            category_rows.append({"ti_uid": row.ti_uid, "mesh_category_code": code})
    category_long = pd.DataFrame(category_rows)
    if category_long.empty:
        raise RuntimeError("No Step 1 indications mapped to MeSH categories")
    category_long = category_long.drop_duplicates()
    category_long = category_long.merge(
        cohort[["ti_uid", "pool", "approval", "linked_gwas", "included_final"]],
        on="ti_uid",
        how="left",
        validate="many_to_one",
    )
    category_long["mesh_category_name"] = category_long["mesh_category_code"].map(code_names).fillna(
        category_long["mesh_category_code"]
    )

    retention_pool = retention_table(cohort, ["pool"])
    retention_phase = retention_table(cohort, ["ccat"])
    retention_indication = retention_table(cohort, ["indication_mesh_id", "indication_mesh_term"])
    retention_category = retention_table(
        category_long.rename(columns={"mesh_category_code": "category_code", "mesh_category_name": "category_name"}),
        ["category_code", "category_name"],
    )

    source_rows = []
    for col in sorted(presence_cols):
        source = col.replace("has_source_", "")
        present = cohort[col].astype(bool)
        source_rows.append(
            {
                "evidence_source": source,
                "n_pairs": int(present.sum()),
                "n_included": int((present & cohort["included_final"]).sum()),
                "retention_rate": float((present & cohort["included_final"]).sum() / present.sum()) if present.sum() else math.nan,
                "n_A": int((present & (cohort["pool"] == "A")).sum()),
                "n_B": int((present & (cohort["pool"] == "B")).sum()),
            }
        )
    retention_source = pd.DataFrame(source_rows).sort_values("n_pairs", ascending=False)

    # Main included-versus-excluded comparison table.
    summary_rows: list[dict] = []
    numeric_vars = [
        "n_support_rows",
        "n_assoc_sources",
        "earliest_assoc_year",
        "n_distinct_assoc_years",
        "n_assoc_traits",
        "n_pairs_for_gene",
    ]
    # Source-specific row counts are valid evidence-volume variables for all pairs.
    numeric_vars.extend(sorted(count_cols))
    for variable in numeric_vars:
        summary_rows.append(compare_numeric(cohort, variable))
    for variable in ["approval", "multi_indication_gene"]:
        summary_rows.append(compare_binary(cohort, variable))
    summary = pd.DataFrame(summary_rows)

    # Categorical omnibus tests.
    tests: list[dict] = []
    for variable in ["pool", "ccat"]:
        table = pd.crosstab(cohort["included_final"], cohort[variable])
        chi2, p, dof, _ = stats.chi2_contingency(table)
        tests.append(
            {
                "variable": variable,
                "test": "Chi-square",
                "statistic": float(chi2),
                "degrees_of_freedom": int(dof),
                "p_value": float(p),
            }
        )
    tests_df = pd.DataFrame(tests)

    # Save.
    cohort.to_parquet(out / "09_cohort_selection_pairs.parquet", index=False)
    summary.to_csv(out / "09_cohort_selection_summary.csv", index=False)
    retention_pool.to_csv(out / "09_retention_by_pool.csv", index=False)
    retention_phase.to_csv(out / "09_retention_by_phase.csv", index=False)
    retention_category.to_csv(out / "09_retention_by_mesh_category.csv", index=False)
    retention_indication.to_csv(out / "09_retention_by_indication.csv", index=False)
    retention_source.to_csv(out / "09_retention_by_evidence_source.csv", index=False)
    tests_df.to_csv(out / "09_cohort_selection_tests.csv", index=False)

    stage_counts = cohort["selection_stage"].value_counts().to_dict()
    result = {
        "step": "9",
        "mesh_release": str(args.mesh_release),
        "support_threshold": float(args.support_threshold),
        "flow": {
            "supported_pairs": supported_pair_total,
            "clinical_stage_pairs": int(len(cohort)),
            "linked_gwas_pairs": int(cohort["linked_gwas"].sum()),
            "complete_ancestry_pairs": int(cohort["included_final"].sum()),
            "excluded_overall": int((~cohort["included_final"]).sum()),
            "selection_stage_counts": {str(k): int(v) for k, v in stage_counts.items()},
        },
        "retention": {
            "overall": float(cohort["included_final"].mean()),
            "pool_A": float(cohort.loc[cohort["pool"] == "A", "included_final"].mean()),
            "pool_B": float(cohort.loc[cohort["pool"] == "B", "included_final"].mean()),
        },
        "mesh": {
            "unique_categories": int(category_long["mesh_category_code"].nunique()),
            "mapped_pairs": int(category_long["ti_uid"].nunique()),
            "unmapped_pairs": int(len(cohort) - category_long["ti_uid"].nunique()),
        },
        "outputs": [
            "09_cohort_selection_pairs.parquet",
            "09_cohort_selection_summary.csv",
            "09_retention_by_pool.csv",
            "09_retention_by_phase.csv",
            "09_retention_by_mesh_category.csv",
            "09_retention_by_indication.csv",
            "09_retention_by_evidence_source.csv",
            "09_cohort_selection_tests.csv",
        ],
    }
    with (out / "09_cohort_selection.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)

    print("\n" + "=" * 72)
    print("STEP 9 COHORT-SELECTION SUMMARY")
    print("=" * 72)
    print(f"Supported pairs:                {supported_pair_total:,}")
    print(f"Clinical-stage supported pairs: {len(cohort):,}")
    print(f"Linked to parseable GWAS:       {cohort['linked_gwas'].sum():,}")
    print(f"Complete ancestry trails:       {cohort['included_final'].sum():,}")
    print(f"Excluded overall:               {(~cohort['included_final']).sum():,}")
    print("\nSelection stages:")
    print(cohort["selection_stage"].value_counts().to_string())
    print("\nRetention by pool:")
    print(retention_pool.to_string(index=False))
    print("\nLargest MeSH category retention differences:")
    category_display = retention_category.sort_values("retention_rate").head(10)
    print(category_display[["category_code", "category_name", "n_total", "n_included", "retention_rate"]].to_string(index=False))
    print("\nSaved outputs:")
    for name in result["outputs"] + ["09_cohort_selection.json"]:
        print(f"  {out / name}")
    print("=" * 72)


if __name__ == "__main__":
    main()
