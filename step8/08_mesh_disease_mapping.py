#!/usr/bin/env python3
"""
Step 8A: Map final target-indication pairs to reference-based MeSH categories.

Inputs (defaults assume execution from target_ancestry/step8):
  ../step4/output/04_trails.parquet
  input/desc2026.gz  (or desc2026.xml)

Outputs:
  output/08_mesh_disease_mapping.csv
  output/08_mesh_disease_mapping.parquet
  output/08_mesh_unmapped_indications.csv
  output/08_pair_mesh_categories_long.parquet
  output/08_pair_mesh_categories.parquet
  output/08_mesh_category_counts.csv
  output/08_mesh_category_cooccurrence.csv
  output/08_mesh_category_overlap.csv
  output/08_mesh_mapping_summary.json

Category rule:
  - Cxx: first branch under the MeSH Diseases hierarchy (for example C04)
  - F03: official MeSH Mental Disorders branch

A descriptor may map to more than one category. No keyword-based disease
classification is used.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import BinaryIO, Iterable, TextIO

import numpy as np
import pandas as pd


REQUIRED_TRAIL_COLUMNS = {
    "ti_uid",
    "gene",
    "pool",
    "indication_mesh_id",
    "indication_mesh_term",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map target indications to official MeSH disease categories."
    )
    parser.add_argument(
        "--trails",
        type=Path,
        default=Path("../step4/output/04_trails.parquet"),
        help="Step 4 pair-level ancestry trail parquet.",
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=Path("input/desc2026.gz"),
        help="Official NLM MeSH descriptor XML (.xml or .gz).",
    )
    parser.add_argument(
        "--mesh-release",
        default="2026",
        help="MeSH release label stored in outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for Step 8A outputs.",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=20,
        help="Minimum pair count for Model 4 category eligibility.",
    )
    parser.add_argument(
        "--min-pool",
        type=int,
        default=5,
        help="Minimum count required in each pool for eligibility.",
    )
    return parser.parse_args()


def normalize_mesh_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    mesh_id = str(value).strip().upper()
    mesh_id = re.sub(r"^MESH\s*:\s*", "", mesh_id)
    return mesh_id or None


def local_name(tag: str) -> str:
    """Remove an optional XML namespace from a tag."""
    return tag.rsplit("}", 1)[-1]


def find_child_text(element: ET.Element, path: Iterable[str]) -> str | None:
    """Namespace-agnostic traversal for a short child path."""
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
    values: list[str] = []
    for child in parent:
        if local_name(child.tag) == child_name and child.text:
            values.append(child.text.strip())
    return values


def open_mesh(path: Path) -> BinaryIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def category_code(tree_number: str) -> str | None:
    """Return an eligible broad category code for a MeSH tree number."""
    if re.match(r"^C\d{2}(?:\.|$)", tree_number):
        return tree_number.split(".", 1)[0]
    if tree_number == "F03" or tree_number.startswith("F03."):
        return "F03"
    return None


def parse_mesh_descriptors(mesh_path: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """Parse descriptor IDs, names, and tree numbers using bounded memory."""
    records: dict[str, dict] = {}
    tree_to_descriptor_name: dict[str, str] = {}

    with open_mesh(mesh_path) as handle:
        for event, element in ET.iterparse(handle, events=("end",)):
            if local_name(element.tag) != "DescriptorRecord":
                continue

            descriptor_ui = find_child_text(element, ["DescriptorUI"])
            descriptor_name = find_child_text(element, ["DescriptorName", "String"])
            tree_numbers = iter_children_text(element, "TreeNumberList", "TreeNumber")

            if descriptor_ui:
                records[descriptor_ui] = {
                    "descriptor_name": descriptor_name,
                    "tree_numbers": tree_numbers,
                }
                if descriptor_name:
                    for tree in tree_numbers:
                        tree_to_descriptor_name.setdefault(tree, descriptor_name)

            element.clear()

    if not records:
        raise RuntimeError(f"No DescriptorRecord elements parsed from {mesh_path}")
    return records, tree_to_descriptor_name


def validate_trails(trails: pd.DataFrame) -> None:
    missing = REQUIRED_TRAIL_COLUMNS - set(trails.columns)
    if missing:
        raise ValueError(f"Missing required trail columns: {sorted(missing)}")
    if trails["ti_uid"].isna().any():
        raise ValueError("ti_uid contains missing values")
    if trails["ti_uid"].duplicated().any():
        examples = trails.loc[trails["ti_uid"].duplicated(False), "ti_uid"].head().tolist()
        raise ValueError(f"Step 4 must have one row per pair; duplicates include {examples}")
    invalid_pools = set(trails["pool"].dropna().astype(str)) - {"A", "B"}
    if invalid_pools:
        raise ValueError(f"Unexpected pool values: {sorted(invalid_pools)}")


def build_indication_mapping(
    indications: pd.DataFrame,
    records: dict[str, dict],
    tree_to_name: dict[str, str],
    mesh_release: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapped_rows: list[dict] = []
    unmapped_rows: list[dict] = []

    for row in indications.itertuples(index=False):
        mesh_id = row.indication_mesh_id
        input_term = row.indication_mesh_term
        record = records.get(mesh_id)

        if record is None:
            unmapped_rows.append(
                {
                    "indication_mesh_id": mesh_id,
                    "indication_mesh_term": input_term,
                    "reason": "mesh_id_not_found_in_descriptor_xml",
                }
            )
            continue

        descriptor_name = record["descriptor_name"]
        by_category: dict[str, list[str]] = defaultdict(list)
        for tree in record["tree_numbers"]:
            code = category_code(tree)
            if code:
                by_category[code].append(tree)

        if not by_category:
            unmapped_rows.append(
                {
                    "indication_mesh_id": mesh_id,
                    "indication_mesh_term": input_term,
                    "mesh_descriptor_name": descriptor_name,
                    "all_tree_numbers": "|".join(sorted(record["tree_numbers"])),
                    "reason": "no_Cxx_or_F03_tree_position",
                }
            )
            continue

        for code, tree_numbers in sorted(by_category.items()):
            category_name = tree_to_name.get(code)
            mapped_rows.append(
                {
                    "mesh_release": str(mesh_release),
                    "indication_mesh_id": mesh_id,
                    "indication_mesh_term": input_term,
                    "mesh_descriptor_name": descriptor_name,
                    "mesh_category_code": code,
                    "mesh_category_name": category_name or code,
                    "tree_numbers": "|".join(sorted(set(tree_numbers))),
                    "n_tree_numbers_in_category": len(set(tree_numbers)),
                    "term_matches_descriptor": (
                        str(input_term).strip().casefold()
                        == str(descriptor_name).strip().casefold()
                        if descriptor_name is not None and input_term is not None
                        else False
                    ),
                }
            )

    mapping = pd.DataFrame(mapped_rows)
    unmapped = pd.DataFrame(unmapped_rows)

    if not mapping.empty:
        mapping = mapping.sort_values(
            ["mesh_category_code", "indication_mesh_term", "indication_mesh_id"]
        ).reset_index(drop=True)
    if not unmapped.empty:
        unmapped = unmapped.sort_values(
            ["reason", "indication_mesh_term", "indication_mesh_id"]
        ).reset_index(drop=True)
    return mapping, unmapped


def build_pair_outputs(
    trails: pd.DataFrame,
    mapping: pd.DataFrame,
    min_pairs: int,
    min_pool: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    pair_base = trails[
        ["ti_uid", "gene", "pool", "indication_mesh_id", "indication_mesh_term"]
    ].copy()

    pair_long = pair_base.merge(
        mapping,
        on=["indication_mesh_id", "indication_mesh_term"],
        how="inner",
        validate="many_to_many",
    )
    pair_long = pair_long.drop_duplicates(
        ["ti_uid", "mesh_category_code"]
    ).sort_values(["ti_uid", "mesh_category_code"])

    categories = sorted(pair_long["mesh_category_code"].unique())
    indicator = pd.crosstab(pair_long["ti_uid"], pair_long["mesh_category_code"])
    indicator = indicator.reindex(columns=categories, fill_value=0).clip(upper=1).astype("int8")
    indicator.columns = [f"mesh_{column}" for column in indicator.columns]
    indicator = indicator.reset_index()

    pair_wide = pair_base.merge(indicator, on="ti_uid", how="left", validate="one_to_one")
    indicator_cols = [column for column in pair_wide.columns if column.startswith("mesh_")]
    pair_wide[indicator_cols] = pair_wide[indicator_cols].fillna(0).astype("int8")
    pair_wide["mesh_category_count"] = pair_wide[indicator_cols].sum(axis=1)
    pair_wide["mesh_category_codes"] = pair_wide.apply(
        lambda row: [column.removeprefix("mesh_") for column in indicator_cols if row[column] == 1],
        axis=1,
    )

    category_counts = (
        pair_long.groupby(["mesh_category_code", "mesh_category_name"], as_index=False)
        .agg(
            n_indications=("indication_mesh_id", "nunique"),
            n_pairs=("ti_uid", "nunique"),
            n_A=("pool", lambda values: (values == "A").sum()),
            n_B=("pool", lambda values: (values == "B").sum()),
            n_genes=("gene", "nunique"),
        )
    )
    category_counts["eligible_model4"] = (
        (category_counts["n_pairs"] >= min_pairs)
        & (category_counts["n_A"] >= min_pool)
        & (category_counts["n_B"] >= min_pool)
    )
    category_counts["eligibility_rule"] = (
        f"n_pairs>={min_pairs}; n_A>={min_pool}; n_B>={min_pool}"
    )
    category_counts = category_counts.sort_values(
        ["eligible_model4", "n_pairs", "mesh_category_code"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    # Pair-level co-occurrence counts for category indicators.
    matrix = pair_wide.set_index("ti_uid")[indicator_cols].astype(int)
    cooccurrence = matrix.T.dot(matrix)
    cooccurrence.index = [index.removeprefix("mesh_") for index in cooccurrence.index]
    cooccurrence.columns = [column.removeprefix("mesh_") for column in cooccurrence.columns]

    # Pairwise overlap diagnostics: exact duplicate and Jaccard similarity.
    overlap_rows: list[dict] = []
    for i, left in enumerate(indicator_cols):
        left_values = matrix[left].astype(bool)
        for right in indicator_cols[i + 1 :]:
            right_values = matrix[right].astype(bool)
            intersection = int((left_values & right_values).sum())
            union = int((left_values | right_values).sum())
            jaccard = intersection / union if union else np.nan
            overlap_rows.append(
                {
                    "category_1": left.removeprefix("mesh_"),
                    "category_2": right.removeprefix("mesh_"),
                    "n_category_1": int(left_values.sum()),
                    "n_category_2": int(right_values.sum()),
                    "intersection": intersection,
                    "union": union,
                    "jaccard": jaccard,
                    "exact_duplicate": bool((left_values == right_values).all()),
                }
            )
    overlap = pd.DataFrame(overlap_rows)
    if not overlap.empty:
        overlap = overlap.sort_values(
            ["exact_duplicate", "jaccard", "intersection"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    eligible_codes = category_counts.loc[
        category_counts["eligible_model4"], "mesh_category_code"
    ].tolist()
    eligible_cols = [f"mesh_{code}" for code in eligible_codes]
    eligible_matrix = pair_wide[eligible_cols].to_numpy(dtype=float) if eligible_cols else np.empty((len(pair_wide), 0))
    rank = int(np.linalg.matrix_rank(eligible_matrix)) if eligible_cols else 0

    diagnostics = {
        "n_pairs": int(pair_base["ti_uid"].nunique()),
        "n_pairs_mapped": int(pair_long["ti_uid"].nunique()),
        "n_pairs_unmapped": int(pair_base["ti_uid"].nunique() - pair_long["ti_uid"].nunique()),
        "n_categories": len(categories),
        "eligible_category_codes": eligible_codes,
        "n_eligible_categories": len(eligible_codes),
        "eligible_indicator_rank": rank,
        "eligible_indicator_full_rank": bool(rank == len(eligible_cols)),
        "exact_duplicate_category_pairs": (
            overlap.loc[overlap["exact_duplicate"], ["category_1", "category_2"]]
            .to_dict("records")
            if not overlap.empty
            else []
        ),
        "highest_nonduplicate_jaccard": (
            float(overlap.loc[~overlap["exact_duplicate"], "jaccard"].max())
            if not overlap.empty and (~overlap["exact_duplicate"]).any()
            else None
        ),
    }
    return pair_long, pair_wide, category_counts, cooccurrence, overlap, diagnostics


def main() -> int:
    args = parse_args()

    if not args.trails.exists():
        raise FileNotFoundError(f"Step 4 trails not found: {args.trails}")
    if not args.mesh.exists():
        raise FileNotFoundError(
            f"MeSH descriptor file not found: {args.mesh}\n"
            "Download the official NLM descriptor file into step8/input first."
        )
    if args.min_pairs < 1 or args.min_pool < 1:
        raise ValueError("--min-pairs and --min-pool must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Step 4 trails: {args.trails}")
    trails = pd.read_parquet(args.trails)
    validate_trails(trails)
    trails = trails.copy()
    trails["indication_mesh_id"] = trails["indication_mesh_id"].map(normalize_mesh_id)
    if trails["indication_mesh_id"].isna().any():
        raise ValueError("Some final pairs have missing indication_mesh_id values")

    indications = (
        trails[["indication_mesh_id", "indication_mesh_term"]]
        .drop_duplicates()
        .sort_values(["indication_mesh_id", "indication_mesh_term"])
        .reset_index(drop=True)
    )
    term_counts = indications.groupby("indication_mesh_id")["indication_mesh_term"].nunique()
    conflicting_terms = term_counts[term_counts > 1]
    if not conflicting_terms.empty:
        raise ValueError(
            "A MeSH ID maps to multiple indication terms in Step 4: "
            f"{conflicting_terms.index.tolist()[:10]}"
        )

    print(f"  Pairs: {len(trails):,}")
    print(f"  Unique indications: {len(indications):,}")
    print(f"Parsing official MeSH descriptors: {args.mesh}")
    records, tree_to_name = parse_mesh_descriptors(args.mesh)
    print(f"  Descriptor records parsed: {len(records):,}")

    mapping, unmapped = build_indication_mapping(
        indications=indications,
        records=records,
        tree_to_name=tree_to_name,
        mesh_release=args.mesh_release,
    )

    pair_long, pair_wide, counts, cooccurrence, overlap, diagnostics = build_pair_outputs(
        trails=trails,
        mapping=mapping,
        min_pairs=args.min_pairs,
        min_pool=args.min_pool,
    )

    # Output paths.
    mapping_csv = args.output_dir / "08_mesh_disease_mapping.csv"
    mapping_parquet = args.output_dir / "08_mesh_disease_mapping.parquet"
    unmapped_csv = args.output_dir / "08_mesh_unmapped_indications.csv"
    pair_long_parquet = args.output_dir / "08_pair_mesh_categories_long.parquet"
    pair_wide_parquet = args.output_dir / "08_pair_mesh_categories.parquet"
    counts_csv = args.output_dir / "08_mesh_category_counts.csv"
    cooccurrence_csv = args.output_dir / "08_mesh_category_cooccurrence.csv"
    overlap_csv = args.output_dir / "08_mesh_category_overlap.csv"
    summary_json = args.output_dir / "08_mesh_mapping_summary.json"

    mapping.to_csv(mapping_csv, index=False)
    mapping.to_parquet(mapping_parquet, index=False)
    unmapped.to_csv(unmapped_csv, index=False)
    pair_long.to_parquet(pair_long_parquet, index=False)
    pair_wide.to_parquet(pair_wide_parquet, index=False)
    counts.to_csv(counts_csv, index=False)
    cooccurrence.to_csv(cooccurrence_csv, index=True, index_label="mesh_category_code")
    overlap.to_csv(overlap_csv, index=False)

    summary = {
        "mesh_release": str(args.mesh_release),
        "mesh_file": str(args.mesh),
        "mesh_file_sha256": sha256_file(args.mesh),
        "trails_file": str(args.trails),
        "n_pairs": int(len(trails)),
        "n_unique_indications": int(len(indications)),
        "n_mapped_indications": int(mapping["indication_mesh_id"].nunique()) if not mapping.empty else 0,
        "n_multicategory_indications": int(
            (mapping.groupby("indication_mesh_id")["mesh_category_code"].nunique() > 1).sum()
        ) if not mapping.empty else 0,
        "n_unmapped_indications": int(len(unmapped)),
        "category_rule": "Cxx first disease branch plus F03 Mental Disorders",
        "model4_min_pairs": int(args.min_pairs),
        "model4_min_each_pool": int(args.min_pool),
        **diagnostics,
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 72)
    print("STEP 8A SUMMARY")
    print("=" * 72)
    print(f"MeSH release:                   {args.mesh_release}")
    print(f"Unique indications:             {summary['n_unique_indications']}")
    print(f"Mapped indications:             {summary['n_mapped_indications']}")
    print(f"Multi-category indications:     {summary['n_multicategory_indications']}")
    print(f"Unmapped indications:           {summary['n_unmapped_indications']}")
    print(f"Pairs with >=1 category:        {summary['n_pairs_mapped']}")
    print(f"Pairs without category:         {summary['n_pairs_unmapped']}")
    print(f"Categories found:               {summary['n_categories']}")
    print(f"Categories eligible for Model 4:{summary['n_eligible_categories']}")
    print(f"Eligible indicator full rank:   {summary['eligible_indicator_full_rank']}")
    print("\nCategory counts:")
    display_cols = [
        "mesh_category_code",
        "mesh_category_name",
        "n_indications",
        "n_pairs",
        "n_A",
        "n_B",
        "n_genes",
        "eligible_model4",
    ]
    print(counts[display_cols].to_string(index=False))

    if not unmapped.empty:
        print("\nUnmapped indications:")
        print(unmapped.to_string(index=False))

    print("\nSaved outputs:")
    for path in [
        mapping_csv,
        mapping_parquet,
        unmapped_csv,
        pair_long_parquet,
        pair_wide_parquet,
        counts_csv,
        cooccurrence_csv,
        overlap_csv,
        summary_json,
    ]:
        print(f"  {path}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
