#!/usr/bin/env python3
"""
Step 13E: Lock the manually adjudicated direct Pan-UKB phenotype mappings.

This script selects exactly one QC-passing Pan-UKB phenotype for each accepted
direct indication. It creates a provenance-rich mapping manifest and the
corresponding subset of target-indication pairs.

Inputs
------
../step4/output/04_trails.parquet
output/13_panukb_direct_candidates_conservative.csv
input/manifests/phenotype_manifest.tsv.bgz

Outputs
-------
output/13_panukb_direct_mapping_locked.csv
output/13_panukb_direct_pairs_locked.parquet
output/13_panukb_direct_pairs_locked.csv
output/13_panukb_direct_coverage_by_population.csv
output/13_panukb_direct_mapping_locked_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


# Manually adjudicated direct mappings.
# Rule: exact disease endpoint preferred; broader/narrower endpoints excluded.
LOCKED = {
    "D019973": {
        "panukb_phenotype_id": "phecode|317|both_sexes||",
        "rationale": "Exact disease endpoint: Alcohol-related disorders.",
    },
    "D001249": {
        "panukb_phenotype_id": "icd10|J45|both_sexes||",
        "rationale": "Exact disease endpoint: ICD-10 J45 Asthma.",
    },
    "D029481": {
        "panukb_phenotype_id": "phecode|496.2|both_sexes||",
        "rationale": "Exact disease endpoint: Chronic bronchitis; excludes narrower obstructive chronic bronchitis.",
    },
    "D003920": {
        "panukb_phenotype_id": "phecode|250|both_sexes||",
        "rationale": "Exact broad endpoint: Diabetes mellitus; excludes the narrower type 2 endpoint.",
    },
    "D003924": {
        "panukb_phenotype_id": "phecode|250.2|both_sexes||",
        "rationale": "Exact disease endpoint: Type 2 diabetes.",
    },
    "D005764": {
        "panukb_phenotype_id": "phecode|530.11|both_sexes||",
        "rationale": "Exact endpoint: GERD; excludes broader esophagitis/related-disease phenotype.",
    },
    "D006973": {
        "panukb_phenotype_id": "phecode|401|both_sexes||",
        "rationale": "Exact broad endpoint: Hypertension; excludes narrower essential hypertension.",
    },
    "D009765": {
        "panukb_phenotype_id": "icd10|E66|both_sexes||",
        "rationale": "Exact disease endpoint: ICD-10 E66 Obesity.",
    },
    "D010003": {
        "panukb_phenotype_id": "phecode|740|both_sexes||",
        "rationale": "Exact endpoint synonym: Osteoarthrosis for osteoarthritis; excludes site-specific hip/knee arthrosis.",
    },
    "D029424": {
        "panukb_phenotype_id": "icd10|J44|both_sexes||",
        "rationale": "Exact disease endpoint: ICD-10 J44 chronic obstructive pulmonary disease.",
    },
    "D014029": {
        "panukb_phenotype_id": "phecode|318|both_sexes||",
        "rationale": "Exact disease endpoint: Tobacco use disorder.",
    },
}

POPS = ["AFR", "AMR", "CSA", "EAS", "EUR", "MID"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--trails",
        type=Path,
        default=Path("../step4/output/04_trails.parquet"),
    )
    p.add_argument(
        "--candidates",
        type=Path,
        default=Path("output/13_panukb_direct_candidates_conservative.csv"),
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("input/manifests/phenotype_manifest.tsv.bgz"),
    )
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    return p.parse_args()


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for path in [args.trails, args.candidates, args.manifest]:
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")

    trails = pd.read_parquet(args.trails)
    candidates = pd.read_csv(args.candidates, low_memory=False)

    required_trails = {
        "ti_uid", "gene", "pool", "indication_mesh_id", "indication_mesh_term"
    }
    missing = sorted(required_trails - set(trails.columns))
    if missing:
        raise SystemExit(f"Trails file missing columns: {missing}")

    required_candidates = {
        "indication_mesh_id", "indication_mesh_term", "panukb_phenotype_id",
        "trait_type", "phenocode", "description", "pops_pass_qc",
        "num_pops_pass_qc", "filename", "filename_tabix",
        "aws_path", "aws_path_tabix",
    }
    missing = sorted(required_candidates - set(candidates.columns))
    if missing:
        raise SystemExit(f"Candidate file missing columns: {missing}")

    # Select exactly one candidate row per locked indication.
    rows = []
    for mesh_id, decision in LOCKED.items():
        match = candidates[
            candidates["indication_mesh_id"].eq(mesh_id)
            & candidates["panukb_phenotype_id"].eq(
                decision["panukb_phenotype_id"]
            )
        ].copy()

        if len(match) != 1:
            raise SystemExit(
                f"Expected exactly one candidate for {mesh_id} -> "
                f"{decision['panukb_phenotype_id']}; found {len(match)}."
            )

        row = match.iloc[0].to_dict()
        row["mapping_tier"] = "Direct"
        row["mapping_status"] = "Locked"
        row["mapping_rationale"] = decision["rationale"]
        row["adjudication_rule"] = (
            "Exact disease endpoint or exact accepted clinical synonym; "
            "broader and narrower alternatives excluded."
        )
        row["source_resource"] = "Pan-UK Biobank"
        row["source_manifest_filename"] = args.manifest.name
        row["source_manifest_md5"] = digest(args.manifest, "md5")
        row["source_manifest_sha256"] = digest(args.manifest, "sha256")
        row["source_manifest_size_bytes"] = args.manifest.stat().st_size
        row["genome_build"] = "GRCh37"
        rows.append(row)

    locked = pd.DataFrame(rows).sort_values("indication_mesh_term")

    if locked["indication_mesh_id"].duplicated().any():
        raise SystemExit("Duplicate indication IDs in locked mappings.")
    if locked["panukb_phenotype_id"].duplicated().any():
        duplicates = locked.loc[
            locked["panukb_phenotype_id"].duplicated(False),
            ["indication_mesh_id", "panukb_phenotype_id"],
        ]
        raise SystemExit(
            "A Pan-UKB phenotype was assigned to multiple indications:\n"
            + duplicates.to_string(index=False)
        )

    locked_path = args.output_dir / "13_panukb_direct_mapping_locked.csv"
    locked.to_csv(locked_path, index=False)

    # Join mappings to the original pair-level universe.
    pair_cols = [
        "indication_mesh_id", "panukb_phenotype_id", "trait_type", "phenocode",
        "description", "pops_pass_qc", "num_pops_pass_qc",
        "filename", "filename_tabix", "aws_path", "aws_path_tabix",
        "mapping_tier", "mapping_status", "mapping_rationale",
        "source_resource", "source_manifest_filename",
        "source_manifest_md5", "source_manifest_sha256", "genome_build",
    ]
    pairs = trails.merge(
        locked[pair_cols],
        on="indication_mesh_id",
        how="inner",
        validate="many_to_one",
    )

    pairs_parquet = args.output_dir / "13_panukb_direct_pairs_locked.parquet"
    pairs_csv = args.output_dir / "13_panukb_direct_pairs_locked.csv"
    pairs.to_parquet(pairs_parquet, index=False)
    pairs.to_csv(pairs_csv, index=False)

    # Coverage by ancestry population using pops_pass_qc only.
    coverage_rows = []
    for pop in POPS:
        mask = pairs["pops_pass_qc"].fillna("").str.split(",").map(
            lambda values, p=pop: p in [str(v).strip() for v in values]
        )
        subset = pairs.loc[mask]
        coverage_rows.append({
            "population": pop,
            "n_pairs": int(subset["ti_uid"].nunique()),
            "n_indications": int(subset["indication_mesh_id"].nunique()),
            "n_genes": int(subset["gene"].nunique()),
            "n_A": int((subset["pool"] == "A").sum()),
            "n_B": int((subset["pool"] == "B").sum()),
        })

    coverage = pd.DataFrame(coverage_rows)
    coverage_path = (
        args.output_dir / "13_panukb_direct_coverage_by_population.csv"
    )
    coverage.to_csv(coverage_path, index=False)

    summary = {
        "n_locked_indications": int(locked["indication_mesh_id"].nunique()),
        "n_locked_phenotypes": int(locked["panukb_phenotype_id"].nunique()),
        "n_pairs": int(pairs["ti_uid"].nunique()),
        "n_genes": int(pairs["gene"].nunique()),
        "n_pool_A": int((pairs["pool"] == "A").sum()),
        "n_pool_B": int((pairs["pool"] == "B").sum()),
        "fraction_of_790_pairs": float(pairs["ti_uid"].nunique() / 790),
        "pool_A_fraction_of_157": float((pairs["pool"] == "A").sum() / 157),
        "pool_B_fraction_of_633": float((pairs["pool"] == "B").sum() / 633),
        "population_coverage": coverage.to_dict("records"),
        "primary_scope": (
            "Direct disease mappings only; ancestry comparisons retained "
            "separately according to Pan-UKB pops_pass_qc."
        ),
        "excluded_scope": (
            "No proxy biomarkers, closely related phenotypes, categorical "
            "self-report fields, or failed-QC ancestry strata."
        ),
    }

    summary_path = (
        args.output_dir / "13_panukb_direct_mapping_locked_summary.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("=" * 76)
    print("STEP 13E LOCKED PAN-UKB DIRECT MAPPINGS")
    print("=" * 76)
    print(f"Locked indications: {summary['n_locked_indications']:,}")
    print(f"Locked phenotypes:  {summary['n_locked_phenotypes']:,}")
    print(f"Pairs:              {summary['n_pairs']:,} / 790")
    print(f"Pool A / Pool B:    {summary['n_pool_A']:,} / {summary['n_pool_B']:,}")
    print(f"Unique genes:       {summary['n_genes']:,}")
    print("\nPopulation coverage:")
    print(coverage.to_string(index=False))
    print("\nOutputs:")
    print(f"  {locked_path}")
    print(f"  {pairs_parquet}")
    print(f"  {pairs_csv}")
    print(f"  {coverage_path}")
    print(f"  {summary_path}")
    print("=" * 76)


if __name__ == "__main__":
    main()
