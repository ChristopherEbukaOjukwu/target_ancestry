#!/usr/bin/env python3
"""
Step 13B: Convert the 1,624 lexical candidate rows from Step 13A into a
manageable one-row-per-indication scientific review table.

Inputs
------
output/13_panukb_candidate_mappings.csv

Outputs
-------
output/13_panukb_mapping_shortlist.csv
output/13_panukb_exact_matches.csv
output/13_panukb_mapping_shortlist_summary.json

The script does not approve mappings. It presents:
- up to three binary/disease candidates per indication;
- up to two quantitative/biomarker candidates per indication;
- ancestry QC coverage and available sample counts;
- blank manual-review fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

POPS = ["AFR", "AMR", "CSA", "EAS", "EUR", "MID"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--candidates",
        type=Path,
        default=Path("output/13_panukb_candidate_mappings.csv"),
    )
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--n-binary", type=int, default=3)
    p.add_argument("--n-quantitative", type=int, default=2)
    return p.parse_args()


def compact_candidate(row: pd.Series) -> str:
    fields = [
        f"id={row.get('panukb_phenotype_id', '')}",
        f"type={row.get('trait_type', '')}",
        f"description={row.get('description', '')}",
        f"pops={row.get('pops_pass_qc', '')}",
    ]
    for pop in POPS:
        ncol = f"n_cases_{pop}"
        if ncol in row.index and pd.notna(row[ncol]):
            try:
                value = int(float(row[ncol]))
            except Exception:
                value = row[ncol]
            fields.append(f"N_{pop}={value}")
    fields.append(f"rank_score={row.get('candidate_rank_score', '')}")
    fields.append(f"containment={row.get('token_containment', '')}")
    fields.append(f"exact={row.get('exact_normalized_match', '')}")
    return " | ".join(str(x) for x in fields)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.candidates.exists():
        raise SystemExit(f"Missing candidate file: {args.candidates}")

    c = pd.read_csv(args.candidates, low_memory=False)

    required = {
        "indication_mesh_id",
        "indication_mesh_term",
        "n_pairs",
        "n_genes",
        "n_A",
        "n_B",
        "candidate_rank",
        "candidate_rank_score",
        "panukb_phenotype_id",
        "description",
        "trait_type",
        "is_binary",
        "is_quantitative",
        "pops_pass_qc",
    }
    missing = sorted(required - set(c.columns))
    if missing:
        raise SystemExit(f"Candidate file missing columns: {missing}")

    c["is_binary"] = c["is_binary"].astype(str).str.lower().eq("true")
    c["is_quantitative"] = c["is_quantitative"].astype(str).str.lower().eq("true")
    c["exact_normalized_match"] = (
        c["exact_normalized_match"].astype(str).str.lower().eq("true")
    )

    exact = c[c["exact_normalized_match"]].copy()
    exact = exact.sort_values(
        ["indication_mesh_term", "candidate_rank_score"],
        ascending=[True, False],
    )
    exact.to_csv(args.output_dir / "13_panukb_exact_matches.csv", index=False)

    rows = []
    group_cols = [
        "indication_mesh_id",
        "indication_mesh_term",
        "n_pairs",
        "n_genes",
        "n_A",
        "n_B",
    ]

    for keys, g in c.groupby(group_cols, dropna=False, sort=False):
        base = dict(zip(group_cols, keys))
        g = g.sort_values(
            ["exact_normalized_match", "candidate_rank_score"],
            ascending=[False, False],
        )

        binary = g[g["is_binary"]].head(args.n_binary)
        quantitative = g[g["is_quantitative"]].head(args.n_quantitative)

        row = dict(base)
        row["has_exact_candidate"] = bool(g["exact_normalized_match"].any())
        row["best_overall_score"] = float(g["candidate_rank_score"].max())

        for i, (_, candidate) in enumerate(binary.iterrows(), start=1):
            row[f"binary_candidate_{i}"] = compact_candidate(candidate)

        for i, (_, candidate) in enumerate(quantitative.iterrows(), start=1):
            row[f"quantitative_candidate_{i}"] = compact_candidate(candidate)

        row["selected_panukb_phenotype_id"] = ""
        row["mapping_tier"] = ""
        row["mapping_rationale"] = ""
        row["reviewer_notes"] = ""
        row["exclude_reason"] = ""
        rows.append(row)

    shortlist = pd.DataFrame(rows).sort_values(
        ["n_pairs", "indication_mesh_term"],
        ascending=[False, True],
    )

    shortlist_path = args.output_dir / "13_panukb_mapping_shortlist.csv"
    shortlist.to_csv(shortlist_path, index=False)

    summary = {
        "n_indications": int(len(shortlist)),
        "n_with_exact_candidate": int(shortlist["has_exact_candidate"].sum()),
        "n_exact_candidate_rows": int(len(exact)),
        "binary_candidates_shown_per_indication": args.n_binary,
        "quantitative_candidates_shown_per_indication": args.n_quantitative,
        "automatic_mapping_decisions": False,
        "allowed_mapping_tiers": [
            "Direct",
            "Closely-related",
            "Proxy",
            "Exclude",
        ],
    }
    summary_path = args.output_dir / "13_panukb_mapping_shortlist_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("=" * 72)
    print("STEP 13B PAN-UKB MAPPING SHORTLIST")
    print("=" * 72)
    print(f"Indications:                 {len(shortlist):,}")
    print(f"With exact text candidate:   {shortlist['has_exact_candidate'].sum():,}")
    print(f"Exact candidate rows:        {len(exact):,}")
    print(f"Shortlist:                   {shortlist_path}")
    print(f"Exact matches:               {args.output_dir / '13_panukb_exact_matches.csv'}")
    print(f"Summary:                     {summary_path}")
    print()
    print("No phenotype mapping was accepted automatically.")
    print("=" * 72)


if __name__ == "__main__":
    main()
