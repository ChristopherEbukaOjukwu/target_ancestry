#!/usr/bin/env python3
"""
Step 13C: Generate Pan-UKB direct-disease mapping candidates using official
2026 MeSH descriptor names and entry terms.

This script does NOT accept mappings automatically.

Inputs
------
../step4/output/04_trails.parquet
input/manifests/desc2026.gz
output/13_panukb_eligible_phenotypes.csv

Outputs
-------
output/13_mesh_terms_for_indications.csv
output/13_panukb_direct_mapping_candidates.csv
output/13_panukb_direct_mapping_review.csv
output/13_panukb_direct_mapping_summary.json

Primary candidate restriction
-----------------------------
Only Pan-UKB phenotype types `icd10` and `phecode` are considered direct-disease
candidates. Categorical/self-report fields and quantitative biomarkers are
excluded from this direct shortlist and may be considered separately later.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DIRECT_TYPES = {"icd10", "phecode"}
POPS = ["AFR", "AMR", "CSA", "EAS", "EUR", "MID"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--trails",
        type=Path,
        default=Path("../step4/output/04_trails.parquet"),
    )
    p.add_argument(
        "--mesh",
        type=Path,
        default=Path("input/manifests/desc2026.gz"),
    )
    p.add_argument(
        "--eligible",
        type=Path,
        default=Path("output/13_panukb_eligible_phenotypes.csv"),
    )
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--top-k", type=int, default=10)
    return p.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize(value: Any) -> str:
    text = clean_text(value).casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"\b(icd|phecode)\b", " ", text)
    text = re.sub(r"^[a-z]\d+(?:\.\d+)?\s+", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


STOPWORDS = {
    "a", "an", "and", "or", "of", "the", "with", "without", "in", "on",
    "other", "specified", "unspecified", "nos", "nec",
}


def tokens(value: Any) -> set[str]:
    return {
        t for t in normalize(value).split()
        if len(t) > 1 and t not in STOPWORDS
    }


def singularize_token(token: str) -> str:
    # Conservative normalization for common English plurals only.
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ses") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def normalized_tokens(value: Any) -> set[str]:
    return {singularize_token(t) for t in tokens(value)}


def phrase_metrics(mesh_term: str, phenotype_text: str) -> dict[str, float]:
    a = normalize(mesh_term)
    b = normalize(phenotype_text)
    ta = normalized_tokens(mesh_term)
    tb = normalized_tokens(phenotype_text)

    exact = float(bool(a) and a == b)
    seq = SequenceMatcher(None, a, b).ratio() if a and b else 0.0

    if ta and tb:
        intersection = len(ta & tb)
        containment = intersection / min(len(ta), len(tb))
        jaccard = intersection / len(ta | tb)
    else:
        containment = 0.0
        jaccard = 0.0

    phrase_inclusion = float(bool(a) and bool(b) and (a in b or b in a))

    score = (
        120.0 * exact
        + 65.0 * containment
        + 35.0 * jaccard
        + 20.0 * phrase_inclusion
        + 15.0 * seq
    )
    return {
        "score": score,
        "exact": exact,
        "containment": containment,
        "jaccard": jaccard,
        "phrase_inclusion": phrase_inclusion,
        "sequence_similarity": seq,
    }


def parse_mesh_descriptors(path: Path) -> dict[str, dict[str, Any]]:
    """
    Parse MeSH DescriptorRecord elements using streaming XML parsing.
    Stores preferred descriptor name and all Term/String entry terms.
    """
    records: dict[str, dict[str, Any]] = {}

    with gzip.open(path, "rb") as handle:
        for event, elem in ET.iterparse(handle, events=("end",)):
            if elem.tag != "DescriptorRecord":
                continue

            ui = clean_text(elem.findtext("DescriptorUI"))
            preferred = clean_text(elem.findtext("DescriptorName/String"))

            terms: set[str] = set()
            if preferred:
                terms.add(preferred)

            for term_node in elem.findall(".//TermList/Term/String"):
                term = clean_text(term_node.text)
                if term:
                    terms.add(term)

            tree_numbers = [
                clean_text(node.text)
                for node in elem.findall(".//TreeNumberList/TreeNumber")
                if clean_text(node.text)
            ]

            records[ui] = {
                "mesh_id": ui,
                "preferred_term": preferred,
                "entry_terms": sorted(terms),
                "tree_numbers": tree_numbers,
            }
            elem.clear()

    return records


def phenotype_search_text(row: pd.Series) -> list[str]:
    values = [
        clean_text(row.get("description")),
        clean_text(row.get("description_more")),
        clean_text(row.get("coding_description")),
    ]
    values = [v for v in values if v]
    return list(dict.fromkeys(values))


def score_candidate(mesh_terms: list[str], phenotype_texts: list[str]) -> dict[str, Any]:
    best: dict[str, Any] | None = None

    for mesh_term in mesh_terms:
        for phenotype_text in phenotype_texts:
            metrics = phrase_metrics(mesh_term, phenotype_text)
            candidate = {
                **metrics,
                "matched_mesh_term": mesh_term,
                "matched_phenotype_text": phenotype_text,
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate

    if best is None:
        return {
            "score": 0.0,
            "exact": 0.0,
            "containment": 0.0,
            "jaccard": 0.0,
            "phrase_inclusion": 0.0,
            "sequence_similarity": 0.0,
            "matched_mesh_term": "",
            "matched_phenotype_text": "",
        }
    return best


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for path in [args.trails, args.mesh, args.eligible]:
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")

    trails = pd.read_parquet(args.trails)
    required_trails = {
        "ti_uid", "gene", "pool", "indication_mesh_id", "indication_mesh_term"
    }
    missing = sorted(required_trails - set(trails.columns))
    if missing:
        raise SystemExit(f"Trails file missing columns: {missing}")

    indications = (
        trails.groupby(["indication_mesh_id", "indication_mesh_term"], dropna=False)
        .agg(
            n_pairs=("ti_uid", "nunique"),
            n_genes=("gene", "nunique"),
            n_A=("pool", lambda s: int((s == "A").sum())),
            n_B=("pool", lambda s: int((s == "B").sum())),
        )
        .reset_index()
    )

    print(f"Parsing official MeSH descriptors: {args.mesh}")
    mesh = parse_mesh_descriptors(args.mesh)
    print(f"  MeSH descriptors parsed: {len(mesh):,}")

    mesh_rows = []
    missing_mesh = []
    for _, row in indications.iterrows():
        mesh_id = clean_text(row["indication_mesh_id"])
        record = mesh.get(mesh_id)
        if record is None:
            missing_mesh.append(mesh_id)
            preferred = clean_text(row["indication_mesh_term"])
            entry_terms = [preferred] if preferred else []
            tree_numbers = []
        else:
            preferred = record["preferred_term"]
            entry_terms = record["entry_terms"]
            tree_numbers = record["tree_numbers"]

        mesh_rows.append({
            "indication_mesh_id": mesh_id,
            "indication_mesh_term_in_trails": row["indication_mesh_term"],
            "mesh2026_preferred_term": preferred,
            "mesh2026_entry_terms": " || ".join(entry_terms),
            "mesh2026_n_terms": len(entry_terms),
            "mesh2026_tree_numbers": " || ".join(tree_numbers),
            "n_pairs": int(row["n_pairs"]),
            "n_genes": int(row["n_genes"]),
            "n_A": int(row["n_A"]),
            "n_B": int(row["n_B"]),
            "mesh2026_record_found": record is not None,
        })

    mesh_df = pd.DataFrame(mesh_rows)
    mesh_df.to_csv(
        args.output_dir / "13_mesh_terms_for_indications.csv",
        index=False,
    )

    eligible = pd.read_csv(args.eligible, low_memory=False)
    if "trait_type" not in eligible.columns:
        raise SystemExit("Eligible phenotype file lacks trait_type.")

    direct = eligible[
        eligible["trait_type"].astype(str).str.casefold().isin(DIRECT_TYPES)
    ].copy()

    required_pheno = {
        "panukb_phenotype_id", "trait_type", "phenocode", "description",
        "pops_pass_qc", "num_pops_pass_qc",
    }
    missing_pheno = sorted(required_pheno - set(direct.columns))
    if missing_pheno:
        raise SystemExit(f"Eligible phenotype file missing columns: {missing_pheno}")

    candidate_rows = []

    for _, ind in mesh_df.iterrows():
        terms = [
            clean_text(x)
            for x in clean_text(ind["mesh2026_entry_terms"]).split(" || ")
            if clean_text(x)
        ]
        if not terms:
            terms = [clean_text(ind["indication_mesh_term_in_trails"])]

        scored = []
        for idx, pheno in direct.iterrows():
            texts = phenotype_search_text(pheno)
            metrics = score_candidate(terms, texts)
            scored.append((float(metrics["score"]), idx, metrics))

        scored.sort(key=lambda x: x[0], reverse=True)

        for rank, (score, idx, metrics) in enumerate(scored[:args.top_k], start=1):
            pheno = direct.loc[idx]
            out = {
                "indication_mesh_id": ind["indication_mesh_id"],
                "indication_mesh_term": ind["indication_mesh_term_in_trails"],
                "mesh2026_preferred_term": ind["mesh2026_preferred_term"],
                "n_pairs": int(ind["n_pairs"]),
                "n_genes": int(ind["n_genes"]),
                "n_A": int(ind["n_A"]),
                "n_B": int(ind["n_B"]),
                "candidate_rank": rank,
                "candidate_score": score,
                "matched_mesh_term": metrics["matched_mesh_term"],
                "matched_phenotype_text": metrics["matched_phenotype_text"],
                "exact_term_match": bool(metrics["exact"]),
                "token_containment": metrics["containment"],
                "token_jaccard": metrics["jaccard"],
                "phrase_inclusion": bool(metrics["phrase_inclusion"]),
                "sequence_similarity": metrics["sequence_similarity"],
                "panukb_phenotype_id": pheno["panukb_phenotype_id"],
                "trait_type": pheno["trait_type"],
                "phenocode": pheno["phenocode"],
                "pheno_sex": pheno.get("pheno_sex", pd.NA),
                "coding": pheno.get("coding", pd.NA),
                "modifier": pheno.get("modifier", pd.NA),
                "description": pheno.get("description", pd.NA),
                "description_more": pheno.get("description_more", pd.NA),
                "coding_description": pheno.get("coding_description", pd.NA),
                "pops_pass_qc": pheno["pops_pass_qc"],
                "num_pops_pass_qc": pheno["num_pops_pass_qc"],
                "filename": pheno.get("filename", pd.NA),
                "filename_tabix": pheno.get("filename_tabix", pd.NA),
                "aws_path": pheno.get("aws_path", pd.NA),
                "aws_path_tabix": pheno.get("aws_path_tabix", pd.NA),
            }
            for pop in POPS:
                for prefix in ["n_cases", "n_controls", "phenotype_qc"]:
                    col = f"{prefix}_{pop}"
                    if col in pheno.index:
                        out[col] = pheno[col]
            candidate_rows.append(out)

    candidates = pd.DataFrame(candidate_rows)
    candidates.to_csv(
        args.output_dir / "13_panukb_direct_mapping_candidates.csv",
        index=False,
    )

    # Condensed manual review: five best candidates in separate columns.
    review_rows = []
    group_cols = [
        "indication_mesh_id", "indication_mesh_term", "mesh2026_preferred_term",
        "n_pairs", "n_genes", "n_A", "n_B",
    ]

    for keys, group in candidates.groupby(group_cols, sort=False, dropna=False):
        row = dict(zip(group_cols, keys))
        group = group.sort_values("candidate_rank")
        for i, (_, cand) in enumerate(group.head(5).iterrows(), start=1):
            row[f"candidate_{i}_id"] = cand["panukb_phenotype_id"]
            row[f"candidate_{i}_description"] = cand["description"]
            row[f"candidate_{i}_pops_pass_qc"] = cand["pops_pass_qc"]
            row[f"candidate_{i}_score"] = cand["candidate_score"]
            row[f"candidate_{i}_matched_mesh_term"] = cand["matched_mesh_term"]

        row["selected_panukb_phenotype_id"] = ""
        row["mapping_decision"] = ""  # Direct or Exclude at this stage
        row["mapping_rationale"] = ""
        row["reviewer_notes"] = ""
        review_rows.append(row)

    review = pd.DataFrame(review_rows).sort_values(
        ["n_pairs", "indication_mesh_term"],
        ascending=[False, True],
    )
    review.to_csv(
        args.output_dir / "13_panukb_direct_mapping_review.csv",
        index=False,
    )

    exact_indications = int(
        candidates.loc[candidates["exact_term_match"], "indication_mesh_id"].nunique()
    )
    high_containment = int(
        candidates.loc[
            candidates["token_containment"] >= 1.0,
            "indication_mesh_id",
        ].nunique()
    )

    summary = {
        "mesh_source_file": str(args.mesh),
        "mesh_production_year": 2026,
        "n_mesh_descriptors_parsed": len(mesh),
        "n_study_indications": int(len(indications)),
        "n_indications_missing_mesh2026_record": len(set(missing_mesh)),
        "n_qc_eligible_icd10_phecode_phenotypes": int(len(direct)),
        "top_k_candidates_per_indication": args.top_k,
        "n_indications_with_exact_mesh_term_candidate": exact_indications,
        "n_indications_with_full_token_containment_candidate": high_containment,
        "automatic_mapping_decisions": False,
        "allowed_decisions_at_this_stage": ["Direct", "Exclude"],
        "excluded_from_direct_candidate_generation": [
            "categorical",
            "continuous",
            "biomarkers",
            "prescriptions",
            "other non-ICD10/non-PheCode phenotype types",
        ],
    }
    (
        args.output_dir / "13_panukb_direct_mapping_summary.json"
    ).write_text(json.dumps(summary, indent=2) + "\n")

    print("=" * 76)
    print("STEP 13C MESH-ENRICHED DIRECT-DISEASE CANDIDATES")
    print("=" * 76)
    print(f"Study indications:                         {len(indications):,}")
    print(f"MeSH 2026 records found:                   {len(indications)-len(set(missing_mesh)):,}")
    print(f"QC-eligible ICD10/PheCode phenotypes:      {len(direct):,}")
    print(f"Indications with exact MeSH-term candidate:{exact_indications:>8,}")
    print(f"Indications with full token containment:   {high_containment:>8,}")
    print("No mapping was accepted automatically.")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
