#!/usr/bin/env python3
"""
Step 13D: Conservative lexical screen for direct Pan-UKB disease mappings.

This script removes the forced-ranking problem from Step 13C. An indication
receives NO candidate unless an ICD-10/PheCode phenotype has strong lexical
support after generic disease words are removed.

It does not approve mappings automatically. Every retained row still requires
manual scientific review.

Input
-----
output/13_panukb_direct_mapping_candidates.csv

Outputs
-------
output/13_panukb_direct_candidates_conservative.csv
output/13_panukb_direct_review_conservative.csv
output/13_panukb_direct_candidates_conservative_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


GENERIC = {
    "a", "an", "and", "or", "of", "the", "with", "without", "in", "on",
    "other", "specified", "unspecified", "nos", "nec",
    "disease", "diseases", "disorder", "disorders", "syndrome", "syndromes",
    "condition", "conditions", "neoplasm", "neoplasms", "cancer", "cancers",
    "tumor", "tumors", "tumour", "tumours", "malignancy", "malignancies",
}

POPS = ["AFR", "AMR", "CSA", "EAS", "EUR", "MID"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--candidates",
        type=Path,
        default=Path("output/13_panukb_direct_mapping_candidates.csv"),
    )
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    return p.parse_args()


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize(value: Any) -> str:
    text = clean(value).casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"^[a-z]\d+(?:\.\d+)?\s+", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def singularize(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ses") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def informative_tokens(value: Any) -> set[str]:
    return {
        singularize(t)
        for t in normalize(value).split()
        if len(t) > 1 and t not in GENERIC
    }


def extract_type_number(value: Any) -> str:
    text = normalize(value)
    match = re.search(r"\btype\s*([12])\b", text)
    if match:
        return match.group(1)
    if "non insulin dependent" in text:
        return "2"
    if (
        "insulin dependent" in text
        and "non insulin dependent" not in text
    ):
        return "1"
    return ""


def compare_terms(mesh_term: str, phenotype_text: str) -> dict[str, Any]:
    left = informative_tokens(mesh_term)
    right = informative_tokens(phenotype_text)
    shared = left & right

    if left and right:
        containment = len(shared) / min(len(left), len(right))
        jaccard = len(shared) / len(left | right)
    else:
        containment = 0.0
        jaccard = 0.0

    norm_left = normalize(mesh_term)
    norm_right = normalize(phenotype_text)
    exact = bool(norm_left and norm_left == norm_right)
    phrase = bool(
        norm_left and norm_right
        and (norm_left in norm_right or norm_right in norm_left)
    )

    left_type = extract_type_number(mesh_term)
    right_type = extract_type_number(phenotype_text)
    subtype_conflict = bool(
        left_type and right_type and left_type != right_type
    )

    strong = (
        not subtype_conflict
        and len(shared) >= 1
        and (
            exact
            or phrase
            or (containment >= 0.80 and jaccard >= 0.50)
        )
    )

    if left == right and left:
        scope_relation = "same_informative_tokens"
    elif left and left < right:
        scope_relation = "phenotype_more_specific"
    elif right and right < left:
        scope_relation = "phenotype_broader"
    elif shared:
        scope_relation = "partial_overlap"
    else:
        scope_relation = "no_informative_overlap"

    return {
        "strong_lexical_candidate": strong,
        "informative_mesh_tokens": " | ".join(sorted(left)),
        "informative_phenotype_tokens": " | ".join(sorted(right)),
        "shared_informative_tokens": " | ".join(sorted(shared)),
        "n_shared_informative_tokens": len(shared),
        "conservative_containment": containment,
        "conservative_jaccard": jaccard,
        "exact_normalized_term": exact,
        "normalized_phrase_inclusion": phrase,
        "subtype_conflict": subtype_conflict,
        "scope_relation": scope_relation,
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.candidates.exists():
        raise SystemExit(f"Missing input: {args.candidates}")

    c = pd.read_csv(args.candidates, low_memory=False)

    required = {
        "indication_mesh_id", "indication_mesh_term", "mesh2026_preferred_term",
        "n_pairs", "n_genes", "n_A", "n_B",
        "panukb_phenotype_id", "trait_type", "phenocode",
        "description", "pops_pass_qc",
    }
    missing = sorted(required - set(c.columns))
    if missing:
        raise SystemExit(f"Input missing columns: {missing}")

    enriched = []
    for _, row in c.iterrows():
        mesh_terms = [
            clean(row.get("matched_mesh_term")),
            clean(row.get("mesh2026_preferred_term")),
            clean(row.get("indication_mesh_term")),
        ]
        mesh_terms = [x for x in dict.fromkeys(mesh_terms) if x]

        phenotype_terms = [
            clean(row.get("description")),
            clean(row.get("description_more")),
            clean(row.get("coding_description")),
        ]
        phenotype_terms = [x for x in dict.fromkeys(phenotype_terms) if x]

        best = None
        for mesh_term in mesh_terms:
            for phenotype_term in phenotype_terms:
                metrics = compare_terms(mesh_term, phenotype_term)
                rank_tuple = (
                    int(metrics["strong_lexical_candidate"]),
                    int(metrics["exact_normalized_term"]),
                    int(metrics["normalized_phrase_inclusion"]),
                    metrics["n_shared_informative_tokens"],
                    metrics["conservative_jaccard"],
                )
                candidate = {
                    **metrics,
                    "conservative_matched_mesh_term": mesh_term,
                    "conservative_matched_phenotype_text": phenotype_term,
                    "_rank_tuple": rank_tuple,
                }
                if best is None or rank_tuple > best["_rank_tuple"]:
                    best = candidate

        record = row.to_dict()
        if best is None:
            best = compare_terms("", "")
            best["conservative_matched_mesh_term"] = ""
            best["conservative_matched_phenotype_text"] = ""
            best["_rank_tuple"] = (0, 0, 0, 0, 0.0)

        record.update({k: v for k, v in best.items() if k != "_rank_tuple"})
        enriched.append(record)

    e = pd.DataFrame(enriched)

    kept = e[e["strong_lexical_candidate"]].copy()
    kept = kept.sort_values(
        [
            "indication_mesh_term",
            "exact_normalized_term",
            "normalized_phrase_inclusion",
            "n_shared_informative_tokens",
            "conservative_jaccard",
        ],
        ascending=[True, False, False, False, False],
    )

    kept_path = args.output_dir / "13_panukb_direct_candidates_conservative.csv"
    kept.to_csv(kept_path, index=False)

    # One compact row per indication with up to three retained candidates.
    review_rows = []
    indication_base = (
        c[
            [
                "indication_mesh_id", "indication_mesh_term",
                "mesh2026_preferred_term", "n_pairs", "n_genes", "n_A", "n_B"
            ]
        ]
        .drop_duplicates()
        .sort_values(["n_pairs", "indication_mesh_term"], ascending=[False, True])
    )

    for _, ind in indication_base.iterrows():
        group = kept[
            kept["indication_mesh_id"].eq(ind["indication_mesh_id"])
        ].copy()

        group = group.sort_values(
            [
                "exact_normalized_term",
                "normalized_phrase_inclusion",
                "n_shared_informative_tokens",
                "conservative_jaccard",
            ],
            ascending=[False, False, False, False],
        )

        out = ind.to_dict()
        out["n_conservative_candidates"] = int(len(group))

        for i, (_, cand) in enumerate(group.head(3).iterrows(), start=1):
            out[f"candidate_{i}_id"] = cand["panukb_phenotype_id"]
            out[f"candidate_{i}_description"] = cand["description"]
            out[f"candidate_{i}_pops_pass_qc"] = cand["pops_pass_qc"]
            out[f"candidate_{i}_scope_relation"] = cand["scope_relation"]
            out[f"candidate_{i}_shared_tokens"] = cand["shared_informative_tokens"]
            out[f"candidate_{i}_subtype_conflict"] = cand["subtype_conflict"]

        out["selected_panukb_phenotype_id"] = ""
        out["mapping_decision"] = ""  # Direct or Exclude
        out["mapping_rationale"] = ""
        out["reviewer_notes"] = ""
        review_rows.append(out)

    review = pd.DataFrame(review_rows)
    review_path = args.output_dir / "13_panukb_direct_review_conservative.csv"
    review.to_csv(review_path, index=False)

    with_candidate = review["n_conservative_candidates"] > 0
    summary = {
        "n_indications_total": int(len(review)),
        "n_indications_with_conservative_candidate": int(with_candidate.sum()),
        "n_indications_without_candidate": int((~with_candidate).sum()),
        "pairs_in_indications_with_candidate": int(
            review.loc[with_candidate, "n_pairs"].sum()
        ),
        "pool_A_pairs_in_indications_with_candidate": int(
            review.loc[with_candidate, "n_A"].sum()
        ),
        "pool_B_pairs_in_indications_with_candidate": int(
            review.loc[with_candidate, "n_B"].sum()
        ),
        "n_retained_candidate_rows": int(len(kept)),
        "automatic_mapping_decisions": False,
        "manual_decisions_required": ["Direct", "Exclude"],
        "generic_terms_removed": sorted(GENERIC),
        "rule": (
            "At least one shared informative token and either exact normalized "
            "term, phrase inclusion, or containment>=0.80 with Jaccard>=0.50; "
            "known type-1/type-2 conflicts are rejected."
        ),
    }

    summary_path = (
        args.output_dir
        / "13_panukb_direct_candidates_conservative_summary.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("=" * 76)
    print("STEP 13D CONSERVATIVE DIRECT-DISEASE SCREEN")
    print("=" * 76)
    print(f"Indications total:                       {len(review):,}")
    print(f"Indications with a conservative match:  {with_candidate.sum():,}")
    print(f"Indications with no retained candidate: {(~with_candidate).sum():,}")
    print(f"Retained candidate rows:                {len(kept):,}")
    print(
        "Pairs represented before manual review:   "
        f"{review.loc[with_candidate, 'n_pairs'].sum():,}"
    )
    print(
        "Pool A / Pool B before manual review:      "
        f"{review.loc[with_candidate, 'n_A'].sum():,} / "
        f"{review.loc[with_candidate, 'n_B'].sum():,}"
    )
    print("No mapping was accepted automatically.")
    print("=" * 76)


if __name__ == "__main__":
    main()
