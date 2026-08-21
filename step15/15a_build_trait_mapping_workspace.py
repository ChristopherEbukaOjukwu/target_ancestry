#!/usr/bin/env python3
"""Build the Step 15A trait-mapping review workspace.

No mappings are accepted automatically. Pan-UKB traits are retained only when
EUR and at least one non-EUR population appear in pops_pass_qc. GBMI endpoints
are included as published GRCh38 candidates, but their file paths and endpoint-
specific ancestry availability remain unverified until the original release is
inspected.
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

import pandas as pd

POPS = ["AFR", "AMR", "CSA", "EAS", "EUR", "MID"]
DIRECT_TYPES = {"icd10", "phecode"}
QUANT_TYPES = {"biomarkers", "continuous"}
GENERIC = {
    "a", "an", "and", "or", "of", "the", "with", "without", "in", "on",
    "other", "specified", "unspecified", "nos", "nec", "disease", "diseases",
    "disorder", "disorders", "condition", "conditions", "syndrome", "syndromes",
}

# Published GBMI pilot endpoint set. Paths and ancestry availability are not
# filled here because those must be independently verified from original files.
GBMI_ENDPOINTS = [
    ("AAA", "Abdominal aortic aneurysm", "disease"),
    ("AcApp", "Acute appendicitis", "disease"),
    ("Asthma", "Asthma", "disease"),
    ("Appendectomy", "Appendectomy", "procedure"),
    ("COPD", "Chronic obstructive pulmonary disease", "disease"),
    ("Gout", "Gout", "disease"),
    ("HCM", "Hypertrophic cardiomyopathy", "disease"),
    ("HF", "Heart failure", "disease"),
    ("IPF", "Idiopathic pulmonary fibrosis", "disease"),
    ("POAG", "Primary open-angle glaucoma", "disease"),
    ("Stroke", "Stroke", "disease"),
    ("ThC", "Thyroid cancer", "disease"),
    ("UtC", "Uterine cancer", "disease"),
    ("VTE", "Venous thromboembolism", "disease"),
]


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trails", type=Path, default=Path("../step4/output/04_trails.parquet"))
    p.add_argument(
        "--panukb-manifest",
        type=Path,
        default=Path("../step13/input/manifests/phenotype_manifest.tsv.bgz"),
    )
    p.add_argument("--mesh", type=Path, default=Path("../step13/input/manifests/desc2026.gz"))
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--top-k-direct", type=int, default=8)
    return p.parse_args()


def clean(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def norm(x: Any) -> str:
    s = clean(x).casefold().replace("&", " and ")
    s = re.sub(r"^[a-z]\d+(?:\.\d+)?\s+", "", s)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s)).strip()


def singular(t: str) -> str:
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("ses") and len(t) > 4:
        return t[:-2]
    if t.endswith("s") and not t.endswith("ss") and len(t) > 3:
        return t[:-1]
    return t


def toks(x: Any) -> set[str]:
    return {singular(t) for t in norm(x).split() if len(t) > 1 and t not in GENERIC}


def split_pops(x: Any) -> list[str]:
    return [p.strip() for p in clean(x).split(",") if p.strip()]


def read_mesh(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with gzip.open(path, "rb") as fh:
        for _, elem in ET.iterparse(fh, events=("end",)):
            if elem.tag != "DescriptorRecord":
                continue
            ui = clean(elem.findtext("DescriptorUI"))
            preferred = clean(elem.findtext("DescriptorName/String"))
            terms = {preferred} if preferred else set()
            terms.update(
                clean(n.text) for n in elem.findall(".//TermList/Term/String") if clean(n.text)
            )
            trees = sorted(
                {clean(n.text) for n in elem.findall(".//TreeNumberList/TreeNumber") if clean(n.text)}
            )
            out[ui] = {"preferred": preferred, "terms": sorted(terms), "trees": trees}
            elem.clear()
    return out


def score(left: str, right: str) -> dict[str, Any]:
    nl, nr = norm(left), norm(right)
    tl, tr = toks(left), toks(right)
    shared = tl & tr
    exact = bool(nl and nr and nl == nr)
    phrase = bool(nl and nr and (nl in nr or nr in nl))
    containment = len(shared) / min(len(tl), len(tr)) if tl and tr else 0.0
    jaccard = len(shared) / len(tl | tr) if tl and tr else 0.0
    sequence = SequenceMatcher(None, nl, nr).ratio() if nl and nr else 0.0
    total = 120 * exact + 45 * phrase + 60 * containment + 35 * jaccard + 15 * sequence
    return {
        "candidate_score": total,
        "exact_normalized_match": exact,
        "phrase_inclusion": phrase,
        "token_containment": containment,
        "token_jaccard": jaccard,
        "sequence_similarity": sequence,
        "shared_informative_tokens": " | ".join(sorted(shared)),
    }


def best_score(ind_terms: list[str], trait_terms: list[str]) -> dict[str, Any]:
    best = None
    for a in ind_terms:
        for b in trait_terms:
            d = score(a, b)
            d["matched_indication_term"] = a
            d["matched_trait_term"] = b
            if best is None or d["candidate_score"] > best["candidate_score"]:
                best = d
    return best or {
        "candidate_score": 0.0,
        "exact_normalized_match": False,
        "phrase_inclusion": False,
        "token_containment": 0.0,
        "token_jaccard": 0.0,
        "sequence_similarity": 0.0,
        "shared_informative_tokens": "",
        "matched_indication_term": "",
        "matched_trait_term": "",
    }


def indication_table(trails: pd.DataFrame, mesh: dict[str, dict[str, Any]]) -> pd.DataFrame:
    g = (
        trails.groupby(["indication_mesh_id", "indication_mesh_term"], dropna=False)
        .agg(
            n_pairs=("ti_uid", "nunique"),
            n_genes=("gene", "nunique"),
            n_A=("pool", lambda s: int((s == "A").sum())),
            n_B=("pool", lambda s: int((s == "B").sum())),
        )
        .reset_index()
    )
    rows = []
    for _, r in g.iterrows():
        mid = clean(r["indication_mesh_id"])
        m = mesh.get(mid, {})
        preferred = clean(m.get("preferred")) or clean(r["indication_mesh_term"])
        terms = m.get("terms") or [preferred]
        rows.append({
            **r.to_dict(),
            "mesh2026_preferred_term": preferred,
            "mesh2026_entry_terms": " || ".join(terms),
            "mesh2026_tree_numbers": " || ".join(m.get("trees", [])),
            "mesh2026_record_found": mid in mesh,
        })
    return pd.DataFrame(rows)


def panukb_catalog(m: pd.DataFrame) -> pd.DataFrame:
    required = {
        "trait_type", "phenocode", "pheno_sex", "coding", "modifier", "description",
        "description_more", "coding_description", "category", "pops_pass_qc",
        "num_pops_pass_qc", "filename", "filename_tabix", "aws_path", "aws_path_tabix",
    }
    missing = sorted(required - set(m.columns))
    if missing:
        raise SystemExit(f"Pan-UKB manifest missing columns: {missing}")
    rows = []
    for _, r in m.iterrows():
        pops = split_pops(r["pops_pass_qc"])
        non_eur = [p for p in pops if p != "EUR"]
        if "EUR" not in pops or not non_eur:
            continue
        ttype = clean(r["trait_type"]).casefold()
        if ttype in DIRECT_TYPES:
            aclass = "direct_disease_candidate"
        elif ttype in QUANT_TYPES:
            aclass = "quantitative_or_biomarker_candidate"
        else:
            aclass = "other_candidate"
        pid = "|".join(clean(r[c]) for c in ["trait_type", "phenocode", "pheno_sex", "coding", "modifier"])
        row = {
            "trait_key": f"Pan-UKB::{pid}",
            "source": "Pan-UKB",
            "genome_build": "GRCh37",
            "phenotype_id": pid,
            "published_endpoint_code": "",
            "trait_type": clean(r["trait_type"]),
            "phenocode": clean(r["phenocode"]),
            "trait_name": clean(r["description"]),
            "description_more": clean(r["description_more"]),
            "coding_description": clean(r["coding_description"]),
            "category": clean(r["category"]),
            "analysis_class": aclass,
            "pops_pass_qc": ",".join(pops),
            "non_eur_pops_pass_qc": ",".join(non_eur),
            "num_non_eur_pops_pass_qc": len(non_eur),
            "ancestry_status": "verified_from_official_manifest",
            "filename": clean(r["filename"]),
            "filename_tabix": clean(r["filename_tabix"]),
            "remote_data_path": clean(r["aws_path"]),
            "remote_index_path": clean(r["aws_path_tabix"]),
            "source_release": "Pan-UKB official phenotype manifest",
            "mapping_status": "UNREVIEWED",
        }
        for pop in POPS:
            for prefix in ["n_cases", "n_controls", "phenotype_qc"]:
                row[f"{prefix}_{pop}"] = r.get(f"{prefix}_{pop}", pd.NA)
        rows.append(row)
    return pd.DataFrame(rows)


def gbmi_catalog() -> pd.DataFrame:
    rows = []
    for code, name, endpoint_type in GBMI_ENDPOINTS:
        rows.append({
            "trait_key": f"GBMI::{code}",
            "source": "GBMI",
            "genome_build": "GRCh38",
            "phenotype_id": code,
            "published_endpoint_code": code,
            "trait_type": endpoint_type,
            "phenocode": "",
            "trait_name": name,
            "description_more": "",
            "coding_description": "",
            "category": "GBMI published pilot endpoint",
            "analysis_class": "direct_disease_candidate" if endpoint_type == "disease" else "procedure_candidate",
            "pops_pass_qc": "",
            "non_eur_pops_pass_qc": "",
            "num_non_eur_pops_pass_qc": pd.NA,
            "ancestry_status": "TO_VERIFY_FROM_ORIGINAL_ENDPOINT_FILES",
            "filename": "",
            "filename_tabix": "",
            "remote_data_path": "",
            "remote_index_path": "",
            "source_release": "GBMI pilot release 052021",
            "mapping_status": "UNREVIEWED",
        })
    return pd.DataFrame(rows)


def direct_candidates(indications: pd.DataFrame, catalog: pd.DataFrame, top_k: int) -> pd.DataFrame:
    direct = catalog[catalog["analysis_class"].isin(["direct_disease_candidate", "procedure_candidate"])]
    rows = []
    for _, ind in indications.iterrows():
        ind_terms = [clean(x) for x in clean(ind["mesh2026_entry_terms"]).split(" || ") if clean(x)]
        scored = []
        for idx, trait in direct.iterrows():
            trait_terms = [clean(trait[c]) for c in ["trait_name", "description_more", "coding_description"] if clean(trait[c])]
            s = best_score(ind_terms, trait_terms)
            scored.append((s["candidate_score"], idx, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        for rank, (_, idx, s) in enumerate(scored[:top_k], 1):
            trait = direct.loc[idx]
            rows.append({
                "indication_mesh_id": ind["indication_mesh_id"],
                "indication_mesh_term": ind["indication_mesh_term"],
                "mesh2026_preferred_term": ind["mesh2026_preferred_term"],
                "n_pairs": int(ind["n_pairs"]),
                "n_genes": int(ind["n_genes"]),
                "n_A": int(ind["n_A"]),
                "n_B": int(ind["n_B"]),
                "candidate_rank": rank,
                **s,
                **trait.to_dict(),
                "manual_mapping_tier": "",
                "manual_mapping_decision": "",
                "manual_mapping_rationale": "",
                "reviewer_notes": "",
            })
    return pd.DataFrame(rows)


def review_template(indications: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ordered = indications.sort_values(["n_pairs", "indication_mesh_term"], ascending=[False, True])
    for _, ind in ordered.iterrows():
        c = candidates[candidates["indication_mesh_id"].eq(ind["indication_mesh_id"])].sort_values("candidate_rank")
        row = {
            "indication_mesh_id": ind["indication_mesh_id"],
            "indication_mesh_term": ind["indication_mesh_term"],
            "mesh2026_preferred_term": ind["mesh2026_preferred_term"],
            "n_pairs": int(ind["n_pairs"]),
            "n_genes": int(ind["n_genes"]),
            "n_A": int(ind["n_A"]),
            "n_B": int(ind["n_B"]),
        }
        for i, (_, cand) in enumerate(c.head(5).iterrows(), 1):
            row[f"direct_candidate_{i}_trait_key"] = cand["trait_key"]
            row[f"direct_candidate_{i}_name"] = cand["trait_name"]
            row[f"direct_candidate_{i}_source"] = cand["source"]
            row[f"direct_candidate_{i}_build"] = cand["genome_build"]
            row[f"direct_candidate_{i}_pops"] = cand["pops_pass_qc"]
            row[f"direct_candidate_{i}_score"] = cand["candidate_score"]
        for i in range(1, 6):
            row[f"selected_trait_key_{i}"] = ""
            row[f"mapping_tier_{i}"] = ""
            row[f"mapping_rationale_{i}"] = ""
            row[f"reviewer_notes_{i}"] = ""
        row["indication_review_status"] = "UNREVIEWED"
        row["exclude_reason"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    a = args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for path in [a.trails, a.panukb_manifest, a.mesh]:
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")

    trails = pd.read_parquet(a.trails)
    required = {"ti_uid", "gene", "pool", "indication_mesh_id", "indication_mesh_term"}
    missing = sorted(required - set(trails.columns))
    if missing:
        raise SystemExit(f"Step 4 trails missing columns: {missing}")

    mesh = read_mesh(a.mesh)
    manifest = pd.read_csv(a.panukb_manifest, sep="\t", compression="gzip", low_memory=False)
    indications = indication_table(trails, mesh)
    panukb = panukb_catalog(manifest)
    gbmi = gbmi_catalog()
    catalog = pd.concat([panukb, gbmi], ignore_index=True, sort=False)
    if catalog["trait_key"].duplicated().any():
        raise SystemExit("Duplicate trait_key values in combined catalog.")

    quantitative = panukb[panukb["analysis_class"].eq("quantitative_or_biomarker_candidate")].copy()
    candidates = direct_candidates(indications, catalog, a.top_k_direct)
    review = review_template(indications, candidates)

    paths = {
        "trait_catalog": a.output_dir / "15a_trait_catalog.csv",
        "panukb_quantitative_catalog": a.output_dir / "15a_panukb_quantitative_catalog.csv",
        "direct_candidates": a.output_dir / "15a_direct_mapping_candidates.csv",
        "review_template": a.output_dir / "15a_indication_mapping_review.csv",
    }
    catalog.to_csv(paths["trait_catalog"], index=False)
    quantitative.to_csv(paths["panukb_quantitative_catalog"], index=False)
    candidates.to_csv(paths["direct_candidates"], index=False)
    review.to_csv(paths["review_template"], index=False)

    summary = {
        "n_target_indication_pairs": int(trails["ti_uid"].nunique()),
        "n_indications": int(indications["indication_mesh_id"].nunique()),
        "n_panukb_traits_with_eur_and_qc_non_eur": int(len(panukb)),
        "n_panukb_direct_disease_candidates": int(panukb["analysis_class"].eq("direct_disease_candidate").sum()),
        "n_panukb_quantitative_or_biomarker_candidates": int(len(quantitative)),
        "n_panukb_other_candidates": int(panukb["analysis_class"].eq("other_candidate").sum()),
        "n_gbmi_published_endpoints": int(len(gbmi)),
        "panukb_genome_build": "GRCh37",
        "gbmi_genome_build": "GRCh38",
        "automatic_mapping_decisions": False,
        "allowed_mapping_tiers": [
            "Direct disease",
            "Direct quantitative readout",
            "Closely related phenotype",
            "Proxy biomarker",
            "Exclude",
        ],
        "gbmi_unresolved_fields": [
            "endpoint-specific ancestry availability",
            "summary-statistic filenames",
            "remote paths",
            "per-ancestry sample sizes",
        ],
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    summary_path = a.output_dir / "15a_trait_mapping_workspace_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("=" * 76)
    print("STEP 15A TRAIT-MAPPING WORKSPACE")
    print("=" * 76)
    print(f"Target-indication pairs:              {summary['n_target_indication_pairs']:,}")
    print(f"Indications:                          {summary['n_indications']:,}")
    print(f"Pan-UKB EUR + QC non-EUR traits:      {summary['n_panukb_traits_with_eur_and_qc_non_eur']:,}")
    print(f"  Direct disease candidates:          {summary['n_panukb_direct_disease_candidates']:,}")
    print(f"  Quantitative/biomarker candidates:  {summary['n_panukb_quantitative_or_biomarker_candidates']:,}")
    print(f"GBMI published endpoints:             {summary['n_gbmi_published_endpoints']:,}")
    print("\nNo mapping was accepted automatically.")
    print("GBMI file paths and ancestry availability remain TO VERIFY.\n")
    for p in paths.values():
        print(f"  {p}")
    print(f"  {summary_path}")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
