#!/usr/bin/env python3
"""Step 13A: Pan-UKB portability feasibility gate.

Validates the official Pan-UKB phenotype manifest, restricts to phenotypes
passing QC in >=2 populations, summarizes coverage, and generates ranked
lexical candidates linking the study's indications to Pan-UKB phenotypes.
No mapping is accepted automatically.
"""
from __future__ import annotations
import argparse, hashlib, json, math, re, sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

POPS = ["AFR", "AMR", "CSA", "EAS", "EUR", "MID"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trails", type=Path, default=Path("../step4/output/04_trails.parquet"))
    p.add_argument("--manifest", type=Path, default=Path("input/manifests/phenotype_manifest.tsv.bgz"))
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--top-k", type=int, default=8)
    return p.parse_args()


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def sha256_file(path, chunk_size=1024*1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def clean_text(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return re.sub(r"\s+", " ", str(v)).strip()


def norm(v):
    t = clean_text(v).casefold().replace("&", " and ")
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def toks(v):
    stop = {"and","or","of","the","with","without","other","specified","unspecified","disease","disorder","syndrome","condition"}
    return {x for x in norm(v).split() if len(x) > 1 and x not in stop}


def jac(a,b):
    x,y=toks(a),toks(b)
    return len(x&y)/len(x|y) if x and y else 0.0


def contain(a,b):
    x,y=toks(a),toks(b)
    return len(x&y)/min(len(x),len(y)) if x and y else 0.0


def lexical(indication, description, description_more):
    i,s,l = norm(indication), norm(description), norm(description_more)
    exact = bool(i and i == s)
    seq = SequenceMatcher(None, i, s).ratio() if i and s else 0.0
    lseq = SequenceMatcher(None, i, l).ratio() if i and l else 0.0
    j = max(jac(i,s), jac(i,l))
    c = max(contain(i,s), contain(i,l))
    score = 100*exact + 45*c + 30*j + 20*seq + 5*lseq
    return dict(lexical_score=score, exact_normalized_match=exact,
                token_containment=c, token_jaccard=j, sequence_similarity=seq)


def parse_pops(v):
    t = clean_text(v)
    return [x.strip().upper() for x in re.split(r"[,;| ]+", t) if x.strip()] if t else []


def json_safe(v: Any):
    if isinstance(v, dict): return {str(k): json_safe(x) for k,x in v.items()}
    if isinstance(v, list): return [json_safe(x) for x in v]
    if isinstance(v, Path): return str(v)
    if isinstance(v, np.generic): return json_safe(v.item())
    if isinstance(v, float) and not np.isfinite(v): return None
    return v


def main():
    a = parse_args()
    if not a.trails.exists(): fail(f"Missing trails: {a.trails}")
    if not a.manifest.exists(): fail(f"Missing official manifest: {a.manifest}")
    if a.top_k < 1: fail("--top-k must be >=1")
    a.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Step 4 trails: {a.trails}")
    trails = pd.read_parquet(a.trails)
    req_t = {"ti_uid","gene","pool","indication_mesh_id","indication_mesh_term"}
    if req_t - set(trails.columns): fail(f"Trails missing: {sorted(req_t-set(trails.columns))}")
    inds = (trails.groupby(["indication_mesh_id","indication_mesh_term"], dropna=False)
            .agg(n_pairs=("ti_uid","nunique"), n_genes=("gene","nunique"),
                 n_A=("pool",lambda x:int((x=="A").sum())),
                 n_B=("pool",lambda x:int((x=="B").sum())))
            .reset_index())

    print(f"Loading official Pan-UKB manifest: {a.manifest}")
    m = pd.read_csv(a.manifest, sep="\t", compression="gzip", low_memory=False)
    req_m = {"trait_type","phenocode","pheno_sex","coding","modifier","description",
             "pops_pass_qc","num_pops_pass_qc","filename","aws_path"}
    if req_m - set(m.columns): fail(f"Manifest missing: {sorted(req_m-set(m.columns))}")
    ids=["trait_type","phenocode","pheno_sex","coding","modifier"]
    if m.duplicated(ids).any(): fail("Manifest phenotype ID fields are not unique")
    m["panukb_phenotype_id"] = m[ids].fillna("").astype(str).agg("|".join, axis=1)
    m["num_pops_pass_qc"] = pd.to_numeric(m["num_pops_pass_qc"], errors="coerce")
    m["pops_pass_qc_list"] = m["pops_pass_qc"].map(parse_pops)
    m["n_parsed_pops_pass_qc"] = m["pops_pass_qc_list"].map(len)
    m["pops_count_consistent"] = m["num_pops_pass_qc"].fillna(-1).astype(int).eq(m["n_parsed_pops_pass_qc"])

    e = m[m["num_pops_pass_qc"] >= 2].copy()
    e["is_binary"] = ~e["trait_type"].isin(["continuous","biomarkers"])
    e["is_quantitative"] = e["trait_type"].isin(["continuous","biomarkers"])
    e["is_both_sexes"] = e["pheno_sex"].eq("both_sexes")

    preferred=["panukb_phenotype_id","trait_type","phenocode","pheno_sex","coding","modifier",
               "description","description_more","coding_description","category","pops","num_pops",
               "pops_pass_qc","num_pops_pass_qc","is_binary","is_quantitative","is_both_sexes",
               "filename","filename_tabix","aws_path","aws_path_tabix"]
    n_fields=[c for c in e.columns if re.fullmatch(r"n_(cases|controls)_(AFR|AMR|CSA|EAS|EUR|MID)", c)]
    qc_fields=[c for c in e.columns if re.fullmatch(r"phenotype_qc_(AFR|AMR|CSA|EAS|EUR|MID)", c)]
    keep=[c for c in preferred+n_fields+qc_fields if c in e.columns]
    e[keep].to_csv(a.output_dir/"13_panukb_eligible_phenotypes.csv", index=False)

    pop_rows=[]
    for pop in POPS:
        mask=e["pops_pass_qc_list"].map(lambda x,p=pop:p in x)
        row={"population":pop,"n_eligible_phenotypes":int(mask.sum()),
             "n_binary":int((mask&e["is_binary"]).sum()),
             "n_quantitative":int((mask&e["is_quantitative"]).sum())}
        cc=f"n_cases_{pop}"; ctl=f"n_controls_{pop}"
        if cc in e.columns:
            bc=pd.to_numeric(e.loc[mask&e["is_binary"],cc],errors="coerce")
            qn=pd.to_numeric(e.loc[mask&e["is_quantitative"],cc],errors="coerce")
            row.update(binary_case_median=bc.median(), binary_case_q25=bc.quantile(.25),
                       binary_case_q75=bc.quantile(.75), quantitative_n_median=qn.median(),
                       quantitative_n_q25=qn.quantile(.25), quantitative_n_q75=qn.quantile(.75))
        if ctl in e.columns:
            ct=pd.to_numeric(e.loc[mask&e["is_binary"],ctl],errors="coerce")
            row["binary_control_median"]=ct.median()
        pop_rows.append(row)
    pop_summary=pd.DataFrame(pop_rows)
    pop_summary.to_csv(a.output_dir/"13_panukb_population_coverage.csv", index=False)

    trait_summary=(e.groupby(["trait_type","is_binary"],dropna=False)
                   .agg(n_phenotypes=("panukb_phenotype_id","size"),
                        n_both_sexes=("is_both_sexes","sum"),
                        median_qc_populations=("num_pops_pass_qc","median"),
                        max_qc_populations=("num_pops_pass_qc","max"))
                   .reset_index().sort_values("n_phenotypes",ascending=False))
    trait_summary.to_csv(a.output_dir/"13_panukb_trait_type_summary.csv", index=False)

    rows=[]; ep=e.reset_index(drop=True)
    for _,ind in inds.iterrows():
        scored=[]
        for idx,p in ep.iterrows():
            met=lexical(ind["indication_mesh_term"],p.get("description",""),p.get("description_more",""))
            scored.append((met["lexical_score"]+(3 if p["is_binary"] else 0),idx,met))
        scored.sort(key=lambda x:x[0],reverse=True)
        for rank,(rs,idx,met) in enumerate(scored[:a.top_k],1):
            p=ep.iloc[idx]
            row={"indication_mesh_id":ind["indication_mesh_id"],
                 "indication_mesh_term":ind["indication_mesh_term"],
                 "n_pairs":int(ind["n_pairs"]),"n_genes":int(ind["n_genes"]),
                 "n_A":int(ind["n_A"]),"n_B":int(ind["n_B"]),
                 "candidate_rank":rank,"candidate_rank_score":rs,**met,
                 "panukb_phenotype_id":p["panukb_phenotype_id"],
                 "trait_type":p["trait_type"],"phenocode":p["phenocode"],
                 "pheno_sex":p["pheno_sex"],"coding":p.get("coding",pd.NA),
                 "modifier":p.get("modifier",pd.NA),"description":p.get("description",pd.NA),
                 "description_more":p.get("description_more",pd.NA),"category":p.get("category",pd.NA),
                 "is_binary":bool(p["is_binary"]),"is_quantitative":bool(p["is_quantitative"]),
                 "pops_pass_qc":p["pops_pass_qc"],"num_pops_pass_qc":int(p["num_pops_pass_qc"]),
                 "filename":p.get("filename",pd.NA),"filename_tabix":p.get("filename_tabix",pd.NA),
                 "aws_path":p.get("aws_path",pd.NA),"aws_path_tabix":p.get("aws_path_tabix",pd.NA)}
            for c in n_fields+qc_fields: row[c]=p.get(c,pd.NA)
            rows.append(row)
    cand=pd.DataFrame(rows)
    cand.to_csv(a.output_dir/"13_panukb_candidate_mappings.csv", index=False)
    review=cand.copy()
    review.insert(review.columns.get_loc("candidate_rank")+1,"selected_for_review",False)
    review["mapping_tier"]=""; review["mapping_rationale"]=""; review["reviewer_notes"]=""; review["exclude_reason"]=""
    review.to_csv(a.output_dir/"13_panukb_mapping_review.csv", index=False)

    exact_n=cand.loc[cand["exact_normalized_match"],"indication_mesh_id"].nunique()
    summary={"step":"13A","input_provenance":{"manifest_path":str(a.manifest),
             "manifest_size_bytes":a.manifest.stat().st_size,"manifest_sha256":sha256_file(a.manifest),
             "trails_path":str(a.trails)},
             "manifest_validation":{"n_rows":len(m),"n_pop_count_inconsistencies":int((~m["pops_count_consistent"]).sum())},
             "feasibility_counts":{"study_pairs":trails["ti_uid"].nunique(),"study_indications":len(inds),
             "eligible_panukb_phenotypes_ge2_qc_pops":len(e),
             "eligible_ge3_qc_pops":int((e["num_pops_pass_qc"]>=3).sum()),
             "eligible_ge4_qc_pops":int((e["num_pops_pass_qc"]>=4).sum()),
             "eligible_binary":int(e["is_binary"].sum()),"eligible_quantitative":int(e["is_quantitative"].sum()),
             "candidate_rows":len(cand),"indications_with_exact_normalized_candidate":int(exact_n)},
             "rules":{"phenotype_eligibility":"num_pops_pass_qc >= 2",
             "ancestry_inclusion":"manifest pops_pass_qc only","automatic_mapping_decisions":False,
             "allowed_manual_mapping_tiers":["Direct","Closely-related","Proxy","Exclude"],
             "primary_candidate_tier":"Direct"}}
    (a.output_dir/"13_panukb_feasibility_summary.json").write_text(json.dumps(json_safe(summary),indent=2)+"\n")

    print("\n"+"="*76)
    print("STEP 13A PAN-UKB FEASIBILITY GATE")
    print("="*76)
    print(f"Study indications:                    {len(inds):,}")
    print(f"Manifest phenotypes:                  {len(m):,}")
    print(f"QC-passing in >=2 populations:        {len(e):,}")
    print(f"QC-passing in >=3 populations:        {(e['num_pops_pass_qc']>=3).sum():,}")
    print(f"QC-passing in >=4 populations:        {(e['num_pops_pass_qc']>=4).sum():,}")
    print(f"Eligible binary phenotypes:           {e['is_binary'].sum():,}")
    print(f"Eligible quantitative phenotypes:     {e['is_quantitative'].sum():,}")
    print(f"Indications with an exact text match: {exact_n:,}")
    print(f"Candidate rows requiring review:      {len(review):,}")
    print("\nPopulation coverage:")
    print(pop_summary.to_string(index=False))
    print("\nIMPORTANT: No indication-to-phenotype mapping was accepted automatically.")
    print("="*76)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
