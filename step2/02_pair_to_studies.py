"""
Step 2: For each supported pair, attach the GWAS studies behind it.

Input:
  - data/processed/01_pairs.parquet   (from Step 1)
  - genetic_support/data/merge2.tsv.gz  (Minikel)

Output:
  - data/processed/02_pair_studies.parquet

Logic:
  - Restrict merge2 to rows where comb_norm >= 0.8 (the support rule)
  - Restrict to assoc_source == "OTG" (GWAS only)
  - Extract study ID from original_link, recognizing four URL patterns:
      GCST*    → GWAS Catalog (ancestry looked up in Step 3)
      NEALE2_* → Neale UKBB (European, assigned in Step 3)
      FINNGEN_*→ FinnGen (European, assigned in Step 3)
      SAIGE_*  → SAIGE UKBB-derived (European, assigned in Step 3)
  - Keep only rows whose ti_uid is in Pool A or B
  - Output: one row per (pair, study), with study_source tagged
"""

from pathlib import Path
import re
import pandas as pd

# paths
PAIRS_IN  = Path("../step1/output/01_pairs.parquet")
MERGE_IN  = Path("../genetic_support/data/merge2.tsv.gz")
OUTPUT    = Path("output/02_pair_studies.parquet")

SIMILARITY_THRESHOLD = 0.8
KEEP_SOURCES = {"OTG"}

# study ID extraction
# Returns (study_id, study_source) or (None, None) if unparseable.
def extract_study_id(link):
    if not isinstance(link, str):
        return None, None

    if "GCST" in link:
        m = re.search(r"(GCST\d+)", link)
        return (m.group(1), "GWAS_CATALOG") if m else (None, None)

    upper = link.upper()
    if "NEALE" in upper:
        m = re.search(r"(NEALE2_[\w\.\-]+)", link)
        return (m.group(1), "NEALE_UKBB") if m else (None, None)
    if "FINNGEN" in upper:
        m = re.search(r"(FINNGEN_[\w]+)", link)
        return (m.group(1), "FINNGEN") if m else (None, None)
    if "SAIGE" in upper:
        m = re.search(r"(SAIGE_\d+)", link)
        return (m.group(1), "SAIGE") if m else (None, None)

    return None, None

# load-
print(f"Loading pairs: {PAIRS_IN}")
pairs = pd.read_parquet(PAIRS_IN)
pool_uids = set(pairs["ti_uid"])
print(f"  {len(pool_uids):,} pairs in Pools A/B")

print(f"Loading associations: {MERGE_IN}")
merge = pd.read_csv(MERGE_IN, sep="\t", low_memory=False)
print(f"  {len(merge):,} association-level rows")

# filter to qualifying associations
m = merge[
    (merge["comb_norm"] >= SIMILARITY_THRESHOLD)
    & (merge["assoc_source"].isin(KEEP_SOURCES))
    & (merge["ti_uid"].isin(pool_uids))
].copy()
print(f"  {len(m):,} rows after support + source + pool filters")

# extract study IDs (now multi-pattern)
extracted = m["original_link"].apply(extract_study_id)
m["study_id"]     = extracted.apply(lambda x: x[0])
m["study_source"] = extracted.apply(lambda x: x[1])

# report extraction quality
n_with_id = m["study_id"].notna().sum()
print(f"  {n_with_id:,} rows have a parseable study ID "
      f"({n_with_id / len(m):.1%})")
print()
print("Study sources:")
print(m["study_source"].value_counts(dropna=False).to_string())

# finalize
result = (
    m.loc[m["study_id"].notna(),
          ["ti_uid", "gene", "indication_mesh_id", "indication_mesh_term",
           "study_id", "study_source",
           "assoc_mesh_id", "assoc_mesh_term",
           "original_trait", "assoc_year", "arow"]]
    .drop_duplicates(["ti_uid", "study_id"])
    .reset_index(drop=True)
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.to_parquet(OUTPUT, index=False)

# report
result_with_pool = result.merge(pairs[["ti_uid", "pool"]], on="ti_uid")

print()
print("=" * 50)
print(f"Output: {OUTPUT}")
print(f"Rows (pair, study): {len(result):,}")
print(f"Unique studies:     {result['study_id'].nunique():,}")
print(f"Unique pairs:       {result['ti_uid'].nunique():,}")
print()
print("Pairs with at least one OTG study, by pool:")
pair_pool = (
    result_with_pool[["ti_uid", "pool"]].drop_duplicates()
    ["pool"].value_counts()
)
print(pair_pool.to_string())
print()
print("Studies per pool (counts):")
print(result_with_pool["pool"].value_counts().to_string())
print()
print("Pairs from Step 1 with no OTG support at all:")
remaining_uids = set(result["ti_uid"])
dropped = pairs[~pairs["ti_uid"].isin(remaining_uids)]
print(dropped["pool"].value_counts().to_string())
print("=" * 50)
