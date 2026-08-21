"""
Step 3: Attach ancestry to each study from Step 2.

Input:
  - data/processed/02_pair_studies.parquet
  - genetic_support/data/gwas_catalog-ancestry_r2022-02-02.tsv

Output:
  - data/processed/03_study_ancestry.parquet

Logic:
  - For GWAS_CATALOG studies: join on STUDY ACCESSION, parse
    BROAD ANCESTRAL CATEGORY, split multi-ancestry rows on comma,
    consolidate to broad categories.
  - For NEALE_UKBB, FINNGEN, SAIGE: synthesize a single
    (stage=initial, ancestry=European) row per study.
  - Output: one row per (study_id, stage, ancestry).
"""

from pathlib import Path
import pandas as pd

# --- paths -----------------------------------------------------------------
STUDIES_IN  = Path("../step2/output/02_pair_studies.parquet")
ANCESTRY_IN = Path("../genetic_support/data/gwas_catalog-ancestry_r2022-02-02.tsv")
OUTPUT      = Path("output/03_study_ancestry.parquet")

# --- ancestry consolidation map ------------------------------------------
ANCESTRY_MAP = {
    # European
    "European": "European",
    # African (all subcategories collapsed)
    "African American or Afro-Caribbean": "African",
    "African unspecified": "African",
    "Sub-Saharan African": "African",
    # East Asian (East + South East collapsed)
    "East Asian": "East Asian",
    "South East Asian": "East Asian",
    # South Asian (kept separate)
    "South Asian": "South Asian",
    # Asian unspecified — can't disambiguate
    "Asian unspecified": "Other Asian",
    # Hispanic / Latin American
    "Hispanic or Latin American": "Hispanic/Latin American",
    # Middle Eastern
    "Greater Middle Eastern": "Middle Eastern",
    # Native American
    "Native American": "Native American",
    # Oceanian
    "Oceanian": "Oceanian",
    # Other
    "Other": "Other",
    "Other admixed ancestry": "Other",
    # NR
    "NR": "NR",
}

def consolidate(raw_ancestry):
    """
    Take a raw BROAD ANCESTRAL CATEGORY value (which may contain commas
    representing mixed samples) and return a set of consolidated categories.
    """
    if not isinstance(raw_ancestry, str):
        return set()

    # Normalize the one known label with an internal comma so it
    # survives the split-on-comma step.
    raw_ancestry = raw_ancestry.replace(
        "Greater Middle Eastern (Middle Eastern, North African or Persian)",
        "Greater Middle Eastern"
    )

    parts = [p.strip() for p in raw_ancestry.split(",")]
    consolidated = set()
    for p in parts:
        if p in ANCESTRY_MAP:
            consolidated.add(ANCESTRY_MAP[p])
        else:
            consolidated.add(f"UNMAPPED: {p}")
    return consolidated

# --- load Step 2 output ---------------------------------------------------
print(f"Loading studies: {STUDIES_IN}")
studies = pd.read_parquet(STUDIES_IN)
unique_studies = studies[["study_id", "study_source"]].drop_duplicates()
print(f"  {len(unique_studies):,} unique studies")
print(f"  by source:")
print(unique_studies["study_source"].value_counts().to_string())

# --- load ancestry file ---------------------------------------------------
print(f"\nLoading ancestry: {ANCESTRY_IN}")
anc = pd.read_csv(ANCESTRY_IN, sep="\t", low_memory=False)
print(f"  {len(anc):,} ancestry rows")

# rename for sanity
anc = anc.rename(columns={
    "STUDY ACCESSION":          "study_id",
    "STAGE":                    "stage",
    "BROAD ANCESTRAL CATEGORY": "raw_ancestry",
    "NUMBER OF INDIVDUALS":     "n_individuals",
})

# --- handle GWAS Catalog studies -----------------------------------------
gc_studies = unique_studies[unique_studies["study_source"] == "GWAS_CATALOG"]
gc_anc = anc[anc["study_id"].isin(gc_studies["study_id"])].copy()
print(f"\nGWAS Catalog ancestry rows matched: {len(gc_anc):,}")

# how many GWAS Catalog studies got NO match?
gc_matched = set(gc_anc["study_id"])
gc_missing = set(gc_studies["study_id"]) - gc_matched
print(f"  GWAS Catalog studies with no ancestry row: {len(gc_missing):,}")
if gc_missing:
    print(f"  (these are likely studies added after the 2022-02-02 snapshot)")

# explode multi-ancestry rows
gc_rows = []
for _, row in gc_anc.iterrows():
    cats = consolidate(row["raw_ancestry"])
    for cat in cats:
        gc_rows.append({
            "study_id":     row["study_id"],
            "study_source": "GWAS_CATALOG",
            "stage":        row["stage"],
            "ancestry":     cat,
            "n_individuals": row["n_individuals"],
        })
gc_out = pd.DataFrame(gc_rows)

# --- handle biobank-recovered studies -------------------------------------
bb_studies = unique_studies[
    unique_studies["study_source"].isin({"NEALE_UKBB", "FINNGEN", "SAIGE"})
]
bb_out = bb_studies.assign(
    stage="initial",
    ancestry="European",
    n_individuals=None,  # not available from URL alone
)[["study_id", "study_source", "stage", "ancestry", "n_individuals"]]
print(f"\nBiobank studies assigned European: {len(bb_out):,}")

# --- combine and save -----------------------------------------------------
result = pd.concat([gc_out, bb_out], ignore_index=True)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.to_parquet(OUTPUT, index=False)

# --- report ---------------------------------------------------------------
print()
print("=" * 50)
print(f"Output: {OUTPUT}")
print(f"Total ancestry rows: {len(result):,}")
print(f"Unique studies covered: {result['study_id'].nunique():,}")
print()
print("Ancestry distribution (all rows):")
print(result["ancestry"].value_counts().to_string())
print()
print("Stage distribution:")
print(result["stage"].value_counts().to_string())
print()
print("Source distribution:")
print(result["study_source"].value_counts().to_string())

# audit for any unmapped categories
unmapped = result[result["ancestry"].str.startswith("UNMAPPED:", na=False)]
if len(unmapped):
    print()
    print(f"WARNING: {len(unmapped):,} rows have unmapped ancestry categories:")
    print(unmapped["ancestry"].value_counts().to_string())
print("=" * 50)
