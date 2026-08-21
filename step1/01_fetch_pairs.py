"""
Step 1: Build the supported target-indication pair universe.

Input:  merge2.tsv.gz from Minikel et al. 2024
Output: data/processed/01_pairs.parquet

Logic:
  - Supported pair: at least one association row with comb_norm >= 0.8
  - Pool A (approved):   ccat == "Launched" AND supported
  - Pool B (tried):      ccat in {Phase I, Phase II, Phase III} AND supported
  - Excluded:            Preclinical, and any unsupported pair
"""

from pathlib import Path
import pandas as pd

# --- paths -----------------------------------------------------------------
INPUT  = Path("../genetic_support/data/merge2.tsv.gz")
OUTDIR = Path("output")
OUTPUT = OUTDIR / "01_pairs.parquet"

SIMILARITY_THRESHOLD = 0.8  # Minikel's rule; logged in decisions.md

# --- load ------------------------------------------------------------------
print(f"Loading {INPUT} ...")
merge = pd.read_csv(INPUT, sep="\t", low_memory=False)
print(f"  {len(merge):,} association-level rows")
print(f"  {merge['ti_uid'].nunique():,} unique target-indication pairs")

# --- support flag (pair level) --------------------------------------------
supported_uids = (
    merge.loc[merge["comb_norm"] >= SIMILARITY_THRESHOLD, "ti_uid"]
    .unique()
)
print(f"  {len(supported_uids):,} pairs are supported (comb_norm >= {SIMILARITY_THRESHOLD})")

# --- collapse to one row per pair -----------------------------------------
pairs = (
    merge[["ti_uid", "gene", "indication_mesh_id",
           "indication_mesh_term", "ccat", "year_launch"]]
    .drop_duplicates("ti_uid")
    .reset_index(drop=True)
)
pairs["supported"] = pairs["ti_uid"].isin(supported_uids)

# --- pool assignment ------------------------------------------------------
def assign_pool(ccat: str, supported: bool):
    if not supported:
        return None
    if ccat == "Launched":
        return "A"
    if ccat in {"Phase I", "Phase II", "Phase III"}:
        return "B"
    return None  # Preclinical and any other value

pairs["pool"] = pairs.apply(
    lambda r: assign_pool(r["ccat"], r["supported"]), axis=1
)

# --- finalize -------------------------------------------------------------
result = pairs[pairs["pool"].notna()].copy()

OUTDIR.mkdir(parents=True, exist_ok=True)
result.to_parquet(OUTPUT, index=False)

# --- report ---------------------------------------------------------------
print()
print("=" * 50)
print(f"Output: {OUTPUT}")
print(f"Total supported pairs in trials: {len(result):,}")
print()
print("Pool sizes:")
print(result["pool"].value_counts().to_string())
print()
print("By phase within each pool:")
print(pd.crosstab(result["ccat"], result["pool"]).to_string())
print("=" * 50)
