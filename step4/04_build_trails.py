"""
Step 4: Build the ancestry trail per pair.

Inputs:
  - data/processed/01_pairs.parquet         (pool labels)
  - data/processed/02_pair_studies.parquet  (pair -> studies)
  - data/processed/03_study_ancestry.parquet (study -> ancestries)

Output:
  - data/processed/04_trails.parquet

Logic:
  - For each pair, collect every (study, stage, ancestry) row that supports it.
  - Build three ancestry sets per pair:
      initial_set       — ancestries from any initial-stage study
      replication_set   — ancestries from any replication-stage study
      all_set           — union of both
  - Set sizes exclude NR (we don't count "we don't know" as an ancestry).
  - Track study volume separately:
      n_studies_total       — all studies supporting the pair
      n_studies_initial     — at least one initial-stage row
      n_studies_replication — at least one replication-stage row
      has_replication       — boolean flag
"""

from pathlib import Path
import pandas as pd

# paths
PAIRS_IN     = Path("../step1/output/01_pairs.parquet")
STUDIES_IN   = Path("../step2/output/02_pair_studies.parquet")
ANCESTRY_IN  = Path("../step3/output/03_study_ancestry.parquet")
OUTPUT       = Path("output/04_trails.parquet")

EXCLUDE_FROM_COUNTS = {"NR"}  # logged in decisions.md

# load-
print(f"Loading pairs:     {PAIRS_IN}")
pairs = pd.read_parquet(PAIRS_IN)
print(f"  {len(pairs):,} pairs in Pool A/B")

print(f"Loading studies:   {STUDIES_IN}")
pair_studies = pd.read_parquet(STUDIES_IN)
print(f"  {len(pair_studies):,} (pair, study) rows")

print(f"Loading ancestry:  {ANCESTRY_IN}")
study_anc = pd.read_parquet(ANCESTRY_IN)
print(f"  {len(study_anc):,} (study, stage, ancestry) rows")

# join to get (pair, study, stage, ancestry)
joined = pair_studies.merge(
    study_anc[["study_id", "stage", "ancestry"]],
    on="study_id",
    how="inner",  # drops pairs whose studies have no ancestry data
)
print(f"\nJoined to (pair, study, stage, ancestry): {len(joined):,} rows")

# how many pairs lost their entire ancestry footprint?
covered_pairs = set(joined["ti_uid"])
lost_pairs = set(pairs["ti_uid"]) - covered_pairs
if lost_pairs:
    lost_by_pool = pairs[pairs["ti_uid"].isin(lost_pairs)]["pool"].value_counts()
    print(f"  {len(lost_pairs):,} pairs lost (no studies with ancestry data):")
    print(lost_by_pool.to_string())

# build ancestry sets per pair
def to_set(series):
    return set(series.dropna())

agg = joined.groupby("ti_uid").apply(
    lambda g: pd.Series({
        "initial_set":     to_set(g.loc[g["stage"] == "initial",     "ancestry"]),
        "replication_set": to_set(g.loc[g["stage"] == "replication", "ancestry"]),
        "all_set":         to_set(g["ancestry"]),
        "n_studies_total":       g["study_id"].nunique(),
        "n_studies_initial":     g.loc[g["stage"] == "initial",     "study_id"].nunique(),
        "n_studies_replication": g.loc[g["stage"] == "replication", "study_id"].nunique(),
    }),
    include_groups=False,
).reset_index()

# set sizes (excluding NR)
def count_excluding_nr(s):
    return len(s - EXCLUDE_FROM_COUNTS)

agg["n_ancestries_initial"]     = agg["initial_set"].apply(count_excluding_nr)
agg["n_ancestries_replication"] = agg["replication_set"].apply(count_excluding_nr)
agg["n_ancestries_all"]         = agg["all_set"].apply(count_excluding_nr)
agg["has_replication"]          = agg["n_studies_replication"] > 0

# attach pair metadata + pool
result = pairs.merge(agg, on="ti_uid", how="inner")

# save (convert sets to sorted lists for parquet safety)
for col in ["initial_set", "replication_set", "all_set"]:
    result[col] = result[col].apply(lambda s: sorted(s))

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.to_parquet(OUTPUT, index=False)

# report
print()
print("=" * 50)
print(f"Output: {OUTPUT}")
print(f"Pairs with full trail: {len(result):,}")
print()
print("Pool sizes:")
print(result["pool"].value_counts().to_string())
print()
print("Studies per pair (mean):")
print(result.groupby("pool")["n_studies_total"].mean().round(2).to_string())
print()
print("Set size (n_ancestries_all, excluding NR) by pool:")
print(result.groupby("pool")["n_ancestries_all"]
      .agg(["mean", "median", "min", "max"]).round(2).to_string())
print()
print("Pairs with any replication evidence:")
print(result.groupby("pool")["has_replication"].sum().to_string())
print()
print("Ancestries appearing in each pool (any stage):")
for pool_name in ["A", "B"]:
    pool_pairs = result[result["pool"] == pool_name]
    counter = {}
    for s in pool_pairs["all_set"]:
        for anc in s:
            counter[anc] = counter.get(anc, 0) + 1
    print(f"\n  Pool {pool_name} (n={len(pool_pairs)} pairs):")
    for anc, count in sorted(counter.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(pool_pairs)
        print(f"    {anc:<28s} {count:>4d} pairs ({pct:5.1f}%)")
print("=" * 50)
