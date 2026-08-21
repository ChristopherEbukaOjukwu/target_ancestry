"""
Step 6: Robustness checks.

Check 1: Does the diversity gap in 6-10 and 11+ bins survive when we
         remove famously diverse approved targets (PCSK9, HLA-B, G6PD)?

Check 2: Stage decomposition — gap by initial vs. replication evidence.

Check 3: Collapse to one row per gene — does the gap survive
         deduplication of multi-indication targets?
"""

from pathlib import Path
import pandas as pd
import numpy as np

# paths
TRAILS_IN = Path("../step4/output/04_trails.parquet")

# Famously diverse approved targets (known from literature/clinical PGx)
KNOWN_DIVERSE_TARGETS = {
    "PCSK9",      # African ancestry discovery (Cohen 2005)
    "HLA-B",      # HLA-B*57:01 (abacavir), HLA-B*15:02 (carbamazepine)
    "G6PD",       # G6PD deficiency — population-stratified
    "CYP2C9",     # warfarin dosing
    "VKORC1",     # warfarin dosing
    "CYP2C19",    # clopidogrel
    "TPMT",       # thiopurines
    "DPYD",       # fluoropyrimidines
    "UGT1A1",     # irinotecan
    "SLCO1B1",    # statins
}

BIN_EDGES  = [0, 2, 5, 10, 10_000]
BIN_LABELS = ["1-2", "3-5", "6-10", "11+"]

# load--
print(f"Loading: {TRAILS_IN}")
df = pd.read_parquet(TRAILS_IN)
df["study_bin"] = pd.cut(
    df["n_studies_total"], bins=BIN_EDGES, labels=BIN_LABELS, include_lowest=True
)
print(f"  {len(df):,} pairs")

# CHECK 1: drop famously diverse targets
print()
print("=" * 60)
print("CHECK 1: Drop known-diverse pharmacogenomic targets")
print("=" * 60)

in_data = df[df["gene"].isin(KNOWN_DIVERSE_TARGETS)]
print(f"\nKnown-diverse targets present in data: "
      f"{sorted(in_data['gene'].unique())}")
print(f"Pairs they account for, by pool:")
print(in_data["pool"].value_counts().to_string())

print("\nMean n_ancestries by bin, BEFORE dropping:")
before = (
    df.groupby(["study_bin", "pool"], observed=True)["n_ancestries_all"]
      .mean().unstack("pool")
)
before["A - B"] = (before["A"] - before["B"]).round(2)
print(before.round(2).to_string())

print("\nMean n_ancestries by bin, AFTER dropping known-diverse targets:")
df_filtered = df[~df["gene"].isin(KNOWN_DIVERSE_TARGETS)]
after = (
    df_filtered.groupby(["study_bin", "pool"], observed=True)["n_ancestries_all"]
      .mean().unstack("pool")
)
after["A - B"] = (after["A"] - after["B"]).round(2)
print(after.round(2).to_string())

print(f"\nPool A pairs removed: {len(df[(df['pool']=='A') & (df['gene'].isin(KNOWN_DIVERSE_TARGETS))])}")
print(f"Pool B pairs removed: {len(df[(df['pool']=='B') & (df['gene'].isin(KNOWN_DIVERSE_TARGETS))])}")

# CHECK 2: stage decomposition
print()
print("=" * 60)
print("CHECK 2: Stage decomposition (initial vs. replication)")
print("=" * 60)

print("\nMean n_ancestries_initial by bin and pool:")
init_summary = (
    df.groupby(["study_bin", "pool"], observed=True)["n_ancestries_initial"]
      .mean().unstack("pool")
)
init_summary["A - B"] = (init_summary["A"] - init_summary["B"]).round(2)
print(init_summary.round(2).to_string())

print("\nMean n_ancestries_replication by bin and pool:")
repl_summary = (
    df.groupby(["study_bin", "pool"], observed=True)["n_ancestries_replication"]
      .mean().unstack("pool")
)
repl_summary["A - B"] = (repl_summary["A"] - repl_summary["B"]).round(2)
print(repl_summary.round(2).to_string())

print("\nFraction of pairs with ANY replication evidence, by bin and pool:")
repl_presence = (
    df.groupby(["study_bin", "pool"], observed=True)["has_replication"]
      .mean().unstack("pool")
)
repl_presence["A - B"] = (repl_presence["A"] - repl_presence["B"]).round(2)
print(repl_presence.round(2).to_string())

# CHECK 3: one row per gene
print()
print("=" * 60)
print("CHECK 3: Collapse to one row per (gene, pool)")
print("=" * 60)

# For each (gene, pool), take the union of ancestries across its indications
# and the max n_studies. Pool conflict (gene in both A and B) — keep separate.
def union_sets(series):
    out = set()
    for s in series:
        out.update(s)
    return sorted(out)

gene_collapsed = (
    df.groupby(["gene", "pool"], observed=True)
      .agg(
          all_set=("all_set", union_sets),
          n_studies_total=("n_studies_total", "max"),
      )
      .reset_index()
)
gene_collapsed["n_ancestries_all"] = gene_collapsed["all_set"].apply(
    lambda s: len([x for x in s if x != "NR"])
)
gene_collapsed["study_bin"] = pd.cut(
    gene_collapsed["n_studies_total"], bins=BIN_EDGES, labels=BIN_LABELS,
    include_lowest=True,
)

print(f"\nUnique (gene, pool) rows: {len(gene_collapsed):,}")
print(f"  Pool A: {(gene_collapsed['pool']=='A').sum()}")
print(f"  Pool B: {(gene_collapsed['pool']=='B').sum()}")

# Note: a gene can appear in both pools (different indications)
both_pools = gene_collapsed.groupby("gene")["pool"].nunique()
print(f"  Genes appearing in BOTH pools: {(both_pools == 2).sum()}")

print("\nMean n_ancestries by bin and pool (gene-level):")
gene_summary = (
    gene_collapsed.groupby(["study_bin", "pool"], observed=True)["n_ancestries_all"]
      .agg(["mean", "count"])
)
print(gene_summary.round(2).to_string())

gene_pivot = (
    gene_collapsed.groupby(["study_bin", "pool"], observed=True)["n_ancestries_all"]
      .mean().unstack("pool")
)
gene_pivot["A - B"] = (gene_pivot["A"] - gene_pivot["B"]).round(2)
print(f"\nMean diff by bin (gene-level):")
print(gene_pivot.round(2).to_string())

print()
print("=" * 60)
print("END OF ROBUSTNESS CHECKS")
print("=" * 60)
