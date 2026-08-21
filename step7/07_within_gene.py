"""
Step 7: Within-gene comparison across indications (V1).

For genes that appear in BOTH Pool A and Pool B (different indications),
compare the ancestry trail of the approved indication(s) against the
ancestry trail of the not-approved indication(s) WITHIN the same gene.

This controls for the gene entirely. Any difference can't be blamed on
"approved targets are different kinds of genes."

Logic:
  - Identify genes present in both pools.
  - For each such gene, build A-side and B-side ancestry sets.
  - Diagnose: how many genes actually have DIFFERENT A and B sets?
    (Degeneracy check — see notes in conversation.)
  - For informative genes, report direction of difference.
"""

from pathlib import Path
import pandas as pd

# --- paths ----------------------------------------------------------------
TRAILS_IN = Path("../step4/output/04_trails.parquet")

# --- load ------------------------------------------------------------------
print(f"Loading: {TRAILS_IN}")
df = pd.read_parquet(TRAILS_IN)
print(f"  {len(df):,} pairs")

# Sets came back from parquet as lists — convert back
df["all_set"] = df["all_set"].apply(set)
df["initial_set"] = df["initial_set"].apply(set)
df["replication_set"] = df["replication_set"].apply(set)

# --- identify dual-pool genes --------------------------------------------
pools_per_gene = df.groupby("gene")["pool"].agg(set)
dual_pool_genes = pools_per_gene[pools_per_gene == {"A", "B"}].index.tolist()
print(f"\nGenes appearing in BOTH pools: {len(dual_pool_genes)}")

# --- build per-gene A and B trails ---------------------------------------
def union_sets(series):
    out = set()
    for s in series:
        out |= s
    return out

def count_excluding_nr(s):
    return len(s - {"NR"})

rows = []
for gene in dual_pool_genes:
    g = df[df["gene"] == gene]
    a_pairs = g[g["pool"] == "A"]
    b_pairs = g[g["pool"] == "B"]

    a_set         = union_sets(a_pairs["all_set"])
    b_set         = union_sets(b_pairs["all_set"])
    a_set_initial = union_sets(a_pairs["initial_set"])
    b_set_initial = union_sets(b_pairs["initial_set"])
    a_set_repl    = union_sets(a_pairs["replication_set"])
    b_set_repl    = union_sets(b_pairs["replication_set"])

    rows.append({
        "gene": gene,
        "n_a_pairs": len(a_pairs),
        "n_b_pairs": len(b_pairs),
        "a_set": sorted(a_set),
        "b_set": sorted(b_set),
        "a_n":     count_excluding_nr(a_set),
        "b_n":     count_excluding_nr(b_set),
        "a_n_initial": count_excluding_nr(a_set_initial),
        "b_n_initial": count_excluding_nr(b_set_initial),
        "a_n_replication": count_excluding_nr(a_set_repl),
        "b_n_replication": count_excluding_nr(b_set_repl),
        "identical": (a_set - {"NR"}) == (b_set - {"NR"}),
        "a_minus_b": sorted((a_set - b_set) - {"NR"}),
        "b_minus_a": sorted((b_set - a_set) - {"NR"}),
    })

per_gene = pd.DataFrame(rows)

# --- DIAGNOSTIC: degeneracy check ----------------------------------------
print()
print("=" * 60)
print("DEGENERACY CHECK")
print("=" * 60)
n_identical = per_gene["identical"].sum()
n_informative = (~per_gene["identical"]).sum()
print(f"Genes where A and B sets are IDENTICAL: {n_identical} "
      f"({100*n_identical/len(per_gene):.1f}%)")
print(f"Genes where A and B sets DIFFER:        {n_informative} "
      f"({100*n_informative/len(per_gene):.1f}%)")
print()
print("(Identical = approved and not-approved indications share the")
print(" same ancestry trail because they draw from overlapping studies.")
print(" Only the 'differ' subset is informative for the within-gene test.)")

if n_informative == 0:
    print("\nAll dual-pool genes are degenerate. V1 is uninformative.")
    print("Fall back to V2 (between-gene comparison).")
else:
    informative = per_gene[~per_gene["identical"]].copy()

    # --- comparison among informative genes ------------------------------
    print()
    print("=" * 60)
    print(f"COMPARISON AMONG {n_informative} INFORMATIVE GENES")
    print("=" * 60)

    informative["diff_all"]         = informative["a_n"] - informative["b_n"]
    informative["diff_initial"]     = informative["a_n_initial"] - informative["b_n_initial"]
    informative["diff_replication"] = informative["a_n_replication"] - informative["b_n_replication"]

    print("\nMean set-size difference (A - B) within informative genes:")
    print(f"  All stages:  {informative['diff_all'].mean():+.2f} "
          f"(median {informative['diff_all'].median():+.1f})")
    print(f"  Initial:     {informative['diff_initial'].mean():+.2f} "
          f"(median {informative['diff_initial'].median():+.1f})")
    print(f"  Replication: {informative['diff_replication'].mean():+.2f} "
          f"(median {informative['diff_replication'].median():+.1f})")

    print("\nDirection counts (all stages):")
    n_a_more = (informative["diff_all"] > 0).sum()
    n_b_more = (informative["diff_all"] < 0).sum()
    n_tie    = (informative["diff_all"] == 0).sum()
    print(f"  A has more ancestries: {n_a_more}")
    print(f"  B has more ancestries: {n_b_more}")
    print(f"  Tied set size:         {n_tie}")

    print("\nDirection counts (initial stage only):")
    n_a_more_i = (informative["diff_initial"] > 0).sum()
    n_b_more_i = (informative["diff_initial"] < 0).sum()
    n_tie_i    = (informative["diff_initial"] == 0).sum()
    print(f"  A has more ancestries: {n_a_more_i}")
    print(f"  B has more ancestries: {n_b_more_i}")
    print(f"  Tied set size:         {n_tie_i}")

    print("\nDirection counts (replication stage only):")
    n_a_more_r = (informative["diff_replication"] > 0).sum()
    n_b_more_r = (informative["diff_replication"] < 0).sum()
    n_tie_r    = (informative["diff_replication"] == 0).sum()
    print(f"  A has more ancestries: {n_a_more_r}")
    print(f"  B has more ancestries: {n_b_more_r}")
    print(f"  Tied set size:         {n_tie_r}")

    # --- which ancestries asymmetric ------------------------------------
    print()
    print("Ancestries unique to A side (across all informative genes):")
    from collections import Counter
    a_unique = Counter()
    b_unique = Counter()
    for _, row in informative.iterrows():
        for anc in row["a_minus_b"]:
            a_unique[anc] += 1
        for anc in row["b_minus_a"]:
            b_unique[anc] += 1

    print(f"  ({n_informative} genes total)")
    print(f"  {'Ancestry':<28s} {'A-only':>8s} {'B-only':>8s}")
    all_ancs = sorted(set(a_unique) | set(b_unique))
    for anc in all_ancs:
        print(f"  {anc:<28s} {a_unique.get(anc, 0):>8d} {b_unique.get(anc, 0):>8d}")

    # --- save informative gene table -----------------------------------
    OUTPUT = Path("output/07_within_gene.parquet")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    informative.to_parquet(OUTPUT, index=False)
    print(f"\nSaved informative-gene table: {OUTPUT}")

print()
print("=" * 60)
print("END")
print("=" * 60)
