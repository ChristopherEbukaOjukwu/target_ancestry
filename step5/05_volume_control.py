"""
Step 5: Volume-control check.

The question: does Pool A's higher mean ancestry count survive when
study count is held fixed? Or is the diversity gap fully explained
by Pool A being studied more on average?

Logic:
  - Bin all pairs by n_studies_total: 1-2, 3-5, 6-10, 11+
  - Within each bin, compare A vs. B on:
      - mean n_ancestries_all
      - per-ancestry presence rate
  - The headline test: in how many bins does A's mean exceed B's?
"""

from pathlib import Path
import pandas as pd
import numpy as np

# --- paths -----------------------------------------------------------------
TRAILS_IN = Path("../step4/output/04_trails.parquet")
OUTPUT    = Path("output/05_volume_control.parquet")

BIN_EDGES  = [0, 2, 5, 10, 10_000]
BIN_LABELS = ["1-2", "3-5", "6-10", "11+"]

ANCESTRIES_TO_REPORT = [
    "European", "East Asian", "African", "South Asian",
    "Hispanic/Latin American", "Other Asian", "Middle Eastern",
    "Native American", "Oceanian", "Other",
]  # NR intentionally excluded

# --- load ------------------------------------------------------------------
print(f"Loading: {TRAILS_IN}")
df = pd.read_parquet(TRAILS_IN)
print(f"  {len(df):,} pairs")

# --- bin -------------------------------------------------------------------
df["study_bin"] = pd.cut(
    df["n_studies_total"],
    bins=BIN_EDGES,
    labels=BIN_LABELS,
    include_lowest=True,
)

# --- summary 1: bin sizes -------------------------------------------------
print()
print("=" * 60)
print("BIN SIZES")
print("=" * 60)
print(pd.crosstab(df["study_bin"], df["pool"], margins=True).to_string())

# --- summary 2: mean n_ancestries within each bin ------------------------
print()
print("=" * 60)
print("MEAN n_ancestries BY BIN AND POOL")
print("=" * 60)
summary = (
    df.groupby(["study_bin", "pool"], observed=True)["n_ancestries_all"]
      .agg(["mean", "median", "std", "count"])
      .round(2)
)
print(summary.to_string())

# A vs. B difference per bin, plain
print()
print("A - B difference in mean n_ancestries, by bin:")
pivot = (
    df.groupby(["study_bin", "pool"], observed=True)["n_ancestries_all"]
      .mean()
      .unstack("pool")
)
pivot["A - B"] = (pivot["A"] - pivot["B"]).round(2)
print(pivot.round(2).to_string())

# --- summary 3: per-ancestry presence rate per bin ------------------------
print()
print("=" * 60)
print("PER-ANCESTRY PRESENCE RATES, BY BIN AND POOL")
print("=" * 60)

def presence_table(group_df):
    """Return a Series of presence rates per ancestry."""
    n = len(group_df)
    if n == 0:
        return pd.Series({a: np.nan for a in ANCESTRIES_TO_REPORT})
    rates = {}
    for anc in ANCESTRIES_TO_REPORT:
        present = group_df["all_set"].apply(lambda s: anc in s).sum()
        rates[anc] = round(100 * present / n, 1)
    return pd.Series(rates)

for bin_label in BIN_LABELS:
    bin_df = df[df["study_bin"] == bin_label]
    if len(bin_df) == 0:
        continue
    a_rates = presence_table(bin_df[bin_df["pool"] == "A"])
    b_rates = presence_table(bin_df[bin_df["pool"] == "B"])
    n_a = (bin_df["pool"] == "A").sum()
    n_b = (bin_df["pool"] == "B").sum()
    table = pd.DataFrame({
        f"A (n={n_a})": a_rates,
        f"B (n={n_b})": b_rates,
        "A - B": (a_rates - b_rates).round(1),
    })
    print(f"\n  Study bin: {bin_label}")
    print(table.to_string())

# --- save -----------------------------------------------------------------
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df_out = df[["ti_uid", "gene", "indication_mesh_term", "pool",
             "n_studies_total", "study_bin", "n_ancestries_all", "all_set"]]
df_out.to_parquet(OUTPUT, index=False)
print()
print("=" * 60)
print(f"Output: {OUTPUT}")
print("=" * 60)
