from pathlib import Path
import numpy as np
import pandas as pd

PORT = Path("step15/output/15g_portability_analysis_table.parquet")
SAMPLE = Path("step15/output/15g2_sample_size_audit.parquet")
OUT = Path("step16/output")

SEED = 42
N_BOOT = 5000
ESTIMATORS = ["fiqt", "deming", "naive"]

rng = np.random.default_rng(SEED)

# ------------------------------------------------------------
# Locked primary portability estimates
# ------------------------------------------------------------
p = pd.read_parquet(PORT)

p = p[
    p["unit_analysis_role"].eq("PRIMARY")
    & p["is_primary_threshold"].eq(True)
].copy()

# ------------------------------------------------------------
# Corrected sample-size metadata from Step 15G2
# ------------------------------------------------------------
s = pd.read_parquet(SAMPLE)

print("=" * 78)
print("STEP 16G — PORTABILITY UNDER PRECISION RESTRICTIONS")
print("=" * 78)

print("\nSample-audit columns:")
print(s.columns.tolist())

# Keep only columns we may need.
sample_cols = [
    c for c in [
        "comparison_uid",
        "effective_n_ratio",
        "total_n_ratio",
        "eur_effective_n",
        "comparison_effective_n",
        "eur_total_n",
        "comparison_total_n",
    ]
    if c in s.columns
]

s = s[sample_cols].drop_duplicates(subset=["comparison_uid"])

s = s.rename(columns={
    "effective_n_ratio": "sample_size_ratio"
})

# Remove stale version from portability table before merge
if "sample_size_ratio" in p.columns:
    p = p.drop(columns=["sample_size_ratio"])

p = p.merge(
    s,
    on="comparison_uid",
    how="left"
)

# numeric cleanup
for c in [
    "slope",
    "n_variants",
    "median_variant_se_ratio",
    "sample_size_ratio",
]:
    if c in p.columns:
        p[c] = pd.to_numeric(p[c], errors="coerce")

# ------------------------------------------------------------
# One comparison-level metadata row
# ------------------------------------------------------------
meta = (
    p[
        [
            "comparison_uid",
            "gene_trait_uid",
            "comparison_population",
            "n_variants",
            "median_variant_se_ratio",
            "sample_size_ratio",
        ]
    ]
    .drop_duplicates(subset=["comparison_uid"])
    .copy()
)

print("\nPrimary comparisons:", meta["comparison_uid"].nunique())

print("\nPrecision metadata coverage:")
print("n_variants:",
      meta["n_variants"].notna().sum())
print("SE ratio:",
      meta["median_variant_se_ratio"].notna().sum())
print("sample-size ratio:",
      meta["sample_size_ratio"].notna().sum())

print("\nSample-size-ratio summary:")
print(
    meta["sample_size_ratio"]
    .describe()
    .to_string()
)

print("\nMedian SE-ratio summary:")
print(
    meta["median_variant_se_ratio"]
    .describe()
    .to_string()
)

# ------------------------------------------------------------
# Restrictions
#
# These are sensitivity restrictions, not a new primary
# analysis universe.
# ------------------------------------------------------------
restrictions = {
    "all_primary":
        pd.Series(True, index=meta.index),

    "variants_ge_5":
        meta["n_variants"] >= 5,

    "variants_ge_10":
        meta["n_variants"] >= 10,

    # comparison ancestry median SE no more than twice EUR
    "se_ratio_le_2":
        meta["median_variant_se_ratio"].notna()
        & (meta["median_variant_se_ratio"] <= 2),

    # non-EUR effective N at least 25% of EUR effective N
    "sample_ratio_ge_0p25":
        meta["sample_size_ratio"].notna()
        & (meta["sample_size_ratio"] >= 0.25),

    # non-EUR effective N at least 50% of EUR effective N
    "sample_ratio_ge_0p50":
        meta["sample_size_ratio"].notna()
        & (meta["sample_size_ratio"] >= 0.50),

    "variants_ge_5_and_sample_ratio_ge_0p25":
        (meta["n_variants"] >= 5)
        & meta["sample_size_ratio"].notna()
        & (meta["sample_size_ratio"] >= 0.25),
}

# data-adaptive precision thresholds
sample_q75 = meta["sample_size_ratio"].quantile(0.75)
se_q25 = meta["median_variant_se_ratio"].quantile(0.25)

restrictions.update({
    "best_sample_balance_quartile":
        meta["sample_size_ratio"].notna()
        & (meta["sample_size_ratio"] >= sample_q75),

    "best_se_balance_quartile":
        meta["median_variant_se_ratio"].notna()
        & (meta["median_variant_se_ratio"] <= se_q25),

    "variants_ge_5_and_best_sample_balance":
        (meta["n_variants"] >= 5)
        & meta["sample_size_ratio"].notna()
        & (meta["sample_size_ratio"] >= sample_q75),

    "variants_ge_5_and_best_se_balance":
        (meta["n_variants"] >= 5)
        & meta["median_variant_se_ratio"].notna()
        & (meta["median_variant_se_ratio"] <= se_q25),
})

def bootstrap_median(values, n_boot=N_BOOT):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return np.nan, np.nan

    medians = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True
        )
        medians[i] = np.median(sample)

    return tuple(
        np.quantile(medians, [0.025, 0.975])
    )


rows = []

for restriction_name, mask in restrictions.items():

    eligible_ids = set(
        meta.loc[mask.fillna(False), "comparison_uid"]
    )

    print("\n" + "=" * 78)
    print(restriction_name)
    print("=" * 78)
    print("Comparisons:", len(eligible_ids))

    for estimator in ESTIMATORS:

        g = p[
            p["comparison_uid"].isin(eligible_ids)
            & p["estimator"].eq(estimator)
        ].copy()

        slopes = (
            pd.to_numeric(g["slope"], errors="coerce")
            .dropna()
            .to_numpy()
        )

        if len(slopes) == 0:
            continue

        med = float(np.median(slopes))
        mean = float(np.mean(slopes))
        lo, hi = bootstrap_median(slopes)

        rows.append({
            "restriction": restriction_name,
            "estimator": estimator,
            "n_comparisons": len(slopes),
            "mean_slope": mean,
            "median_slope": med,
            "bootstrap_ci_low": lo,
            "bootstrap_ci_high": hi,
        })

        print(
            f"{estimator:7s} "
            f"n={len(slopes):3d} "
            f"median={med:.3f} "
            f"95% bootstrap CI {lo:.3f}-{hi:.3f} "
            f"mean={mean:.3f}"
        )

results = pd.DataFrame(rows)

# ------------------------------------------------------------
# Population composition under restrictions
# ------------------------------------------------------------
population_rows = []

for restriction_name, mask in restrictions.items():

    q = meta.loc[mask.fillna(False)].copy()

    counts = (
        q["comparison_population"]
        .value_counts()
    )

    for population, n in counts.items():
        population_rows.append({
            "restriction": restriction_name,
            "comparison_population": population,
            "n_comparisons": int(n),
        })

population = pd.DataFrame(population_rows)

print("\n" + "=" * 78)
print("FIQT SUMMARY")
print("=" * 78)

print(
    results[
        results["estimator"].eq("fiqt")
    ][
        [
            "restriction",
            "n_comparisons",
            "median_slope",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
        ]
    ].to_string(index=False)
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
results.to_csv(
    OUT / "16g_precision_restriction_results.csv",
    index=False
)

population.to_csv(
    OUT / "16g_precision_restriction_population_counts.csv",
    index=False
)

meta.to_csv(
    OUT / "16g_precision_metadata.csv",
    index=False
)

print("\nSaved:")
print(" ", OUT / "16g_precision_restriction_results.csv")
print(" ", OUT / "16g_precision_restriction_population_counts.csv")
print(" ", OUT / "16g_precision_metadata.csv")
