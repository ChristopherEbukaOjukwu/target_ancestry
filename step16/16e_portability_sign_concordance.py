from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest

IN = Path("step15/output/15g_portability_analysis_table.parquet")
OUT = Path("step16/output")

x = pd.read_parquet(IN)


# One row per actual EUR-vs-other-ancestry comparison.
#
# sign_concordance, n_variants, lead_sign_concordant etc. are
# comparison-level quantities repeated across the estimators.

keep = [
    "comparison_uid",
    "gene_trait_uid",
    "gene",
    "candidate_trait_name",
    "candidate_source",
    "comparison_population",
    "comparison_reference_population",
    "n_variants",
    "sign_concordance",
    "lead_sign_concordant",
    "pearson_r",
    "median_abs_eur_z",
    "median_variant_se_ratio",
    "median_eur_variant_se",
    "median_comparison_variant_se",
    "n_variant_effect_rows",
    "unit_analysis_role",
    "is_primary_threshold",
    "primary_approval_eligible",
]

cols = [c for c in keep if c in x.columns]

d = (
    x[cols]
    .drop_duplicates(subset=["comparison_uid"])
    .copy()
)

# Primary locked portability comparisons only
if "unit_analysis_role" in d.columns:
    d = d[d["unit_analysis_role"].eq("PRIMARY")].copy()

if "is_primary_threshold" in d.columns:
    d = d[d["is_primary_threshold"].eq(True)].copy()

d["n_variants"] = pd.to_numeric(d["n_variants"], errors="coerce")
d["sign_concordance"] = pd.to_numeric(
    d["sign_concordance"], errors="coerce"
)

d = d[
    d["n_variants"].notna()
    & (d["n_variants"] > 0)
    & d["sign_concordance"].notna()
].copy()


# Recover approximate concordant/discordant variant counts.
#
# sign_concordance was calculated as concordant / n_variants,
# so multiplication should be essentially integral.

d["n_concordant"] = np.rint(
    d["sign_concordance"] * d["n_variants"]
).astype(int)

d["n_discordant"] = (
    d["n_variants"].astype(int) - d["n_concordant"]
)

total_variants = int(d["n_variants"].sum())
total_concordant = int(d["n_concordant"].sum())

pooled_concordance = total_concordant / total_variants

# Exact binomial test against random direction (50%)
pooled_test = binomtest(
    total_concordant,
    total_variants,
    p=0.5,
    alternative="greater"
)

# Comparison-level summary
mean_comparison = d["sign_concordance"].mean()
median_comparison = d["sign_concordance"].median()

# Lead variants
lead = d["lead_sign_concordant"].dropna()

if len(lead):
    lead = lead.astype(bool)
    n_lead = len(lead)
    n_lead_concordant = int(lead.sum())
    lead_rate = n_lead_concordant / n_lead

    lead_test = binomtest(
        n_lead_concordant,
        n_lead,
        p=0.5,
        alternative="greater"
    )
else:
    n_lead = 0
    n_lead_concordant = 0
    lead_rate = np.nan
    lead_test = None

print("=" * 75)
print("STEP 16E — CROSS-ANCESTRY SIGN CONCORDANCE")
print("=" * 75)

print("Independent ancestry comparisons:", len(d))
print("Total LD-pruned variant comparisons:", total_variants)

print("\nPooled variant-level sign concordance:")
print(
    f"{total_concordant}/{total_variants} = "
    f"{pooled_concordance:.3f}"
)
print(
    f"Exact binomial p vs 0.50: "
    f"{pooled_test.pvalue:.3g}"
)

print("\nComparison-level sign concordance:")
print(f"Mean:   {mean_comparison:.3f}")
print(f"Median: {median_comparison:.3f}")

print("\nLead-variant direction:")
if n_lead:
    print(
        f"{n_lead_concordant}/{n_lead} = "
        f"{lead_rate:.3f}"
    )
    print(
        f"Exact binomial p vs 0.50: "
        f"{lead_test.pvalue:.3g}"
    )
else:
    print("No usable lead-sign records.")

print("\nBy comparison ancestry:")
by_pop = (
    d.groupby("comparison_population")
    .agg(
        comparisons=("comparison_uid", "size"),
        variants=("n_variants", "sum"),
        concordant=("n_concordant", "sum"),
        mean_comparison_concordance=("sign_concordance", "mean"),
        median_comparison_concordance=("sign_concordance", "median"),
    )
    .reset_index()
)

by_pop["pooled_sign_concordance"] = (
    by_pop["concordant"] / by_pop["variants"]
)

print(by_pop.to_string(index=False))

print("\nBy variant-count precision:")
# Descriptive strata; not used to alter the primary universe.
d["variant_stratum"] = pd.cut(
    d["n_variants"],
    bins=[0, 4, 9, 19, np.inf],
    labels=["1-4", "5-9", "10-19", "20+"]
)

by_n = (
    d.groupby("variant_stratum", observed=True)
    .agg(
        comparisons=("comparison_uid", "size"),
        variants=("n_variants", "sum"),
        concordant=("n_concordant", "sum"),
        mean_comparison_concordance=("sign_concordance", "mean"),
    )
    .reset_index()
)

by_n["pooled_sign_concordance"] = (
    by_n["concordant"] / by_n["variants"]
)

print(by_n.to_string(index=False))


# Save

d.to_csv(
    OUT / "16e_sign_concordance_comparisons.csv",
    index=False
)

by_pop.to_csv(
    OUT / "16e_sign_concordance_by_population.csv",
    index=False
)

by_n.to_csv(
    OUT / "16e_sign_concordance_by_variant_count.csv",
    index=False
)

summary = pd.DataFrame([{
    "n_comparisons": len(d),
    "n_variants": total_variants,
    "n_concordant": total_concordant,
    "pooled_sign_concordance": pooled_concordance,
    "pooled_binomial_p": pooled_test.pvalue,
    "mean_comparison_sign_concordance": mean_comparison,
    "median_comparison_sign_concordance": median_comparison,
    "n_lead_comparisons": n_lead,
    "n_lead_concordant": n_lead_concordant,
    "lead_sign_concordance": lead_rate,
    "lead_binomial_p":
        lead_test.pvalue if lead_test is not None else np.nan,
}])

summary.to_csv(
    OUT / "16e_sign_concordance_summary.csv",
    index=False
)
