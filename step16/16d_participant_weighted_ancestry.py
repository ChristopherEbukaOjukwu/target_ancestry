from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

PAIR_STUDIES = Path("step2/output/02_pair_studies.parquet")
TRAILS = Path("step4/output/04_trails.parquet")
RAW = Path("genetic_support/data/gwas_catalog-ancestry_r2022-02-02.tsv")
OUT = Path("step16/output")

ANCESTRY_MAP = {
    "European": "European",
    "African American or Afro-Caribbean": "African",
    "African unspecified": "African",
    "Sub-Saharan African": "African",
    "East Asian": "East Asian",
    "South East Asian": "East Asian",
    "South Asian": "South Asian",
    "Asian unspecified": "Other Asian",
    "Hispanic or Latin American": "Hispanic/Latin American",
    "Greater Middle Eastern": "Middle Eastern",
    "Native American": "Native American",
    "Oceanian": "Oceanian",
    "Other": "Other",
    "Other admixed ancestry": "Other",
    "NR": "NR",
}

def resolve(raw):
    if not isinstance(raw, str):
        return set()

    raw = raw.replace(
        "Greater Middle Eastern (Middle Eastern, North African or Persian)",
        "Greater Middle Eastern"
    )

    out = set()
    for p in [x.strip() for x in raw.split(",")]:
        out.add(ANCESTRY_MAP.get(p, f"UNMAPPED:{p}"))

    return out


pairs = pd.read_parquet(PAIR_STUDIES)
trails = pd.read_parquet(TRAILS)

raw = pd.read_csv(RAW, sep="\t", low_memory=False).rename(columns={
    "STUDY ACCESSION": "study_id",
    "BROAD ANCESTRAL CATEGORY": "raw_ancestry",
    "NUMBER OF INDIVDUALS": "n_individuals",
})

raw["n_individuals"] = pd.to_numeric(
    raw["n_individuals"],
    errors="coerce"
)

# ------------------------------------------------------------
# Final validated cohort
# ------------------------------------------------------------

used = (
    pairs[pairs["ti_uid"].isin(trails["ti_uid"])]
    [["ti_uid", "study_id"]]
    .drop_duplicates()
)

x = used.merge(
    raw[["study_id", "raw_ancestry", "n_individuals"]],
    on="study_id",
    how="left"
)

x = x[
    x["n_individuals"].notna()
    & (x["n_individuals"] > 0)
].copy()

x["resolved"] = x["raw_ancestry"].apply(resolve)

x["has_nr"] = x["resolved"].apply(
    lambda s: "NR" in s
)

x["has_unmapped"] = x["resolved"].apply(
    lambda s: any(v.startswith("UNMAPPED:") for v in s)
)

x["informative"] = x["resolved"].apply(
    lambda s: sorted(
        v for v in s
        if v != "NR" and not v.startswith("UNMAPPED:")
    )
)

x["n_informative"] = x["informative"].str.len()

# ------------------------------------------------------------
# Cumulative participant volume
#
# Each original ancestry row contributes N once.
# This is independent of ancestry allocation.
# ------------------------------------------------------------

cum_n = (
    x.groupby("ti_uid", as_index=False)["n_individuals"]
    .sum()
    .rename(columns={"n_individuals": "cumulative_n"})
)

# ============================================================
# PRIMARY ANALYSIS:
# fully assignable ancestry records only
# ============================================================

# A pair is clean only when EVERY usable-N raw ancestry record:
#   - contains no NR
#   - contains no unmapped label
#   - resolves to exactly one informative ancestry
x["primary_valid_row"] = (
    ~x["has_nr"]
    & ~x["has_unmapped"]
    & (x["n_informative"] == 1)
)

pair_validity = (
    x.groupby("ti_uid")["primary_valid_row"]
    .all()
)

primary_pairs = set(
    pair_validity[pair_validity].index
)

primary = x[x["ti_uid"].isin(primary_pairs)].copy()
primary["ancestry"] = primary["informative"].str[0]

primary_counts = (
    primary.groupby(
        ["ti_uid", "ancestry"],
        as_index=False
    )["n_individuals"]
    .sum()
)

primary_counts["total_n"] = (
    primary_counts.groupby("ti_uid")["n_individuals"]
    .transform("sum")
)

primary_counts["p"] = (
    primary_counts["n_individuals"]
    / primary_counts["total_n"]
)

def summarize_weights(g):
    p = g["p"].to_numpy()

    effective = np.exp(
        -(p * np.log(p)).sum()
    )

    non_eur = (
        g.loc[
            g["ancestry"] != "European",
            "n_individuals"
        ].sum()
        / g["n_individuals"].sum()
    )

    return pd.Series({
        "effective_ancestries": effective,
        "non_european_share": non_eur,
        "weighted_n": g["n_individuals"].sum(),
        "n_observed_ancestries": g["ancestry"].nunique(),
    })

weighted_primary = (
    primary_counts.groupby("ti_uid")
    .apply(summarize_weights)
    .reset_index()
)

# ============================================================
# SENSITIVITY:
# equal split of genuinely multi-ancestry records
#
# NR/unmapped records are still not used because their
# ancestry allocation is fundamentally unknown.
# ============================================================

sens_source = x[
    ~x["has_nr"]
    & ~x["has_unmapped"]
    & (x["n_informative"] >= 1)
].copy()

sens_rows = []

for _, row in sens_source.iterrows():
    ancestries = row["informative"]
    allocated_n = row["n_individuals"] / len(ancestries)

    for ancestry in ancestries:
        sens_rows.append({
            "ti_uid": row["ti_uid"],
            "ancestry": ancestry,
            "allocated_n": allocated_n,
        })

sens = pd.DataFrame(sens_rows)

sens_counts = (
    sens.groupby(
        ["ti_uid", "ancestry"],
        as_index=False
    )["allocated_n"]
    .sum()
)

sens_counts["total_n"] = (
    sens_counts.groupby("ti_uid")["allocated_n"]
    .transform("sum")
)

sens_counts["p"] = (
    sens_counts["allocated_n"]
    / sens_counts["total_n"]
)

def summarize_sensitivity(g):
    p = g["p"].to_numpy()

    effective = np.exp(
        -(p * np.log(p)).sum()
    )

    non_eur = (
        g.loc[
            g["ancestry"] != "European",
            "allocated_n"
        ].sum()
        / g["allocated_n"].sum()
    )

    return pd.Series({
        "effective_ancestries_equal_split": effective,
        "non_european_share_equal_split": non_eur,
    })

weighted_sens = (
    sens_counts.groupby("ti_uid")
    .apply(summarize_sensitivity)
    .reset_index()
)

# ============================================================
# Regression helper
# ============================================================

def fit_logit(data, exposure, adjust_sample=False):

    vars_ = [exposure]

    if adjust_sample:
        vars_.append("log_cumulative_n")

    X = sm.add_constant(data[vars_])
    m = sm.Logit(data["launched"], X).fit(disp=False)

    b = m.params[exposure]
    se = m.bse[exposure]

    return {
        "exposure": exposure,
        "adjust_sample_size": adjust_sample,
        "n": len(data),
        "n_launched": int(data["launched"].sum()),
        "or": np.exp(b),
        "ci_low": np.exp(b - 1.96 * se),
        "ci_high": np.exp(b + 1.96 * se),
        "p": m.pvalues[exposure],
    }


# ============================================================
# PRIMARY DATASET
# ============================================================

d = (
    trails.merge(
        weighted_primary,
        on="ti_uid",
        how="inner"
    )
    .merge(
        cum_n,
        on="ti_uid",
        how="left"
    )
)

d["launched"] = (d["pool"] == "A").astype(int)
d["log_cumulative_n"] = np.log1p(d["cumulative_n"])

print("=" * 75)
print("STEP 16D — PARTICIPANT-WEIGHTED ANCESTRY")
print("=" * 75)

print("\nPRIMARY: fully assignable ancestry records")
print("Pairs:", len(d))
print("Launched:", d["launched"].sum())
print("Phase I-III:", (1 - d["launched"]).sum())

print("\nEffective ancestry number:")
print(
    d.groupby("pool")["effective_ancestries"]
    .agg(["count", "mean", "median"])
    .to_string()
)

print("\nNon-European participant share:")
print(
    d.groupby("pool")["non_european_share"]
    .agg(["count", "mean", "median"])
    .to_string()
)

results = []

for exposure in [
    "effective_ancestries",
    "non_european_share",
]:
    r1 = fit_logit(d, exposure, False)
    r2 = fit_logit(d, exposure, True)

    results.extend([r1, r2])

    print(f"\n{exposure}")

    print(
        "Unadjusted: "
        f"OR={r1['or']:.3f} "
        f"95% CI {r1['ci_low']:.3f}-{r1['ci_high']:.3f} "
        f"p={r1['p']:.4g}"
    )

    print(
        "Adjusted for cumulative N: "
        f"OR={r2['or']:.3f} "
        f"95% CI {r2['ci_low']:.3f}-{r2['ci_high']:.3f} "
        f"p={r2['p']:.4g}"
    )


# ============================================================
# EQUAL-SPLIT SENSITIVITY
# ============================================================

s = (
    trails.merge(
        weighted_sens,
        on="ti_uid",
        how="inner"
    )
    .merge(
        cum_n,
        on="ti_uid",
        how="left"
    )
)

s["launched"] = (s["pool"] == "A").astype(int)
s["log_cumulative_n"] = np.log1p(s["cumulative_n"])

print("\n" + "=" * 75)
print("SENSITIVITY: equal split across known multi-ancestry labels")
print("=" * 75)

print("Pairs:", len(s))
print("Launched:", s["launched"].sum())
print("Phase I-III:", (1 - s["launched"]).sum())

print("\nEffective ancestry number:")
print(
    s.groupby("pool")["effective_ancestries_equal_split"]
    .agg(["count", "mean", "median"])
    .to_string()
)

print("\nNon-European participant share:")
print(
    s.groupby("pool")["non_european_share_equal_split"]
    .agg(["count", "mean", "median"])
    .to_string()
)

for exposure in [
    "effective_ancestries_equal_split",
    "non_european_share_equal_split",
]:
    r1 = fit_logit(s, exposure, False)
    r2 = fit_logit(s, exposure, True)

    results.extend([r1, r2])

    print(f"\n{exposure}")

    print(
        "Unadjusted: "
        f"OR={r1['or']:.3f} "
        f"95% CI {r1['ci_low']:.3f}-{r1['ci_high']:.3f} "
        f"p={r1['p']:.4g}"
    )

    print(
        "Adjusted for cumulative N: "
        f"OR={r2['or']:.3f} "
        f"95% CI {r2['ci_low']:.3f}-{r2['ci_high']:.3f} "
        f"p={r2['p']:.4g}"
    )

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

pd.DataFrame(results).to_csv(
    OUT / "16d_weighted_ancestry_models.csv",
    index=False
)

d.to_parquet(
    OUT / "16d_weighted_ancestry_primary.parquet",
    index=False
)

s.to_parquet(
    OUT / "16d_weighted_ancestry_equal_split.parquet",
    index=False
)

primary_counts.to_csv(
    OUT / "16d_primary_ancestry_counts.csv",
    index=False
)

sens_counts.to_csv(
    OUT / "16d_equal_split_ancestry_counts.csv",
    index=False
)
