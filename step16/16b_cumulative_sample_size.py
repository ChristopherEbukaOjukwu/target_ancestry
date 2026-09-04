from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

PAIR_STUDIES = Path("step2/output/02_pair_studies.parquet")
TRAILS = Path("step4/output/04_trails.parquet")
RAW = Path("genetic_support/data/gwas_catalog-ancestry_r2022-02-02.tsv")
OUT = Path("step16/output")

pairs = pd.read_parquet(PAIR_STUDIES)
trails = pd.read_parquet(TRAILS)

raw = pd.read_csv(RAW, sep="\t", low_memory=False).rename(columns={
    "STUDY ACCESSION": "study_id",
    "NUMBER OF INDIVDUALS": "n_individuals",
})

raw["n_individuals"] = pd.to_numeric(raw["n_individuals"], errors="coerce")

# Final 790-pair cohort only
used = (
    pairs[pairs["ti_uid"].isin(trails["ti_uid"])]
    [["ti_uid", "study_id"]]
    .drop_duplicates()
)

# Important:
# each raw ancestry row contributes its reported N once.
# We do NOT use exploded Step 3 rows here.
x = used.merge(
    raw[["study_id", "n_individuals"]],
    on="study_id",
    how="left"
)

x = x[x["n_individuals"].notna() & (x["n_individuals"] > 0)].copy()

cum_n = (
    x.groupby("ti_uid", as_index=False)["n_individuals"]
    .sum()
    .rename(columns={"n_individuals": "cumulative_n"})
)

d = trails.merge(cum_n, on="ti_uid", how="left")
d["launched"] = (d["pool"] == "A").astype(int)

# Complete-case sample-size analysis
a = d[d["cumulative_n"].notna()].copy()
a["log_cumulative_n"] = np.log1p(a["cumulative_n"])

print("=" * 70)
print("STEP 16B — CUMULATIVE PARTICIPANT SAMPLE SIZE")
print("=" * 70)
print("Final cohort:", len(d))
print("Pairs with usable cumulative N:", len(a))
print("Launched:", a["launched"].sum())
print("Phase I-III:", (1-a["launched"]).sum())

print("\nCumulative N by pool:")
print(
    a.groupby("pool")["cumulative_n"]
    .agg(["count", "median", "mean"])
    .to_string()
)

# M1: ancestry breadth only, on same 735-pair complete-case cohort
X1 = sm.add_constant(a[["n_ancestries_all"]])
m1 = sm.Logit(a["launched"], X1).fit(disp=False)

# M2: cumulative sample size only
X2 = sm.add_constant(a[["log_cumulative_n"]])
m2 = sm.Logit(a["launched"], X2).fit(disp=False)

# M3: ancestry breadth + cumulative N
X3 = sm.add_constant(a[["n_ancestries_all", "log_cumulative_n"]])
m3 = sm.Logit(a["launched"], X3).fit(disp=False)

def report(model, vars):
    for v in vars:
        b = model.params[v]
        se = model.bse[v]
        print(
            f"{v}: "
            f"OR={np.exp(b):.3f} "
            f"95% CI {np.exp(b-1.96*se):.3f}-{np.exp(b+1.96*se):.3f} "
            f"p={model.pvalues[v]:.4g}"
        )

print("\nM1: launch ~ ancestry breadth")
report(m1, ["n_ancestries_all"])

print("\nM2: launch ~ cumulative participant N")
report(m2, ["log_cumulative_n"])

print("\nM3: launch ~ ancestry breadth + cumulative participant N")
report(m3, ["n_ancestries_all", "log_cumulative_n"])

# Compare with original study-count model on same 735 pairs
a["log_studies"] = np.log1p(a["n_studies_total"])

X4 = sm.add_constant(a[["n_ancestries_all", "log_studies"]])
m4 = sm.Logit(a["launched"], X4).fit(disp=False)

print("\nM4: original study-count adjustment on same 735 pairs")
report(m4, ["n_ancestries_all", "log_studies"])

# Association between ancestry breadth and evidence quantity
print("\nCorrelations:")
print(
    "n_ancestries_all vs log cumulative N:",
    a["n_ancestries_all"].corr(a["log_cumulative_n"], method="spearman")
)
print(
    "n_ancestries_all vs log study count:",
    a["n_ancestries_all"].corr(a["log_studies"], method="spearman")
)

rows = []
for name, model, variables in [
    ("M1_ancestry_only", m1, ["n_ancestries_all"]),
    ("M2_sample_size_only", m2, ["log_cumulative_n"]),
    ("M3_ancestry_plus_sample_size", m3,
     ["n_ancestries_all", "log_cumulative_n"]),
    ("M4_ancestry_plus_study_count", m4,
     ["n_ancestries_all", "log_studies"]),
]:
    for v in variables:
        b = model.params[v]
        se = model.bse[v]
        rows.append({
            "model": name,
            "variable": v,
            "n": len(a),
            "or": np.exp(b),
            "ci_low": np.exp(b - 1.96*se),
            "ci_high": np.exp(b + 1.96*se),
            "p": model.pvalues[v],
        })

pd.DataFrame(rows).to_csv(
    OUT / "16b_cumulative_sample_size_models.csv",
    index=False
)

a.to_parquet(
    OUT / "16b_cumulative_sample_size_pair_table.parquet",
    index=False
)
