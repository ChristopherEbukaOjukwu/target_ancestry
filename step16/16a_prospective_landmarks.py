from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

PAIR_STUDIES = Path("step2/output/02_pair_studies.parquet")
ANCESTRY = Path("step3/output/03_study_ancestry.parquet")
TRAILS = Path("step4/output/04_trails.parquet")
OUT = Path("step16/output")

CUTOFFS = [2012, 2014, 2016, 2018]

pairs = pd.read_parquet(PAIR_STUDIES)
anc = pd.read_parquet(ANCESTRY)
trails = pd.read_parquet(TRAILS)

# Final validated 790-pair cohort only
pairs = pairs[pairs["ti_uid"].isin(trails["ti_uid"])].copy()

# Keep study-year relationship
study_year = (
    pairs[["ti_uid", "study_id", "assoc_year"]]
    .drop_duplicates()
)

# Attach ancestry.
# NR is not counted as an ancestry.
anc_clean = anc[
    anc["ancestry"].notna()
    & (anc["ancestry"] != "NR")
    & ~anc["ancestry"].astype(str).str.startswith("UNMAPPED:")
].copy()

evidence = study_year.merge(
    anc_clean[["study_id", "ancestry"]].drop_duplicates(),
    on="study_id",
    how="left"
)

base = trails[
    [
        "ti_uid",
        "gene",
        "indication_mesh_id",
        "indication_mesh_term",
        "pool",
        "year_launch",
    ]
].copy()

base["launched"] = (base["pool"] == "A").astype(int)

results = []
landmark_tables = []

for cutoff in CUTOFFS:

    # Evidence that existed by the landmark year
    e = evidence[
        evidence["assoc_year"].notna()
        & (evidence["assoc_year"] <= cutoff)
    ].copy()

    # Study count by landmark
    nstudies = (
        e[["ti_uid", "study_id"]]
        .drop_duplicates()
        .groupby("ti_uid")
        .size()
        .rename("n_studies")
    )

    # Ancestry breadth by landmark
    nanc = (
        e.dropna(subset=["ancestry"])
        [["ti_uid", "ancestry"]]
        .drop_duplicates()
        .groupby("ti_uid")
        .size()
        .rename("n_ancestries")
    )

    d = (
        base.merge(nstudies, on="ti_uid", how="left")
        .merge(nanc, on="ti_uid", how="left")
    )

    d["n_studies"] = d["n_studies"].fillna(0).astype(int)
    d["n_ancestries"] = d["n_ancestries"].fillna(0).astype(int)

    # Only pairs with some qualifying genetic evidence by cutoff
    d = d[d["n_studies"] > 0].copy()

    d["log_studies"] = np.log1p(d["n_studies"])
    d["cutoff"] = cutoff

    landmark_tables.append(d)

    # -------------------------------
    # Model 1: ancestry breadth only
    # -------------------------------
    X1 = sm.add_constant(d[["n_ancestries"]])
    m1 = sm.Logit(d["launched"], X1).fit(disp=False)

    b = m1.params["n_ancestries"]
    se = m1.bse["n_ancestries"]

    results.append({
        "cutoff": cutoff,
        "model": "M1_ancestry_only",
        "n_pairs": len(d),
        "n_launched": int(d["launched"].sum()),
        "n_not_launched": int((1-d["launched"]).sum()),
        "or_ancestry": np.exp(b),
        "ci_low": np.exp(b - 1.96*se),
        "ci_high": np.exp(b + 1.96*se),
        "p": m1.pvalues["n_ancestries"],
    })

    # -----------------------------------------
    # Model 2: ancestry + pre-cutoff study volume
    # -----------------------------------------
    X2 = sm.add_constant(d[["n_ancestries", "log_studies"]])
    m2 = sm.Logit(d["launched"], X2).fit(disp=False)

    b = m2.params["n_ancestries"]
    se = m2.bse["n_ancestries"]

    results.append({
        "cutoff": cutoff,
        "model": "M2_plus_study_volume",
        "n_pairs": len(d),
        "n_launched": int(d["launched"].sum()),
        "n_not_launched": int((1-d["launched"]).sum()),
        "or_ancestry": np.exp(b),
        "ci_low": np.exp(b - 1.96*se),
        "ci_high": np.exp(b + 1.96*se),
        "p": m2.pvalues["n_ancestries"],
    })

    print("=" * 70)
    print(f"LANDMARK {cutoff}")
    print("=" * 70)
    print("Eligible pairs:", len(d))
    print("Launched:", d["launched"].sum())
    print("Phase I-III:", (1-d["launched"]).sum())

    print("\nMean ancestry breadth:")
    print(d.groupby("pool")["n_ancestries"].mean().to_string())

    print("\nMedian study count:")
    print(d.groupby("pool")["n_studies"].median().to_string())

    print("\nM1 ancestry only:")
    print(
        f"OR={np.exp(m1.params['n_ancestries']):.3f} "
        f"95% CI "
        f"{np.exp(m1.params['n_ancestries']-1.96*m1.bse['n_ancestries']):.3f}-"
        f"{np.exp(m1.params['n_ancestries']+1.96*m1.bse['n_ancestries']):.3f} "
        f"p={m1.pvalues['n_ancestries']:.4g}"
    )

    print("\nM2 + study volume:")
    print(
        f"OR={np.exp(m2.params['n_ancestries']):.3f} "
        f"95% CI "
        f"{np.exp(m2.params['n_ancestries']-1.96*m2.bse['n_ancestries']):.3f}-"
        f"{np.exp(m2.params['n_ancestries']+1.96*m2.bse['n_ancestries']):.3f} "
        f"p={m2.pvalues['n_ancestries']:.4g}"
    )
    print()

res = pd.DataFrame(results)
res.to_csv(OUT / "16a_landmark_models.csv", index=False)

all_landmarks = pd.concat(landmark_tables, ignore_index=True)
all_landmarks.to_parquet(
    OUT / "16a_landmark_pair_table.parquet",
    index=False
)

print("\nSUMMARY")
print(res.to_string(index=False))
