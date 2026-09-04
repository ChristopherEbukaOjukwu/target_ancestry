from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

TRAILS = Path("step4/output/04_trails.parquet")
SAMPLE = Path("step16/output/16b_cumulative_sample_size_pair_table.parquet")
OUT = Path("step16/output")

trails = pd.read_parquet(TRAILS)
d = pd.read_parquet(SAMPLE)

# d contains the 735 complete-case pairs from Step 16B
d["log_studies"] = np.log1p(d["n_studies_total"])
d["log_cumulative_n"] = np.log1p(d["cumulative_n"])

print("=" * 72)
print("STEP 16C — ANCESTRY BREADTH VS EVIDENCE VOLUME")
print("=" * 72)
print("Pairs:", len(d))

# ------------------------------------------------------------
# 1. Correlations
# ------------------------------------------------------------
rho_study, p_study = spearmanr(
    d["n_ancestries_all"], d["n_studies_total"]
)

rho_n, p_n = spearmanr(
    d["n_ancestries_all"], d["cumulative_n"]
)

print("\nSpearman correlations")
print(
    f"Ancestry breadth vs study count: "
    f"rho={rho_study:.3f}, p={p_study:.3g}"
)
print(
    f"Ancestry breadth vs cumulative participants: "
    f"rho={rho_n:.3f}, p={p_n:.3g}"
)

# ------------------------------------------------------------
# 2. Poisson models
# ------------------------------------------------------------
m_studies = smf.glm(
    "n_ancestries_all ~ log_studies",
    data=d,
    family=sm.families.Poisson()
).fit()

m_sample = smf.glm(
    "n_ancestries_all ~ log_cumulative_n",
    data=d,
    family=sm.families.Poisson()
).fit()

m_both = smf.glm(
    "n_ancestries_all ~ log_studies + log_cumulative_n",
    data=d,
    family=sm.families.Poisson()
).fit()

def report(model, variables):
    for v in variables:
        b = model.params[v]
        se = model.bse[v]
        print(
            f"{v}: "
            f"IRR={np.exp(b):.3f} "
            f"95% CI {np.exp(b-1.96*se):.3f}-"
            f"{np.exp(b+1.96*se):.3f} "
            f"p={model.pvalues[v]:.3g}"
        )

print("\nModel 1: ancestry breadth ~ study count")
report(m_studies, ["log_studies"])

print("\nModel 2: ancestry breadth ~ cumulative participants")
report(m_sample, ["log_cumulative_n"])

print("\nModel 3: both evidence-volume measures")
report(m_both, ["log_studies", "log_cumulative_n"])

# ------------------------------------------------------------
# 3. Descriptive quantile bins
# ------------------------------------------------------------
# qcut may drop duplicate boundaries if distributions are tied
d["study_bin"] = pd.qcut(
    d["n_studies_total"],
    q=4,
    duplicates="drop"
)

d["sample_bin"] = pd.qcut(
    d["cumulative_n"],
    q=4,
    duplicates="drop"
)

study_summary = (
    d.groupby("study_bin", observed=True)
    .agg(
        n=("ti_uid", "size"),
        median_studies=("n_studies_total", "median"),
        mean_ancestries=("n_ancestries_all", "mean"),
        median_ancestries=("n_ancestries_all", "median"),
    )
    .reset_index()
)

sample_summary = (
    d.groupby("sample_bin", observed=True)
    .agg(
        n=("ti_uid", "size"),
        median_cumulative_n=("cumulative_n", "median"),
        mean_ancestries=("n_ancestries_all", "mean"),
        median_ancestries=("n_ancestries_all", "median"),
    )
    .reset_index()
)

print("\nAncestry breadth across study-count quartiles:")
print(study_summary.to_string(index=False))

print("\nAncestry breadth across cumulative-N quartiles:")
print(sample_summary.to_string(index=False))

# ------------------------------------------------------------
# 4. Save
# ------------------------------------------------------------
rows = []

for name, model, variables in [
    ("study_count_only", m_studies, ["log_studies"]),
    ("sample_size_only", m_sample, ["log_cumulative_n"]),
    ("both", m_both, ["log_studies", "log_cumulative_n"]),
]:
    for v in variables:
        b = model.params[v]
        se = model.bse[v]
        rows.append({
            "model": name,
            "variable": v,
            "irr": np.exp(b),
            "ci_low": np.exp(b - 1.96*se),
            "ci_high": np.exp(b + 1.96*se),
            "p": model.pvalues[v],
        })

pd.DataFrame(rows).to_csv(
    OUT / "16c_ancestry_evidence_models.csv",
    index=False
)

study_summary.to_csv(
    OUT / "16c_study_count_quartiles.csv",
    index=False
)

sample_summary.to_csv(
    OUT / "16c_sample_size_quartiles.csv",
    index=False
)
