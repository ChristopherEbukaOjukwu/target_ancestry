from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

ROOT = Path(".")
COMP_DIR = ROOT / "step15/intermediate/15b4/comparisons"
PORT_TABLE = ROOT / "step15/output/15g_portability_analysis_table.parquet"
RETAINED = ROOT / "step15/output/15c3_portability_variants_primary_retained.parquet"
STEP15D = ROOT / "step15/15d_estimate_portability.py"
OUT = ROOT / "step16/output"

N_SIM = 1000
SEED = 42


# Import exact Step 15D estimators


spec = importlib.util.spec_from_file_location("step15d", STEP15D)
step15d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step15d)

estimate_all = step15d.estimate_all


# Locked primary comparison universe


port = pd.read_parquet(PORT_TABLE)

primary = port[
    port["unit_analysis_role"].eq("PRIMARY")
    & port["is_primary_threshold"].eq(True)
][["comparison_uid", "estimator", "slope"]].copy()

comparison_ids = sorted(primary["comparison_uid"].unique())

print("=" * 78)
print("STEP 16F — PERFECT-PORTABILITY NULL CALIBRATION")
print("=" * 78)
print("Primary comparisons:", len(comparison_ids))
print("Simulation replicates:", N_SIM)
print("Seed:", SEED)


# Exact LD-pruned retained variants


ret = pd.read_parquet(RETAINED)

print("\nRetained-table columns:")
print(ret.columns.tolist())

required = {"comparison_uid", "variant_id"}
missing = required - set(ret.columns)

if missing:
    raise RuntimeError(
        f"Retained table missing required columns: {sorted(missing)}"
    )

ret = ret[
    ret["comparison_uid"].isin(comparison_ids)
].copy()

ret = ret[
    ["comparison_uid", "variant_id"]
].drop_duplicates()

print("\nExact retained variant-comparisons:", len(ret))
print("Comparisons represented:", ret["comparison_uid"].nunique())

# This should match Step 16E
if len(ret) != 714:
    raise RuntimeError(
        f"Expected 714 retained variant-comparisons, found {len(ret)}"
    )

if ret["comparison_uid"].nunique() != 104:
    raise RuntimeError(
        "Expected 104 primary ancestry comparisons."
    )


# Recover beta/SE only for the exact retained variants


comparison_data = {}

for uid in comparison_ids:

    retained_ids = set(
        ret.loc[
            ret["comparison_uid"].eq(uid),
            "variant_id"
        ]
    )

    f = COMP_DIR / f"{uid}.parquet"

    if not f.exists():
        raise FileNotFoundError(f)

    x = pd.read_parquet(f)

    q = x[
        x["variant_id"].isin(retained_ids)
    ].copy()

    # one row per retained variant
    q = q.drop_duplicates(subset=["variant_id"])

    for c in [
        "eur_beta",
        "eur_se",
        "comparison_beta",
        "comparison_se",
    ]:
        q[c] = pd.to_numeric(q[c], errors="coerce")

    q = q[
        np.isfinite(q["eur_beta"])
        & np.isfinite(q["eur_se"])
        & np.isfinite(q["comparison_beta"])
        & np.isfinite(q["comparison_se"])
        & (q["eur_se"] > 0)
        & (q["comparison_se"] > 0)
    ].copy()

    if len(q) != len(retained_ids):
        raise RuntimeError(
            f"{uid}: retained={len(retained_ids)}, "
            f"usable beta/SE rows={len(q)}"
        )

    
    # Shared latent effect under perfect portability.
    #
    # Neither ancestry is treated as truth.
    # theta is the inverse-variance estimate of a common
    # underlying effect under H0: beta_EUR = beta_other.
    

    bx = q["eur_beta"].to_numpy(dtype=float)
    by = q["comparison_beta"].to_numpy(dtype=float)

    sx = q["eur_se"].to_numpy(dtype=float)
    sy = q["comparison_se"].to_numpy(dtype=float)

    wx = 1.0 / np.square(sx)
    wy = 1.0 / np.square(sy)

    theta = (
        wx * bx + wy * by
    ) / (wx + wy)

    comparison_data[uid] = {
        "theta": theta,
        "sx": sx,
        "sy": sy,
        "n_variants": len(q),
    }

total_variants = sum(
    d["n_variants"]
    for d in comparison_data.values()
)

print("Variants loaded into calibration:", total_variants)

if total_variants != 714:
    raise RuntimeError(
        f"Simulation universe should contain 714 variants, "
        f"found {total_variants}"
    )


# Observed slopes


obs = (
    primary.groupby("estimator")["slope"]
    .agg(["count", "mean", "median"])
)

print("\nObserved comparison-level slopes:")
print(obs.to_string())


# Perfect-portability simulations


rng = np.random.default_rng(SEED)

simulation_rows = []

for rep in range(N_SIM):

    slopes = {
        "naive": [],
        "fiqt": [],
        "deming": [],
    }

    for uid, dat in comparison_data.items():

        theta = dat["theta"]
        sx = dat["sx"]
        sy = dat["sy"]

        # Perfect portability:
        # both ancestries share exactly the same true beta.
        #
        # Their observed estimates differ only because of
        # ancestry-specific sampling uncertainty.
        x_sim = theta + rng.normal(
            0.0,
            sx,
            size=len(theta)
        )

        y_sim = theta + rng.normal(
            0.0,
            sy,
            size=len(theta)
        )

        est = estimate_all(
            x_sim,
            y_sim,
            sx,
            sy,
        )

        for estimator in ["naive", "fiqt", "deming"]:
            value = est[estimator]

            if np.isfinite(value):
                slopes[estimator].append(value)

    for estimator, values in slopes.items():

        values = np.asarray(values, dtype=float)

        simulation_rows.append({
            "replicate": rep,
            "estimator": estimator,
            "n_valid_comparisons": len(values),
            "mean_slope": float(np.mean(values)),
            "median_slope": float(np.median(values)),
        })

sim = pd.DataFrame(simulation_rows)


# Compare observed median with null distribution


summary_rows = []

for estimator in ["naive", "fiqt", "deming"]:

    null = sim.loc[
        sim["estimator"].eq(estimator),
        "median_slope"
    ].to_numpy()

    observed = float(
        primary.loc[
            primary["estimator"].eq(estimator),
            "slope"
        ].median()
    )

    lo, hi = np.quantile(
        null,
        [0.025, 0.975]
    )

    # empirical one-sided probability
    empirical_p = (
        np.sum(null <= observed) + 1
    ) / (len(null) + 1)

    summary_rows.append({
        "estimator": estimator,
        "observed_median": observed,
        "null_median_mean": float(np.mean(null)),
        "null_median_median": float(np.median(null)),
        "null_ci_low": float(lo),
        "null_ci_high": float(hi),
        "empirical_p_observed_or_lower": empirical_p,
        "n_simulations": len(null),
    })

summary = pd.DataFrame(summary_rows)

print("\nPerfect-portability calibration:")
print(summary.to_string(index=False))


# Save


sim.to_csv(
    OUT / "16f_null_calibration_replicates.csv",
    index=False
)

summary.to_csv(
    OUT / "16f_null_calibration_summary.csv",
    index=False
)

ret.to_csv(
    OUT / "16f_exact_variant_universe.csv",
    index=False
)

print("\nSaved:")
print(" ", OUT / "16f_null_calibration_replicates.csv")
print(" ", OUT / "16f_null_calibration_summary.csv")
print(" ", OUT / "16f_exact_variant_universe.csv")
