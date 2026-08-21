#!/usr/bin/env python3
"""
Step 15G3 — Does sample-size balance explain cross-ancestry portability?

Primary question
----------------
After accounting for which non-European ancestry is being compared with EUR,
is portability associated with the non-EUR/EUR effective-sample-size ratio?

Primary estimator: FIQT
Sensitivity estimators: Deming and naive

Models
------
M0: slope ~ log10(effective_N_ratio)
M1: slope ~ log10(effective_N_ratio) + comparison_population

Each model is run:
  (a) unweighted with cluster-robust SEs by gene_trait_uid
  (b) inverse-variance weighted using the approximate slope SE derived from
      the Step 15D bootstrap CI, again clustered by gene_trait_uid

Important:
This is a robustness/diagnostic analysis. It does not replace the original
bootstrap portability estimates.

Inputs
------
output/15d_primary_comparison_portability.parquet
output/15g2_sample_size_audit.csv

Outputs
-------
output/15g3_sample_balance_models.csv
output/15g3_sample_balance_descriptives.csv
output/15g3_sample_balance_summary.txt
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path("output")
PORTABILITY = ROOT / "15d_primary_comparison_portability.parquet"
SAMPLE_AUDIT = ROOT / "15g2_sample_size_audit.csv"

OUT_MODELS = ROOT / "15g3_sample_balance_models.csv"
OUT_DESC = ROOT / "15g3_sample_balance_descriptives.csv"
OUT_TXT = ROOT / "15g3_sample_balance_summary.txt"

ESTIMATORS = ["fiqt", "deming", "naive"]
Z975 = 1.959963984540054


def fit_model(df, formula, weighted):
    d = df.copy()

    if weighted:
        d = d[np.isfinite(d["slope_se_approx"]) & (d["slope_se_approx"] > 0)].copy()
        weights = 1.0 / np.square(d["slope_se_approx"])
        fit = smf.wls(formula, data=d, weights=weights).fit()
    else:
        fit = smf.ols(formula, data=d).fit()

    robust = fit.get_robustcov_results(
        cov_type="cluster",
        groups=d["gene_trait_uid"],
        use_correction=True,
    )

    names = list(fit.model.exog_names)
    params = pd.Series(np.asarray(robust.params), index=names)
    ses = pd.Series(np.asarray(robust.bse), index=names)
    pvals = pd.Series(np.asarray(robust.pvalues), index=names)
    ci = np.asarray(robust.conf_int())

    term = "log10_effective_n_ratio"
    idx = names.index(term)

    return {
        "n_comparisons": int(len(d)),
        "n_units": int(d["gene_trait_uid"].nunique()),
        "beta_log10_ratio": float(params[term]),
        "se_log10_ratio": float(ses[term]),
        "ci_lower": float(ci[idx, 0]),
        "ci_upper": float(ci[idx, 1]),
        "p_value": float(pvals[term]),
        "r_squared": float(getattr(fit, "rsquared", math.nan)),
    }


def main():
    if not PORTABILITY.exists():
        raise SystemExit(f"Missing: {PORTABILITY}")
    if not SAMPLE_AUDIT.exists():
        raise SystemExit(f"Missing: {SAMPLE_AUDIT}")

    port = pd.read_parquet(PORTABILITY)
    samp = pd.read_csv(SAMPLE_AUDIT)

    needed_port = {
        "comparison_uid",
        "gene_trait_uid",
        "estimator",
        "slope",
        "ci_lower",
        "ci_upper",
    }
    needed_samp = {
        "comparison_uid",
        "comparison_population",
        "effective_n_ratio",
        "eur_effective_n",
        "comparison_effective_n",
    }

    missing = needed_port - set(port.columns)
    if missing:
        raise SystemExit(f"Portability file missing: {sorted(missing)}")

    missing = needed_samp - set(samp.columns)
    if missing:
        raise SystemExit(f"Sample audit missing: {sorted(missing)}")

    # Keep primary Step 15D r2 threshold if present.
    if "r2_threshold" in port.columns:
        r2 = pd.to_numeric(port["r2_threshold"], errors="coerce")
        port = port[np.isclose(r2, 0.10, rtol=0.0, atol=1e-12)].copy()

    port = port[port["estimator"].isin(ESTIMATORS)].copy()

    for c in ["slope", "ci_lower", "ci_upper"]:
        port[c] = pd.to_numeric(port[c], errors="coerce")

    port["slope_se_approx"] = (
        port["ci_upper"] - port["ci_lower"]
    ) / (2.0 * Z975)

    samp = samp.drop_duplicates("comparison_uid").copy()
    samp["effective_n_ratio"] = pd.to_numeric(
        samp["effective_n_ratio"], errors="coerce"
    )

    # Step 15D already contains comparison_population. Do not merge a second
    # copy from 15G2, otherwise pandas creates comparison_population_x/_y
    # and the downstream ancestry-adjusted model cannot find the canonical name.
    sample_cols = [
        "comparison_uid",
        "effective_n_ratio",
        "eur_effective_n",
        "comparison_effective_n",
    ]

    if "comparison_population" not in port.columns:
        sample_cols.append("comparison_population")

    df = port.merge(
        samp[sample_cols],
        on="comparison_uid",
        how="inner",
        validate="many_to_one",
    )

    if "comparison_population" not in df.columns:
        raise SystemExit(
            "comparison_population was not found after merging Step 15D and Step 15G2 inputs."
        )

    df = df[
        np.isfinite(df["effective_n_ratio"])
        & (df["effective_n_ratio"] > 0)
        & np.isfinite(df["slope"])
    ].copy()

    df["log10_effective_n_ratio"] = np.log10(df["effective_n_ratio"])

    # Descriptives by ancestry, based on one row per comparison.
    one = (
        df.sort_values(["comparison_uid", "estimator"])
        .drop_duplicates("comparison_uid")
        .copy()
    )
    desc = (
        one.groupby("comparison_population", as_index=False)
        .agg(
            n_comparisons=("comparison_uid", "nunique"),
            n_units=("gene_trait_uid", "nunique"),
            median_effective_n_ratio=("effective_n_ratio", "median"),
            min_effective_n_ratio=("effective_n_ratio", "min"),
            max_effective_n_ratio=("effective_n_ratio", "max"),
            sd_log10_effective_n_ratio=("log10_effective_n_ratio", "std"),
        )
    )
    desc.to_csv(OUT_DESC, index=False)

    rows = []

    for estimator in ESTIMATORS:
        d = df[df["estimator"] == estimator].copy()

        models = [
            (
                "M0_unadjusted",
                "slope ~ log10_effective_n_ratio",
            ),
            (
                "M1_ancestry_adjusted",
                "slope ~ log10_effective_n_ratio + C(comparison_population)",
            ),
        ]

        for model_name, formula in models:
            for weighted in [False, True]:
                result = fit_model(d, formula, weighted)
                rows.append(
                    {
                        "estimator": estimator,
                        "model": model_name,
                        "weighting": (
                            "inverse_variance_approx"
                            if weighted
                            else "unweighted"
                        ),
                        **result,
                    }
                )

    results = pd.DataFrame(rows)
    results.to_csv(OUT_MODELS, index=False)

    lines = []
    lines.append("=" * 88)
    lines.append("STEP 15G3 — SAMPLE-SIZE BALANCE VS PORTABILITY")
    lines.append("=" * 88)
    lines.append("")
    lines.append("Interpretation of beta_log10_ratio:")
    lines.append(
        "  Change in portability slope for a 10-fold increase in the "
        "non-EUR/EUR effective-N ratio."
    )
    lines.append(
        "  Positive beta: more sample-size-balanced comparisons tend to "
        "have larger portability slopes."
    )
    lines.append(
        "  The ancestry-adjusted model asks whether that pattern remains "
        "within ancestry groups."
    )
    lines.append("")
    lines.append("Ancestry-specific effective-N-ratio variation:")
    lines.append(desc.to_string(index=False))
    lines.append("")
    lines.append("Model results:")
    lines.append(
        results[
            [
                "estimator",
                "model",
                "weighting",
                "n_comparisons",
                "n_units",
                "beta_log10_ratio",
                "ci_lower",
                "ci_upper",
                "p_value",
                "r_squared",
            ]
        ].to_string(index=False)
    )
    lines.append("")
    lines.append("Primary result to inspect:")
    lines.append(
        "  FIQT + M1_ancestry_adjusted + inverse_variance_approx."
    )
    lines.append(
        "  Deming and naive are sensitivity analyses."
    )
    lines.append("=" * 88)

    text = "\n".join(lines) + "\n"
    OUT_TXT.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
