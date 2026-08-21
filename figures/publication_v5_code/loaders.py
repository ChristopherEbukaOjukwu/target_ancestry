#!/usr/bin/env python3
"""Input loaders and schema adapters for publication_v5 figures."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from style import ROOT, find_first, normalize_pool, pick_column, read_table


def load_trails() -> pd.DataFrame:
    frame = read_table(ROOT / "step4/output/04_trails.parquet")

    pool = pick_column(
        frame,
        ["pool", "analysis_pool", "pool_label", "pool_ab", "ccat"],
        label="Step 4 approval pool",
    )
    all_ancestries = pick_column(
        frame,
        ["n_ancestries_all", "ancestry_count_all", "n_ancestries"],
        label="all-stage ancestry count",
    )
    initial = pick_column(
        frame,
        ["n_ancestries_initial", "ancestry_count_initial"],
        label="initial-stage ancestry count",
    )
    replication = pick_column(
        frame,
        ["n_ancestries_replication", "ancestry_count_replication"],
        label="replication-stage ancestry count",
    )
    studies = pick_column(
        frame,
        ["n_studies_total", "study_count", "n_studies"],
        label="study count",
    )
    indication = pick_column(
        frame,
        [
            "indication",
            "indication_name",
            "disease",
            "disease_name",
            "disease_id",
            "indication_id",
        ],
        label="indication",
        required=False,
    )
    gene = pick_column(
        frame,
        ["gene", "target_symbol", "target", "gene_symbol"],
        label="gene",
        required=False,
    )

    data = pd.DataFrame(
        {
            "pool": normalize_pool(frame[pool]),
            "n_ancestries_all": pd.to_numeric(
                frame[all_ancestries], errors="coerce"
            ),
            "n_ancestries_initial": pd.to_numeric(
                frame[initial], errors="coerce"
            ),
            "n_ancestries_replication": pd.to_numeric(
                frame[replication], errors="coerce"
            ),
            "n_studies_total": pd.to_numeric(
                frame[studies], errors="coerce"
            ),
        }
    )

    if indication:
        data["indication"] = frame[indication].astype(str)

    if gene:
        data["gene"] = frame[gene].astype(str)

    data = data[data["pool"].isin(["A", "B"])].copy()
    return data


def load_ancestry_rows() -> pd.DataFrame:
    frame = read_table(ROOT / "step3/output/03_study_ancestry.parquet")

    ancestry = pick_column(
        frame,
        [
            "ancestry",
            "ancestry_group",
            "ancestry_clean",
            "broad_ancestry",
            "consolidated_ancestry",
        ],
        label="Step 3 ancestry",
    )

    stage = pick_column(
        frame,
        ["stage", "study_stage", "ancestry_stage"],
        label="Step 3 evidence stage",
        required=False,
    )

    data = pd.DataFrame(
        {"ancestry": frame[ancestry].astype(str).str.strip()}
    )

    if stage:
        data["stage"] = frame[stage].astype(str).str.strip()

    return data


def standardize_regression(frame: pd.DataFrame) -> pd.DataFrame:
    model = pick_column(
        frame,
        ["model_id", "model", "model_name", "specification"],
        label="model",
    )
    exposure = pick_column(
        frame,
        ["exposure", "predictor", "term", "variable"],
        label="exposure",
    )
    estimate = pick_column(
        frame,
        ["odds_ratio", "or", "OR", "estimate_or"],
        label="odds ratio",
    )
    lower = pick_column(
        frame,
        [
            "ci_low",
            "ci_lower",
            "lower_ci",
            "or_ci_lower",
            "conf_low",
            "lcl",
        ],
        label="lower confidence bound",
    )
    upper = pick_column(
        frame,
        [
            "ci_high",
            "ci_upper",
            "upper_ci",
            "or_ci_upper",
            "conf_high",
            "ucl",
        ],
        label="upper confidence bound",
    )
    pvalue = pick_column(
        frame,
        ["p_value", "pvalue", "p", "pval"],
        label="p-value",
        required=False,
    )

    columns = [model, exposure, estimate, lower, upper]

    if pvalue:
        columns.append(pvalue)

    data = frame[columns].copy()
    data.columns = ["model", "exposure", "estimate", "lower", "upper"] + (
        ["p_value"] if pvalue else []
    )
    data["model"] = data["model"].astype(str)
    data["exposure"] = data["exposure"].astype(str)

    return data


def _normalized_exposure(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.casefold()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )


def load_pair_models() -> pd.DataFrame:
    frame = standardize_regression(
        read_table(ROOT / "step8/output/08_approval_regression.parquet")
    )

    normalized = _normalized_exposure(frame["exposure"])
    exact = frame[normalized.eq("n_ancestries_all")]

    if not exact.empty:
        return exact.copy()

    return frame[
        normalized.str.contains("ancestr")
        & normalized.str.contains("all")
        & ~normalized.str.contains("initial")
        & ~normalized.str.contains("replication")
        & ~normalized.str.contains("per_sd")
    ].copy()


def load_within_indication_models() -> pd.DataFrame:
    frame = standardize_regression(
        read_table(
            ROOT / "step10/output/10_within_indication_regression.csv"
        )
    )

    normalized = _normalized_exposure(frame["exposure"])
    exact = frame[normalized.eq("n_ancestries_all")]

    if not exact.empty:
        return exact.copy()

    return frame[
        normalized.str.contains("ancestr")
        & normalized.str.contains("all")
        & ~normalized.str.contains("initial")
        & ~normalized.str.contains("replication")
        & ~normalized.str.contains("per_sd")
    ].copy()


def load_practical_effects() -> pd.DataFrame:
    output = ROOT / "step12/output"

    path = find_first(
        output,
        [
            "*practical*significance*.csv",
            "*approval*probability*.csv",
            "*probability*difference*.csv",
            "*practical*.parquet",
            "*probability*.parquet",
        ],
    )

    if path is None:
        raise FileNotFoundError(
            "Could not locate the Step 12 practical-effect output."
        )

    frame = read_table(path)

    model = pick_column(
        frame,
        ["model_id", "model", "model_name"],
        label="Step 12 model",
    )
    exposure = pick_column(
        frame,
        ["exposure", "predictor", "variable"],
        label="Step 12 exposure",
    )
    estimate = pick_column(
        frame,
        [
            "approval_probability_difference",
            "probability_difference",
            "difference",
            "risk_difference",
            "difference_pp",
        ],
        label="approval-probability difference",
    )
    lower = pick_column(
        frame,
        [
            "difference_ci_low",
            "difference_ci_lower",
            "ci_low",
            "ci_lower",
            "lower_pp",
        ],
        label="practical-effect lower CI",
    )
    upper = pick_column(
        frame,
        [
            "difference_ci_high",
            "difference_ci_upper",
            "ci_high",
            "ci_upper",
            "upper_pp",
        ],
        label="practical-effect upper CI",
    )

    normalized = _normalized_exposure(frame[exposure])

    data = frame[
        normalized.eq("n_ancestries_all")
        | (
            normalized.str.contains("ancestr")
            & normalized.str.contains("all")
        )
    ].copy()

    if {"low_value", "high_value"}.issubset(data.columns):
        low = pd.to_numeric(data["low_value"], errors="coerce")
        high = pd.to_numeric(data["high_value"], errors="coerce")
        exact = np.isclose(low, 1) & np.isclose(high, 5)

        if exact.any():
            data = data[exact].copy()

    data = data[[model, estimate, lower, upper]].copy()
    data.columns = ["model", "estimate", "lower", "upper"]

    for column in ["estimate", "lower", "upper"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

        if data[column].abs().max() <= 1:
            data[column] *= 100

    return data.dropna().drop_duplicates(subset="model")


def load_within_gene() -> pd.DataFrame:
    frame = read_table(ROOT / "step7/output/07_within_gene.parquet")

    informative = pick_column(
        frame,
        ["informative", "is_informative", "sets_differ"],
        label="within-gene informative indicator",
        required=False,
    )
    difference = pick_column(
        frame,
        [
            "diff_all",
            "all_diff",
            "n_ancestries_all_diff",
            "difference_all",
            "set_size_diff_all",
        ],
        label="within-gene A minus B difference",
        required=False,
    )
    gene = pick_column(
        frame,
        ["gene", "target_symbol", "gene_symbol", "target"],
        label="within-gene symbol",
        required=False,
    )

    data = frame.copy()

    if informative:
        raw = data[informative]

        keep = (
            raw
            if raw.dtype == bool
            else raw.astype(str)
            .str.casefold()
            .isin({"true", "1", "yes"})
        )

        data = data[keep].copy()

    if difference is None:
        a_col = pick_column(
            data,
            ["n_ancestries_all_A", "a_n_ancestries_all", "all_A"],
            label="Pool A ancestry count",
        )
        b_col = pick_column(
            data,
            ["n_ancestries_all_B", "b_n_ancestries_all", "all_B"],
            label="Pool B ancestry count",
        )

        data["difference"] = (
            pd.to_numeric(data[a_col], errors="coerce")
            - pd.to_numeric(data[b_col], errors="coerce")
        )
    else:
        data["difference"] = pd.to_numeric(
            data[difference], errors="coerce"
        )

    if gene:
        data["gene_label"] = data[gene].astype(str)
    else:
        data["gene_label"] = [
            f"Gene {index + 1}" for index in range(len(data))
        ]

    return data.dropna(subset=["difference"]).copy()


def normalize_coloc_category(value: object) -> str:
    text = (
        str(value)
        .strip()
        .casefold()
        .replace("_", " ")
        .replace("-", " ")
    )

    text = re.sub(r"\s+", " ", text)

    if "robust" in text and "shared" in text:
        return "robust shared"

    if "robust" in text and "distinct" in text:
        return "robust distinct"

    if "ancestry" in text and "discord" in text:
        return "ancestry-discordant"

    if "mixed" in text and "prior" in text:
        return "mixed with prior sensitivity"

    if "prior" in text and "sensitive" in text:
        return "prior-sensitive"

    return text
