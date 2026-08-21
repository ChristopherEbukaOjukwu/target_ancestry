#!/usr/bin/env python3
"""
Step 15A3: Lock the broader portability/colocalization trait mappings.

Inputs
------
output/15a2_mapping_candidates.csv
../step4/output/04_trails.parquet

Outputs
-------
output/15a3_mapping_decisions.csv
output/15_trait_mapping_locked.csv
output/15_trait_mapping_primary.csv
output/15_trait_mapping_sensitivity.csv
output/15_trait_mapping_rejected.csv
output/15_pair_trait_mapping_locked.csv
output/15_pair_trait_mapping_locked.parquet
output/15_gene_trait_units_locked.csv
output/15_gene_trait_units_locked.parquet
output/15a3_mapping_summary.json

Principles
----------
- At most one PRIMARY mapping per indication.
- Verified direct GBMI disease endpoints take precedence.
- Direct quantitative traits are preferred when no direct disease endpoint exists.
- Additional correlated measures are SENSITIVITY only.
- Specific organ-function proxies may be primary when no stronger phenotype exists.
- Broad/nonspecific CRP mappings are rejected.
- Unverified source mappings cannot be accepted.
- Mixed-pool gene-trait-source units are retained and flagged. They are excluded
  from the primary approval comparison and available for sensitivity analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


# Explicit scientific adjudication.
#
# Keys are normalized indication names. Each value gives accepted trait choices
# using (source, canonical_group). Every remaining candidate is rejected.
DECISIONS: dict[str, dict[str, Any]] = {
    "obesity": {
        "primary": [("Pan-UKB", "body_mass_index")],
        "sensitivity": [("Pan-UKB", "waist_circumference")],
    },
    "asthma": {
        "primary": [("GBMI", "Asthma")],
        "sensitivity": [
            ("Pan-UKB", "eosinophil_count"),
            ("Pan-UKB", "fev1"),
            ("Pan-UKB", "fvc"),
        ],
    },
    "hyperlipidemias": {
        "primary": [("Pan-UKB", "ldl_cholesterol")],
        "sensitivity": [
            ("Pan-UKB", "apolipoprotein_b"),
            ("Pan-UKB", "total_cholesterol"),
        ],
    },
    "pulmonary disease chronic obstructive": {
        "primary": [("GBMI", "COPD")],
        "sensitivity": [
            ("Pan-UKB", "fev1"),
            ("Pan-UKB", "fvc"),
        ],
    },
    "hypertension": {
        "primary": [("Pan-UKB", "systolic_blood_pressure")],
        "sensitivity": [
            ("Pan-UKB", "diastolic_blood_pressure"),
        ],
    },
    "diabetes mellitus type 2": {
        "primary": [("Pan-UKB", "hba1c")],
        "sensitivity": [],
    },
    "hypercholesterolemia": {
        "primary": [("Pan-UKB", "ldl_cholesterol")],
        "sensitivity": [
            ("Pan-UKB", "apolipoprotein_b"),
            ("Pan-UKB", "total_cholesterol"),
        ],
    },
    "hypertriglyceridemia": {
        "primary": [("Pan-UKB", "triglycerides")],
        "sensitivity": [],
    },
    "diabetes mellitus": {
        "primary": [("Pan-UKB", "hba1c")],
        "sensitivity": [],
    },
    "anemia": {
        "primary": [("Pan-UKB", "haemoglobin")],
        "sensitivity": [
            ("Pan-UKB", "haematocrit"),
            ("Pan-UKB", "mean_corpuscular_volume"),
        ],
    },
    "thrombocytopenia": {
        "primary": [("Pan-UKB", "platelet_count")],
        "sensitivity": [],
    },
    "thrombocytosis": {
        "primary": [("Pan-UKB", "platelet_count")],
        "sensitivity": [],
    },
    "hyperlipoproteinemia type ii": {
        "primary": [("Pan-UKB", "ldl_cholesterol")],
        "sensitivity": [
            ("Pan-UKB", "apolipoprotein_b"),
            ("Pan-UKB", "total_cholesterol"),
        ],
    },
    "dyslipidemias": {
        "primary": [("Pan-UKB", "ldl_cholesterol")],
        "sensitivity": [
            ("Pan-UKB", "apolipoprotein_b"),
            ("Pan-UKB", "total_cholesterol"),
        ],
    },
    "renal insufficiency": {
        "primary": [("Pan-UKB", "egfr_creatinine")],
        "sensitivity": [
            ("Pan-UKB", "egfr_creatinine_cystatin_c"),
            ("Pan-UKB", "egfr_cystatin_c"),
            ("Pan-UKB", "serum_creatinine"),
            ("Pan-UKB", "cystatin_c"),
        ],
    },
    # HbA1c measures glycemia but is not specific to type 1 diabetes etiology.
    "diabetes mellitus type 1": {
        "primary": [],
        "sensitivity": [("Pan-UKB", "hba1c")],
        "tier_override": {
            ("Pan-UKB", "hba1c"): "Proxy biomarker",
        },
    },
    "neutropenia": {
        "primary": [("Pan-UKB", "neutrophil_count")],
        "sensitivity": [],
    },
    "venous thrombosis": {
        "primary": [("GBMI", "VTE")],
        "sensitivity": [],
    },
    "glomerulosclerosis focal segmental": {
        "primary": [("Pan-UKB", "egfr_creatinine")],
        "sensitivity": [
            ("Pan-UKB", "egfr_creatinine_cystatin_c"),
            ("Pan-UKB", "egfr_cystatin_c"),
            ("Pan-UKB", "serum_creatinine"),
            ("Pan-UKB", "cystatin_c"),
        ],
    },
    "liver diseases": {
        "primary": [],
        "sensitivity": [],
        "reject_reason": (
            "Broad indication; available liver markers cannot identify a "
            "specific disease process."
        ),
    },
    "osteoporosis": {
        "primary": [("Pan-UKB", "heel_bone_mineral_density")],
        "sensitivity": [],
    },
    "stroke": {
        "primary": [("GBMI", "Stroke")],
        "sensitivity": [],
    },
    "anemia sickle cell": {
        "primary": [],
        "sensitivity": [
            ("Pan-UKB", "haemoglobin"),
            ("Pan-UKB", "haematocrit"),
            ("Pan-UKB", "mean_corpuscular_volume"),
        ],
        "tier_override": {
            ("Pan-UKB", "haemoglobin"): "Proxy biomarker",
            ("Pan-UKB", "haematocrit"): "Proxy biomarker",
            ("Pan-UKB", "mean_corpuscular_volume"): "Proxy biomarker",
        },
    },
    "glomerulonephritis iga": {
        "primary": [("Pan-UKB", "egfr_creatinine")],
        "sensitivity": [
            ("Pan-UKB", "egfr_creatinine_cystatin_c"),
            ("Pan-UKB", "egfr_cystatin_c"),
            ("Pan-UKB", "serum_creatinine"),
            ("Pan-UKB", "cystatin_c"),
        ],
    },
    "gout": {
        "primary": [("GBMI", "Gout")],
        "sensitivity": [],
    },
    "rhinitis allergic seasonal": {
        "primary": [],
        "sensitivity": [("Pan-UKB", "eosinophil_count")],
    },
    "diabetic nephropathies": {
        "primary": [("Pan-UKB", "egfr_creatinine")],
        "sensitivity": [
            ("Pan-UKB", "egfr_creatinine_cystatin_c"),
            ("Pan-UKB", "egfr_cystatin_c"),
            ("Pan-UKB", "serum_creatinine"),
            ("Pan-UKB", "cystatin_c"),
        ],
    },
    "heart failure": {
        "primary": [("GBMI", "HF")],
        "sensitivity": [],
    },
    "hypercalcemia": {
        "primary": [("Pan-UKB", "calcium")],
        "sensitivity": [],
    },
    "hypocalcemia": {
        "primary": [("Pan-UKB", "calcium")],
        "sensitivity": [],
    },
    "hypophosphatemia": {
        "primary": [("Pan-UKB", "phosphate")],
        "sensitivity": [],
    },
    "leukopenia": {
        "primary": [("Pan-UKB", "white_blood_cell_count")],
        "sensitivity": [],
    },
    "liver cirrhosis": {
        "primary": [("Pan-UKB", "albumin")],
        "sensitivity": [
            ("Pan-UKB", "alanine_aminotransferase"),
            ("Pan-UKB", "alkaline_phosphatase"),
            ("Pan-UKB", "gamma_glutamyltransferase"),
        ],
    },
    "liver cirrhosis biliary": {
        "primary": [("Pan-UKB", "alkaline_phosphatase")],
        "sensitivity": [
            ("Pan-UKB", "gamma_glutamyltransferase"),
            ("Pan-UKB", "albumin"),
            ("Pan-UKB", "alanine_aminotransferase"),
        ],
    },
    "non alcoholic fatty liver disease": {
        "primary": [("Pan-UKB", "alanine_aminotransferase")],
        "sensitivity": [
            ("Pan-UKB", "gamma_glutamyltransferase"),
        ],
    },
    "polycythemia vera": {
        "primary": [("Pan-UKB", "haematocrit")],
        "sensitivity": [
            ("Pan-UKB", "haemoglobin"),
            ("Pan-UKB", "red_blood_cell_count"),
        ],
    },
    "pulmonary embolism": {
        "primary": [("GBMI", "VTE")],
        "sensitivity": [],
    },
    # Allergic inflammatory biomarkers are retained only as sensitivities.
    "dermatitis atopic": {
        "primary": [],
        "sensitivity": [("Pan-UKB", "eosinophil_count")],
    },
    "rhinitis allergic": {
        "primary": [],
        "sensitivity": [("Pan-UKB", "eosinophil_count")],
    },
}

# Broad/nonspecific CRP candidates are explicitly rejected.
CRP_REJECT_INDICATIONS = {
    "colitis ulcerative",
    "crohn disease",
    "arthritis rheumatoid",
    "psoriasis",
    "autoimmune diseases",
    "lupus erythematosus systemic",
    "inflammatory bowel diseases",
}

UNVERIFIED_GBMI_REJECT = {
    "endometrial neoplasms",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--candidates",
        type=Path,
        default=Path("output/15a2_mapping_candidates.csv"),
    )
    p.add_argument(
        "--trails",
        type=Path,
        default=Path("../step4/output/04_trails.parquet"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
    )
    return p.parse_args()


def clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def norm(value: Any) -> str:
    text = clean(value).casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def make_id(prefix: str, *values: Any) -> str:
    raw = "||".join(clean(v) for v in values)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def decision_for(row: pd.Series) -> tuple[str, str, str]:
    indication = norm(row["indication_mesh_term"])
    source = clean(row["candidate_source"])
    group = clean(row["canonical_group"])
    key = (source, group)

    if not bool(row["source_verified"]):
        return (
            "REJECT",
            clean(row["mapping_tier"]),
            "Source lacks verified EUR plus non-EUR cross-ancestry data.",
        )

    if indication in CRP_REJECT_INDICATIONS and group == "c_reactive_protein":
        return (
            "REJECT",
            clean(row["mapping_tier"]),
            "CRP is too nonspecific to represent this disease indication.",
        )

    if indication in UNVERIFIED_GBMI_REJECT:
        return (
            "REJECT",
            clean(row["mapping_tier"]),
            "GBMI endpoint is not cross-ancestry eligible in this release.",
        )

    spec = DECISIONS.get(indication)
    if spec is None:
        return (
            "REJECT",
            clean(row["mapping_tier"]),
            "No prespecified accepted mapping for this indication.",
        )

    tier = spec.get("tier_override", {}).get(
        key,
        clean(row["mapping_tier"]),
    )

    if key in spec.get("primary", []):
        return (
            "PRIMARY",
            tier,
            "Locked as the single primary representation of this indication.",
        )

    if key in spec.get("sensitivity", []):
        return (
            "SENSITIVITY",
            tier,
            "Retained only for prespecified sensitivity analysis.",
        )

    return (
        "REJECT",
        tier,
        spec.get(
            "reject_reason",
            "Candidate is redundant, less specific, or not selected.",
        ),
    )


def adjudicate(candidates: pd.DataFrame) -> pd.DataFrame:
    required = {
        "indication_mesh_id",
        "indication_mesh_term",
        "candidate_source",
        "candidate_trait_key",
        "candidate_trait_name",
        "phenotype_id",
        "genome_build",
        "mapping_tier",
        "canonical_group",
        "pops_available",
        "non_eur_pops_available",
        "source_verified",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise SystemExit(f"Candidate table missing columns: {missing}")

    rows = []
    for _, row in candidates.iterrows():
        role, locked_tier, reason = decision_for(row)
        out = row.to_dict()
        out.update(
            {
                "analysis_role": role,
                "locked_mapping_tier": locked_tier,
                "locking_reason": reason,
                "mapping_uid": make_id(
                    "map",
                    row["indication_mesh_id"],
                    row["candidate_trait_key"],
                ),
            }
        )
        rows.append(out)

    decisions = pd.DataFrame(rows)

    primary_counts = (
        decisions[decisions["analysis_role"].eq("PRIMARY")]
        .groupby("indication_mesh_id")
        .size()
    )
    bad = primary_counts[primary_counts > 1]
    if len(bad):
        raise SystemExit(
            "More than one PRIMARY mapping for indication(s): "
            + ", ".join(map(str, bad.index))
        )

    accepted_unverified = decisions[
        decisions["analysis_role"].isin(["PRIMARY", "SENSITIVITY"])
        & ~decisions["source_verified"].astype(bool)
    ]
    if len(accepted_unverified):
        raise SystemExit(
            "Accepted mapping has unverified source:\n"
            + accepted_unverified[
                [
                    "indication_mesh_term",
                    "candidate_trait_name",
                    "candidate_source",
                ]
            ].to_string(index=False)
        )

    return decisions


def expand_to_pairs(
    locked: pd.DataFrame,
    trails: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "ti_uid",
        "gene",
        "pool",
        "indication_mesh_id",
        "indication_mesh_term",
    }
    missing = sorted(required - set(trails.columns))
    if missing:
        raise SystemExit(f"Step 4 trails missing columns: {missing}")

    # One row per target-indication pair before expansion.
    pairs = trails.drop_duplicates("ti_uid").copy()

    merged = pairs.merge(
        locked,
        on="indication_mesh_id",
        how="inner",
        suffixes=("_pair", "_mapping"),
        validate="many_to_many",
    )

    # Keep the pair's original indication label authoritative.
    if "indication_mesh_term_pair" in merged.columns:
        merged["indication_mesh_term"] = merged[
            "indication_mesh_term_pair"
        ]
    elif "indication_mesh_term" not in merged.columns:
        raise SystemExit("Merged pair table lacks indication term.")

    merged["pair_trait_uid"] = [
        make_id(
            "pt",
            ti_uid,
            trait_key,
            role,
        )
        for ti_uid, trait_key, role in zip(
            merged["ti_uid"],
            merged["candidate_trait_key"],
            merged["analysis_role"],
        )
    ]

    keep_front = [
        "pair_trait_uid",
        "ti_uid",
        "gene",
        "pool",
        "indication_mesh_id",
        "indication_mesh_term",
        "analysis_role",
        "locked_mapping_tier",
        "candidate_source",
        "candidate_trait_key",
        "candidate_trait_name",
        "phenotype_id",
        "genome_build",
        "canonical_group",
        "pops_available",
        "non_eur_pops_available",
        "mapping_uid",
        "mapping_rationale",
        "locking_reason",
    ]
    keep_front = [c for c in keep_front if c in merged.columns]
    remaining = [c for c in merged.columns if c not in keep_front]
    return merged[keep_front + remaining].copy()


def collapse_gene_trait_units(pair_traits: pd.DataFrame) -> pd.DataFrame:
    role_rank = {"PRIMARY": 1, "SENSITIVITY": 2}
    tier_rank = {
        "Direct disease": 1,
        "Direct quantitative readout": 2,
        "Closely related phenotype": 3,
        "Proxy biomarker": 4,
    }

    group_cols = [
        "gene",
        "candidate_source",
        "candidate_trait_key",
        "candidate_trait_name",
        "phenotype_id",
        "genome_build",
        "canonical_group",
    ]

    rows = []
    for keys, group in pair_traits.groupby(group_cols, dropna=False):
        data = dict(zip(group_cols, keys))
        pools = sorted(set(group["pool"].dropna().astype(str)))
        indications = sorted(
            set(group["indication_mesh_term"].dropna().astype(str))
        )
        mesh_ids = sorted(
            set(group["indication_mesh_id"].dropna().astype(str))
        )
        roles = sorted(
            set(group["analysis_role"]),
            key=lambda x: role_rank.get(x, 99),
        )
        unit_role = (
            "PRIMARY"
            if "PRIMARY" in roles
            else "SENSITIVITY"
        )

        # A gene-trait-source unit can be reused across indications with
        # different mapping tiers. Assign the unit the tier associated with its
        # highest-priority analysis role, then retain all contributing tiers
        # for auditability.
        role_subset = group[
            group["analysis_role"].eq(unit_role)
        ]
        if role_subset.empty:
            role_subset = group

        contributing_tiers = sorted(
            {
                clean(value)
                for value in group["locked_mapping_tier"]
                if clean(value)
            },
            key=lambda value: tier_rank.get(value, 99),
        )
        role_tiers = sorted(
            {
                clean(value)
                for value in role_subset["locked_mapping_tier"]
                if clean(value)
            },
            key=lambda value: tier_rank.get(value, 99),
        )
        unit_mapping_tier = (
            role_tiers[0]
            if role_tiers
            else (
                contributing_tiers[0]
                if contributing_tiers
                else ""
            )
        )

        mixed = "A" in pools and "B" in pools

        data.update(
            {
                "gene_trait_uid": make_id(
                    "gt",
                    data["gene"],
                    data["candidate_source"],
                    data["candidate_trait_key"],
                ),
                "unit_analysis_role": unit_role,
                "locked_mapping_tier": unit_mapping_tier,
                "mapping_tiers_contributing": " || ".join(
                    contributing_tiers
                ),
                "roles_contributing": ",".join(roles),
                "n_pair_trait_rows": int(len(group)),
                "n_target_indication_pairs": int(
                    group["ti_uid"].nunique()
                ),
                "n_indications": len(indications),
                "indication_mesh_ids": " || ".join(mesh_ids),
                "indications": " || ".join(indications),
                "pools": ",".join(pools),
                "n_A_pairs": int((group["pool"] == "A").sum()),
                "n_B_pairs": int((group["pool"] == "B").sum()),
                "mixed_pool": mixed,
                "primary_approval_eligible": (
                    unit_role == "PRIMARY" and not mixed
                ),
                "sensitivity_pool_assignment": (
                    "A" if "A" in pools else "B"
                ),
                "pops_available": " || ".join(
                    sorted(
                        set(
                            group["pops_available"]
                            .dropna()
                            .astype(str)
                        )
                    )
                ),
                "non_eur_pops_available": " || ".join(
                    sorted(
                        set(
                            group["non_eur_pops_available"]
                            .dropna()
                            .astype(str)
                        )
                    )
                ),
                "mapping_uids": " || ".join(
                    sorted(set(group["mapping_uid"]))
                ),
            }
        )
        rows.append(data)

    units = pd.DataFrame(rows)
    if len(units):
        units = units.sort_values(
            [
                "unit_analysis_role",
                "candidate_source",
                "candidate_trait_name",
                "gene",
            ]
        ).reset_index(drop=True)
    return units


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in [args.candidates, args.trails]:
        if not input_path.exists():
            raise SystemExit(f"Missing required input: {input_path}")

    candidates = pd.read_csv(args.candidates, low_memory=False)
    trails = pd.read_parquet(args.trails)

    decisions = adjudicate(candidates)

    locked = decisions[
        decisions["analysis_role"].isin(["PRIMARY", "SENSITIVITY"])
    ].copy()
    primary = locked[locked["analysis_role"].eq("PRIMARY")].copy()
    sensitivity = locked[
        locked["analysis_role"].eq("SENSITIVITY")
    ].copy()
    rejected = decisions[
        decisions["analysis_role"].eq("REJECT")
    ].copy()

    pair_traits = expand_to_pairs(locked, trails)
    units = collapse_gene_trait_units(pair_traits)

    outputs = {
        "decisions": args.output_dir / "15a3_mapping_decisions.csv",
        "locked": args.output_dir / "15_trait_mapping_locked.csv",
        "primary": args.output_dir / "15_trait_mapping_primary.csv",
        "sensitivity": (
            args.output_dir / "15_trait_mapping_sensitivity.csv"
        ),
        "rejected": args.output_dir / "15_trait_mapping_rejected.csv",
        "pair_csv": (
            args.output_dir / "15_pair_trait_mapping_locked.csv"
        ),
        "pair_parquet": (
            args.output_dir / "15_pair_trait_mapping_locked.parquet"
        ),
        "unit_csv": (
            args.output_dir / "15_gene_trait_units_locked.csv"
        ),
        "unit_parquet": (
            args.output_dir / "15_gene_trait_units_locked.parquet"
        ),
    }

    decisions.to_csv(outputs["decisions"], index=False)
    locked.to_csv(outputs["locked"], index=False)
    primary.to_csv(outputs["primary"], index=False)
    sensitivity.to_csv(outputs["sensitivity"], index=False)
    rejected.to_csv(outputs["rejected"], index=False)
    pair_traits.to_csv(outputs["pair_csv"], index=False)
    pair_traits.to_parquet(outputs["pair_parquet"], index=False)
    units.to_csv(outputs["unit_csv"], index=False)
    units.to_parquet(outputs["unit_parquet"], index=False)

    primary_indications = primary["indication_mesh_id"].nunique()
    sensitivity_only_ids = set(
        sensitivity["indication_mesh_id"]
    ) - set(primary["indication_mesh_id"])

    primary_units = units[
        units["unit_analysis_role"].eq("PRIMARY")
    ]
    summary = {
        "step": "15A3",
        "status": "LOCKED",
        "n_candidate_rows": int(len(decisions)),
        "n_primary_mapping_rows": int(len(primary)),
        "n_sensitivity_mapping_rows": int(len(sensitivity)),
        "n_rejected_mapping_rows": int(len(rejected)),
        "n_indications_with_primary_mapping": int(primary_indications),
        "n_indications_with_sensitivity_only": int(
            len(sensitivity_only_ids)
        ),
        "sensitivity_only_indication_ids": sorted(
            map(str, sensitivity_only_ids)
        ),
        "n_pair_trait_rows_locked": int(len(pair_traits)),
        "n_target_indication_pairs_covered": int(
            pair_traits["ti_uid"].nunique()
        ),
        "n_primary_target_indication_pairs_covered": int(
            pair_traits.loc[
                pair_traits["analysis_role"].eq("PRIMARY"),
                "ti_uid",
            ].nunique()
        ),
        "n_genes_covered": int(pair_traits["gene"].nunique()),
        "n_gene_trait_units": int(len(units)),
        "n_primary_gene_trait_units": int(len(primary_units)),
        "n_sensitivity_only_gene_trait_units": int(
            units["unit_analysis_role"].eq("SENSITIVITY").sum()
        ),
        "n_mixed_pool_gene_trait_units": int(
            units["mixed_pool"].sum()
        ),
        "n_primary_mixed_pool_gene_trait_units": int(
            primary_units["mixed_pool"].sum()
        ),
        "n_primary_approval_eligible_gene_trait_units": int(
            units["primary_approval_eligible"].sum()
        ),
        "primary_units_by_source": {
            str(k): int(v)
            for k, v in primary_units[
                "candidate_source"
            ].value_counts().items()
        },
        "primary_units_by_mapping_tier": {
            str(k): int(v)
            for k, v in primary_units[
                "locked_mapping_tier"
            ].value_counts().items()
        },
        "pool_policy": {
            "primary": (
                "Exclude gene-trait-source units represented in both "
                "Pool A and Pool B."
            ),
            "sensitivity": (
                "Retain mixed units with assignment to Pool A if any "
                "Pool A mapping is present; otherwise Pool B."
            ),
        },
        "outputs": {k: str(v) for k, v in outputs.items()},
    }

    summary_path = (
        args.output_dir / "15a3_mapping_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 78)
    print("STEP 15A3 — LOCKED TRAIT MAPPINGS")
    print("=" * 78)
    print(f"Candidate rows adjudicated:                 {len(decisions):,}")
    print(f"Primary mapping rows:                       {len(primary):,}")
    print(f"Sensitivity mapping rows:                   {len(sensitivity):,}")
    print(f"Rejected mapping rows:                      {len(rejected):,}")
    print(f"Indications with a primary mapping:         {primary_indications:,}")
    print(
        f"Indications with sensitivity mappings only: "
        f"{len(sensitivity_only_ids):,}"
    )
    print(f"Target-indication pairs covered:            {summary['n_target_indication_pairs_covered']:,}")
    print(f"Primary target-indication pairs covered:    {summary['n_primary_target_indication_pairs_covered']:,}")
    print(f"Genes covered:                              {summary['n_genes_covered']:,}")
    print(f"Gene-trait-source units:                    {summary['n_gene_trait_units']:,}")
    print(f"Primary gene-trait-source units:            {summary['n_primary_gene_trait_units']:,}")
    print(f"Mixed-pool units:                           {summary['n_mixed_pool_gene_trait_units']:,}")
    print(f"Primary approval-eligible units:            {summary['n_primary_approval_eligible_gene_trait_units']:,}")
    print()
    print("Outputs:")
    for output in outputs.values():
        print(f"  {output}")
    print(f"  {summary_path}")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
