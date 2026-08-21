#!/usr/bin/env python3
"""
Step 15B1 — Build native-build coordinates and the extraction manifest.

Pan-UKB
    GENCODE v19
    GRCh37.p13
    one multi-ancestry BGZF file per phenotype

GBMI
    GENCODE v38
    GRCh38.p13
    separate BGZF files by endpoint and ancestry

The frozen regional definition is the GENCODE gene body ±100 kb.

This step does not extract variants and does not calculate portability or
colocalization.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y"]
POP_ORDER = ["AFR", "AMR", "CSA", "EAS", "MID"]

# Historical HGNC symbols used by older GENCODE releases.
# The left-hand side remains the study gene symbol; the right-hand side is
# used only to retrieve coordinates in the specified annotation.
GENE_SYMBOL_ALIASES = {
    "GRCh37.p13": {
        "ANGPTL8": "C19orf80",
    },
    "GRCh38.p13": {},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--units",
        type=Path,
        default=Path("output/15_gene_trait_units_locked.parquet"),
    )
    p.add_argument(
        "--panukb-catalog",
        type=Path,
        default=Path("output/15a_trait_catalog.csv"),
    )
    p.add_argument(
        "--gbmi-files",
        type=Path,
        default=Path("output/15a2_gbmi_verified_files.csv"),
    )
    p.add_argument(
        "--gencode37",
        type=Path,
        default=Path(
            "../step14/input/reference/gencode.v19.annotation.gtf.gz"
        ),
    )
    p.add_argument(
        "--gencode38",
        type=Path,
        default=Path(
            "input/reference/gencode.v38.annotation.gtf.gz"
        ),
    )
    p.add_argument("--window-bp", type=int, default=100_000)
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    return p.parse_args()


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return clean(value).casefold() in {"true", "1", "yes", "y"}


def strip_chr(value: Any) -> str:
    chrom = clean(value)
    if chrom.casefold().startswith("chr"):
        chrom = chrom[3:]
    return chrom


def parse_gtf_attributes(text: str) -> dict[str, str]:
    return {
        match.group(1): match.group(2)
        for match in re.finditer(
            r'(?:^|;\s*)([A-Za-z0-9_]+)\s+"([^"]*)"',
            text,
        )
    }


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def parse_gene_coordinates(
    path: Path,
    *,
    release: str,
    build: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    with open_text(path) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue

            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "gene":
                continue

            chrom = strip_chr(fields[0])
            if chrom not in CHROMS:
                continue

            attrs = parse_gtf_attributes(fields[8])
            gene = clean(attrs.get("gene_name"))
            if not gene:
                continue

            gene_id = clean(attrs.get("gene_id"))
            rows.append(
                {
                    "gene": gene,
                    "gene_id": gene_id,
                    "gene_type": clean(
                        attrs.get("gene_type")
                        or attrs.get("gene_biotype")
                    ),
                    "chrom": chrom,
                    "start_1based": int(fields[3]),
                    "end_1based": int(fields[4]),
                    "strand": fields[6],
                }
            )

    raw = pd.DataFrame(rows)
    if raw.empty:
        raise SystemExit(f"No canonical gene records parsed from {path}")

    collapsed: list[dict[str, Any]] = []
    for (gene, chrom), group in raw.groupby(
        ["gene", "chrom"],
        sort=False,
    ):
        collapsed.append(
            {
                "gene": gene,
                "chrom": chrom,
                "chrom_ucsc": f"chr{chrom}",
                "start_1based": int(group["start_1based"].min()),
                "end_1based": int(group["end_1based"].max()),
                "start_0based": int(group["start_1based"].min()) - 1,
                "strand": " || ".join(
                    sorted(set(group["strand"].astype(str)))
                ),
                "gene_ids": " || ".join(
                    sorted(set(group["gene_id"].astype(str)))
                ),
                "gene_types": " || ".join(
                    sorted(set(group["gene_type"].astype(str)))
                ),
                "n_gene_records_collapsed": int(len(group)),
                "gencode_release": release,
                "genome_build": build,
            }
        )

    coordinates = pd.DataFrame(collapsed)
    n_chrom = (
        coordinates.groupby("gene")["chrom"]
        .nunique()
        .rename("n_canonical_chromosomes")
    )
    coordinates = coordinates.merge(
        n_chrom,
        on="gene",
        how="left",
        validate="many_to_one",
    )
    coordinates["multichromosome_symbol"] = (
        coordinates["n_canonical_chromosomes"] > 1
    )
    return coordinates


def add_symbol_aliases(
    coordinates: pd.DataFrame,
    *,
    build: str,
) -> pd.DataFrame:
    result = coordinates.copy()
    result["coordinate_gene_symbol"] = result["gene"]
    result["coordinate_match_method"] = "exact_gene_name"

    alias_rows = []
    for current_symbol, historical_symbol in GENE_SYMBOL_ALIASES.get(
        build,
        {},
    ).items():
        source = result[result["gene"].eq(historical_symbol)].copy()
        if source.empty:
            continue

        source["gene"] = current_symbol
        source["coordinate_gene_symbol"] = historical_symbol
        source["coordinate_match_method"] = (
            "historical_symbol_alias"
        )
        alias_rows.append(source)

    if alias_rows:
        result = pd.concat(
            [result, *alias_rows],
            ignore_index=True,
            sort=False,
        )

    return result


def lock_genes(
    coordinates: pd.DataFrame,
    genes: list[str],
    *,
    window_bp: int,
) -> pd.DataFrame:
    requested = pd.DataFrame({"gene": sorted(set(genes))})
    ambiguous = set(
        coordinates.loc[
            coordinates["multichromosome_symbol"],
            "gene",
        ]
    )
    unique = coordinates[
        ~coordinates["multichromosome_symbol"]
    ].copy()

    if unique["gene"].duplicated().any():
        duplicate = unique.loc[
            unique["gene"].duplicated(False),
            ["gene", "chrom", "start_1based", "end_1based"],
        ]
        raise SystemExit(
            "Nonunique coordinate rows:\n"
            + duplicate.to_string(index=False)
        )

    result = requested.merge(
        unique,
        on="gene",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    result["coordinate_status"] = (
        result["_merge"]
        .map(
            {
                "both": "MATCHED",
                "left_only": "UNMATCHED_GENE_SYMBOL",
                "right_only": "UNEXPECTED",
            }
        )
        .astype("string")
    )
    result = result.drop(columns="_merge")
    result.loc[
        result["gene"].isin(ambiguous),
        "coordinate_status",
    ] = "AMBIGUOUS_MULTICHROMOSOME_SYMBOL"

    result["window_bp"] = window_bp
    valid = result["coordinate_status"].eq("MATCHED")
    result["window_start_1based"] = pd.NA
    result["window_end_1based"] = pd.NA
    result["window_start_0based"] = pd.NA

    result.loc[valid, "window_start_1based"] = (
        result.loc[valid, "start_1based"]
        .astype(int)
        .sub(window_bp)
        .clip(lower=1)
    )
    result.loc[valid, "window_end_1based"] = (
        result.loc[valid, "end_1based"].astype(int) + window_bp
    )
    result.loc[valid, "window_start_0based"] = (
        result.loc[valid, "window_start_1based"].astype(int) - 1
    )

    int_cols = [
        "start_1based",
        "end_1based",
        "start_0based",
        "window_start_1based",
        "window_end_1based",
        "window_start_0based",
        "window_bp",
        "n_gene_records_collapsed",
        "n_canonical_chromosomes",
    ]
    for col in int_cols:
        result[col] = pd.to_numeric(
            result[col],
            errors="coerce",
        ).astype("Int64")

    return result


def split_pops(value: Any) -> list[str]:
    tokens = re.split(
        r"\s*\|\|\s*|\s*,\s*",
        clean(value),
    )
    found = {
        token.upper()
        for token in tokens
        if token.upper() in POP_ORDER
    }
    return [pop for pop in POP_ORDER if pop in found]


def make_uid(prefix: str, *values: Any) -> str:
    raw = "||".join(clean(v) for v in values)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def coordinate_dict(table: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {
        clean(row["gene"]): row.to_dict()
        for _, row in table.iterrows()
    }


def common_fields(
    unit: pd.Series,
    coord: dict[str, Any] | None,
    population: str,
) -> dict[str, Any]:
    c = coord or {}
    return {
        "comparison_uid": make_uid(
            "cmp",
            unit["gene_trait_uid"],
            population,
        ),
        "gene_trait_uid": unit["gene_trait_uid"],
        "gene": unit["gene"],
        "candidate_source": unit["candidate_source"],
        "candidate_trait_key": unit["candidate_trait_key"],
        "candidate_trait_name": unit["candidate_trait_name"],
        "phenotype_id": unit["phenotype_id"],
        "genome_build": unit["genome_build"],
        "canonical_group": unit["canonical_group"],
        "unit_analysis_role": unit["unit_analysis_role"],
        "locked_mapping_tier": unit["locked_mapping_tier"],
        "mapping_tiers_contributing": unit.get(
            "mapping_tiers_contributing",
            "",
        ),
        "mixed_pool": as_bool(unit["mixed_pool"]),
        "primary_approval_eligible": as_bool(
            unit["primary_approval_eligible"]
        ),
        "sensitivity_pool_assignment": unit[
            "sensitivity_pool_assignment"
        ],
        "pools": unit["pools"],
        "n_A_pairs": unit["n_A_pairs"],
        "n_B_pairs": unit["n_B_pairs"],
        "n_target_indication_pairs": unit[
            "n_target_indication_pairs"
        ],
        "n_indications": unit["n_indications"],
        "indications": unit["indications"],
        "eur_population": "EUR",
        "comparison_population": population,
        "chrom": c.get("chrom", pd.NA),
        "chrom_ucsc": c.get("chrom_ucsc", pd.NA),
        "gene_start_1based": c.get("start_1based", pd.NA),
        "gene_end_1based": c.get("end_1based", pd.NA),
        "gene_start_0based": c.get("start_0based", pd.NA),
        "window_start_1based": c.get(
            "window_start_1based",
            pd.NA,
        ),
        "window_end_1based": c.get(
            "window_end_1based",
            pd.NA,
        ),
        "window_start_0based": c.get(
            "window_start_0based",
            pd.NA,
        ),
        "window_bp": c.get("window_bp", pd.NA),
        "strand": c.get("strand", ""),
        "gene_ids": c.get("gene_ids", ""),
        "gene_types": c.get("gene_types", ""),
        "coordinate_gene_symbol": c.get(
            "coordinate_gene_symbol",
            "",
        ),
        "coordinate_match_method": c.get(
            "coordinate_match_method",
            "",
        ),
        "gencode_release": c.get("gencode_release", ""),
        "coordinate_status": c.get(
            "coordinate_status",
            "UNRESOLVED",
        ),
    }


def build_panukb_rows(
    units: pd.DataFrame,
    catalog: pd.DataFrame,
    coords: pd.DataFrame,
) -> list[dict[str, Any]]:
    catalog = catalog[catalog["source"].eq("Pan-UKB")].copy()
    if catalog["trait_key"].duplicated().any():
        raise SystemExit("Pan-UKB trait_key is not unique.")

    traits = catalog.set_index("trait_key").to_dict("index")
    coordinate_lookup = coordinate_dict(coords)
    rows: list[dict[str, Any]] = []

    for _, unit in units[
        units["candidate_source"].eq("Pan-UKB")
    ].iterrows():
        trait = traits.get(clean(unit["candidate_trait_key"]))
        pops = (
            split_pops(trait["non_eur_pops_pass_qc"])
            if trait is not None
            else split_pops(unit["non_eur_pops_available"])
        )

        for pop in pops or ["UNRESOLVED"]:
            row = common_fields(
                unit,
                coordinate_lookup.get(clean(unit["gene"])),
                pop,
            )
            data_url = (
                clean(trait["remote_data_path"])
                if trait is not None
                else ""
            )
            index_url = (
                clean(trait["remote_index_path"])
                if trait is not None
                else ""
            )
            row.update(
                {
                    "summary_stat_layout": (
                        "single_multianalysis_bgzf"
                    ),
                    "eur_data_url": data_url,
                    "eur_index_url": index_url,
                    "comparison_data_url": data_url,
                    "comparison_index_url": index_url,
                    "source_population_label": pop,
                    "source_schema_status": "TO_VALIDATE_IN_15B2",
                    "source_metadata_status": (
                        "MATCHED_TRAIT_CATALOG"
                        if trait is not None
                        else "MISSING_TRAIT_CATALOG_ROW"
                    ),
                    "source_file_index_verified": bool(
                        data_url and index_url
                    ),
                }
            )
            rows.append(row)

    return rows


def build_gbmi_rows(
    units: pd.DataFrame,
    files: pd.DataFrame,
    coords: pd.DataFrame,
) -> list[dict[str, Any]]:
    files = files[files["data_exists"].map(as_bool)].copy()
    files["harmonized_population"] = (
        files["harmonized_population"]
        .astype(str)
        .str.upper()
    )

    duplicate = files.duplicated(
        ["endpoint_code", "harmonized_population"],
        keep=False,
    )
    if duplicate.any():
        raise SystemExit(
            "GBMI has multiple files for the same endpoint/population:\n"
            + files.loc[
                duplicate,
                [
                    "endpoint_code",
                    "harmonized_population",
                    "data_url",
                ],
            ].to_string(index=False)
        )

    lookup = {
        (
            clean(row["endpoint_code"]),
            clean(row["harmonized_population"]),
        ): row.to_dict()
        for _, row in files.iterrows()
    }
    coordinate_lookup = coordinate_dict(coords)
    rows: list[dict[str, Any]] = []

    for _, unit in units[
        units["candidate_source"].eq("GBMI")
    ].iterrows():
        endpoint = clean(unit["phenotype_id"])
        available = [
            pop
            for pop in POP_ORDER
            if (endpoint, pop) in lookup
        ]
        eur = lookup.get((endpoint, "EUR"))

        for pop in available or ["UNRESOLVED"]:
            non_eur = lookup.get((endpoint, pop))
            row = common_fields(
                unit,
                coordinate_lookup.get(clean(unit["gene"])),
                pop,
            )
            row.update(
                {
                    "summary_stat_layout": (
                        "separate_ancestry_bgzf"
                    ),
                    "eur_data_url": (
                        clean(eur["data_url"]) if eur else ""
                    ),
                    "eur_index_url": (
                        clean(eur["index_url"]) if eur else ""
                    ),
                    "comparison_data_url": (
                        clean(non_eur["data_url"])
                        if non_eur
                        else ""
                    ),
                    "comparison_index_url": (
                        clean(non_eur["index_url"])
                        if non_eur
                        else ""
                    ),
                    "source_population_label": (
                        clean(
                            non_eur.get(
                                "source_population",
                                pop,
                            )
                        )
                        if non_eur
                        else pop
                    ),
                    "source_schema_status": (
                        "PASS"
                        if eur
                        and non_eur
                        and clean(eur["schema_status"]) == "PASS"
                        and clean(non_eur["schema_status"]) == "PASS"
                        else "REVIEW_OR_MISSING"
                    ),
                    "source_metadata_status": (
                        "MATCHED_EUR_AND_NON_EUR_FILES"
                        if eur and non_eur
                        else "MISSING_EUR_OR_NON_EUR_FILE"
                    ),
                    "source_file_index_verified": bool(
                        eur
                        and non_eur
                        and as_bool(eur["index_exists"])
                        and as_bool(non_eur["index_exists"])
                    ),
                }
            )
            rows.append(row)

    return rows


def add_readiness(manifest: pd.DataFrame) -> pd.DataFrame:
    result = manifest.copy()
    result["coordinate_ready"] = result[
        "coordinate_status"
    ].eq("MATCHED")
    result["eur_source_ready"] = (
        result["eur_data_url"].fillna("").ne("")
        & result["eur_index_url"].fillna("").ne("")
    )
    result["comparison_source_ready"] = (
        result["comparison_population"].ne("UNRESOLVED")
        & result["comparison_data_url"].fillna("").ne("")
        & result["comparison_index_url"].fillna("").ne("")
    )
    result["extraction_ready"] = (
        result["coordinate_ready"]
        & result["eur_source_ready"]
        & result["comparison_source_ready"]
        & result["source_file_index_verified"].map(as_bool)
    )

    def reason(row: pd.Series) -> str:
        failures = []
        if not row["coordinate_ready"]:
            failures.append(
                f"coordinate:{clean(row['coordinate_status'])}"
            )
        if not row["eur_source_ready"]:
            failures.append("missing_EUR_data_or_index")
        if not row["comparison_source_ready"]:
            failures.append("missing_nonEUR_data_or_index")
        if not as_bool(row["source_file_index_verified"]):
            failures.append("source_index_not_verified")
        return " | ".join(failures)

    result["unresolved_reason"] = result.apply(reason, axis=1)

    for col in [
        "gene_start_1based",
        "gene_end_1based",
        "gene_start_0based",
        "window_start_1based",
        "window_end_1based",
        "window_start_0based",
        "window_bp",
        "n_A_pairs",
        "n_B_pairs",
        "n_target_indication_pairs",
        "n_indications",
    ]:
        result[col] = pd.to_numeric(
            result[col],
            errors="coerce",
        ).astype("Int64")

    return result


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.window_bp < 0:
        raise SystemExit("--window-bp must be nonnegative.")

    required_paths = [
        args.units,
        args.panukb_catalog,
        args.gbmi_files,
        args.gencode37,
        args.gencode38,
    ]
    for path in required_paths:
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")

    units = pd.read_parquet(args.units)
    required_unit_columns = {
        "gene_trait_uid",
        "gene",
        "candidate_source",
        "candidate_trait_key",
        "candidate_trait_name",
        "phenotype_id",
        "genome_build",
        "canonical_group",
        "unit_analysis_role",
        "locked_mapping_tier",
        "mixed_pool",
        "primary_approval_eligible",
        "sensitivity_pool_assignment",
        "pools",
        "n_A_pairs",
        "n_B_pairs",
        "n_target_indication_pairs",
        "n_indications",
        "indications",
        "non_eur_pops_available",
    }
    missing = sorted(required_unit_columns - set(units.columns))
    if missing:
        raise SystemExit(
            f"Locked units missing columns: {missing}"
        )

    source_build_errors = units[
        (
            units["candidate_source"].eq("Pan-UKB")
            & ~units["genome_build"].eq("GRCh37")
        )
        | (
            units["candidate_source"].eq("GBMI")
            & ~units["genome_build"].eq("GRCh38")
        )
    ]
    if len(source_build_errors):
        raise SystemExit(
            "Source/build mismatch:\n"
            + source_build_errors[
                [
                    "gene_trait_uid",
                    "candidate_source",
                    "genome_build",
                ]
            ].to_string(index=False)
        )

    genes = sorted(set(units["gene"].dropna().astype(str)))
    panukb_genes = sorted(
        set(
            units.loc[
                units["candidate_source"].eq("Pan-UKB"),
                "gene",
            ].astype(str)
        )
    )
    gbmi_genes = sorted(
        set(
            units.loc[
                units["candidate_source"].eq("GBMI"),
                "gene",
            ].astype(str)
        )
    )

    print(f"Parsing GENCODE v19: {args.gencode37}")
    all37 = parse_gene_coordinates(
        args.gencode37,
        release="GENCODE v19",
        build="GRCh37.p13",
    )
    all37 = add_symbol_aliases(
        all37,
        build="GRCh37.p13",
    )
    coord37 = lock_genes(
        all37,
        genes,
        window_bp=args.window_bp,
    )

    print(f"Parsing GENCODE v38: {args.gencode38}")
    all38 = parse_gene_coordinates(
        args.gencode38,
        release="GENCODE v38",
        build="GRCh38.p13",
    )
    all38 = add_symbol_aliases(
        all38,
        build="GRCh38.p13",
    )
    coord38 = lock_genes(
        all38,
        genes,
        window_bp=args.window_bp,
    )

    panukb_catalog = pd.read_csv(
        args.panukb_catalog,
        low_memory=False,
    )
    gbmi_files = pd.read_csv(
        args.gbmi_files,
        low_memory=False,
    )

    rows = build_panukb_rows(
        units,
        panukb_catalog,
        coord37,
    )
    rows.extend(
        build_gbmi_rows(
            units,
            gbmi_files,
            coord38,
        )
    )

    manifest = add_readiness(pd.DataFrame(rows))
    if manifest["comparison_uid"].duplicated().any():
        raise SystemExit("comparison_uid is not unique.")

    primary = manifest[
        manifest["unit_analysis_role"].eq("PRIMARY")
    ].copy()
    unresolved = manifest[
        ~manifest["extraction_ready"]
    ].copy()

    coord_audit = pd.concat(
        [
            coord37.assign(
                required_for_source=coord37["gene"].isin(
                    panukb_genes
                ),
                source_requiring_build="Pan-UKB",
            ),
            coord38.assign(
                required_for_source=coord38["gene"].isin(
                    gbmi_genes
                ),
                source_requiring_build="GBMI",
            ),
        ],
        ignore_index=True,
        sort=False,
    )

    outputs = {
        "coord37_csv": args.output_dir
        / "15b1_gene_coordinates_grch37.csv",
        "coord37_parquet": args.output_dir
        / "15b1_gene_coordinates_grch37.parquet",
        "coord38_csv": args.output_dir
        / "15b1_gene_coordinates_grch38.csv",
        "coord38_parquet": args.output_dir
        / "15b1_gene_coordinates_grch38.parquet",
        "coordinate_audit": args.output_dir
        / "15b1_gene_coordinate_audit.csv",
        "manifest_csv": args.output_dir
        / "15b1_extraction_manifest.csv",
        "manifest_parquet": args.output_dir
        / "15b1_extraction_manifest.parquet",
        "primary_csv": args.output_dir
        / "15b1_extraction_manifest_primary.csv",
        "primary_parquet": args.output_dir
        / "15b1_extraction_manifest_primary.parquet",
        "unresolved": args.output_dir
        / "15b1_extraction_manifest_unresolved.csv",
    }

    coord37.to_csv(outputs["coord37_csv"], index=False)
    coord37.to_parquet(
        outputs["coord37_parquet"],
        index=False,
    )
    coord38.to_csv(outputs["coord38_csv"], index=False)
    coord38.to_parquet(
        outputs["coord38_parquet"],
        index=False,
    )
    coord_audit.to_csv(
        outputs["coordinate_audit"],
        index=False,
    )
    manifest.to_csv(outputs["manifest_csv"], index=False)
    manifest.to_parquet(
        outputs["manifest_parquet"],
        index=False,
    )
    primary.to_csv(outputs["primary_csv"], index=False)
    primary.to_parquet(
        outputs["primary_parquet"],
        index=False,
    )
    unresolved.to_csv(outputs["unresolved"], index=False)

    required37 = coord37[
        coord37["gene"].isin(panukb_genes)
    ]
    required38 = coord38[
        coord38["gene"].isin(gbmi_genes)
    ]

    summary = {
        "step": "15B1",
        "window_definition": "GENCODE gene body ±100 kb",
        "window_bp": args.window_bp,
        "gene_symbol_aliases": GENE_SYMBOL_ALIASES,
        "coordinate_sources": {
            "Pan-UKB": {
                "annotation": "GENCODE v19",
                "assembly": "GRCh37.p13",
                "native_summary_stat_build": "GRCh37",
            },
            "GBMI": {
                "annotation": "GENCODE v38",
                "assembly": "GRCh38.p13",
                "native_summary_stat_build": "GRCh38",
            },
        },
        "n_locked_gene_trait_units": int(len(units)),
        "n_unique_genes": int(len(genes)),
        "n_panukb_units": int(
            units["candidate_source"].eq("Pan-UKB").sum()
        ),
        "n_gbmi_units": int(
            units["candidate_source"].eq("GBMI").sum()
        ),
        "coordinates": {
            "panukb_required_genes": int(len(panukb_genes)),
            "panukb_matched_genes": int(
                required37["coordinate_status"]
                .eq("MATCHED")
                .sum()
            ),
            "panukb_unresolved_genes": int(
                (~required37["coordinate_status"].eq("MATCHED"))
                .sum()
            ),
            "gbmi_required_genes": int(len(gbmi_genes)),
            "gbmi_matched_genes": int(
                required38["coordinate_status"]
                .eq("MATCHED")
                .sum()
            ),
            "gbmi_unresolved_genes": int(
                (~required38["coordinate_status"].eq("MATCHED"))
                .sum()
            ),
        },
        "comparisons": {
            "all_rows": int(len(manifest)),
            "primary_rows": int(len(primary)),
            "all_extraction_ready": int(
                manifest["extraction_ready"].sum()
            ),
            "primary_extraction_ready": int(
                primary["extraction_ready"].sum()
            ),
            "unresolved_rows": int(len(unresolved)),
            "by_source": {
                str(k): int(v)
                for k, v in manifest[
                    "candidate_source"
                ].value_counts().items()
            },
            "by_non_eur_population": {
                str(k): int(v)
                for k, v in manifest[
                    "comparison_population"
                ].value_counts().items()
            },
        },
        "outputs": {
            key: str(value)
            for key, value in outputs.items()
        },
    }

    summary_path = (
        args.output_dir
        / "15b1_extraction_manifest_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 78)
    print("STEP 15B1 — DUAL-BUILD EXTRACTION MANIFEST")
    print("=" * 78)
    print(f"Locked gene-trait units:              {len(units):,}")
    print(f"Unique genes:                         {len(genes):,}")
    print(
        "Pan-UKB required genes matched:       "
        f"{summary['coordinates']['panukb_matched_genes']:,}/"
        f"{summary['coordinates']['panukb_required_genes']:,}"
    )
    print(
        "GBMI required genes matched:          "
        f"{summary['coordinates']['gbmi_matched_genes']:,}/"
        f"{summary['coordinates']['gbmi_required_genes']:,}"
    )
    print(f"EUR-vs-non-EUR rows:                  {len(manifest):,}")
    print(
        "Extraction-ready rows:                "
        f"{manifest['extraction_ready'].sum():,}"
    )
    print(f"Primary rows:                         {len(primary):,}")
    print(
        "Primary extraction-ready rows:        "
        f"{primary['extraction_ready'].sum():,}"
    )
    print(f"Unresolved rows:                      {len(unresolved):,}")
    print()
    print("No variants were extracted in Step 15B1.")
    print("Outputs:")
    for output in outputs.values():
        print(f"  {output}")
    print(f"  {summary_path}")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
