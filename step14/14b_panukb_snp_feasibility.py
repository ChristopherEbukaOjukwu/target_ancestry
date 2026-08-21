#!/usr/bin/env python3
"""
Step 14B: Pan-UKB remote-schema validation and SNP-count feasibility gate.

Purpose
-------
Before fitting any cross-ancestry portability model, determine whether the
locked target-indication units contain enough usable EUR-associated variants.

This step:
1. uses the locked Step 13 phenotype mappings;
2. uses the validated GENCODE v19 GRCh37 coordinates from Step 14A;
3. queries Pan-UKB BGZF files remotely using their tabix indexes;
4. validates the current file schema;
5. counts usable variants for each target-indication-comparison-ancestry unit;
6. does not estimate portability.

Precommitted analysis rules
---------------------------
Primary interval:
    GENCODE v19 gene body.

Prespecified sensitivity interval:
    Gene body plus 10 kb on both sides.

Variant-level filters:
    - biallelic A/C/G/T SNP;
    - finite beta and SE in EUR and comparison ancestry;
    - SE > 0 in both ancestries;
    - low_confidence is false in both ancestries;
    - pooled MAF >= 0.01 in both ancestries;
    - EUR genome-wide significance:
          neglog10_pval_EUR >= -log10(5e-8) = 7.30102999566.

Raw-count feasibility gate:
    >= 10 usable EUR-significant SNPs.

This is a necessary, not sufficient, gate. The final portability estimator
will require >= 10 LD-independent variants after a later, prespecified LD
pruning/clumping step. A unit with fewer than 10 raw variants cannot satisfy
that final requirement.

MHC handling:
    Windows overlapping chr6:25,000,000-34,000,000 (GRCh37) are flagged and
    excluded from primary summaries, but their counts are retained.

Inputs
------
../step13/output/13_panukb_direct_mapping_locked.csv
../step13/output/13_panukb_direct_pairs_locked.parquet
output/14_locked_gene_coordinates_grch37.parquet

Outputs
-------
output/14b_panukb_schema_validation.json
output/14b_panukb_snp_count_feasibility.csv
output/14b_panukb_snp_count_summary.csv
output/14b_panukb_snp_count_summary.json
output/14b_panukb_windows.csv
input/panukb_tabix_indexes/*.tbi

Dependencies
------------
pip install pysam pandas pyarrow
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    import pysam
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'pysam'. Install it in the active environment:\n"
        "  python3 -m pip install pysam"
    ) from exc


POPS = ["AFR", "AMR", "CSA", "EAS", "EUR", "MID"]
EUR_GWS_P = 5e-8
EUR_GWS_NEGLOG10 = -math.log10(EUR_GWS_P)
MAF_MIN = 0.01
RAW_MIN_SNPS = 10
MHC_CHROM = "6"
MHC_START_0BASED = 25_000_000
MHC_END_0BASED = 34_000_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mapping",
        type=Path,
        default=Path("../step13/output/13_panukb_direct_mapping_locked.csv"),
    )
    p.add_argument(
        "--pairs",
        type=Path,
        default=Path("../step13/output/13_panukb_direct_pairs_locked.parquet"),
    )
    p.add_argument(
        "--coordinates",
        type=Path,
        default=Path("output/14_locked_gene_coordinates_grch37.parquet"),
    )
    p.add_argument(
        "--index-dir",
        type=Path,
        default=Path("input/panukb_tabix_indexes"),
    )
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument(
        "--mode",
        choices=["validate", "count"],
        default="validate",
        help=(
            "'validate' tests one phenotype-window and writes the actual schema. "
            "'count' runs the complete frozen feasibility gate."
        ),
    )
    p.add_argument(
        "--window-set",
        choices=["gene_body", "both"],
        default="both",
        help="Use the primary gene body only, or also count the +10 kb sensitivity.",
    )
    p.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="Timeout for downloading each tabix index.",
    )
    return p.parse_args()


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def split_pops(value: Any) -> list[str]:
    return [x.strip() for x in clean(value).split(",") if x.strip()]


def s3_to_https(value: str) -> str:
    value = clean(value)
    if value.startswith("https://") or value.startswith("http://"):
        return value
    if not value.startswith("s3://"):
        raise ValueError(f"Unsupported remote path: {value}")
    body = value[len("s3://"):]
    bucket, key = body.split("/", 1)
    return f"https://{bucket}.s3.amazonaws.com/{key}"


def download_file(url: str, destination: Path, timeout: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return

    temp = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "target-ancestry-step14b/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        with temp.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    temp.replace(destination)


def parse_bool(value: Any) -> bool | None:
    text = clean(value).casefold()
    if text in {"true", "t", "1", "yes"}:
        return True
    if text in {"false", "f", "0", "no"}:
        return False
    return None


def parse_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def maf_from_af(af: float) -> float:
    if not math.isfinite(af) or af < 0 or af > 1:
        return math.nan
    return min(af, 1.0 - af)


def pooled_binary_af(
    record: dict[str, str],
    pop: str,
    mapping_row: pd.Series,
) -> float:
    af_cases = parse_float(record.get(f"af_cases_{pop}"))
    af_controls = parse_float(record.get(f"af_controls_{pop}"))
    n_cases = parse_float(mapping_row.get(f"n_cases_{pop}"))
    n_controls = parse_float(mapping_row.get(f"n_controls_{pop}"))

    numerator = 0.0
    denominator = 0.0

    if math.isfinite(af_cases) and math.isfinite(n_cases) and n_cases > 0:
        numerator += af_cases * n_cases
        denominator += n_cases

    if (
        math.isfinite(af_controls)
        and math.isfinite(n_controls)
        and n_controls > 0
    ):
        numerator += af_controls * n_controls
        denominator += n_controls

    if denominator == 0:
        return math.nan
    return numerator / denominator


def is_biallelic_snp(record: dict[str, str]) -> bool:
    ref = clean(record.get("ref")).upper()
    alt = clean(record.get("alt")).upper()
    return (
        len(ref) == 1
        and len(alt) == 1
        and ref in {"A", "C", "G", "T"}
        and alt in {"A", "C", "G", "T"}
        and ref != alt
    )


def complete_effect(record: dict[str, str], pop: str) -> bool:
    beta = parse_float(record.get(f"beta_{pop}"))
    se = parse_float(record.get(f"se_{pop}"))
    p = parse_float(record.get(f"neglog10_pval_{pop}"))
    return (
        math.isfinite(beta)
        and math.isfinite(se)
        and se > 0
        and math.isfinite(p)
        and p >= 0
    )


def not_low_confidence(record: dict[str, str], pop: str) -> bool:
    value = parse_bool(record.get(f"low_confidence_{pop}"))
    return value is False


def window_overlaps_mhc(chrom: str, start0: int, end0: int) -> bool:
    return (
        str(chrom) == MHC_CHROM
        and start0 < MHC_END_0BASED
        and end0 > MHC_START_0BASED
    )


def build_windows(coords: pd.DataFrame, window_set: str) -> pd.DataFrame:
    rows = []
    for _, row in coords.iterrows():
        gene_start0 = int(row["start_1based"]) - 1
        gene_end0 = int(row["end_1based"])
        gene_length = gene_end0 - gene_start0

        definitions = [("gene_body", gene_start0, gene_end0, 0)]
        if window_set == "both":
            definitions.append(
                (
                    "gene_body_plus_10kb",
                    max(0, gene_start0 - 10_000),
                    gene_end0 + 10_000,
                    10_000,
                )
            )

        for name, start0, end0, flank in definitions:
            window_length = end0 - start0
            rows.append(
                {
                    "gene": row["gene"],
                    "chrom": str(row["chrom"]),
                    "gene_start_1based": int(row["start_1based"]),
                    "gene_end_1based": int(row["end_1based"]),
                    "gene_length_bp": gene_length,
                    "window_type": name,
                    "flank_each_side_bp": flank,
                    "query_start_0based": start0,
                    "query_end_0based_exclusive": end0,
                    "window_length_bp": window_length,
                    "gene_fraction_of_window": gene_length / window_length,
                    "overlaps_extended_mhc": window_overlaps_mhc(
                        str(row["chrom"]), start0, end0
                    ),
                }
            )
    return pd.DataFrame(rows)


class RemotePhenotype:
    def __init__(
        self,
        mapping_row: pd.Series,
        index_dir: Path,
        timeout: int,
    ):
        self.mapping_row = mapping_row
        self.phenotype_id = clean(mapping_row["panukb_phenotype_id"])
        self.data_url = s3_to_https(clean(mapping_row["aws_path"]))
        self.index_url = s3_to_https(clean(mapping_row["aws_path_tabix"]))
        self.local_index = index_dir / clean(mapping_row["filename_tabix"])

        download_file(self.index_url, self.local_index, timeout)

        try:
            self.tbx = pysam.TabixFile(
                self.data_url,
                index=str(self.local_index),
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not open the remote BGZF file with its local tabix index.\n"
                f"Phenotype: {self.phenotype_id}\n"
                f"Data URL: {self.data_url}\n"
                f"Index: {self.local_index}\n"
                "The active pysam/htslib build may lack remote HTTP support."
            ) from exc

        self.columns = self._extract_columns()
        self.contigs = set(self.tbx.contigs)

    def _extract_columns(self) -> list[str]:
        # Some Pan-UKB flat files have an ordinary first-line column header
        # that was not stored as tabix metadata. In that case, tbx.header is
        # empty even though the remote file and index are valid.
        header_lines = list(self.tbx.header)
        if header_lines:
            header = header_lines[-1].lstrip("#").rstrip("\r\n")
        else:
            request = urllib.request.Request(
                self.data_url,
                headers={"User-Agent": "target-ancestry-step14b/1.1"},
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    # BGZF is gzip-compatible for sequential reading. Only the
                    # first decompressed line is needed to recover the columns.
                    with gzip.GzipFile(fileobj=response, mode="rb") as handle:
                        first_line = handle.readline()
            except Exception as exc:
                raise RuntimeError(
                    f"Tabix metadata contained no header and the first BGZF "
                    f"line could not be read for {self.phenotype_id}."
                ) from exc

            if not first_line:
                raise RuntimeError(
                    f"Remote Pan-UKB file is empty for {self.phenotype_id}."
                )
            try:
                header = first_line.decode("utf-8-sig").lstrip("#").rstrip("\r\n")
            except UnicodeDecodeError as exc:
                raise RuntimeError(
                    f"Could not decode the first BGZF line for "
                    f"{self.phenotype_id} as UTF-8."
                ) from exc

        columns = header.split("\t")
        if not {"chr", "pos", "ref", "alt"}.issubset(columns):
            raise RuntimeError(
                f"Unexpected first-line/header schema for "
                f"{self.phenotype_id}:\n{header}"
            )
        return columns

    def validate_schema(self, qc_pops: Iterable[str]) -> None:
        required = {"chr", "pos", "ref", "alt"}
        for pop in sorted(set(qc_pops) | {"EUR"}):
            required.update(
                {
                    f"beta_{pop}",
                    f"se_{pop}",
                    f"neglog10_pval_{pop}",
                    f"low_confidence_{pop}",
                    f"af_cases_{pop}",
                    f"af_controls_{pop}",
                }
            )

        missing = sorted(required - set(self.columns))
        if missing:
            legacy_p = [
                x for x in self.columns
                if x.startswith("pval_") or x.startswith("log_pval_")
            ]
            raise RuntimeError(
                f"Schema validation failed for {self.phenotype_id}.\n"
                f"Missing expected current-schema columns: {missing}\n"
                f"Potential legacy p-value columns present: {legacy_p}\n"
                "No p-value transformation will be guessed."
            )

    def reference_name(self, chrom: str) -> str:
        chrom = str(chrom)
        if chrom in self.contigs:
            return chrom
        prefixed = f"chr{chrom}"
        if prefixed in self.contigs:
            return prefixed
        raise RuntimeError(
            f"Chromosome {chrom} is absent from tabix contigs for "
            f"{self.phenotype_id}. First contigs: {sorted(self.contigs)[:10]}"
        )

    def fetch(
        self,
        chrom: str,
        start0: int,
        end0: int,
    ) -> list[dict[str, str]]:
        reference = self.reference_name(chrom)
        rows = []
        for line in self.tbx.fetch(reference, start0, end0):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != len(self.columns):
                raise RuntimeError(
                    f"Column-count mismatch for {self.phenotype_id}: "
                    f"{len(fields)} fields versus {len(self.columns)} columns."
                )
            rows.append(dict(zip(self.columns, fields)))
        return rows

    def close(self) -> None:
        self.tbx.close()


def phenotype_metadata(mapping: pd.DataFrame) -> dict[str, pd.Series]:
    if mapping["panukb_phenotype_id"].duplicated().any():
        duplicates = mapping.loc[
            mapping["panukb_phenotype_id"].duplicated(False),
            ["indication_mesh_id", "panukb_phenotype_id"],
        ]
        raise ValueError(
            "Locked phenotype mapping is not one row per phenotype:\n"
            + duplicates.to_string(index=False)
        )
    return {
        clean(row["panukb_phenotype_id"]): row
        for _, row in mapping.iterrows()
    }


def choose_validation_unit(
    pairs: pd.DataFrame,
    windows: pd.DataFrame,
) -> pd.Series:
    units = pairs[
        [
            "ti_uid",
            "gene",
            "indication_mesh_id",
            "indication_mesh_term",
            "panukb_phenotype_id",
            "pops_pass_qc",
        ]
    ].drop_duplicates()

    candidates = units.merge(
        windows[windows["window_type"] == "gene_body"],
        on="gene",
        how="inner",
        validate="many_to_one",
    )
    candidates = candidates[
        candidates["pops_pass_qc"].map(
            lambda x: len([p for p in split_pops(x) if p != "EUR"]) > 0
        )
    ].copy()

    if candidates.empty:
        raise RuntimeError("No locked unit has EUR plus a non-EUR QC-pass ancestry.")

    # Largest gene body maximizes the chance of receiving records during
    # remote-access validation without selecting on association strength.
    candidates = candidates.sort_values(
        ["gene_length_bp", "gene", "indication_mesh_id"],
        ascending=[False, True, True],
    )
    return candidates.iloc[0]


def schema_validation_payload(
    remote: RemotePhenotype,
    unit: pd.Series,
    records: list[dict[str, str]],
) -> dict[str, Any]:
    pops = split_pops(unit["pops_pass_qc"])
    pval_summary = {}
    for pop in pops:
        values = [
            parse_float(r.get(f"neglog10_pval_{pop}"))
            for r in records[:1000]
        ]
        values = [v for v in values if math.isfinite(v)]
        pval_summary[pop] = {
            "n_sampled": len(values),
            "minimum": min(values) if values else None,
            "median": statistics.median(values) if values else None,
            "maximum": max(values) if values else None,
        }

    return {
        "validation_unit": {
            "ti_uid": clean(unit["ti_uid"]),
            "gene": clean(unit["gene"]),
            "indication_mesh_id": clean(unit["indication_mesh_id"]),
            "indication_mesh_term": clean(unit["indication_mesh_term"]),
            "panukb_phenotype_id": clean(unit["panukb_phenotype_id"]),
            "pops_pass_qc": pops,
            "chrom": clean(unit["chrom"]),
            "query_start_0based": int(unit["query_start_0based"]),
            "query_end_0based_exclusive": int(
                unit["query_end_0based_exclusive"]
            ),
        },
        "remote_data_url": remote.data_url,
        "local_tabix_index": str(remote.local_index),
        "tabix_contigs": list(remote.tbx.contigs),
        "columns": remote.columns,
        "n_records_in_validation_window": len(records),
        "pvalue_encoding_required": "neglog10_pval_{pop}",
        "genome_build": "GRCh37",
        "sampled_neglog10_pvalue_ranges": pval_summary,
        "status": "PASS",
    }


def count_records(
    records: list[dict[str, str]],
    pop: str,
    mapping_row: pd.Series,
) -> dict[str, int]:
    counts = defaultdict(int)
    seen_variants = set()

    for record in records:
        counts["n_records_raw"] += 1

        variant = (
            clean(record.get("chr")),
            clean(record.get("pos")),
            clean(record.get("ref")).upper(),
            clean(record.get("alt")).upper(),
        )
        if variant in seen_variants:
            counts["n_duplicate_variant_rows"] += 1
            continue
        seen_variants.add(variant)

        if not is_biallelic_snp(record):
            continue
        counts["n_biallelic_snps"] += 1

        if not complete_effect(record, "EUR"):
            continue
        counts["n_eur_effect_complete"] += 1

        if not complete_effect(record, pop):
            continue
        counts["n_cross_effect_complete"] += 1

        if not (
            not_low_confidence(record, "EUR")
            and not_low_confidence(record, pop)
        ):
            continue
        counts["n_cross_not_low_confidence"] += 1

        af_eur = pooled_binary_af(record, "EUR", mapping_row)
        af_pop = pooled_binary_af(record, pop, mapping_row)
        maf_eur = maf_from_af(af_eur)
        maf_pop = maf_from_af(af_pop)

        if not (
            math.isfinite(maf_eur)
            and math.isfinite(maf_pop)
            and maf_eur >= MAF_MIN
            and maf_pop >= MAF_MIN
        ):
            continue
        counts["n_cross_maf_ge_0_01"] += 1

        eur_neglog10 = parse_float(record.get("neglog10_pval_EUR"))
        if eur_neglog10 >= 5.0:
            counts["n_eur_p_le_1e_5_cross_usable"] += 1
        if eur_neglog10 >= EUR_GWS_NEGLOG10:
            counts["n_eur_gws_cross_usable"] += 1

    for key in [
        "n_records_raw",
        "n_duplicate_variant_rows",
        "n_biallelic_snps",
        "n_eur_effect_complete",
        "n_cross_effect_complete",
        "n_cross_not_low_confidence",
        "n_cross_maf_ge_0_01",
        "n_eur_p_le_1e_5_cross_usable",
        "n_eur_gws_cross_usable",
    ]:
        counts[key] = int(counts[key])

    return dict(counts)


def run_validation(
    mapping: pd.DataFrame,
    pairs: pd.DataFrame,
    windows: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    metadata = phenotype_metadata(mapping)
    unit = choose_validation_unit(pairs, windows)
    phenotype_id = clean(unit["panukb_phenotype_id"])
    mapping_row = metadata[phenotype_id]

    remote = RemotePhenotype(mapping_row, args.index_dir, args.timeout_seconds)
    try:
        pops = split_pops(mapping_row["pops_pass_qc"])
        remote.validate_schema(pops)
        records = remote.fetch(
            clean(unit["chrom"]),
            int(unit["query_start_0based"]),
            int(unit["query_end_0based_exclusive"]),
        )
        payload = schema_validation_payload(remote, unit, records)
    finally:
        remote.close()

    output = args.output_dir / "14b_panukb_schema_validation.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")

    print("=" * 76)
    print("STEP 14B REMOTE PAN-UKB VALIDATION")
    print("=" * 76)
    print(f"Phenotype:       {payload['validation_unit']['panukb_phenotype_id']}")
    print(f"Gene:            {payload['validation_unit']['gene']}")
    print(f"Indication:      {payload['validation_unit']['indication_mesh_term']}")
    print(f"QC populations:  {','.join(payload['validation_unit']['pops_pass_qc'])}")
    print(f"Records fetched: {payload['n_records_in_validation_window']:,}")
    print(f"Columns:         {len(payload['columns']):,}")
    print("P-value schema:  neglog10_pval_{pop}")
    print(f"Output:          {output}")
    print("=" * 76)


def run_count(
    mapping: pd.DataFrame,
    pairs: pd.DataFrame,
    windows: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    metadata = phenotype_metadata(mapping)

    pair_units = pairs[
        [
            "ti_uid",
            "gene",
            "pool",
            "indication_mesh_id",
            "indication_mesh_term",
            "panukb_phenotype_id",
            "pops_pass_qc",
        ]
    ].drop_duplicates()

    units = pair_units.merge(
        windows,
        on="gene",
        how="inner",
        validate="many_to_many",
    )

    output_rows = []
    cache: dict[tuple[str, str, int, int], list[dict[str, str]]] = {}

    for phenotype_id, pheno_units in units.groupby(
        "panukb_phenotype_id", sort=True
    ):
        phenotype_id = clean(phenotype_id)
        mapping_row = metadata[phenotype_id]
        qc_pops = [p for p in split_pops(mapping_row["pops_pass_qc"]) if p != "EUR"]

        if not qc_pops:
            continue

        print(f"Opening {phenotype_id} ({','.join(qc_pops)} vs EUR)")
        remote = RemotePhenotype(
            mapping_row,
            args.index_dir,
            args.timeout_seconds,
        )
        try:
            remote.validate_schema(["EUR", *qc_pops])

            for _, unit in pheno_units.iterrows():
                key = (
                    phenotype_id,
                    clean(unit["chrom"]),
                    int(unit["query_start_0based"]),
                    int(unit["query_end_0based_exclusive"]),
                )
                if key not in cache:
                    cache[key] = remote.fetch(
                        key[1],
                        key[2],
                        key[3],
                    )
                records = cache[key]

                unit_qc_pops = [
                    p for p in split_pops(unit["pops_pass_qc"])
                    if p != "EUR"
                ]
                for pop in unit_qc_pops:
                    counts = count_records(records, pop, mapping_row)
                    row = {
                        "ti_uid": unit["ti_uid"],
                        "gene": unit["gene"],
                        "pool": unit["pool"],
                        "indication_mesh_id": unit["indication_mesh_id"],
                        "indication_mesh_term": unit["indication_mesh_term"],
                        "panukb_phenotype_id": phenotype_id,
                        "comparison_population": pop,
                        "window_type": unit["window_type"],
                        "chrom": unit["chrom"],
                        "gene_start_1based": unit["gene_start_1based"],
                        "gene_end_1based": unit["gene_end_1based"],
                        "gene_length_bp": unit["gene_length_bp"],
                        "query_start_0based": unit["query_start_0based"],
                        "query_end_0based_exclusive": unit[
                            "query_end_0based_exclusive"
                        ],
                        "window_length_bp": unit["window_length_bp"],
                        "gene_fraction_of_window": unit[
                            "gene_fraction_of_window"
                        ],
                        "overlaps_extended_mhc": unit[
                            "overlaps_extended_mhc"
                        ],
                        **counts,
                    }
                    row["passes_raw_count_gate"] = (
                        counts["n_eur_gws_cross_usable"] >= RAW_MIN_SNPS
                    )
                    row["raw_gate_minimum"] = RAW_MIN_SNPS
                    row["final_ld_independent_minimum"] = RAW_MIN_SNPS
                    output_rows.append(row)
        finally:
            remote.close()

    results = pd.DataFrame(output_rows)
    if results.empty:
        raise RuntimeError("No feasibility rows were produced.")

    results_path = (
        args.output_dir / "14b_panukb_snp_count_feasibility.csv"
    )
    results.to_csv(results_path, index=False)

    primary = results[~results["overlaps_extended_mhc"]].copy()

    summary_rows = []
    for (window_type, pop), group in primary.groupby(
        ["window_type", "comparison_population"],
        sort=True,
    ):
        counts = group["n_eur_gws_cross_usable"]
        summary_rows.append(
            {
                "window_type": window_type,
                "comparison_population": pop,
                "n_target_indication_units": int(group["ti_uid"].nunique()),
                "n_genes": int(group["gene"].nunique()),
                "n_indications": int(
                    group["indication_mesh_id"].nunique()
                ),
                "n_pool_A_units": int(
                    group.loc[group["pool"] == "A", "ti_uid"].nunique()
                ),
                "n_pool_B_units": int(
                    group.loc[group["pool"] == "B", "ti_uid"].nunique()
                ),
                "units_with_at_least_1_eur_gws_snp": int((counts >= 1).sum()),
                "units_passing_raw_10_snp_gate": int(
                    (counts >= RAW_MIN_SNPS).sum()
                ),
                "median_eur_gws_cross_usable": float(counts.median()),
                "q25_eur_gws_cross_usable": float(counts.quantile(0.25)),
                "q75_eur_gws_cross_usable": float(counts.quantile(0.75)),
                "maximum_eur_gws_cross_usable": int(counts.max()),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = args.output_dir / "14b_panukb_snp_count_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    summary_json = {
        "analysis": "Step 14B Pan-UKB SNP-count feasibility gate",
        "genome_build": "GRCh37",
        "primary_interval": "GENCODE v19 gene body",
        "sensitivity_interval": (
            "GENCODE v19 gene body plus 10 kb on each side"
            if args.window_set == "both"
            else None
        ),
        "eur_significance_threshold": {
            "p": EUR_GWS_P,
            "neglog10_p": EUR_GWS_NEGLOG10,
        },
        "variant_filters": {
            "variant_type": "biallelic A/C/G/T SNP",
            "low_confidence_EUR": False,
            "low_confidence_comparison_population": False,
            "maf_min_EUR": MAF_MIN,
            "maf_min_comparison_population": MAF_MIN,
            "finite_beta_and_positive_se_both": True,
        },
        "raw_count_gate": RAW_MIN_SNPS,
        "final_requirement_not_yet_evaluated": (
            "At least 10 LD-independent variants after prespecified LD "
            "pruning/clumping."
        ),
        "extended_mhc_primary_exclusion": {
            "chromosome": MHC_CHROM,
            "start_0based": MHC_START_0BASED,
            "end_0based_exclusive": MHC_END_0BASED,
        },
        "n_result_rows": int(len(results)),
        "n_mhc_flagged_rows": int(
            results["overlaps_extended_mhc"].sum()
        ),
        "summary": summary_df.to_dict("records"),
        "interpretation_rule": (
            "Passing the raw 10-SNP gate only advances a unit to the LD stage. "
            "It is not evidence that portability is estimable."
        ),
    }
    summary_json_path = (
        args.output_dir / "14b_panukb_snp_count_summary.json"
    )
    summary_json_path.write_text(json.dumps(summary_json, indent=2) + "\n")

    print("=" * 76)
    print("STEP 14B PAN-UKB SNP-COUNT FEASIBILITY")
    print("=" * 76)
    print(summary_df.to_string(index=False))
    print("\nOutputs:")
    print(f"  {results_path}")
    print(f"  {summary_csv}")
    print(f"  {summary_json_path}")
    print("=" * 76)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.index_dir.mkdir(parents=True, exist_ok=True)

    for path in [args.mapping, args.pairs, args.coordinates]:
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")

    mapping = pd.read_csv(args.mapping, low_memory=False)
    pairs = pd.read_parquet(args.pairs)
    coords = pd.read_parquet(args.coordinates)

    required_mapping = {
        "indication_mesh_id",
        "panukb_phenotype_id",
        "pops_pass_qc",
        "filename_tabix",
        "aws_path",
        "aws_path_tabix",
    }
    missing = sorted(required_mapping - set(mapping.columns))
    if missing:
        raise SystemExit(f"Locked mapping file missing columns: {missing}")

    required_pairs = {
        "ti_uid",
        "gene",
        "pool",
        "indication_mesh_id",
        "indication_mesh_term",
        "panukb_phenotype_id",
        "pops_pass_qc",
    }
    missing = sorted(required_pairs - set(pairs.columns))
    if missing:
        raise SystemExit(f"Locked pair file missing columns: {missing}")

    required_coords = {
        "gene",
        "chrom",
        "start_1based",
        "end_1based",
    }
    missing = sorted(required_coords - set(coords.columns))
    if missing:
        raise SystemExit(f"Coordinate file missing columns: {missing}")

    windows = build_windows(coords, args.window_set)
    windows_path = args.output_dir / "14b_panukb_windows.csv"
    windows.to_csv(windows_path, index=False)

    if args.mode == "validate":
        run_validation(mapping, pairs, windows, args)
    else:
        validation_file = (
            args.output_dir / "14b_panukb_schema_validation.json"
        )
        if not validation_file.exists():
            raise SystemExit(
                "Run validation first:\n"
                "  python3 14b_panukb_snp_feasibility.py --mode validate"
            )
        run_count(mapping, pairs, windows, args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
