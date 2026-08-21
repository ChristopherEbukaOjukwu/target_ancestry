#!/usr/bin/env python3
"""
Step 15B2 — Validate one Pan-UKB and one GBMI regional query.

The script downloads only the required tabix indexes, queries the locked
native-build gene ±100 kb windows remotely, recovers the actual headers,
identifies required fields, and tests EUR/non-EUR GBMI variant matching.
It does not retain full regional data or calculate portability/colocalization.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
import pysam

POP_ORDER = ["AFR", "AMR", "CSA", "EAS", "MID"]
ALIASES = {
    "chrom": ["chr", "#chr", "chrom", "#chrom", "chromosome"],
    "pos": ["pos", "position", "bp", "base_pair_location"],
    "ref": ["ref", "reference_allele", "other_allele", "nea", "a0"],
    "alt": ["alt", "alternate_allele", "effect_allele", "ea", "a1"],
    "beta": ["beta", "effect", "effect_size", "estimate", "b"],
    "se": ["se", "stderr", "standard_error", "sebeta", "se_beta"],
    "p": ["inv_var_meta_p", "p", "pval", "pvalue", "p_value", "p-value"],
    "af": ["af", "eaf", "effect_allele_frequency", "alt_af", "freq"],
    "n": ["n", "n_total", "sample_size", "samplesize", "n_samples"],
    "n_cases": ["n_case", "n_cases", "cases", "ncase", "num_cases"],
    "n_controls": ["n_ctrl", "n_control", "n_controls", "controls", "ncontrol", "num_controls"],
}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("output/15b1_extraction_manifest_primary.parquet"),
    )
    p.add_argument(
        "--trait-catalog",
        type=Path,
        default=Path("output/15a_trait_catalog.csv"),
    )
    p.add_argument(
        "--index-dir",
        type=Path,
        default=Path("input/tabix_indexes/15b2"),
    )
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--panukb-comparison-uid", default="")
    p.add_argument("--gbmi-comparison-uid", default="")
    p.add_argument("--max-records", type=int, default=100_000)
    p.add_argument("--timeout", type=float, default=60.0)
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


def truthy(value: Any) -> bool:
    return value is True or clean(value).casefold() in {"true", "1", "yes", "y"}


def ncol(value: Any) -> str:
    value = clean(value).casefold().lstrip("#")
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def nchrom(value: Any) -> str:
    value = clean(value)
    return value[3:] if value.casefold().startswith("chr") else value


def https(url: str) -> str:
    url = clean(url)
    if url.startswith("s3://"):
        parsed = urllib.parse.urlparse(url)
        key = urllib.parse.quote(parsed.path.lstrip("/"), safe="/")
        return f"https://{parsed.netloc}.s3.amazonaws.com/{key}"
    return url


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_index(url: str, directory: Path, label: str, timeout: float) -> tuple[Path, dict[str, Any]]:
    url = https(url)
    directory.mkdir(parents=True, exist_ok=True)
    basename = Path(urllib.parse.urlparse(url).path).name or "index.tbi"
    digest = hashlib.sha256(url.encode()).hexdigest()[:12]
    destination = directory / f"{label}_{digest}_{basename}"
    downloaded = False
    if not destination.exists() or destination.stat().st_size == 0:
        temp = destination.with_suffix(destination.suffix + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "target-ancestry-step15b2/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response, temp.open("wb") as out:
                shutil.copyfileobj(response, out)
            temp.replace(destination)
            downloaded = True
        except Exception:
            temp.unlink(missing_ok=True)
            raise
    return destination, {
        "url": url,
        "path": str(destination),
        "downloaded": downloaded,
        "size_bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def first_line(url: str, timeout: float) -> str:
    request = urllib.request.Request(
        https(url),
        headers={"User-Agent": "target-ancestry-step15b2/1.0", "Accept-Encoding": "identity"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        with gzip.GzipFile(fileobj=response) as stream:
            line = stream.readline()
    if not line:
        raise RuntimeError(f"No decompressed line returned from {url}")
    return line.decode("utf-8", errors="replace").rstrip("\r\n")


def split_header(line: str) -> list[str]:
    line = line.lstrip("\ufeff").lstrip("#")
    columns = line.split("\t")
    if len(columns) <= 1:
        columns = re.split(r"\s+", line.strip())
    columns = [clean(column) for column in columns if clean(column)]
    if len(columns) < 4:
        raise RuntimeError(f"Could not parse header: {line[:250]}")
    return columns


def recover_header(tabix: pysam.TabixFile, data_url: str, timeout: float) -> tuple[list[str], str, str]:
    metadata = list(tabix.header)
    if metadata:
        raw = metadata[-1]
        return split_header(raw), raw, "tabix_metadata_header"
    raw = first_line(data_url, timeout)
    return split_header(raw), raw, "first_bgzf_line"


def open_tabix(data_url: str, index_url: str, index_dir: Path, label: str, timeout: float):
    local_index, metadata = download_index(index_url, index_dir, label, timeout)
    data_url = https(data_url)
    return pysam.TabixFile(data_url, index=str(local_index)), {
        "data_url": data_url,
        "index": metadata,
    }


def choose_contig(tabix: pysam.TabixFile, row: pd.Series) -> tuple[str, dict[str, Any]]:
    contigs = list(tabix.contigs)
    target = nchrom(row["chrom"])
    candidates = [clean(row.get("chrom_ucsc")), target, f"chr{target}"]
    for candidate in dict.fromkeys(candidate for candidate in candidates if candidate):
        if candidate in contigs:
            return candidate, {
                "target_chromosome": target,
                "selected_contig": candidate,
                "selection_method": "exact_contig_match",
                "n_tabix_contigs": len(contigs),
                "tabix_contig_examples": contigs[:25],
            }
    normalized = {nchrom(contig): contig for contig in contigs}
    if target in normalized:
        return normalized[target], {
            "target_chromosome": target,
            "selected_contig": normalized[target],
            "selection_method": "normalized_contig_match",
            "n_tabix_contigs": len(contigs),
            "tabix_contig_examples": contigs[:25],
        }
    raise RuntimeError(f"No tabix contig matched chromosome {target}; examples: {contigs[:25]}")


def fetch(tabix: pysam.TabixFile, row: pd.Series, contig: str, maximum: int) -> list[str]:
    records = []
    for record in tabix.fetch(
        contig,
        int(row["window_start_0based"]),
        int(row["window_end_1based"]),
    ):
        records.append(record)
        if len(records) > maximum:
            raise RuntimeError(f"Validation query exceeded --max-records ({maximum:,})")
    return records


def frame(records: list[str], columns: list[str]) -> tuple[pd.DataFrame, int]:
    rows, malformed = [], 0
    for record in records:
        fields = record.split("\t")
        if len(fields) != len(columns):
            fields = re.split(r"\s+", record.strip())
        if len(fields) != len(columns):
            malformed += 1
        else:
            rows.append(fields)
    return pd.DataFrame(rows, columns=columns), malformed


def find(columns: list[str], semantic: str) -> str:
    lookup = {ncol(column): column for column in columns}
    for alias in map(ncol, ALIASES[semantic]):
        if alias in lookup:
            return lookup[alias]
    candidates = []
    for normalized, original in lookup.items():
        for alias in map(ncol, ALIASES[semantic]):
            if normalized.startswith(alias + "_") or normalized.endswith("_" + alias):
                candidates.append(original)
                break
    candidates = list(dict.fromkeys(candidates))
    return candidates[0] if len(candidates) == 1 else ""


def pop_field(columns: list[str], stem: str, population: str) -> str:
    lookup = {ncol(column): column for column in columns}
    return lookup.get(ncol(f"{stem}_{population}"), "")


def pop_frequency(columns: list[str], population: str) -> list[str]:
    suffix = "_" + population.casefold()
    result = []
    for column in columns:
        normalized = ncol(column)
        if normalized.endswith(suffix) and (
            normalized.startswith("af_")
            or normalized.startswith("maf_")
            or "_af_" in normalized
            or normalized.startswith("freq_")
        ):
            result.append(column)
    return result


def pop_sample(columns: list[str], population: str) -> list[str]:
    suffix = "_" + population.casefold()
    return [
        column for column in columns
        if ncol(column).endswith(suffix)
        and (ncol(column).startswith("n_") or ncol(column).startswith("num_"))
    ]


def numeric_summary(data: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    output = {}
    for column in dict.fromkeys(column for column in columns if column):
        values = pd.to_numeric(data[column], errors="coerce")
        output[column] = {
            "finite_fraction": float(values.notna().mean()) if len(values) else None,
            "minimum": float(values.min()) if values.notna().any() else None,
            "maximum": float(values.max()) if values.notna().any() else None,
        }
    return output


def choose_row(manifest: pd.DataFrame, source: str, override: str) -> pd.Series:
    subset = manifest[
        manifest["candidate_source"].eq(source)
        & manifest["extraction_ready"].map(truthy)
        & manifest["unit_analysis_role"].eq("PRIMARY")
    ].copy()
    if override:
        subset = subset[subset["comparison_uid"].eq(override)]
        if len(subset) != 1:
            raise SystemExit(f"{source} override matched {len(subset)} rows")
        return subset.iloc[0]
    groups = (
        ["hba1c", "ldl_cholesterol", "body_mass_index"]
        if source == "Pan-UKB"
        else ["Asthma", "COPD", "VTE"]
    )
    pops = ["CSA", "AFR", "EAS", "MID", "AMR"] if source == "Pan-UKB" else ["AFR", "EAS", "AMR", "CSA", "MID"]
    grank = {value: i for i, value in enumerate(groups)}
    prank = {value: i for i, value in enumerate(pops)}
    subset["_g"] = subset["canonical_group"].map(lambda x: grank.get(clean(x), 999))
    subset["_p"] = subset["comparison_population"].map(lambda x: prank.get(clean(x), 999))
    subset["_length"] = pd.to_numeric(subset["window_end_1based"], errors="coerce") - pd.to_numeric(subset["window_start_1based"], errors="coerce")
    subset = subset.sort_values(["_g", "_p", "_length", "gene", "comparison_uid"])
    if subset.empty:
        raise SystemExit(f"No extraction-ready primary {source} row")
    return subset.iloc[0]


def validate_panukb(row: pd.Series, catalog: pd.DataFrame, a: argparse.Namespace) -> dict[str, Any]:
    tabix, source = open_tabix(row["eur_data_url"], row["eur_index_url"], a.index_dir, "PanUKB", a.timeout)
    try:
        columns, raw_header, method = recover_header(tabix, row["eur_data_url"], a.timeout)
        contig, contig_meta = choose_contig(tabix, row)
        records = fetch(tabix, row, contig, a.max_records)
    finally:
        tabix.close()
    data, malformed = frame(records, columns)
    base = {semantic: find(columns, semantic) for semantic in ["chrom", "pos", "ref", "alt"]}
    populations = ["EUR", clean(row["comparison_population"]).upper()]
    fields = {}
    missing = [f"base:{key}" for key, value in base.items() if not value]
    numeric = []
    for population in populations:
        mapping = {
            "beta": pop_field(columns, "beta", population),
            "se": pop_field(columns, "se", population),
            "p": pop_field(columns, "neglog10_pval", population),
            "low_confidence": pop_field(columns, "low_confidence", population),
            "frequency_fields": pop_frequency(columns, population),
            "sample_fields_in_data": pop_sample(columns, population),
        }
        fields[population] = mapping
        for required in ["beta", "se", "p", "low_confidence"]:
            if not mapping[required]:
                missing.append(f"{population}:{required}")
        if not mapping["frequency_fields"]:
            missing.append(f"{population}:frequency_fields")
        numeric += [mapping["beta"], mapping["se"], mapping["p"], *mapping["frequency_fields"]]
    trait = catalog[catalog["trait_key"].eq(row["candidate_trait_key"])]
    sample_metadata = {}
    if len(trait) == 1:
        record = trait.iloc[0]
        for population in populations:
            sample_metadata[population] = {
                field: None if field not in trait.columns or pd.isna(record.get(field)) else record.get(field)
                for field in [f"n_cases_{population}", f"n_controls_{population}", f"phenotype_qc_{population}"]
            }
    header_path = a.output_dir / "15b2_panukb_header.txt"
    header_path.write_text(raw_header + "\n", encoding="utf-8")
    pos = pd.to_numeric(data[base["pos"]], errors="coerce") if len(data) and base["pos"] else pd.Series(dtype=float)
    status = "PASS" if records and not missing and malformed == 0 else "FAIL"
    return {
        "status": status,
        "comparison_uid": clean(row["comparison_uid"]),
        "gene": clean(row["gene"]),
        "trait": clean(row["candidate_trait_name"]),
        "trait_key": clean(row["candidate_trait_key"]),
        "genome_build": clean(row["genome_build"]),
        "comparison_population": populations[1],
        "window": {
            "chromosome": clean(row["chrom"]),
            "start_1based": int(row["window_start_1based"]),
            "end_1based": int(row["window_end_1based"]),
            "start_0based": int(row["window_start_0based"]),
        },
        "source": source,
        "contig": contig_meta,
        "header": {"method": method, "n_columns": len(columns), "columns": columns, "saved_to": str(header_path)},
        "field_mapping": {
            "base": base,
            "populations": fields,
            "p_value_representation": "negative_log10_p",
            "effect_allele_note": "The file exposes REF/ALT and population beta fields; ALT orientation remains tied to the official Pan-UKB specification.",
        },
        "trait_sample_metadata": sample_metadata,
        "regional_query": {
            "n_records": len(records),
            "n_parsed_records": len(data),
            "n_malformed_records": malformed,
            "minimum_position": int(pos.min()) if pos.notna().any() else None,
            "maximum_position": int(pos.max()) if pos.notna().any() else None,
            "numeric_field_summary": numeric_summary(data, numeric),
        },
        "missing_requirements": missing,
    }


def gbmi_fields(columns: list[str]) -> dict[str, str]:
    return {semantic: find(columns, semantic) for semantic in ["chrom", "pos", "ref", "alt", "beta", "se", "p", "af", "n", "n_cases", "n_controls"]}


def keys(data: pd.DataFrame, mapping: dict[str, str]) -> tuple[set[tuple[str, int, str, str]], int]:
    required = [mapping["chrom"], mapping["pos"], mapping["ref"], mapping["alt"]]
    if not all(required):
        return set(), 0
    rows = []
    for chrom, pos, ref, alt in zip(data[required[0]], data[required[1]], data[required[2]], data[required[3]]):
        try:
            position = int(float(pos))
        except (TypeError, ValueError):
            continue
        ref, alt = clean(ref).upper(), clean(alt).upper()
        if ref and alt:
            rows.append((nchrom(chrom), position, ref, alt))
    return set(rows), len(rows) - len(set(rows))


def validate_gbmi(row: pd.Series, a: argparse.Namespace) -> dict[str, Any]:
    non_eur = clean(row["comparison_population"]).upper()
    datasets = {}
    for label, data_column, index_column in [
        ("EUR", "eur_data_url", "eur_index_url"),
        (non_eur, "comparison_data_url", "comparison_index_url"),
    ]:
        tabix, source = open_tabix(row[data_column], row[index_column], a.index_dir, f"GBMI_{label}", a.timeout)
        try:
            columns, raw_header, method = recover_header(tabix, row[data_column], a.timeout)
            contig, contig_meta = choose_contig(tabix, row)
            records = fetch(tabix, row, contig, a.max_records)
        finally:
            tabix.close()
        data, malformed = frame(records, columns)
        mapping = gbmi_fields(columns)
        variant_keys, duplicates = keys(data, mapping)
        missing = [semantic for semantic in ["chrom", "pos", "ref", "alt", "beta", "se", "p"] if not mapping[semantic]]
        numeric = [mapping[k] for k in ["beta", "se", "p", "af", "n", "n_cases", "n_controls"] if mapping[k]]
        header_path = a.output_dir / ("15b2_gbmi_eur_header.txt" if label == "EUR" else "15b2_gbmi_noneur_header.txt")
        header_path.write_text(raw_header + "\n", encoding="utf-8")
        datasets[label] = {
            "source": source,
            "contig": contig_meta,
            "header": {"method": method, "n_columns": len(columns), "columns": columns, "saved_to": str(header_path)},
            "field_mapping": mapping,
            "regional_query": {
                "n_records": len(records),
                "n_parsed_records": len(data),
                "n_malformed_records": malformed,
                "n_unique_variant_keys": len(variant_keys),
                "n_duplicate_variant_key_rows": duplicates,
                "numeric_field_summary": numeric_summary(data, numeric),
            },
            "missing_requirements": missing,
            "_keys": variant_keys,
        }
    eur_keys = datasets["EUR"].pop("_keys")
    non_keys = datasets[non_eur].pop("_keys")
    exact = eur_keys & non_keys
    swapped = eur_keys & {(chrom, pos, alt, ref) for chrom, pos, ref, alt in non_keys}
    positions = {(chrom, pos) for chrom, pos, _, _ in eur_keys} & {(chrom, pos) for chrom, pos, _, _ in non_keys}
    all_required = all(
        not item["missing_requirements"]
        and item["regional_query"]["n_records"] > 0
        and item["regional_query"]["n_malformed_records"] == 0
        for item in datasets.values()
    )
    return {
        "status": "PASS" if all_required and exact else "FAIL",
        "comparison_uid": clean(row["comparison_uid"]),
        "gene": clean(row["gene"]),
        "trait": clean(row["candidate_trait_name"]),
        "trait_key": clean(row["candidate_trait_key"]),
        "genome_build": clean(row["genome_build"]),
        "comparison_population": non_eur,
        "source_population_label": clean(row["source_population_label"]),
        "window": {
            "chromosome": clean(row["chrom"]),
            "start_1based": int(row["window_start_1based"]),
            "end_1based": int(row["window_end_1based"]),
            "start_0based": int(row["window_start_0based"]),
        },
        "datasets": datasets,
        "cross_ancestry_matching": {
            "n_eur_unique_variant_keys": len(eur_keys),
            "n_noneur_unique_variant_keys": len(non_keys),
            "n_shared_positions": len(positions),
            "n_exact_chr_pos_ref_alt_matches": len(exact),
            "n_swapped_ref_alt_matches": len(swapped),
            "exact_match_fraction_of_eur": len(exact) / len(eur_keys) if eur_keys else None,
            "exact_match_fraction_of_noneur": len(exact) / len(non_keys) if non_keys else None,
            "recommended_primary_key": "normalized chromosome + position + REF + ALT",
            "allele_action": "Use exact REF/ALT matches as primary. Swapped matches require explicit sign-flip harmonization after confirming source orientation.",
        },
    }


def main() -> int:
    a = args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    a.index_dir.mkdir(parents=True, exist_ok=True)
    for path in [a.manifest, a.trait_catalog]:
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")
    manifest = pd.read_parquet(a.manifest)
    catalog = pd.read_csv(a.trait_catalog, low_memory=False)
    panukb = choose_row(manifest, "Pan-UKB", a.panukb_comparison_uid)
    gbmi = choose_row(manifest, "GBMI", a.gbmi_comparison_uid)
    selected_columns = [
        "comparison_uid", "gene_trait_uid", "gene", "candidate_source",
        "candidate_trait_name", "canonical_group", "comparison_population",
        "source_population_label", "genome_build", "chrom",
        "window_start_1based", "window_end_1based", "eur_data_url",
        "comparison_data_url",
    ]
    selected = pd.DataFrame([panukb, gbmi])[selected_columns]
    selected_path = a.output_dir / "15b2_selected_validation_rows.csv"
    selected.to_csv(selected_path, index=False)
    print("Validating Pan-UKB:", panukb["gene"], panukb["candidate_trait_name"], panukb["comparison_population"])
    panukb_result = validate_panukb(panukb, catalog, a)
    print("Validating GBMI:", gbmi["gene"], gbmi["candidate_trait_name"], gbmi["comparison_population"])
    gbmi_result = validate_gbmi(gbmi, a)
    panukb_path = a.output_dir / "15b2_panukb_schema_validation.json"
    gbmi_path = a.output_dir / "15b2_gbmi_schema_validation.json"
    panukb_path.write_text(json.dumps(panukb_result, indent=2) + "\n", encoding="utf-8")
    gbmi_path.write_text(json.dumps(gbmi_result, indent=2) + "\n", encoding="utf-8")
    overall = "PASS" if panukb_result["status"] == "PASS" and gbmi_result["status"] == "PASS" else "FAIL"
    summary = {
        "step": "15B2",
        "status": overall,
        "panukb_status": panukb_result["status"],
        "gbmi_status": gbmi_result["status"],
        "selected_rows": str(selected_path),
        "panukb_validation": str(panukb_path),
        "gbmi_validation": str(gbmi_path),
        "index_directory": str(a.index_dir),
        "scientific_decisions_frozen": {
            "panukb_p_value_field": "neglog10_pval_{population}",
            "gbmi_primary_variant_match": "normalized chromosome + position + REF + ALT",
            "gbmi_swapped_alleles": "retain for explicit sign-flip review; do not merge silently",
            "builds": {"Pan-UKB": "GRCh37", "GBMI": "GRCh38"},
        },
    }
    summary_path = a.output_dir / "15b2_schema_validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    non_eur = gbmi_result["comparison_population"]
    print("=" * 78)
    print("STEP 15B2 — REGIONAL SCHEMA VALIDATION")
    print("=" * 78)
    print(f"Pan-UKB status:                         {panukb_result['status']}")
    print(f"Pan-UKB regional records:               {panukb_result['regional_query']['n_records']:,}")
    print(f"Pan-UKB columns:                        {panukb_result['header']['n_columns']:,}")
    print(f"GBMI status:                            {gbmi_result['status']}")
    print(f"GBMI EUR regional records:              {gbmi_result['datasets']['EUR']['regional_query']['n_records']:,}")
    print(f"GBMI {non_eur} regional records:              {gbmi_result['datasets'][non_eur]['regional_query']['n_records']:,}")
    print(f"GBMI exact cross-ancestry variant keys: {gbmi_result['cross_ancestry_matching']['n_exact_chr_pos_ref_alt_matches']:,}")
    print(f"Overall status:                         {overall}")
    print()
    print("Outputs:")
    for path in [selected_path, panukb_path, gbmi_path, summary_path]:
        print(f"  {path}")
    print("=" * 78)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
