#!/usr/bin/env python3
"""
Step 15B3 — Extract and cache all locked native-build regional datasets.

This step performs source-native extraction only.

Pan-UKB
-------
- One regional extract per gene-trait-source unit.
- The extract retains the complete multi-ancestry Pan-UKB row schema.
- Native build: GRCh37.
- Region: GENCODE v19 gene body ±100 kb.

GBMI
----
- One regional extract per gene-trait-source unit and ancestry.
- EUR and non-EUR files remain separate.
- Native build: GRCh38.
- Region: GENCODE v38 gene body ±100 kb.

No allele harmonization, ancestry QC, genome-wide-significance filtering,
LD processing, portability calculation, or colocalization is performed here.

The script is resumable. Each completed extraction has an atomic Parquet file
and JSON sidecar. Re-running the script skips valid completed extracts unless
--force is supplied.

Inputs
------
output/15b1_extraction_manifest.parquet
output/15b2_schema_validation_summary.json
output/15b2_panukb_schema_validation.json
output/15b2_gbmi_schema_validation.json

Outputs
-------
intermediate/15b3/Pan-UKB/*.parquet
intermediate/15b3/Pan-UKB/*.json
intermediate/15b3/GBMI/*.parquet
intermediate/15b3/GBMI/*.json
input/tabix_indexes/15b3/*.tbi
output/15b3_extraction_inventory.csv
output/15b3_extraction_inventory.parquet
output/15b3_comparison_extraction_map.csv
output/15b3_comparison_extraction_map.parquet
output/15b3_extraction_failures.csv
output/15b3_extraction_summary.json
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pysam


SCRIPT_VERSION = "15B3.2"
PANUKB_SOURCE = "Pan-UKB"
GBMI_SOURCE = "GBMI"

PANUKB_FIXED_COLUMNS = [
    "chr",
    "pos",
    "ref",
    "alt",
    "af_meta_hq",
    "beta_meta_hq",
    "se_meta_hq",
    "neglog10_pval_meta_hq",
    "neglog10_pval_heterogeneity_hq",
    "af_meta",
    "beta_meta",
    "se_meta",
    "neglog10_pval_meta",
    "neglog10_pval_heterogeneity",
]

PANUKB_POPULATION_STEMS = [
    "af",
    "beta",
    "se",
    "neglog10_pval",
    "low_confidence",
]

PANUKB_KNOWN_POPULATIONS = [
    "AFR",
    "AMR",
    "CSA",
    "EAS",
    "EUR",
    "MID",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("output/15b1_extraction_manifest.parquet"),
    )
    parser.add_argument(
        "--schema-summary",
        type=Path,
        default=Path("output/15b2_schema_validation_summary.json"),
    )
    parser.add_argument(
        "--panukb-schema",
        type=Path,
        default=Path("output/15b2_panukb_schema_validation.json"),
    )
    parser.add_argument(
        "--gbmi-schema",
        type=Path,
        default=Path("output/15b2_gbmi_schema_validation.json"),
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=Path("intermediate/15b3"),
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("input/tabix_indexes/15b3"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=90.0,
    )
    parser.add_argument(
        "--max-records-per-region",
        type=int,
        default=250_000,
    )
    parser.add_argument(
        "--max-file-groups",
        type=int,
        default=0,
        help=(
            "Optional debugging limit on unique remote data files. "
            "Zero means all files."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract files even when valid cached outputs exist.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Continue to later remote files after a file-level failure. "
            "The default is to continue, but the option is retained for "
            "command-line clarity."
        ),
    )
    return parser.parse_args()


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


def normalize_column(value: Any) -> str:
    text = clean(value).casefold().lstrip("#")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def normalize_chrom(value: Any) -> str:
    chrom = clean(value)
    if chrom.casefold().startswith("chr"):
        chrom = chrom[3:]
    if chrom == "M":
        chrom = "MT"
    return chrom


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_https(url: str) -> str:
    url = clean(url)
    if url.startswith("s3://"):
        parsed = urllib.parse.urlparse(url)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        return (
            f"https://{bucket}.s3.amazonaws.com/"
            f"{urllib.parse.quote(key, safe='/')}"
        )
    return url


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_write(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def atomic_parquet_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(
        temp,
        index=False,
        compression="zstd",
    )
    temp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def index_local_path(
    index_dir: Path,
    *,
    source: str,
    url: str,
) -> Path:
    parsed = urllib.parse.urlparse(url)
    basename = Path(parsed.path).name or "index.tbi"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    safe_source = re.sub(r"[^A-Za-z0-9]+", "_", source).strip("_")
    return index_dir / f"{safe_source}_{digest}_{basename}"


def download_file(
    url: str,
    destination: Path,
    *,
    timeout: float,
    retries: int = 4,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = to_https(url)

    if destination.exists() and destination.stat().st_size > 0:
        return {
            "url": url,
            "path": str(destination),
            "downloaded": False,
            "size_bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }

    temp = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        temp.unlink(missing_ok=True)
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "target-ancestry-step15b3/1.0"
                ),
                "Accept-Encoding": "identity",
            },
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout,
            ) as response, temp.open("wb") as output:
                shutil.copyfileobj(response, output)

            if temp.stat().st_size <= 0:
                raise RuntimeError("Downloaded index is empty.")

            temp.replace(destination)
            return {
                "url": url,
                "path": str(destination),
                "downloaded": True,
                "size_bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        except Exception as exc:
            last_error = exc
            temp.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2 ** attempt, 15))

    raise RuntimeError(
        f"Failed to download {url} after {retries} attempts: "
        f"{last_error}"
    )


def read_first_decompressed_line(
    url: str,
    *,
    timeout: float,
) -> str:
    request = urllib.request.Request(
        to_https(url),
        headers={
            "User-Agent": "target-ancestry-step15b3/1.0",
            "Accept-Encoding": "identity",
        },
    )
    with urllib.request.urlopen(
        request,
        timeout=timeout,
    ) as response:
        with gzip.GzipFile(fileobj=response) as stream:
            raw = stream.readline()
            if not raw:
                raise RuntimeError(
                    f"No decompressed line returned from {url}"
                )
            return raw.decode(
                "utf-8",
                errors="replace",
            ).rstrip("\r\n")


def split_header(line: str) -> list[str]:
    line = line.lstrip("\ufeff")
    if line.startswith("##"):
        raise RuntimeError(
            "First BGZF line is metadata, not a column header."
        )
    line = line.lstrip("#")
    fields = line.split("\t")
    if len(fields) <= 1:
        fields = re.split(r"\s+", line.strip())
    fields = [clean(field) for field in fields if clean(field)]
    if len(fields) < 4:
        raise RuntimeError(
            f"Could not parse header: {line[:250]}"
        )
    return fields


def header_from_tabix_or_stream(
    tabix: pysam.TabixFile,
    data_url: str,
    *,
    timeout: float,
) -> tuple[list[str], str]:
    metadata_header = list(tabix.header)
    if metadata_header:
        return split_header(metadata_header[-1]), "tabix_metadata_header"

    first_line = read_first_decompressed_line(
        data_url,
        timeout=timeout,
    )
    return split_header(first_line), "first_bgzf_line"


def contig_for_target(
    tabix: pysam.TabixFile,
    chrom: str,
    chrom_ucsc: str,
) -> tuple[str, str]:
    contigs = list(tabix.contigs)
    target = normalize_chrom(chrom)
    candidates = [
        clean(chrom_ucsc),
        target,
        f"chr{target}",
    ]

    for candidate in dict.fromkeys(candidates):
        if candidate and candidate in contigs:
            return candidate, "exact_contig_match"

    normalized = {
        normalize_chrom(contig): contig
        for contig in contigs
    }
    if target in normalized:
        return normalized[target], "normalized_contig_match"

    raise RuntimeError(
        f"No tabix contig matched chromosome {target}. "
        f"First contigs: {contigs[:25]}"
    )


def fetch_records(
    tabix: pysam.TabixFile,
    *,
    contig: str,
    start_0based: int,
    end_1based: int,
    max_records: int,
) -> list[str]:
    records: list[str] = []
    for record in tabix.fetch(
        contig,
        start_0based,
        end_1based,
    ):
        records.append(record)
        if len(records) > max_records:
            raise RuntimeError(
                "Region exceeded --max-records-per-region "
                f"({max_records:,})."
            )
    return records


def parse_records(
    records: Iterable[str],
    columns: list[str],
) -> tuple[pd.DataFrame, int]:
    rows: list[list[str]] = []
    malformed = 0

    for line in records:
        fields = line.rstrip("\n").split("\t")
        if len(fields) != len(columns):
            fields = re.split(r"\s+", line.strip())
        if len(fields) != len(columns):
            malformed += 1
            continue
        rows.append(fields)

    frame = pd.DataFrame(rows, columns=columns)
    return frame, malformed


def column_by_normalized_name(
    columns: list[str],
    *candidate_names: str,
) -> str:
    lookup = {
        normalize_column(column): column
        for column in columns
    }
    for candidate in candidate_names:
        normalized = normalize_column(candidate)
        if normalized in lookup:
            return lookup[normalized]
    return ""


def raw_variant_key_summary(
    frame: pd.DataFrame,
    *,
    source: str,
) -> dict[str, Any]:
    columns = list(frame.columns)
    if source == PANUKB_SOURCE:
        chrom_col = column_by_normalized_name(
            columns,
            "chr",
            "chrom",
        )
        pos_col = column_by_normalized_name(columns, "pos")
        ref_col = column_by_normalized_name(columns, "ref")
        alt_col = column_by_normalized_name(columns, "alt")
    else:
        chrom_col = column_by_normalized_name(columns, "CHR")
        pos_col = column_by_normalized_name(columns, "POS")
        ref_col = column_by_normalized_name(columns, "REF")
        alt_col = column_by_normalized_name(columns, "ALT")

    required = [chrom_col, pos_col, ref_col, alt_col]
    if not all(required):
        return {
            "key_columns_found": False,
            "chrom_column": chrom_col,
            "position_column": pos_col,
            "ref_column": ref_col,
            "alt_column": alt_col,
            "n_valid_key_rows": 0,
            "n_unique_variant_keys": 0,
            "n_duplicate_variant_key_rows": 0,
            "minimum_position": None,
            "maximum_position": None,
        }

    keys: list[tuple[str, int, str, str]] = []
    positions: list[int] = []

    for chrom, pos, ref, alt in zip(
        frame[chrom_col],
        frame[pos_col],
        frame[ref_col],
        frame[alt_col],
    ):
        try:
            position = int(float(pos))
        except (TypeError, ValueError):
            continue

        ref_value = clean(ref).upper()
        alt_value = clean(alt).upper()
        if not ref_value or not alt_value:
            continue

        keys.append(
            (
                normalize_chrom(chrom),
                position,
                ref_value,
                alt_value,
            )
        )
        positions.append(position)

    unique_keys = set(keys)
    return {
        "key_columns_found": True,
        "chrom_column": chrom_col,
        "position_column": pos_col,
        "ref_column": ref_col,
        "alt_column": alt_col,
        "n_valid_key_rows": len(keys),
        "n_unique_variant_keys": len(unique_keys),
        "n_duplicate_variant_key_rows": (
            len(keys) - len(unique_keys)
        ),
        "minimum_position": min(positions) if positions else None,
        "maximum_position": max(positions) if positions else None,
    }


def validate_panukb_header(
    columns: list[str],
    *,
    required_populations: list[str],
) -> dict[str, Any]:
    normalized = {
        normalize_column(column): column
        for column in columns
    }

    missing_fixed = [
        column
        for column in PANUKB_FIXED_COLUMNS
        if normalize_column(column) not in normalized
    ]

    population_fields: dict[str, dict[str, str]] = {}
    incomplete_present_populations: dict[str, list[str]] = {}

    for population in PANUKB_KNOWN_POPULATIONS:
        fields = {
            stem: normalized.get(
                normalize_column(f"{stem}_{population}"),
                "",
            )
            for stem in PANUKB_POPULATION_STEMS
        }
        present_count = sum(bool(value) for value in fields.values())

        if present_count:
            population_fields[population] = fields
        if 0 < present_count < len(PANUKB_POPULATION_STEMS):
            incomplete_present_populations[population] = [
                stem
                for stem, value in fields.items()
                if not value
            ]

    required = sorted(
        {
            clean(population).upper()
            for population in required_populations
            if clean(population)
        }
        | {"EUR"}
    )
    missing_required_populations = [
        population
        for population in required
        if population not in population_fields
        or not all(population_fields[population].values())
    ]

    observed_expected_order = [
        normalize_column(column)
        for column in columns
    ]
    canonical_order = [
        normalize_column(column)
        for column in PANUKB_FIXED_COLUMNS
    ]
    for population in PANUKB_KNOWN_POPULATIONS:
        if population in population_fields:
            canonical_order.extend(
                normalize_column(f"{stem}_{population}")
                for stem in PANUKB_POPULATION_STEMS
            )

    # Pan-UKB stores population blocks by statistic rather than by population:
    # all AF columns, then all beta columns, etc. Construct the expected order
    # using only ancestries actually present in this file.
    present_populations = [
        population
        for population in PANUKB_KNOWN_POPULATIONS
        if population in population_fields
    ]
    canonical_order = [
        normalize_column(column)
        for column in PANUKB_FIXED_COLUMNS
    ]
    for stem in PANUKB_POPULATION_STEMS:
        canonical_order.extend(
            normalize_column(f"{stem}_{population}")
            for population in present_populations
        )

    unexpected_columns = [
        column
        for column in observed_expected_order
        if column not in canonical_order
    ]
    order_matches = observed_expected_order == canonical_order

    status = (
        "PASS"
        if not missing_fixed
        and not incomplete_present_populations
        and not missing_required_populations
        and not unexpected_columns
        and order_matches
        else "FAIL"
    )

    return {
        "status": status,
        "missing_fixed_columns": missing_fixed,
        "present_populations": present_populations,
        "required_populations": required,
        "missing_required_populations": (
            missing_required_populations
        ),
        "incomplete_present_populations": (
            incomplete_present_populations
        ),
        "unexpected_columns": unexpected_columns,
        "column_order_matches_source_pattern": order_matches,
    }


def extraction_paths(
    extract_dir: Path,
    *,
    source: str,
    extraction_key: str,
) -> tuple[Path, Path]:
    source_dir = extract_dir / source
    parquet = source_dir / f"{extraction_key}.parquet"
    sidecar = source_dir / f"{extraction_key}.json"
    return parquet, sidecar


def cache_is_valid(
    parquet_path: Path,
    sidecar_path: Path,
    *,
    expected_key: str,
) -> bool:
    if not parquet_path.exists() or not sidecar_path.exists():
        return False
    try:
        sidecar = read_json(sidecar_path)
    except Exception:
        return False

    return (
        clean(sidecar.get("extraction_key")) == expected_key
        and clean(sidecar.get("status"))
        in {"EXTRACTED", "EMPTY_REGION"}
        and int(sidecar.get("output_size_bytes", -1))
        == parquet_path.stat().st_size
    )


def task_key_panukb(row: pd.Series) -> str:
    return clean(row["gene_trait_uid"])


def task_key_gbmi(row: pd.Series, population: str) -> str:
    return (
        f"{clean(row['gene_trait_uid'])}"
        f"__{clean(population).upper()}"
    )


def create_tasks(
    manifest: pd.DataFrame,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    tasks: list[dict[str, Any]] = []

    panukb = manifest[
        manifest["candidate_source"].eq(PANUKB_SOURCE)
    ].copy()
    for _, group in panukb.groupby(
        "gene_trait_uid",
        sort=True,
    ):
        row = group.iloc[0]
        invariant = [
            "eur_data_url",
            "eur_index_url",
            "chrom",
            "chrom_ucsc",
            "window_start_0based",
            "window_start_1based",
            "window_end_1based",
        ]
        for column in invariant:
            if group[column].astype(str).nunique() != 1:
                raise SystemExit(
                    "Pan-UKB comparison rows disagree within "
                    f"{row['gene_trait_uid']} for {column}."
                )

        key = task_key_panukb(row)
        tasks.append(
            {
                "extraction_key": key,
                "source": PANUKB_SOURCE,
                "gene_trait_uid": clean(row["gene_trait_uid"]),
                "gene": clean(row["gene"]),
                "candidate_trait_key": clean(
                    row["candidate_trait_key"]
                ),
                "candidate_trait_name": clean(
                    row["candidate_trait_name"]
                ),
                "population": "MULTI_ANCESTRY",
                "required_populations": sorted(
                    {
                        "EUR",
                        *[
                            clean(value).upper()
                            for value in group[
                                "comparison_population"
                            ].tolist()
                            if clean(value)
                        ],
                    }
                ),
                "genome_build": clean(row["genome_build"]),
                "data_url": to_https(clean(row["eur_data_url"])),
                "index_url": to_https(clean(row["eur_index_url"])),
                "chrom": clean(row["chrom"]),
                "chrom_ucsc": clean(row["chrom_ucsc"]),
                "window_start_0based": int(
                    row["window_start_0based"]
                ),
                "window_start_1based": int(
                    row["window_start_1based"]
                ),
                "window_end_1based": int(
                    row["window_end_1based"]
                ),
                "unit_analysis_role": clean(
                    row["unit_analysis_role"]
                ),
                "mixed_pool": as_bool(row["mixed_pool"]),
            }
        )

    gbmi = manifest[
        manifest["candidate_source"].eq(GBMI_SOURCE)
    ].copy()
    for gene_trait_uid, group in gbmi.groupby(
        "gene_trait_uid",
        sort=True,
    ):
        row = group.iloc[0]

        eur_key = task_key_gbmi(row, "EUR")
        tasks.append(
            {
                "extraction_key": eur_key,
                "source": GBMI_SOURCE,
                "gene_trait_uid": clean(gene_trait_uid),
                "gene": clean(row["gene"]),
                "candidate_trait_key": clean(
                    row["candidate_trait_key"]
                ),
                "candidate_trait_name": clean(
                    row["candidate_trait_name"]
                ),
                "population": "EUR",
                "required_populations": [],
                "genome_build": clean(row["genome_build"]),
                "data_url": to_https(clean(row["eur_data_url"])),
                "index_url": to_https(clean(row["eur_index_url"])),
                "chrom": clean(row["chrom"]),
                "chrom_ucsc": clean(row["chrom_ucsc"]),
                "window_start_0based": int(
                    row["window_start_0based"]
                ),
                "window_start_1based": int(
                    row["window_start_1based"]
                ),
                "window_end_1based": int(
                    row["window_end_1based"]
                ),
                "unit_analysis_role": clean(
                    row["unit_analysis_role"]
                ),
                "mixed_pool": as_bool(row["mixed_pool"]),
            }
        )

        for population, pop_group in group.groupby(
            "comparison_population",
            sort=True,
        ):
            pop_row = pop_group.iloc[0]
            key = task_key_gbmi(pop_row, population)
            tasks.append(
                {
                    "extraction_key": key,
                    "source": GBMI_SOURCE,
                    "gene_trait_uid": clean(gene_trait_uid),
                    "gene": clean(pop_row["gene"]),
                    "candidate_trait_key": clean(
                        pop_row["candidate_trait_key"]
                    ),
                    "candidate_trait_name": clean(
                        pop_row["candidate_trait_name"]
                    ),
                    "population": clean(population).upper(),
                    "required_populations": [],
                    "genome_build": clean(
                        pop_row["genome_build"]
                    ),
                    "data_url": to_https(
                        clean(pop_row["comparison_data_url"])
                    ),
                    "index_url": to_https(
                        clean(pop_row["comparison_index_url"])
                    ),
                    "chrom": clean(pop_row["chrom"]),
                    "chrom_ucsc": clean(
                        pop_row["chrom_ucsc"]
                    ),
                    "window_start_0based": int(
                        pop_row["window_start_0based"]
                    ),
                    "window_start_1based": int(
                        pop_row["window_start_1based"]
                    ),
                    "window_end_1based": int(
                        pop_row["window_end_1based"]
                    ),
                    "unit_analysis_role": clean(
                        pop_row["unit_analysis_role"]
                    ),
                    "mixed_pool": as_bool(
                        pop_row["mixed_pool"]
                    ),
                }
            )

    task_frame = pd.DataFrame(tasks)
    duplicate = task_frame[
        task_frame["extraction_key"].duplicated(False)
    ]
    if len(duplicate):
        conflicting = (
            duplicate.groupby("extraction_key")[
                [
                    "data_url",
                    "index_url",
                    "chrom",
                    "window_start_0based",
                    "window_end_1based",
                ]
            ]
            .nunique()
        )
        if (conflicting > 1).any(axis=None):
            raise SystemExit(
                "Conflicting duplicate extraction tasks:\n"
                + duplicate.to_string(index=False)
            )
        task_frame = task_frame.drop_duplicates(
            "extraction_key",
            keep="first",
        )

    comparison_map_rows = []
    for _, row in manifest.iterrows():
        if clean(row["candidate_source"]) == PANUKB_SOURCE:
            extraction_key = task_key_panukb(row)
            eur_key = extraction_key
            comparison_key = extraction_key
        else:
            eur_key = task_key_gbmi(row, "EUR")
            comparison_key = task_key_gbmi(
                row,
                clean(row["comparison_population"]),
            )

        comparison_map_rows.append(
            {
                "comparison_uid": clean(row["comparison_uid"]),
                "gene_trait_uid": clean(row["gene_trait_uid"]),
                "gene": clean(row["gene"]),
                "candidate_source": clean(
                    row["candidate_source"]
                ),
                "candidate_trait_key": clean(
                    row["candidate_trait_key"]
                ),
                "candidate_trait_name": clean(
                    row["candidate_trait_name"]
                ),
                "comparison_population": clean(
                    row["comparison_population"]
                ),
                "unit_analysis_role": clean(
                    row["unit_analysis_role"]
                ),
                "mixed_pool": as_bool(row["mixed_pool"]),
                "primary_approval_eligible": as_bool(
                    row["primary_approval_eligible"]
                ),
                "eur_extraction_key": eur_key,
                "comparison_extraction_key": comparison_key,
            }
        )

    comparison_map = pd.DataFrame(comparison_map_rows)
    if comparison_map["comparison_uid"].duplicated().any():
        raise SystemExit(
            "comparison_uid is not unique in extraction map."
        )

    return (
        task_frame.sort_values(
            [
                "source",
                "data_url",
                "chrom",
                "window_start_0based",
                "extraction_key",
            ]
        ).to_dict("records"),
        comparison_map,
    )


def inventory_row_from_sidecar(
    sidecar: dict[str, Any],
) -> dict[str, Any]:
    return {
        "extraction_key": clean(sidecar.get("extraction_key")),
        "source": clean(sidecar.get("source")),
        "gene_trait_uid": clean(sidecar.get("gene_trait_uid")),
        "gene": clean(sidecar.get("gene")),
        "candidate_trait_key": clean(
            sidecar.get("candidate_trait_key")
        ),
        "candidate_trait_name": clean(
            sidecar.get("candidate_trait_name")
        ),
        "population": clean(sidecar.get("population")),
        "genome_build": clean(sidecar.get("genome_build")),
        "unit_analysis_role": clean(
            sidecar.get("unit_analysis_role")
        ),
        "mixed_pool": as_bool(sidecar.get("mixed_pool")),
        "status": clean(sidecar.get("status")),
        "cache_action": clean(sidecar.get("cache_action")),
        "data_url": clean(sidecar.get("data_url")),
        "index_url": clean(sidecar.get("index_url")),
        "local_index_path": clean(
            sidecar.get("local_index_path")
        ),
        "local_index_sha256": clean(
            sidecar.get("local_index_sha256")
        ),
        "chrom": clean(sidecar.get("chrom")),
        "tabix_contig": clean(sidecar.get("tabix_contig")),
        "contig_match_method": clean(
            sidecar.get("contig_match_method")
        ),
        "window_start_1based": sidecar.get(
            "window_start_1based"
        ),
        "window_end_1based": sidecar.get(
            "window_end_1based"
        ),
        "n_raw_records": sidecar.get("n_raw_records"),
        "n_parsed_records": sidecar.get(
            "n_parsed_records"
        ),
        "n_malformed_records": sidecar.get(
            "n_malformed_records"
        ),
        "n_unique_variant_keys": sidecar.get(
            "n_unique_variant_keys"
        ),
        "n_duplicate_variant_key_rows": sidecar.get(
            "n_duplicate_variant_key_rows"
        ),
        "minimum_position": sidecar.get("minimum_position"),
        "maximum_position": sidecar.get("maximum_position"),
        "n_outside_locked_window": sidecar.get(
            "n_outside_locked_window"
        ),
        "header_method": clean(sidecar.get("header_method")),
        "n_columns": sidecar.get("n_columns"),
        "output_parquet": clean(sidecar.get("output_parquet")),
        "output_sidecar": clean(sidecar.get("output_sidecar")),
        "output_size_bytes": sidecar.get(
            "output_size_bytes"
        ),
        "output_sha256": clean(sidecar.get("output_sha256")),
        "failure_reason": clean(sidecar.get("failure_reason")),
        "completed_at_utc": clean(
            sidecar.get("completed_at_utc")
        ),
    }


def process_file_group(
    tasks: list[dict[str, Any]],
    *,
    expected_headers: dict[str, list[str]],
    extract_dir: Path,
    index_dir: Path,
    timeout: float,
    max_records: int,
    force: bool,
) -> list[dict[str, Any]]:
    first = tasks[0]
    source = first["source"]
    data_url = first["data_url"]
    index_url = first["index_url"]

    local_index = index_local_path(
        index_dir,
        source=source,
        url=index_url,
    )
    index_metadata = download_file(
        index_url,
        local_index,
        timeout=timeout,
    )

    tabix = pysam.TabixFile(
        data_url,
        index=str(local_index),
    )

    inventory: list[dict[str, Any]] = []
    try:
        columns, header_method = header_from_tabix_or_stream(
            tabix,
            data_url,
            timeout=timeout,
        )

        expected = expected_headers[source]
        if source == PANUKB_SOURCE:
            required_populations = sorted(
                {
                    population
                    for task in tasks
                    for population in task.get(
                        "required_populations",
                        [],
                    )
                }
            )
            panukb_schema_check = validate_panukb_header(
                columns,
                required_populations=required_populations,
            )
            if panukb_schema_check["status"] != "PASS":
                raise RuntimeError(
                    "Pan-UKB header failed flexible ancestry-block "
                    "validation.\n"
                    + json.dumps(
                        panukb_schema_check,
                        indent=2,
                    )
                    + f"\nObserved: {columns}"
                )
        elif [normalize_column(x) for x in columns] != [
            normalize_column(x) for x in expected
        ]:
            raise RuntimeError(
                f"{source} header differs from the Step 15B2 "
                "validated schema.\n"
                f"Expected: {expected}\nObserved: {columns}"
            )

        for task in tasks:
            parquet_path, sidecar_path = extraction_paths(
                extract_dir,
                source=source,
                extraction_key=task["extraction_key"],
            )

            if (
                not force
                and cache_is_valid(
                    parquet_path,
                    sidecar_path,
                    expected_key=task["extraction_key"],
                )
            ):
                sidecar = read_json(sidecar_path)
                sidecar["cache_action"] = "REUSED"
                inventory.append(
                    inventory_row_from_sidecar(sidecar)
                )
                continue

            started = utc_now()
            try:
                contig, contig_method = contig_for_target(
                    tabix,
                    task["chrom"],
                    task["chrom_ucsc"],
                )
                records = fetch_records(
                    tabix,
                    contig=contig,
                    start_0based=task["window_start_0based"],
                    end_1based=task["window_end_1based"],
                    max_records=max_records,
                )
                frame, malformed = parse_records(
                    records,
                    columns,
                )

                if malformed:
                    raise RuntimeError(
                        f"{malformed:,} rows did not match the "
                        "validated column count."
                    )

                key_summary = raw_variant_key_summary(
                    frame,
                    source=source,
                )
                min_pos = key_summary["minimum_position"]
                max_pos = key_summary["maximum_position"]
                outside = 0

                if min_pos is not None and (
                    min_pos < task["window_start_1based"]
                ):
                    pos_col = key_summary["position_column"]
                    numeric_pos = pd.to_numeric(
                        frame[pos_col],
                        errors="coerce",
                    )
                    outside += int(
                        (
                            numeric_pos
                            < task["window_start_1based"]
                        ).sum()
                    )
                if max_pos is not None and (
                    max_pos > task["window_end_1based"]
                ):
                    pos_col = key_summary["position_column"]
                    numeric_pos = pd.to_numeric(
                        frame[pos_col],
                        errors="coerce",
                    )
                    outside += int(
                        (
                            numeric_pos
                            > task["window_end_1based"]
                        ).sum()
                    )

                if outside:
                    raise RuntimeError(
                        f"{outside:,} returned variants fall outside "
                        "the locked interval."
                    )

                # Preserve all source columns as strings. Step 15B4 will
                # perform typed conversion and harmonization.
                atomic_parquet_write(frame, parquet_path)
                output_sha256 = sha256_file(parquet_path)

                status = (
                    "EXTRACTED"
                    if len(frame)
                    else "EMPTY_REGION"
                )
                sidecar = {
                    "script_version": SCRIPT_VERSION,
                    "extraction_key": task["extraction_key"],
                    "source": source,
                    "gene_trait_uid": task["gene_trait_uid"],
                    "gene": task["gene"],
                    "candidate_trait_key": (
                        task["candidate_trait_key"]
                    ),
                    "candidate_trait_name": (
                        task["candidate_trait_name"]
                    ),
                    "population": task["population"],
                    "genome_build": task["genome_build"],
                    "unit_analysis_role": (
                        task["unit_analysis_role"]
                    ),
                    "mixed_pool": task["mixed_pool"],
                    "status": status,
                    "cache_action": "EXTRACTED_NOW",
                    "data_url": data_url,
                    "index_url": index_url,
                    "local_index_path": str(local_index),
                    "local_index_sha256": (
                        index_metadata["sha256"]
                    ),
                    "chrom": task["chrom"],
                    "tabix_contig": contig,
                    "contig_match_method": contig_method,
                    "window_start_0based": (
                        task["window_start_0based"]
                    ),
                    "window_start_1based": (
                        task["window_start_1based"]
                    ),
                    "window_end_1based": (
                        task["window_end_1based"]
                    ),
                    "header_method": header_method,
                    "columns": columns,
                    "n_columns": len(columns),
                    "panukb_present_populations": (
                        panukb_schema_check[
                            "present_populations"
                        ]
                        if source == PANUKB_SOURCE
                        else []
                    ),
                    "panukb_required_populations": (
                        panukb_schema_check[
                            "required_populations"
                        ]
                        if source == PANUKB_SOURCE
                        else []
                    ),
                    "n_raw_records": len(records),
                    "n_parsed_records": len(frame),
                    "n_malformed_records": malformed,
                    "n_unique_variant_keys": (
                        key_summary["n_unique_variant_keys"]
                    ),
                    "n_duplicate_variant_key_rows": (
                        key_summary[
                            "n_duplicate_variant_key_rows"
                        ]
                    ),
                    "minimum_position": min_pos,
                    "maximum_position": max_pos,
                    "n_outside_locked_window": outside,
                    "output_parquet": str(parquet_path),
                    "output_sidecar": str(sidecar_path),
                    "output_size_bytes": (
                        parquet_path.stat().st_size
                    ),
                    "output_sha256": output_sha256,
                    "failure_reason": "",
                    "started_at_utc": started,
                    "completed_at_utc": utc_now(),
                }
                atomic_json_write(sidecar, sidecar_path)
                inventory.append(
                    inventory_row_from_sidecar(sidecar)
                )

            except Exception as exc:
                parquet_path.unlink(missing_ok=True)
                failure_sidecar = {
                    "script_version": SCRIPT_VERSION,
                    "extraction_key": task["extraction_key"],
                    "source": source,
                    "gene_trait_uid": task["gene_trait_uid"],
                    "gene": task["gene"],
                    "candidate_trait_key": (
                        task["candidate_trait_key"]
                    ),
                    "candidate_trait_name": (
                        task["candidate_trait_name"]
                    ),
                    "population": task["population"],
                    "genome_build": task["genome_build"],
                    "unit_analysis_role": (
                        task["unit_analysis_role"]
                    ),
                    "mixed_pool": task["mixed_pool"],
                    "status": "FAILED",
                    "cache_action": "FAILED_NOW",
                    "data_url": data_url,
                    "index_url": index_url,
                    "local_index_path": str(local_index),
                    "local_index_sha256": (
                        index_metadata["sha256"]
                    ),
                    "chrom": task["chrom"],
                    "window_start_0based": (
                        task["window_start_0based"]
                    ),
                    "window_start_1based": (
                        task["window_start_1based"]
                    ),
                    "window_end_1based": (
                        task["window_end_1based"]
                    ),
                    "n_raw_records": None,
                    "n_parsed_records": None,
                    "n_malformed_records": None,
                    "n_unique_variant_keys": None,
                    "n_duplicate_variant_key_rows": None,
                    "minimum_position": None,
                    "maximum_position": None,
                    "n_outside_locked_window": None,
                    "header_method": header_method,
                    "n_columns": len(columns),
                    "output_parquet": str(parquet_path),
                    "output_sidecar": str(sidecar_path),
                    "output_size_bytes": None,
                    "output_sha256": "",
                    "failure_reason": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                    "started_at_utc": started,
                    "completed_at_utc": utc_now(),
                }
                atomic_json_write(
                    failure_sidecar,
                    sidecar_path,
                )
                inventory.append(
                    inventory_row_from_sidecar(
                        failure_sidecar
                    )
                )
    finally:
        tabix.close()

    return inventory


def write_final_outputs(
    inventory: pd.DataFrame,
    comparison_map: pd.DataFrame,
    *,
    extract_dir: Path,
    index_dir: Path,
    output_dir: Path,
    n_expected_tasks: int,
    n_processed_file_groups: int,
    n_total_file_groups: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory = inventory.sort_values(
        ["source", "candidate_trait_key", "gene", "population"]
    )
    inventory_csv = output_dir / "15b3_extraction_inventory.csv"
    inventory_parquet = (
        output_dir / "15b3_extraction_inventory.parquet"
    )
    inventory.to_csv(inventory_csv, index=False)
    inventory.to_parquet(
        inventory_parquet,
        index=False,
        compression="zstd",
    )

    path_lookup = inventory.set_index("extraction_key")[
        ["status", "output_parquet"]
    ].to_dict("index")

    comparison_map = comparison_map.copy()
    comparison_map["eur_extraction_status"] = (
        comparison_map["eur_extraction_key"].map(
            lambda key: path_lookup.get(key, {}).get(
                "status",
                "NOT_PROCESSED",
            )
        )
    )
    comparison_map["eur_extraction_parquet"] = (
        comparison_map["eur_extraction_key"].map(
            lambda key: path_lookup.get(key, {}).get(
                "output_parquet",
                "",
            )
        )
    )
    comparison_map["comparison_extraction_status"] = (
        comparison_map["comparison_extraction_key"].map(
            lambda key: path_lookup.get(key, {}).get(
                "status",
                "NOT_PROCESSED",
            )
        )
    )
    comparison_map["comparison_extraction_parquet"] = (
        comparison_map["comparison_extraction_key"].map(
            lambda key: path_lookup.get(key, {}).get(
                "output_parquet",
                "",
            )
        )
    )
    comparison_map["both_extractions_available"] = (
        comparison_map["eur_extraction_status"].isin(
            ["EXTRACTED", "EMPTY_REGION"]
        )
        & comparison_map[
            "comparison_extraction_status"
        ].isin(["EXTRACTED", "EMPTY_REGION"])
    )

    comparison_csv = (
        output_dir / "15b3_comparison_extraction_map.csv"
    )
    comparison_parquet = (
        output_dir / "15b3_comparison_extraction_map.parquet"
    )
    comparison_map.to_csv(comparison_csv, index=False)
    comparison_map.to_parquet(
        comparison_parquet,
        index=False,
        compression="zstd",
    )

    failures = inventory[
        inventory["status"].eq("FAILED")
    ].copy()
    failures_csv = output_dir / "15b3_extraction_failures.csv"
    failures.to_csv(failures_csv, index=False)

    counts = {
        str(key): int(value)
        for key, value in inventory["status"]
        .value_counts(dropna=False)
        .items()
    }
    by_source = {
        str(source): {
            str(key): int(value)
            for key, value in group["status"]
            .value_counts(dropna=False)
            .items()
        }
        for source, group in inventory.groupby("source")
    }

    summary = {
        "step": "15B3",
        "script_version": SCRIPT_VERSION,
        "status": (
            "PASS"
            if len(failures) == 0
            and len(inventory) == n_expected_tasks
            else (
                "PARTIAL"
                if len(failures) == 0
                else "FAIL"
            )
        ),
        "scientific_scope": (
            "Source-native regional extraction only; no QC or "
            "harmonization."
        ),
        "region_definition": "locked GENCODE gene body ±100 kb",
        "native_builds": {
            "Pan-UKB": "GRCh37",
            "GBMI": "GRCh38",
        },
        "n_expected_extraction_tasks": n_expected_tasks,
        "n_inventory_rows": int(len(inventory)),
        "n_total_remote_file_groups": n_total_file_groups,
        "n_processed_remote_file_groups": (
            n_processed_file_groups
        ),
        "status_counts": counts,
        "status_counts_by_source": by_source,
        "n_comparison_rows": int(len(comparison_map)),
        "n_comparisons_with_both_extractions": int(
            comparison_map[
                "both_extractions_available"
            ].sum()
        ),
        "n_failures": int(len(failures)),
        "n_empty_regions": int(
            inventory["status"].eq("EMPTY_REGION").sum()
        ),
        "total_raw_records": int(
            pd.to_numeric(
                inventory["n_raw_records"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        ),
        "cache_actions": {
            str(key): int(value)
            for key, value in inventory["cache_action"]
            .value_counts(dropna=False)
            .items()
        },
        "directories": {
            "extract_dir": str(extract_dir),
            "index_dir": str(index_dir),
        },
        "outputs": {
            "inventory_csv": str(inventory_csv),
            "inventory_parquet": str(inventory_parquet),
            "comparison_map_csv": str(comparison_csv),
            "comparison_map_parquet": str(
                comparison_parquet
            ),
            "failures_csv": str(failures_csv),
        },
        "completed_at_utc": utc_now(),
    }

    summary_path = output_dir / "15b3_extraction_summary.json"
    atomic_json_write(summary, summary_path)
    summary["outputs"]["summary_json"] = str(summary_path)
    return summary


def main() -> int:
    args = parse_args()
    args.extract_dir.mkdir(parents=True, exist_ok=True)
    args.index_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    required = [
        args.manifest,
        args.schema_summary,
        args.panukb_schema,
        args.gbmi_schema,
    ]
    for path in required:
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")

    schema_summary = read_json(args.schema_summary)
    panukb_schema = read_json(args.panukb_schema)
    gbmi_schema = read_json(args.gbmi_schema)

    if clean(schema_summary.get("status")) != "PASS":
        raise SystemExit(
            "Step 15B2 schema summary is not PASS."
        )
    if clean(panukb_schema.get("status")) != "PASS":
        raise SystemExit(
            "Pan-UKB Step 15B2 schema validation is not PASS."
        )
    if clean(gbmi_schema.get("status")) != "PASS":
        raise SystemExit(
            "GBMI Step 15B2 schema validation is not PASS."
        )

    expected_headers = {
        PANUKB_SOURCE: panukb_schema["header"]["columns"],
        GBMI_SOURCE: gbmi_schema["datasets"]["EUR"][
            "header"
        ]["columns"],
    }

    gbmi_non_eur = clean(
        gbmi_schema["comparison_population"]
    )
    gbmi_non_eur_header = gbmi_schema["datasets"][
        gbmi_non_eur
    ]["header"]["columns"]
    if [
        normalize_column(x)
        for x in expected_headers[GBMI_SOURCE]
    ] != [
        normalize_column(x)
        for x in gbmi_non_eur_header
    ]:
        raise SystemExit(
            "Validated GBMI EUR and non-EUR headers differ."
        )

    manifest = pd.read_parquet(args.manifest)
    if not manifest["extraction_ready"].map(as_bool).all():
        raise SystemExit(
            "The Step 15B1 manifest contains non-ready rows."
        )

    tasks, comparison_map = create_tasks(manifest)
    n_expected_tasks = len(tasks)

    file_groups: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for task in tasks:
        key = (
            task["source"],
            task["data_url"],
            task["index_url"],
        )
        file_groups[key].append(task)

    ordered_groups = sorted(
        file_groups.items(),
        key=lambda item: item[0],
    )
    n_total_file_groups = len(ordered_groups)
    if args.max_file_groups > 0:
        ordered_groups = ordered_groups[
            : args.max_file_groups
        ]

    print("=" * 78)
    print("STEP 15B3 — NATIVE REGIONAL EXTRACTION")
    print("=" * 78)
    print(f"Expected extraction tasks:             {n_expected_tasks:,}")
    print(
        f"Unique remote file groups:             "
        f"{n_total_file_groups:,}"
    )
    print(
        f"Remote file groups in this run:        "
        f"{len(ordered_groups):,}"
    )
    print(f"Locked comparison rows:                {len(comparison_map):,}")
    print(f"Resume mode:                           {not args.force}")
    print("=" * 78)

    inventory_rows: list[dict[str, Any]] = []

    for group_number, ((source, data_url, _), group_tasks) in enumerate(
        ordered_groups,
        start=1,
    ):
        trait_names = sorted(
            {
                task["candidate_trait_name"]
                for task in group_tasks
            }
        )
        display_trait = (
            trait_names[0]
            if len(trait_names) == 1
            else f"{len(trait_names)} traits"
        )
        print(
            f"[{group_number:>3}/{len(ordered_groups):>3}] "
            f"{source:<8} | {display_trait} | "
            f"{len(group_tasks):>3} extracts"
        )

        try:
            rows = process_file_group(
                group_tasks,
                expected_headers=expected_headers,
                extract_dir=args.extract_dir,
                index_dir=args.index_dir,
                timeout=args.timeout,
                max_records=args.max_records_per_region,
                force=args.force,
            )
            inventory_rows.extend(rows)
        except Exception as exc:
            # A file-level access or schema failure affects every task in
            # that remote file group. Record all tasks explicitly.
            for task in group_tasks:
                parquet_path, sidecar_path = extraction_paths(
                    args.extract_dir,
                    source=task["source"],
                    extraction_key=task["extraction_key"],
                )
                failure = {
                    "script_version": SCRIPT_VERSION,
                    "extraction_key": task["extraction_key"],
                    "source": task["source"],
                    "gene_trait_uid": task["gene_trait_uid"],
                    "gene": task["gene"],
                    "candidate_trait_key": (
                        task["candidate_trait_key"]
                    ),
                    "candidate_trait_name": (
                        task["candidate_trait_name"]
                    ),
                    "population": task["population"],
                    "genome_build": task["genome_build"],
                    "unit_analysis_role": (
                        task["unit_analysis_role"]
                    ),
                    "mixed_pool": task["mixed_pool"],
                    "status": "FAILED",
                    "cache_action": "FAILED_FILE_GROUP",
                    "data_url": task["data_url"],
                    "index_url": task["index_url"],
                    "local_index_path": "",
                    "local_index_sha256": "",
                    "chrom": task["chrom"],
                    "window_start_0based": (
                        task["window_start_0based"]
                    ),
                    "window_start_1based": (
                        task["window_start_1based"]
                    ),
                    "window_end_1based": (
                        task["window_end_1based"]
                    ),
                    "n_raw_records": None,
                    "n_parsed_records": None,
                    "n_malformed_records": None,
                    "n_unique_variant_keys": None,
                    "n_duplicate_variant_key_rows": None,
                    "minimum_position": None,
                    "maximum_position": None,
                    "n_outside_locked_window": None,
                    "header_method": "",
                    "n_columns": None,
                    "output_parquet": str(parquet_path),
                    "output_sidecar": str(sidecar_path),
                    "output_size_bytes": None,
                    "output_sha256": "",
                    "failure_reason": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                    "started_at_utc": utc_now(),
                    "completed_at_utc": utc_now(),
                }
                atomic_json_write(failure, sidecar_path)
                inventory_rows.append(
                    inventory_row_from_sidecar(failure)
                )
            print(
                f"    FILE-GROUP FAILURE: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    inventory = pd.DataFrame(inventory_rows)

    # In a limited debugging run, include valid cached sidecars for tasks
    # outside the current file-group limit where available.
    processed_keys = set(
        inventory["extraction_key"]
        if len(inventory)
        else []
    )
    for task in tasks:
        key = task["extraction_key"]
        if key in processed_keys:
            continue
        parquet_path, sidecar_path = extraction_paths(
            args.extract_dir,
            source=task["source"],
            extraction_key=key,
        )
        if cache_is_valid(
            parquet_path,
            sidecar_path,
            expected_key=key,
        ):
            sidecar = read_json(sidecar_path)
            sidecar["cache_action"] = "REUSED_OUTSIDE_RUN_LIMIT"
            inventory_rows.append(
                inventory_row_from_sidecar(sidecar)
            )

    inventory = pd.DataFrame(inventory_rows)
    if len(inventory) and inventory["extraction_key"].duplicated().any():
        raise SystemExit(
            "Extraction inventory contains duplicate keys."
        )

    summary = write_final_outputs(
        inventory,
        comparison_map,
        extract_dir=args.extract_dir,
        index_dir=args.index_dir,
        output_dir=args.output_dir,
        n_expected_tasks=n_expected_tasks,
        n_processed_file_groups=len(ordered_groups),
        n_total_file_groups=n_total_file_groups,
    )

    print("=" * 78)
    print("STEP 15B3 — EXTRACTION SUMMARY")
    print("=" * 78)
    print(
        f"Overall status:                        "
        f"{summary['status']}"
    )
    print(
        f"Inventory rows:                        "
        f"{summary['n_inventory_rows']:,}/"
        f"{summary['n_expected_extraction_tasks']:,}"
    )
    for status, count in sorted(
        summary["status_counts"].items()
    ):
        print(f"{status + ':':<39}{count:>10,}")
    print(
        f"Locked comparisons with both extracts: "
        f"{summary['n_comparisons_with_both_extractions']:,}/"
        f"{summary['n_comparison_rows']:,}"
    )
    print(
        f"Total source-native regional rows:     "
        f"{summary['total_raw_records']:,}"
    )
    print(
        f"Failures:                              "
        f"{summary['n_failures']:,}"
    )
    print(
        f"Empty regions:                         "
        f"{summary['n_empty_regions']:,}"
    )
    print()
    print("No QC, harmonization, slopes, or PP4 were calculated.")
    print("=" * 78)

    if summary["status"] == "PASS":
        return 0
    if summary["status"] == "PARTIAL":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
