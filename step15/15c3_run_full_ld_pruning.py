#!/usr/bin/env python3
"""
Step 15C3 — Full ancestry-paired LD pruning.

Applies the frozen Step 15C1 pruning rule to every primary,
nonmixed, pre-LD-eligible portability comparison:

    1. Rank candidate variants by EUR association p-value, then
       position, REF, and ALT.
    2. Resolve exact variants in both ancestry-specific LD references.
    3. Greedily retain a candidate only when its pairwise r² is
       strictly below the threshold against every previously retained
       variant in BOTH ancestry references.
    4. Evaluate thresholds r² < 0.01, r² < 0.10, and r² < 0.20.
    5. Require at least three retained variants for a comparison to
       enter the corresponding portability-analysis universe.

LD references
-------------
Pan-UKB / GRCh37:
    Public ancestry-specific Pan-UKB variant-index Hail Tables and
    covariate-adjusted LD BlockMatrices.

GBMI / GRCh38:
    Official phased 1000 Genomes GRCh38 chromosome VCFs, using EUR and
    the frozen comparison reference population from Step 15C1.

The script is resumable. Each comparison's exact reference resolution
and ancestry-specific r² matrices are cached under output/15c3_cache.
Re-running the script skips valid completed caches unless
--rebuild-cache is supplied.

This step performs LD pruning only. It does not estimate portability
slopes.

Required environment for the Pan-UKB branch
--------------------------------------------
    export PYSPARK_SUBMIT_ARGS="--packages org.apache.hadoop:hadoop-aws:3.3.4 pyspark-shell"
    export SPARK_LOCAL_IP=127.0.0.1
    export SPARK_LOCAL_DIRS=/tmp/target_ancestry_spark

Default inputs
--------------
output/15c1_portability_comparison_manifest.parquet
output/15c1_portability_variant_requests_primary.parquet

Primary outputs
---------------
output/15c3_variant_reference_resolution.parquet
output/15c3_variant_pruning_decisions.parquet
output/15c3_comparison_pruning_results.parquet
output/15c3_portability_variants_primary_retained.parquet
output/15c3_portability_comparisons_primary_final.parquet
output/15c3_portability_variants_r2_0p01_retained.parquet
output/15c3_portability_variants_r2_0p20_retained.parquet
output/15c3_ld_pruning_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import traceback
import urllib.request
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pysam


SCRIPT_VERSION = "15C3.2"
CACHE_COMPATIBLE_VERSIONS = {"15C3.1", "15C3.2"}
GBMI_REMOTE_FETCH_ATTEMPTS = 4
GBMI_REMOTE_FETCH_BASE_DELAY_SECONDS = 15
PANUKB = "Pan-UKB"
GBMI = "GBMI"
DEFAULT_THRESHOLDS = (0.01, 0.10, 0.20)
PRIMARY_THRESHOLD = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "output/15c1_portability_comparison_manifest.parquet"
        ),
    )
    parser.add_argument(
        "--requests",
        type=Path,
        default=Path(
            "output/15c1_portability_variant_requests_primary.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("input/reference/1000G"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("output/15c3_cache"),
    )
    parser.add_argument(
        "--thresholds",
        default="0.01,0.10,0.20",
    )
    parser.add_argument(
        "--primary-threshold",
        type=float,
        default=PRIMARY_THRESHOLD,
    )
    parser.add_argument(
        "--minimum-retained",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--source",
        choices=["both", "panukb", "gbmi"],
        default="both",
        help=(
            "Process both sources or only one branch. Final outputs "
            "require valid caches for every manifest comparison."
        ),
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignore and overwrite valid comparison caches.",
    )
    parser.add_argument(
        "--hail-master",
        default="local[2]",
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        default=0,
        help=(
            "Optional debugging limit on newly processed comparisons. "
            "Zero means no limit."
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
    return clean(value).casefold() in {
        "true",
        "1",
        "yes",
        "y",
    }


def normalize_chrom(value: Any) -> str:
    chrom = clean(value)
    if chrom.casefold().startswith("chr"):
        chrom = chrom[3:]
    if chrom == "M":
        chrom = "MT"
    return chrom


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_thresholds(text: str) -> tuple[float, ...]:
    values = []
    for token in text.split(","):
        value = float(token.strip())
        if not 0.0 < value < 1.0:
            raise SystemExit(
                f"LD threshold must be between 0 and 1: {value}"
            )
        values.append(value)
    unique = tuple(sorted(set(values)))
    if not unique:
        raise SystemExit("At least one LD threshold is required.")
    return unique


def threshold_label(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(
            f"{label} missing required columns: {missing}"
        )


def atomic_json_write(
    data: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def download_file(
    url: str,
    destination: Path,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        return {
            "url": url,
            "path": str(destination),
            "downloaded_now": False,
            "size_bytes": int(destination.stat().st_size),
            "sha256": sha256_file(destination),
        }

    temporary = destination.with_suffix(
        destination.suffix + ".part"
    )
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "target-ancestry-step15c3/1.0",
        },
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=120,
        ) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    temporary.replace(destination)
    return {
        "url": url,
        "path": str(destination),
        "downloaded_now": True,
        "size_bytes": int(destination.stat().st_size),
        "sha256": sha256_file(destination),
    }


def to_s3a(path: Any) -> str:
    text = clean(path)
    if text.startswith("s3://"):
        return "s3a://" + text[len("s3://") :]
    if text.startswith("s3a://"):
        return text
    raise RuntimeError(
        f"Expected a Pan-UKB S3 path, observed: {text}"
    )


def sorted_requests(
    requests: pd.DataFrame,
    comparison_uid: str,
) -> pd.DataFrame:
    selected = requests[
        requests["comparison_uid"].eq(comparison_uid)
    ].copy()
    selected = selected.sort_values(
        [
            "eur_p_rank",
            "eur_p",
            "pos",
            "ref",
            "alt",
            "variant_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    selected["request_order"] = np.arange(
        len(selected),
        dtype=int,
    )
    if selected["variant_id"].duplicated().any():
        duplicates = selected.loc[
            selected["variant_id"].duplicated(
                keep=False
            ),
            "variant_id",
        ].tolist()
        raise RuntimeError(
            "Duplicate variant IDs within comparison "
            f"{comparison_uid}: {duplicates[:10]}"
        )
    return selected


def cache_paths(
    cache_dir: Path,
    comparison_uid: str,
) -> tuple[Path, Path]:
    safe_uid = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        comparison_uid,
    )
    return (
        cache_dir / f"{safe_uid}.json",
        cache_dir / f"{safe_uid}.npz",
    )


def cache_is_valid(
    metadata_path: Path,
    matrix_path: Path,
    selected: pd.DataFrame,
) -> bool:
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return False

    if metadata.get("script_version") not in CACHE_COMPATIBLE_VERSIONS:
        return False
    if metadata.get("request_variant_ids") != selected[
        "variant_id"
    ].tolist():
        return False

    status = metadata.get("processing_status")
    if status == "COMPLETE":
        return matrix_path.exists()
    if status in {
        "INSUFFICIENT_REFERENCE_VARIANTS",
        "PROCESSING_ERROR",
    }:
        return True
    return False


def save_complete_cache(
    metadata_path: Path,
    matrix_path: Path,
    *,
    metadata: dict[str, Any],
    usable_variant_ids: list[str],
    r2_eur: np.ndarray,
    r2_comparison: np.ndarray,
) -> None:
    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    matrix_temporary = matrix_path.with_suffix(
        matrix_path.suffix + ".part"
    )
    with matrix_temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            usable_variant_ids=np.asarray(
                usable_variant_ids,
                dtype=str,
            ),
            r2_eur=np.asarray(
                r2_eur,
                dtype=float,
            ),
            r2_comparison=np.asarray(
                r2_comparison,
                dtype=float,
            ),
        )
    matrix_temporary.replace(matrix_path)

    metadata = {
        **metadata,
        "script_version": SCRIPT_VERSION,
        "processing_status": "COMPLETE",
        "matrix_cache": str(matrix_path),
        "completed_at_utc": utc_now(),
    }
    atomic_json_write(metadata, metadata_path)


def save_nonmatrix_cache(
    metadata_path: Path,
    matrix_path: Path,
    *,
    metadata: dict[str, Any],
    status: str,
) -> None:
    matrix_path.unlink(missing_ok=True)
    metadata = {
        **metadata,
        "script_version": SCRIPT_VERSION,
        "processing_status": status,
        "matrix_cache": None,
        "completed_at_utc": utc_now(),
    }
    atomic_json_write(metadata, metadata_path)


def load_cache(
    metadata_path: Path,
    matrix_path: Path,
) -> dict[str, Any]:
    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )
    if metadata["processing_status"] == "COMPLETE":
        with np.load(
            matrix_path,
            allow_pickle=False,
        ) as archive:
            metadata["usable_variant_ids"] = (
                archive[
                    "usable_variant_ids"
                ].astype(str).tolist()
            )
            metadata["r2_eur"] = np.asarray(
                archive["r2_eur"],
                dtype=float,
            )
            metadata["r2_comparison"] = (
                np.asarray(
                    archive["r2_comparison"],
                    dtype=float,
                )
            )
    else:
        metadata["usable_variant_ids"] = []
        metadata["r2_eur"] = np.empty((0, 0))
        metadata["r2_comparison"] = (
            np.empty((0, 0))
        )
    return metadata


def reconstruct_panukb_symmetric(
    raw_sorted: np.ndarray,
) -> np.ndarray:
    if (
        raw_sorted.ndim != 2
        or raw_sorted.shape[0]
        != raw_sorted.shape[1]
    ):
        raise RuntimeError(
            "Pan-UKB LD subset is not square: "
            f"{raw_sorted.shape}"
        )
    return (
        raw_sorted
        + raw_sorted.T
        - np.diag(np.diag(raw_sorted))
    )


def matrix_diagnostics(
    matrix: np.ndarray,
) -> dict[str, Any]:
    if matrix.size == 0:
        return {
            "shape": list(matrix.shape),
            "all_entries_finite": True,
            "symmetric": True,
            "minimum": None,
            "maximum": None,
            "diagonal_minimum": None,
            "diagonal_maximum": None,
        }
    diagonal = np.diag(matrix)
    return {
        "shape": list(matrix.shape),
        "all_entries_finite": bool(
            np.isfinite(matrix).all()
        ),
        "symmetric": bool(
            np.allclose(
                matrix,
                matrix.T,
                atol=1e-10,
                equal_nan=True,
            )
        ),
        "minimum": float(np.nanmin(matrix)),
        "maximum": float(np.nanmax(matrix)),
        "diagonal_minimum": float(
            np.nanmin(diagonal)
        ),
        "diagonal_maximum": float(
            np.nanmax(diagonal)
        ),
    }


def build_hail_request_table(
    hl,
    request_frame: pd.DataFrame,
):
    frame = request_frame[
        [
            "variant_id",
            "chrom",
            "pos",
            "ref",
            "alt",
        ]
    ].drop_duplicates(
        "variant_id"
    ).copy()
    frame["chrom"] = frame[
        "chrom"
    ].map(normalize_chrom)
    frame["pos"] = pd.to_numeric(
        frame["pos"],
        errors="raise",
    ).astype(int)
    frame["ref"] = frame[
        "ref"
    ].astype(str).str.upper()
    frame["alt"] = frame[
        "alt"
    ].astype(str).str.upper()

    request_ht = hl.Table.from_pandas(frame)
    request_ht = request_ht.annotate(
        locus=hl.locus(
            request_ht.chrom,
            hl.int32(request_ht.pos),
            reference_genome="GRCh37",
        ),
        alleles=hl.array(
            [
                request_ht.ref,
                request_ht.alt,
            ]
        ),
    )
    return request_ht.key_by(
        "locus",
        "alleles",
    )


def resolve_panukb_population(
    hl,
    request_frame: pd.DataFrame,
    index_path: str,
) -> tuple[dict[str, int], dict[str, Any]]:
    request_ht = build_hail_request_table(
        hl,
        request_frame,
    )
    index_ht = hl.read_table(index_path)
    joined = request_ht.annotate(
        panel_idx=index_ht[
            request_ht.key
        ].idx
    )
    rows = joined.select(
        "variant_id",
        "panel_idx",
    ).collect()
    mapping = {
        clean(row.variant_id): int(row.panel_idx)
        for row in rows
        if row.panel_idx is not None
    }

    globals_value = hl.eval(
        index_ht.index_globals()
    )
    globals_dict = {
        field: globals_value[field]
        for field in globals_value
    }

    return mapping, {
        "index_path": index_path,
        "n_requested_unique": int(
            request_frame[
                "variant_id"
            ].nunique()
        ),
        "n_resolved_unique": int(
            len(mapping)
        ),
        "index_globals": globals_dict,
    }


def process_panukb(
    manifest: pd.DataFrame,
    requests: pd.DataFrame,
    *,
    cache_dir: Path,
    minimum_retained: int,
    rebuild_cache: bool,
    hail_master: str,
    stop_after: int,
) -> int:
    target_manifest = manifest[
        manifest["candidate_source"].eq(PANUKB)
    ].copy()
    if target_manifest.empty:
        return 0

    pending_rows = []
    selected_by_uid: dict[str, pd.DataFrame] = {}

    for row in target_manifest.itertuples(
        index=False
    ):
        comparison_uid = clean(
            row.comparison_uid
        )
        selected = sorted_requests(
            requests,
            comparison_uid,
        )
        selected_by_uid[comparison_uid] = selected
        metadata_path, matrix_path = (
            cache_paths(
                cache_dir,
                comparison_uid,
            )
        )
        valid = cache_is_valid(
            metadata_path,
            matrix_path,
            selected,
        )
        if rebuild_cache or not valid:
            pending_rows.append(row)

    if not pending_rows:
        print(
            "Pan-UKB: all comparison caches "
            "are already valid."
        )
        return 0

    try:
        import hail as hl
        import pyspark
        from hail.linalg import BlockMatrix
    except Exception as exc:
        raise RuntimeError(
            f"Hail import failed: {exc}"
        ) from exc

    print()
    print(
        "Pan-UKB pending comparisons:",
        len(pending_rows),
    )
    print("Hail:", hl.__version__)
    print("PySpark:", pyspark.__version__)

    os.environ.setdefault(
        "SPARK_LOCAL_IP",
        "127.0.0.1",
    )
    os.environ.setdefault(
        "SPARK_LOCAL_DIRS",
        "/tmp/target_ancestry_spark",
    )
    Path(
        "/tmp/target_ancestry_spark"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )
    Path(
        "/tmp/target_ancestry_hail"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    hl.init(
        master=hail_master,
        tmp_dir=(
            "file:///tmp/target_ancestry_hail"
        ),
        local_tmpdir=(
            "/tmp/target_ancestry_hail"
        ),
        log=(
            "/tmp/"
            "target_ancestry_15c3_panukb.log"
        ),
        spark_conf={
            "spark.hadoop.fs.s3a.impl":
                "org.apache.hadoop.fs.s3a."
                "S3AFileSystem",
            "spark.hadoop.fs.s3a."
            "aws.credentials.provider":
                "org.apache.hadoop.fs.s3a."
                "AnonymousAWSCredentialsProvider",
            "spark.hadoop.fs.s3a.endpoint":
                "s3.us-east-1.amazonaws.com",
            "spark.hadoop.fs.s3a."
            "endpoint.region":
                "us-east-1",
            "spark.hadoop.fs.s3a."
            "path.style.access":
                "false",
            "spark.hadoop.fs.s3a."
            "connection.ssl.enabled":
                "true",
        },
        quiet=False,
    )
    hl.default_reference("GRCh37")

    processed = 0
    try:
        pending_uids = {
            clean(row.comparison_uid)
            for row in pending_rows
        }
        pending_requests = requests[
            requests["comparison_uid"].isin(
                pending_uids
            )
        ].copy()

        populations = ["EUR"] + sorted(
            {
                clean(
                    row.comparison_population
                ).upper()
                for row in pending_rows
            }
        )

        index_maps: dict[
            str,
            dict[str, int],
        ] = {}
        index_metadata: dict[
            str,
            dict[str, Any],
        ] = {}
        matrices: dict[str, Any] = {}

        for population in populations:
            if population == "EUR":
                population_requests = (
                    pending_requests
                )
                sample_row = pending_rows[0]
                index_path = to_s3a(
                    sample_row.eur_variant_index
                )
                matrix_path = to_s3a(
                    sample_row.eur_ld_matrix
                )
            else:
                population_uids = {
                    clean(row.comparison_uid)
                    for row in pending_rows
                    if clean(
                        row.comparison_population
                    ).upper()
                    == population
                }
                population_requests = (
                    pending_requests[
                        pending_requests[
                            "comparison_uid"
                        ].isin(population_uids)
                    ]
                )
                sample_row = next(
                    row
                    for row in pending_rows
                    if clean(
                        row.comparison_population
                    ).upper()
                    == population
                )
                index_path = to_s3a(
                    sample_row.
                    comparison_variant_index
                )
                matrix_path = to_s3a(
                    sample_row.
                    comparison_ld_matrix
                )

            print()
            print(
                f"Resolving Pan-UKB {population} "
                "variant indices..."
            )
            mapping, metadata = (
                resolve_panukb_population(
                    hl,
                    population_requests,
                    index_path,
                )
            )
            print(
                "  resolved:",
                metadata[
                    "n_resolved_unique"
                ],
                "/",
                metadata[
                    "n_requested_unique"
                ],
            )
            index_maps[population] = mapping
            index_metadata[population] = {
                **metadata,
                "matrix_path": matrix_path,
            }
            matrices[population] = (
                BlockMatrix.read(matrix_path)
            )

        total = len(pending_rows)
        for ordinal, row in enumerate(
            pending_rows,
            start=1,
        ):
            comparison_uid = clean(
                row.comparison_uid
            )
            selected = selected_by_uid[
                comparison_uid
            ]
            comparison_population = clean(
                row.comparison_population
            ).upper()
            metadata_path, matrix_path = (
                cache_paths(
                    cache_dir,
                    comparison_uid,
                )
            )

            print(
                f"[Pan-UKB {ordinal}/{total}] "
                f"{comparison_uid} | "
                f"{clean(row.gene)} | "
                f"{comparison_population} | "
                f"{len(selected)} candidates"
            )

            try:
                eur_map = index_maps["EUR"]
                comparison_map = index_maps[
                    comparison_population
                ]

                resolution = []
                usable_variant_ids = []
                eur_indices = []
                comparison_indices = []

                for variant in selected.itertuples(
                    index=False
                ):
                    variant_id = clean(
                        variant.variant_id
                    )
                    eur_index = eur_map.get(
                        variant_id
                    )
                    comparison_index = (
                        comparison_map.get(
                            variant_id
                        )
                    )
                    eur_present = (
                        eur_index is not None
                    )
                    comparison_present = (
                        comparison_index
                        is not None
                    )
                    usable = (
                        eur_present
                        and comparison_present
                    )

                    if usable:
                        reason = "USABLE"
                        usable_variant_ids.append(
                            variant_id
                        )
                        eur_indices.append(
                            int(eur_index)
                        )
                        comparison_indices.append(
                            int(
                                comparison_index
                            )
                        )
                    elif (
                        not eur_present
                        and not comparison_present
                    ):
                        reason = (
                            "MISSING_BOTH_REFERENCES"
                        )
                    elif not eur_present:
                        reason = (
                            "MISSING_EUR_REFERENCE"
                        )
                    else:
                        reason = (
                            "MISSING_COMPARISON_REFERENCE"
                        )

                    resolution.append(
                        {
                            "variant_id": variant_id,
                            "request_order": int(
                                variant.request_order
                            ),
                            "reference_usable": (
                                usable
                            ),
                            "reference_exclusion_reason": (
                                reason
                            ),
                            "eur_panel_index": (
                                eur_index
                            ),
                            "comparison_panel_index": (
                                comparison_index
                            ),
                            "eur_reference_polymorphic": (
                                None
                            ),
                            "comparison_reference_polymorphic": (
                                None
                            ),
                        }
                    )

                base_metadata = {
                    "comparison_uid": (
                        comparison_uid
                    ),
                    "gene_trait_uid": clean(
                        row.gene_trait_uid
                    ),
                    "gene": clean(row.gene),
                    "trait": clean(
                        row.candidate_trait_name
                    ),
                    "candidate_source": PANUKB,
                    "comparison_population": (
                        comparison_population
                    ),
                    "comparison_reference_population": (
                        comparison_population
                    ),
                    "request_variant_ids": selected[
                        "variant_id"
                    ].tolist(),
                    "n_requested": int(
                        len(selected)
                    ),
                    "n_reference_usable": int(
                        len(usable_variant_ids)
                    ),
                    "reference_resolution": (
                        resolution
                    ),
                    "eur_reference": (
                        index_metadata["EUR"]
                    ),
                    "comparison_reference": (
                        index_metadata[
                            comparison_population
                        ]
                    ),
                }

                if (
                    len(usable_variant_ids)
                    < minimum_retained
                ):
                    save_nonmatrix_cache(
                        metadata_path,
                        matrix_path,
                        metadata=base_metadata,
                        status=(
                            "INSUFFICIENT_REFERENCE_VARIANTS"
                        ),
                    )
                    processed += 1
                    continue

                if (
                    len(set(eur_indices))
                    != len(eur_indices)
                    or len(
                        set(comparison_indices)
                    )
                    != len(comparison_indices)
                ):
                    raise RuntimeError(
                        "Duplicate panel indices "
                        "within comparison."
                    )

                def extract_r2(
                    matrix,
                    indices: list[int],
                ) -> np.ndarray:
                    order = np.argsort(
                        np.asarray(indices),
                        kind="stable",
                    )
                    sorted_indices = [
                        indices[int(position)]
                        for position in order
                    ]
                    restore = np.argsort(
                        order,
                        kind="stable",
                    )
                    raw_sorted = matrix.filter(
                        sorted_indices,
                        sorted_indices,
                    ).to_numpy()
                    symmetric_sorted = (
                        reconstruct_panukb_symmetric(
                            raw_sorted
                        )
                    )
                    symmetric = (
                        symmetric_sorted[
                            np.ix_(
                                restore,
                                restore,
                            )
                        ]
                    )
                    return np.square(symmetric)

                r2_eur = extract_r2(
                    matrices["EUR"],
                    eur_indices,
                )
                r2_comparison = extract_r2(
                    matrices[
                        comparison_population
                    ],
                    comparison_indices,
                )

                eur_diagnostics = (
                    matrix_diagnostics(r2_eur)
                )
                comparison_diagnostics = (
                    matrix_diagnostics(
                        r2_comparison
                    )
                )
                if (
                    not eur_diagnostics[
                        "all_entries_finite"
                    ]
                    or not comparison_diagnostics[
                        "all_entries_finite"
                    ]
                    or not eur_diagnostics[
                        "symmetric"
                    ]
                    or not comparison_diagnostics[
                        "symmetric"
                    ]
                ):
                    raise RuntimeError(
                        "Nonfinite or asymmetric "
                        "Pan-UKB r² subset."
                    )

                save_complete_cache(
                    metadata_path,
                    matrix_path,
                    metadata={
                        **base_metadata,
                        "eur_r2_diagnostics": (
                            eur_diagnostics
                        ),
                        "comparison_r2_diagnostics": (
                            comparison_diagnostics
                        ),
                    },
                    usable_variant_ids=(
                        usable_variant_ids
                    ),
                    r2_eur=r2_eur,
                    r2_comparison=(
                        r2_comparison
                    ),
                )

            except Exception as exc:
                save_nonmatrix_cache(
                    metadata_path,
                    matrix_path,
                    metadata={
                        "comparison_uid": (
                            comparison_uid
                        ),
                        "gene_trait_uid": clean(
                            row.gene_trait_uid
                        ),
                        "gene": clean(row.gene),
                        "trait": clean(
                            row.candidate_trait_name
                        ),
                        "candidate_source": (
                            PANUKB
                        ),
                        "comparison_population": (
                            comparison_population
                        ),
                        "comparison_reference_population": (
                            comparison_population
                        ),
                        "request_variant_ids": (
                            selected[
                                "variant_id"
                            ].tolist()
                        ),
                        "n_requested": int(
                            len(selected)
                        ),
                        "n_reference_usable": 0,
                        "reference_resolution": [],
                        "error_type": (
                            type(exc).__name__
                        ),
                        "error_message": str(exc),
                        "traceback": (
                            traceback.format_exc()
                        ),
                    },
                    status="PROCESSING_ERROR",
                )
                print(
                    "  ERROR:",
                    type(exc).__name__,
                    str(exc),
                )

            processed += 1
            if (
                stop_after > 0
                and processed >= stop_after
            ):
                print(
                    "Stopping after requested "
                    f"{stop_after} new comparisons."
                )
                break
    finally:
        hl.stop()

    return processed


def read_sample_panel(
    path: Path,
) -> pd.DataFrame:
    panel = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
    )
    panel.columns = [
        re.sub(
            r"[^a-z0-9]+",
            "_",
            clean(column).casefold(),
        ).strip("_")
        for column in panel.columns
    ]

    aliases = {
        "sample": [
            "sample",
            "sample_id",
        ],
        "population": [
            "pop",
            "population",
        ],
        "super_population": [
            "super_pop",
            "super_population",
        ],
    }
    resolved: dict[str, str] = {}
    for semantic, options in aliases.items():
        for option in options:
            if option in panel.columns:
                resolved[semantic] = option
                break

    missing = sorted(
        set(aliases) - set(resolved)
    )
    if missing:
        raise RuntimeError(
            "1000 Genomes sample panel "
            "missing fields "
            f"{missing}; observed="
            f"{panel.columns.tolist()}"
        )

    panel = panel.rename(
        columns={
            resolved["sample"]: "sample",
            resolved["population"]: (
                "population"
            ),
            resolved["super_population"]: (
                "super_population"
            ),
        }
    )
    for column in [
        "sample",
        "population",
        "super_population",
    ]:
        panel[column] = panel[
            column
        ].map(clean)
    return panel


def resolve_contig(
    vcf: pysam.VariantFile,
    chrom: str,
) -> str:
    target = normalize_chrom(chrom)
    contigs = list(vcf.header.contigs)
    for candidate in [
        target,
        f"chr{target}",
    ]:
        if candidate in contigs:
            return candidate

    normalized = {
        normalize_chrom(contig): contig
        for contig in contigs
    }
    if target in normalized:
        return normalized[target]

    raise RuntimeError(
        f"Chromosome {target} absent "
        "from VCF header."
    )


def allele_dosages(
    record,
    samples: list[str],
    alternate_allele_index: int,
) -> np.ndarray:
    dosages = np.full(
        len(samples),
        np.nan,
        dtype=float,
    )
    for index, sample in enumerate(samples):
        genotype = record.samples[
            sample
        ].get("GT")
        if (
            genotype is None
            or len(genotype) != 2
            or genotype[0] is None
            or genotype[1] is None
        ):
            continue
        dosages[index] = float(
            int(
                genotype[0]
                == alternate_allele_index
            )
            + int(
                genotype[1]
                == alternate_allele_index
            )
        )
    return dosages


def dosage_metrics(
    dosages: np.ndarray,
) -> dict[str, Any]:
    called = dosages[
        np.isfinite(dosages)
    ]
    n_total = int(dosages.size)
    n_called = int(called.size)
    if n_called == 0:
        return {
            "n_total": n_total,
            "n_called": 0,
            "call_rate": 0.0,
            "alternate_allele_frequency": (
                None
            ),
            "minor_allele_frequency": None,
            "dosage_variance": None,
            "polymorphic": False,
        }

    alternate_frequency = float(
        np.mean(called) / 2.0
    )
    minor_frequency = float(
        min(
            alternate_frequency,
            1.0 - alternate_frequency,
        )
    )
    variance = (
        float(np.var(called, ddof=1))
        if n_called > 1
        else 0.0
    )
    return {
        "n_total": n_total,
        "n_called": n_called,
        "call_rate": float(
            n_called / n_total
        ),
        "alternate_allele_frequency": (
            alternate_frequency
        ),
        "minor_allele_frequency": (
            minor_frequency
        ),
        "dosage_variance": variance,
        "polymorphic": bool(
            n_called >= 3
            and variance > 0.0
        ),
    }


def pairwise_correlation(
    genotype_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_variants = genotype_matrix.shape[0]
    correlation = np.full(
        (n_variants, n_variants),
        np.nan,
        dtype=float,
    )
    pairwise_n = np.zeros(
        (n_variants, n_variants),
        dtype=int,
    )

    for first in range(n_variants):
        for second in range(
            first,
            n_variants,
        ):
            x = genotype_matrix[first]
            y = genotype_matrix[second]
            complete = (
                np.isfinite(x)
                & np.isfinite(y)
            )
            n_complete = int(
                complete.sum()
            )
            pairwise_n[
                first,
                second,
            ] = n_complete
            pairwise_n[
                second,
                first,
            ] = n_complete

            if n_complete < 3:
                continue

            x_complete = x[complete]
            y_complete = y[complete]
            x_sd = float(
                np.std(
                    x_complete,
                    ddof=1,
                )
            )
            y_sd = float(
                np.std(
                    y_complete,
                    ddof=1,
                )
            )
            if x_sd == 0.0 or y_sd == 0.0:
                continue

            value = float(
                np.corrcoef(
                    x_complete,
                    y_complete,
                )[0, 1]
            )
            correlation[
                first,
                second,
            ] = value
            correlation[
                second,
                first,
            ] = value

    return correlation, pairwise_n


def open_remote_vcf(
    vcf_url: str,
    index_url: str,
) -> pysam.VariantFile:
    try:
        return pysam.VariantFile(
            vcf_url,
            index_filename=index_url,
        )
    except (TypeError, ValueError):
        return pysam.VariantFile(
            vcf_url
        )


def close_variant_file_safely(
    vcf: pysam.VariantFile | None,
) -> str | None:
    """
    Close a remote VariantFile without allowing an htslib close error
    to mask the original fetch result.
    """
    if vcf is None:
        return None
    try:
        vcf.close()
    except OSError as exc:
        return str(exc)
    return None


def fetch_gbmi_region_with_retries(
    *,
    vcf_url: str,
    index_url: str,
    selected: pd.DataFrame,
    eur_panel_samples: list[str],
    comparison_panel_samples: list[str],
    comparison_reference: str,
    max_attempts: int = GBMI_REMOTE_FETCH_ATTEMPTS,
    base_delay_seconds: int = (
        GBMI_REMOTE_FETCH_BASE_DELAY_SECONDS
    ),
) -> dict[str, Any]:
    """
    Fetch one indexed remote 1000 Genomes region with bounded retries.

    A short HTTP/range-read failure can surface from htslib as
    ``OSError: truncated file``. Each retry opens a fresh remote handle.
    A close-time Illegal seek is recorded but does not invalidate a
    fetch that iterated through the requested region successfully.
    """
    chromosomes = {
        normalize_chrom(value)
        for value in selected["chrom"]
    }
    if len(chromosomes) != 1:
        raise RuntimeError(
            "Comparison spans multiple chromosomes: "
            f"{chromosomes}"
        )
    chromosome = next(iter(chromosomes))

    positions = pd.to_numeric(
        selected["pos"],
        errors="raise",
    ).astype(int)
    start_zero_based = max(
        0,
        int(positions.min()) - 1,
    )
    end_one_based = int(positions.max())

    request_lookup: dict[
        tuple[str, int, str, str],
        str,
    ] = {
        (
            normalize_chrom(variant.chrom),
            int(variant.pos),
            clean(variant.ref).upper(),
            clean(variant.alt).upper(),
        ): clean(variant.variant_id)
        for variant in selected.itertuples(
            index=False
        )
    }

    failures: list[dict[str, Any]] = []

    for attempt in range(1, max_attempts + 1):
        vcf: pysam.VariantFile | None = None
        fetch_completed = False
        close_warning: str | None = None

        try:
            vcf = open_remote_vcf(
                vcf_url,
                index_url,
            )
            available_samples = set(
                vcf.header.samples
            )
            eur_samples = [
                sample
                for sample in eur_panel_samples
                if sample in available_samples
            ]
            comparison_samples = [
                sample
                for sample in comparison_panel_samples
                if sample in available_samples
            ]

            if len(eur_samples) < 3:
                raise RuntimeError(
                    "Fewer than three EUR samples "
                    "present in VCF."
                )
            if len(comparison_samples) < 3:
                raise RuntimeError(
                    "Fewer than three "
                    f"{comparison_reference} samples "
                    "present in VCF."
                )

            selected_samples = list(
                dict.fromkeys(
                    eur_samples
                    + comparison_samples
                )
            )
            vcf.subset_samples(
                selected_samples
            )

            contig = resolve_contig(
                vcf,
                chromosome,
            )

            found: dict[
                str,
                dict[str, Any],
            ] = {}
            records_scanned = 0

            for record in vcf.fetch(
                contig,
                start_zero_based,
                end_one_based,
            ):
                records_scanned += 1
                if not record.alts:
                    continue

                for alternate_index, alternate in (
                    enumerate(
                        record.alts,
                        start=1,
                    )
                ):
                    key = (
                        normalize_chrom(
                            record.contig
                        ),
                        int(record.pos),
                        clean(
                            record.ref
                        ).upper(),
                        clean(
                            alternate
                        ).upper(),
                    )
                    variant_id = (
                        request_lookup.get(key)
                    )
                    if not variant_id:
                        continue

                    found[variant_id] = {
                        "vcf_id": clean(
                            record.id
                        ),
                        "alternate_allele_index": (
                            int(alternate_index)
                        ),
                        "eur_dosage": (
                            allele_dosages(
                                record,
                                eur_samples,
                                alternate_index,
                            )
                        ),
                        "comparison_dosage": (
                            allele_dosages(
                                record,
                                comparison_samples,
                                alternate_index,
                            )
                        ),
                    }

            fetch_completed = True

        except OSError as exc:
            failures.append(
                {
                    "attempt": attempt,
                    "error_type": (
                        type(exc).__name__
                    ),
                    "error_message": str(exc),
                }
            )
            print(
                "  Remote VCF fetch attempt "
                f"{attempt}/{max_attempts} failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        finally:
            close_warning = (
                close_variant_file_safely(vcf)
            )

        if fetch_completed:
            if close_warning:
                print(
                    "  Remote VCF close warning "
                    "after successful region fetch: "
                    f"{close_warning}",
                    flush=True,
                )
            return {
                "found": found,
                "records_scanned": int(
                    records_scanned
                ),
                "eur_samples": eur_samples,
                "comparison_samples": (
                    comparison_samples
                ),
                "fetch_attempts_used": int(
                    attempt
                ),
                "prior_fetch_failures": (
                    failures
                ),
                "close_warning": (
                    close_warning
                ),
                "contig": contig,
                "start_zero_based": int(
                    start_zero_based
                ),
                "end_one_based": int(
                    end_one_based
                ),
            }

        if attempt < max_attempts:
            delay = (
                base_delay_seconds
                * attempt
            )
            print(
                f"  Retrying in {delay} seconds...",
                flush=True,
            )
            time.sleep(delay)

    last_failure = (
        failures[-1]
        if failures
        else {
            "error_type": "UnknownError",
            "error_message": (
                "Remote fetch did not complete."
            ),
        }
    )
    raise RuntimeError(
        "Remote 1000 Genomes region fetch failed "
        f"after {max_attempts} attempts. "
        f"Last error: "
        f"{last_failure['error_type']}: "
        f"{last_failure['error_message']}"
    )


def process_one_gbmi_comparison(
    *,
    row,
    selected: pd.DataFrame,
    panel: pd.DataFrame,
    cache_dir: Path,
    minimum_retained: int,
) -> None:
    comparison_uid = clean(
        row.comparison_uid
    )
    comparison_population = clean(
        row.comparison_population
    ).upper()
    comparison_reference = clean(
        row.
        comparison_reference_population
    ).upper()
    metadata_path, matrix_path = cache_paths(
        cache_dir,
        comparison_uid,
    )

    vcf_url = clean(row.kgp_vcf_url)
    index_url = clean(
        row.kgp_vcf_index_url
    )

    eur_panel_samples = panel.loc[
        panel["super_population"].eq(
            "EUR"
        ),
        "sample",
    ].tolist()
    comparison_panel_samples = panel.loc[
        panel["super_population"].eq(
            comparison_reference
        ),
        "sample",
    ].tolist()

    if not eur_panel_samples:
        raise RuntimeError(
            "No EUR samples in 1000 Genomes panel."
        )
    if not comparison_panel_samples:
        raise RuntimeError(
            "No samples in 1000 Genomes "
            f"panel for {comparison_reference}."
        )

    fetch_result = fetch_gbmi_region_with_retries(
        vcf_url=vcf_url,
        index_url=index_url,
        selected=selected,
        eur_panel_samples=eur_panel_samples,
        comparison_panel_samples=(
            comparison_panel_samples
        ),
        comparison_reference=(
            comparison_reference
        ),
    )
    found = fetch_result["found"]
    records_scanned = fetch_result[
        "records_scanned"
    ]
    eur_samples = fetch_result[
        "eur_samples"
    ]
    comparison_samples = fetch_result[
        "comparison_samples"
    ]
    resolution = []
    usable_variant_ids = []
    eur_genotypes = []
    comparison_genotypes = []

    for variant in selected.itertuples(
        index=False
    ):
        variant_id = clean(
            variant.variant_id
        )
        data = found.get(variant_id)
        if data is None:
            resolution.append(
                {
                    "variant_id": variant_id,
                    "request_order": int(
                        variant.request_order
                    ),
                    "reference_usable": False,
                    "reference_exclusion_reason": (
                        "MISSING_BOTH_REFERENCES"
                    ),
                    "eur_panel_index": None,
                    "comparison_panel_index": (
                        None
                    ),
                    "eur_reference_polymorphic": (
                        False
                    ),
                    "comparison_reference_polymorphic": (
                        False
                    ),
                    "eur_reference_af": None,
                    "comparison_reference_af": (
                        None
                    ),
                    "eur_reference_call_rate": (
                        None
                    ),
                    "comparison_reference_call_rate": (
                        None
                    ),
                }
            )
            continue

        eur_metrics = dosage_metrics(
            data["eur_dosage"]
        )
        comparison_metrics = (
            dosage_metrics(
                data[
                    "comparison_dosage"
                ]
            )
        )
        eur_polymorphic = bool(
            eur_metrics["polymorphic"]
        )
        comparison_polymorphic = bool(
            comparison_metrics[
                "polymorphic"
            ]
        )
        usable = (
            eur_polymorphic
            and comparison_polymorphic
        )

        if usable:
            reason = "USABLE"
            usable_variant_ids.append(
                variant_id
            )
            eur_genotypes.append(
                data["eur_dosage"]
            )
            comparison_genotypes.append(
                data[
                    "comparison_dosage"
                ]
            )
        elif (
            not eur_polymorphic
            and not comparison_polymorphic
        ):
            reason = (
                "MONOMORPHIC_BOTH_REFERENCES"
            )
        elif not eur_polymorphic:
            reason = (
                "MONOMORPHIC_EUR_REFERENCE"
            )
        else:
            reason = (
                "MONOMORPHIC_COMPARISON_REFERENCE"
            )

        resolution.append(
            {
                "variant_id": variant_id,
                "request_order": int(
                    variant.request_order
                ),
                "reference_usable": usable,
                "reference_exclusion_reason": (
                    reason
                ),
                "eur_panel_index": None,
                "comparison_panel_index": (
                    None
                ),
                "eur_reference_polymorphic": (
                    eur_polymorphic
                ),
                "comparison_reference_polymorphic": (
                    comparison_polymorphic
                ),
                "eur_reference_af": (
                    eur_metrics[
                        "alternate_allele_frequency"
                    ]
                ),
                "comparison_reference_af": (
                    comparison_metrics[
                        "alternate_allele_frequency"
                    ]
                ),
                "eur_reference_call_rate": (
                    eur_metrics["call_rate"]
                ),
                "comparison_reference_call_rate": (
                    comparison_metrics[
                        "call_rate"
                    ]
                ),
                "vcf_id": data["vcf_id"],
                "alternate_allele_index": (
                    data[
                        "alternate_allele_index"
                    ]
                ),
            }
        )

    base_metadata = {
        "comparison_uid": comparison_uid,
        "gene_trait_uid": clean(
            row.gene_trait_uid
        ),
        "gene": clean(row.gene),
        "trait": clean(
            row.candidate_trait_name
        ),
        "candidate_source": GBMI,
        "comparison_population": (
            comparison_population
        ),
        "comparison_reference_population": (
            comparison_reference
        ),
        "request_variant_ids": selected[
            "variant_id"
        ].tolist(),
        "n_requested": int(
            len(selected)
        ),
        "n_reference_usable": int(
            len(usable_variant_ids)
        ),
        "reference_resolution": (
            resolution
        ),
        "eur_reference": {
            "vcf_url": vcf_url,
            "vcf_index_url": index_url,
            "reference_population": (
                "EUR"
            ),
            "n_samples": int(
                len(eur_samples)
            ),
        },
        "comparison_reference": {
            "vcf_url": vcf_url,
            "vcf_index_url": index_url,
            "reference_population": (
                comparison_reference
            ),
            "n_samples": int(
                len(
                    comparison_samples
                )
            ),
        },
        "records_scanned": int(
            records_scanned
        ),
        "remote_fetch_attempts_used": int(
            fetch_result[
                "fetch_attempts_used"
            ]
        ),
        "remote_fetch_prior_failures": (
            fetch_result[
                "prior_fetch_failures"
            ]
        ),
        "remote_close_warning": (
            fetch_result[
                "close_warning"
            ]
        ),
        "remote_fetch_region": {
            "contig": fetch_result[
                "contig"
            ],
            "start_zero_based": int(
                fetch_result[
                    "start_zero_based"
                ]
            ),
            "end_one_based": int(
                fetch_result[
                    "end_one_based"
                ]
            ),
        },
    }

    if (
        len(usable_variant_ids)
        < minimum_retained
    ):
        save_nonmatrix_cache(
            metadata_path,
            matrix_path,
            metadata=base_metadata,
            status=(
                "INSUFFICIENT_REFERENCE_VARIANTS"
            ),
        )
        return

    eur_matrix = np.vstack(
        eur_genotypes
    )
    comparison_matrix = np.vstack(
        comparison_genotypes
    )
    eur_r, eur_pairwise_n = (
        pairwise_correlation(
            eur_matrix
        )
    )
    comparison_r, comparison_pairwise_n = (
        pairwise_correlation(
            comparison_matrix
        )
    )

    if (
        not np.isfinite(eur_r).all()
        or not np.isfinite(
            comparison_r
        ).all()
    ):
        raise RuntimeError(
            "Nonfinite 1000 Genomes LD "
            "correlations after polymorphism QC."
        )

    r2_eur = np.square(eur_r)
    r2_comparison = np.square(
        comparison_r
    )
    eur_diagnostics = (
        matrix_diagnostics(r2_eur)
    )
    comparison_diagnostics = (
        matrix_diagnostics(
            r2_comparison
        )
    )

    save_complete_cache(
        metadata_path,
        matrix_path,
        metadata={
            **base_metadata,
            "eur_pairwise_n_minimum": int(
                np.min(eur_pairwise_n)
            ),
            "comparison_pairwise_n_minimum": (
                int(
                    np.min(
                        comparison_pairwise_n
                    )
                )
            ),
            "eur_r2_diagnostics": (
                eur_diagnostics
            ),
            "comparison_r2_diagnostics": (
                comparison_diagnostics
            ),
        },
        usable_variant_ids=(
            usable_variant_ids
        ),
        r2_eur=r2_eur,
        r2_comparison=r2_comparison,
    )


def process_gbmi(
    manifest: pd.DataFrame,
    requests: pd.DataFrame,
    *,
    cache_dir: Path,
    reference_dir: Path,
    minimum_retained: int,
    rebuild_cache: bool,
    stop_after: int,
) -> int:
    target_manifest = manifest[
        manifest["candidate_source"].eq(GBMI)
    ].copy()
    if target_manifest.empty:
        return 0

    pending_rows = []
    selected_by_uid: dict[
        str,
        pd.DataFrame,
    ] = {}

    for row in target_manifest.itertuples(
        index=False
    ):
        comparison_uid = clean(
            row.comparison_uid
        )
        selected = sorted_requests(
            requests,
            comparison_uid,
        )
        selected_by_uid[
            comparison_uid
        ] = selected
        metadata_path, matrix_path = (
            cache_paths(
                cache_dir,
                comparison_uid,
            )
        )
        valid = cache_is_valid(
            metadata_path,
            matrix_path,
            selected,
        )
        if rebuild_cache or not valid:
            pending_rows.append(row)

    if not pending_rows:
        print(
            "GBMI: all comparison caches "
            "are already valid."
        )
        return 0

    panel_urls = {
        clean(row.kgp_sample_panel_url)
        for row in pending_rows
    }
    if len(panel_urls) != 1:
        raise RuntimeError(
            "Expected one locked 1000 Genomes "
            f"sample-panel URL; observed {panel_urls}"
        )
    panel_url = next(iter(panel_urls))
    panel_path = (
        reference_dir
        / (
            "integrated_call_samples_"
            "v3.20130502.ALL.panel"
        )
    )
    panel_download = download_file(
        panel_url,
        panel_path,
    )
    panel = read_sample_panel(
        panel_path
    )

    print()
    print(
        "GBMI pending comparisons:",
        len(pending_rows),
    )
    print(
        "1000 Genomes panel:",
        panel_download["path"],
    )

    processed = 0
    total = len(pending_rows)
    for ordinal, row in enumerate(
        pending_rows,
        start=1,
    ):
        comparison_uid = clean(
            row.comparison_uid
        )
        selected = selected_by_uid[
            comparison_uid
        ]
        comparison_reference = clean(
            row.
            comparison_reference_population
        ).upper()
        metadata_path, matrix_path = (
            cache_paths(
                cache_dir,
                comparison_uid,
            )
        )

        print(
            f"[GBMI {ordinal}/{total}] "
            f"{comparison_uid} | "
            f"{clean(row.gene)} | "
            f"{comparison_reference} | "
            f"{len(selected)} candidates"
        )

        try:
            process_one_gbmi_comparison(
                row=row,
                selected=selected,
                panel=panel,
                cache_dir=cache_dir,
                minimum_retained=(
                    minimum_retained
                ),
            )
        except Exception as exc:
            save_nonmatrix_cache(
                metadata_path,
                matrix_path,
                metadata={
                    "comparison_uid": (
                        comparison_uid
                    ),
                    "gene_trait_uid": clean(
                        row.gene_trait_uid
                    ),
                    "gene": clean(row.gene),
                    "trait": clean(
                        row.candidate_trait_name
                    ),
                    "candidate_source": GBMI,
                    "comparison_population": (
                        clean(
                            row.
                            comparison_population
                        ).upper()
                    ),
                    "comparison_reference_population": (
                        comparison_reference
                    ),
                    "request_variant_ids": (
                        selected[
                            "variant_id"
                        ].tolist()
                    ),
                    "n_requested": int(
                        len(selected)
                    ),
                    "n_reference_usable": 0,
                    "reference_resolution": [],
                    "error_type": (
                        type(exc).__name__
                    ),
                    "error_message": str(exc),
                    "traceback": (
                        traceback.format_exc()
                    ),
                },
                status="PROCESSING_ERROR",
            )
            print(
                "  ERROR:",
                type(exc).__name__,
                str(exc),
            )

        processed += 1
        if (
            stop_after > 0
            and processed >= stop_after
        ):
            print(
                "Stopping after requested "
                f"{stop_after} new comparisons."
            )
            break

    return processed


def greedy_prune(
    variant_ids: list[str],
    r2_eur: np.ndarray,
    r2_comparison: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    n_variants = len(variant_ids)
    expected_shape = (
        n_variants,
        n_variants,
    )
    if (
        r2_eur.shape != expected_shape
        or r2_comparison.shape
        != expected_shape
    ):
        raise RuntimeError(
            "LD matrix shape does not match "
            f"variant count {n_variants}: "
            f"EUR={r2_eur.shape}; "
            f"comparison="
            f"{r2_comparison.shape}"
        )

    retained_indices: list[int] = []
    decisions: list[
        dict[str, Any]
    ] = []

    for index, variant_id in enumerate(
        variant_ids
    ):
        if not retained_indices:
            retained_indices.append(index)
            decisions.append(
                {
                    "variant_id": variant_id,
                    "retained": True,
                    "pruning_reason": (
                        "FIRST_RANKED_VARIANT"
                    ),
                    "max_r2_eur": None,
                    "max_r2_comparison": (
                        None
                    ),
                    "blocking_variant_eur": (
                        None
                    ),
                    "blocking_variant_comparison": (
                        None
                    ),
                }
            )
            continue

        eur_values = r2_eur[
            index,
            retained_indices,
        ]
        comparison_values = (
            r2_comparison[
                index,
                retained_indices,
            ]
        )

        if (
            not np.isfinite(
                eur_values
            ).all()
            or not np.isfinite(
                comparison_values
            ).all()
        ):
            decisions.append(
                {
                    "variant_id": variant_id,
                    "retained": False,
                    "pruning_reason": (
                        "NONFINITE_PAIRWISE_LD"
                    ),
                    "max_r2_eur": None,
                    "max_r2_comparison": (
                        None
                    ),
                    "blocking_variant_eur": (
                        None
                    ),
                    "blocking_variant_comparison": (
                        None
                    ),
                }
            )
            continue

        eur_max_position = int(
            np.argmax(eur_values)
        )
        comparison_max_position = int(
            np.argmax(
                comparison_values
            )
        )
        max_eur = float(
            eur_values[
                eur_max_position
            ]
        )
        max_comparison = float(
            comparison_values[
                comparison_max_position
            ]
        )
        passes_eur = bool(
            np.all(
                eur_values < threshold
            )
        )
        passes_comparison = bool(
            np.all(
                comparison_values
                < threshold
            )
        )
        retained = (
            passes_eur
            and passes_comparison
        )

        if retained:
            reason = "RETAINED_BELOW_THRESHOLD"
            retained_indices.append(index)
        elif (
            not passes_eur
            and not passes_comparison
        ):
            reason = (
                "PRUNED_LD_BOTH_REFERENCES"
            )
        elif not passes_eur:
            reason = (
                "PRUNED_LD_EUR_REFERENCE"
            )
        else:
            reason = (
                "PRUNED_LD_COMPARISON_REFERENCE"
            )

        decisions.append(
            {
                "variant_id": variant_id,
                "retained": retained,
                "pruning_reason": reason,
                "max_r2_eur": max_eur,
                "max_r2_comparison": (
                    max_comparison
                ),
                "blocking_variant_eur": (
                    variant_ids[
                        retained_indices[
                            eur_max_position
                        ]
                    ]
                ),
                "blocking_variant_comparison": (
                    variant_ids[
                        retained_indices[
                            comparison_max_position
                        ]
                    ]
                ),
            }
        )

    return decisions


def aggregate_outputs(
    manifest: pd.DataFrame,
    requests: pd.DataFrame,
    *,
    cache_dir: Path,
    output_dir: Path,
    thresholds: tuple[float, ...],
    primary_threshold: float,
    minimum_retained: int,
) -> dict[str, Any]:
    resolution_rows: list[
        dict[str, Any]
    ] = []
    decision_rows: list[
        dict[str, Any]
    ] = []
    comparison_rows: list[
        dict[str, Any]
    ] = []
    missing_caches = []
    processing_errors = []

    manifest_lookup = (
        manifest.set_index(
            "comparison_uid",
            drop=False,
        )
    )

    for comparison_uid in manifest[
        "comparison_uid"
    ]:
        comparison_uid = clean(
            comparison_uid
        )
        row = manifest_lookup.loc[
            comparison_uid
        ]
        selected = sorted_requests(
            requests,
            comparison_uid,
        )
        metadata_path, matrix_path = (
            cache_paths(
                cache_dir,
                comparison_uid,
            )
        )

        if not cache_is_valid(
            metadata_path,
            matrix_path,
            selected,
        ):
            missing_caches.append(
                comparison_uid
            )
            continue

        cache = load_cache(
            metadata_path,
            matrix_path,
        )
        processing_status = cache[
            "processing_status"
        ]

        if processing_status == (
            "PROCESSING_ERROR"
        ):
            processing_errors.append(
                {
                    "comparison_uid": (
                        comparison_uid
                    ),
                    "candidate_source": clean(
                        row["candidate_source"]
                    ),
                    "error_type": cache.get(
                        "error_type"
                    ),
                    "error_message": (
                        cache.get(
                            "error_message"
                        )
                    ),
                }
            )

        resolution_map = {
            record["variant_id"]: record
            for record in cache.get(
                "reference_resolution",
                [],
            )
        }

        for variant in selected.itertuples(
            index=False
        ):
            variant_id = clean(
                variant.variant_id
            )
            record = resolution_map.get(
                variant_id,
                {},
            )
            resolution_rows.append(
                {
                    "comparison_uid": (
                        comparison_uid
                    ),
                    "gene_trait_uid": clean(
                        row["gene_trait_uid"]
                    ),
                    "gene": clean(
                        row["gene"]
                    ),
                    "candidate_source": clean(
                        row[
                            "candidate_source"
                        ]
                    ),
                    "candidate_trait_name": clean(
                        row[
                            "candidate_trait_name"
                        ]
                    ),
                    "comparison_population": clean(
                        row[
                            "comparison_population"
                        ]
                    ),
                    "comparison_reference_population": clean(
                        row.get(
                            "comparison_reference_population",
                            row[
                                "comparison_population"
                            ],
                        )
                    ),
                    "variant_id": variant_id,
                    "chrom": normalize_chrom(
                        variant.chrom
                    ),
                    "pos": int(
                        variant.pos
                    ),
                    "ref": clean(
                        variant.ref
                    ).upper(),
                    "alt": clean(
                        variant.alt
                    ).upper(),
                    "eur_p": float(
                        variant.eur_p
                    ),
                    "eur_p_rank": int(
                        variant.eur_p_rank
                    ),
                    "request_order": int(
                        variant.request_order
                    ),
                    "processing_status": (
                        processing_status
                    ),
                    "reference_usable": bool(
                        record.get(
                            "reference_usable",
                            False,
                        )
                    ),
                    "reference_exclusion_reason": (
                        record.get(
                            "reference_exclusion_reason",
                            (
                                "PROCESSING_ERROR"
                                if processing_status
                                == "PROCESSING_ERROR"
                                else "NO_REFERENCE_RECORD"
                            ),
                        )
                    ),
                    "eur_panel_index": (
                        record.get(
                            "eur_panel_index"
                        )
                    ),
                    "comparison_panel_index": (
                        record.get(
                            "comparison_panel_index"
                        )
                    ),
                    "eur_reference_polymorphic": (
                        record.get(
                            "eur_reference_polymorphic"
                        )
                    ),
                    "comparison_reference_polymorphic": (
                        record.get(
                            "comparison_reference_polymorphic"
                        )
                    ),
                    "eur_reference_af": (
                        record.get(
                            "eur_reference_af"
                        )
                    ),
                    "comparison_reference_af": (
                        record.get(
                            "comparison_reference_af"
                        )
                    ),
                    "eur_reference_call_rate": (
                        record.get(
                            "eur_reference_call_rate"
                        )
                    ),
                    "comparison_reference_call_rate": (
                        record.get(
                            "comparison_reference_call_rate"
                        )
                    ),
                }
            )

        usable_variant_ids = cache[
            "usable_variant_ids"
        ]
        usable_order = {
            variant_id: index
            for index, variant_id in enumerate(
                usable_variant_ids
            )
        }

        for threshold in thresholds:
            if processing_status == "COMPLETE":
                usable_decisions = greedy_prune(
                    usable_variant_ids,
                    cache["r2_eur"],
                    cache[
                        "r2_comparison"
                    ],
                    threshold,
                )
                usable_decision_map = {
                    decision["variant_id"]: (
                        decision
                    )
                    for decision in (
                        usable_decisions
                    )
                }
            else:
                usable_decision_map = {}

            n_retained = 0
            n_ld_pruned = 0
            n_reference_excluded = 0

            for variant in selected.itertuples(
                index=False
            ):
                variant_id = clean(
                    variant.variant_id
                )
                reference_record = (
                    resolution_map.get(
                        variant_id,
                        {},
                    )
                )
                reference_usable = bool(
                    reference_record.get(
                        "reference_usable",
                        False,
                    )
                )

                if reference_usable:
                    decision = (
                        usable_decision_map[
                            variant_id
                        ]
                    )
                    retained = bool(
                        decision["retained"]
                    )
                    if retained:
                        n_retained += 1
                    else:
                        n_ld_pruned += 1
                    reason = decision[
                        "pruning_reason"
                    ]
                    max_r2_eur = decision[
                        "max_r2_eur"
                    ]
                    max_r2_comparison = (
                        decision[
                            "max_r2_comparison"
                        ]
                    )
                    blocker_eur = decision[
                        "blocking_variant_eur"
                    ]
                    blocker_comparison = (
                        decision[
                            "blocking_variant_comparison"
                        ]
                    )
                    ld_usable_order = (
                        usable_order[
                            variant_id
                        ]
                    )
                else:
                    retained = False
                    n_reference_excluded += 1
                    reason = (
                        reference_record.get(
                            "reference_exclusion_reason",
                            (
                                "PROCESSING_ERROR"
                                if processing_status
                                == "PROCESSING_ERROR"
                                else "REFERENCE_UNAVAILABLE"
                            ),
                        )
                    )
                    max_r2_eur = None
                    max_r2_comparison = None
                    blocker_eur = None
                    blocker_comparison = None
                    ld_usable_order = None

                decision_rows.append(
                    {
                        "comparison_uid": (
                            comparison_uid
                        ),
                        "gene_trait_uid": clean(
                            row[
                                "gene_trait_uid"
                            ]
                        ),
                        "gene": clean(
                            row["gene"]
                        ),
                        "candidate_source": clean(
                            row[
                                "candidate_source"
                            ]
                        ),
                        "candidate_trait_name": clean(
                            row[
                                "candidate_trait_name"
                            ]
                        ),
                        "comparison_population": clean(
                            row[
                                "comparison_population"
                            ]
                        ),
                        "variant_id": variant_id,
                        "chrom": normalize_chrom(
                            variant.chrom
                        ),
                        "pos": int(
                            variant.pos
                        ),
                        "ref": clean(
                            variant.ref
                        ).upper(),
                        "alt": clean(
                            variant.alt
                        ).upper(),
                        "eur_p": float(
                            variant.eur_p
                        ),
                        "eur_p_rank": int(
                            variant.eur_p_rank
                        ),
                        "request_order": int(
                            variant.request_order
                        ),
                        "ld_usable_order": (
                            ld_usable_order
                        ),
                        "r2_threshold": float(
                            threshold
                        ),
                        "reference_usable": (
                            reference_usable
                        ),
                        "retained": retained,
                        "pruning_reason": (
                            reason
                        ),
                        "max_r2_eur": (
                            max_r2_eur
                        ),
                        "max_r2_comparison": (
                            max_r2_comparison
                        ),
                        "blocking_variant_eur": (
                            blocker_eur
                        ),
                        "blocking_variant_comparison": (
                            blocker_comparison
                        ),
                    }
                )

            passes = (
                processing_status == "COMPLETE"
                and n_retained
                >= minimum_retained
            )
            comparison_rows.append(
                {
                    "comparison_uid": (
                        comparison_uid
                    ),
                    "gene_trait_uid": clean(
                        row["gene_trait_uid"]
                    ),
                    "gene": clean(
                        row["gene"]
                    ),
                    "candidate_source": clean(
                        row[
                            "candidate_source"
                        ]
                    ),
                    "candidate_trait_name": clean(
                        row[
                            "candidate_trait_name"
                        ]
                    ),
                    "comparison_population": clean(
                        row[
                            "comparison_population"
                        ]
                    ),
                    "comparison_reference_population": clean(
                        row.get(
                            "comparison_reference_population",
                            row[
                                "comparison_population"
                            ],
                        )
                    ),
                    "processing_status": (
                        processing_status
                    ),
                    "r2_threshold": float(
                        threshold
                    ),
                    "n_requested_variants": int(
                        len(selected)
                    ),
                    "n_reference_usable": int(
                        len(
                            usable_variant_ids
                        )
                    ),
                    "n_reference_excluded": int(
                        n_reference_excluded
                    ),
                    "n_ld_pruned": int(
                        n_ld_pruned
                    ),
                    "n_retained": int(
                        n_retained
                    ),
                    "minimum_retained_required": int(
                        minimum_retained
                    ),
                    "portability_eligible": bool(
                        passes
                    ),
                    "comparison_exclusion_reason": (
                        "ELIGIBLE"
                        if passes
                        else (
                            "PROCESSING_ERROR"
                            if processing_status
                            == "PROCESSING_ERROR"
                            else (
                                "INSUFFICIENT_REFERENCE_VARIANTS"
                                if processing_status
                                == "INSUFFICIENT_REFERENCE_VARIANTS"
                                else "INSUFFICIENT_RETAINED_VARIANTS"
                            )
                        )
                    ),
                }
            )

    resolution_frame = pd.DataFrame(
        resolution_rows
    )
    decisions_frame = pd.DataFrame(
        decision_rows
    )
    comparisons_frame = pd.DataFrame(
        comparison_rows
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    resolution_path = (
        output_dir
        / (
            "15c3_variant_reference_"
            "resolution.parquet"
        )
    )
    decisions_path = (
        output_dir
        / (
            "15c3_variant_pruning_"
            "decisions.parquet"
        )
    )
    comparisons_path = (
        output_dir
        / (
            "15c3_comparison_pruning_"
            "results.parquet"
        )
    )
    resolution_frame.to_parquet(
        resolution_path,
        index=False,
    )
    decisions_frame.to_parquet(
        decisions_path,
        index=False,
    )
    comparisons_frame.to_parquet(
        comparisons_path,
        index=False,
    )

    retained_paths = {}
    for threshold in thresholds:
        label = threshold_label(
            threshold
        )
        retained = decisions_frame[
            decisions_frame[
                "r2_threshold"
            ].eq(threshold)
            & decisions_frame[
                "retained"
            ]
        ].copy()

        if math.isclose(
            threshold,
            primary_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            retained_path = (
                output_dir
                / (
                    "15c3_portability_variants_"
                    "primary_retained.parquet"
                )
            )
        else:
            retained_path = (
                output_dir
                / (
                    "15c3_portability_variants_"
                    f"r2_{label}_retained.parquet"
                )
            )
        retained.to_parquet(
            retained_path,
            index=False,
        )
        retained_paths[
            str(threshold)
        ] = str(retained_path)

    primary_results = comparisons_frame[
        comparisons_frame[
            "r2_threshold"
        ].eq(primary_threshold)
    ].copy()
    final_comparison_frame = (
        manifest.merge(
            primary_results[
                [
                    "comparison_uid",
                    "processing_status",
                    "n_requested_variants",
                    "n_reference_usable",
                    "n_reference_excluded",
                    "n_ld_pruned",
                    "n_retained",
                    "minimum_retained_required",
                    "portability_eligible",
                    "comparison_exclusion_reason",
                ]
            ],
            on="comparison_uid",
            how="left",
            validate="one_to_one",
        )
    )
    final_comparison_frame = (
        final_comparison_frame[
            final_comparison_frame[
                "portability_eligible"
            ].fillna(False)
        ].copy()
    )
    final_comparison_path = (
        output_dir
        / (
            "15c3_portability_comparisons_"
            "primary_final.parquet"
        )
    )
    final_comparison_frame.to_parquet(
        final_comparison_path,
        index=False,
    )

    threshold_summaries = {}
    for threshold in thresholds:
        subset = comparisons_frame[
            comparisons_frame[
                "r2_threshold"
            ].eq(threshold)
        ]
        threshold_summaries[
            str(threshold)
        ] = {
            "n_comparisons": int(
                len(subset)
            ),
            "n_portability_eligible": int(
                subset[
                    "portability_eligible"
                ].sum()
            ),
            "n_not_eligible": int(
                (
                    ~subset[
                        "portability_eligible"
                    ]
                ).sum()
            ),
            "total_variants_requested": int(
                subset[
                    "n_requested_variants"
                ].sum()
            ),
            "total_variants_reference_usable": int(
                subset[
                    "n_reference_usable"
                ].sum()
            ),
            "total_variants_retained": int(
                subset[
                    "n_retained"
                ].sum()
            ),
            "eligible_by_source": {
                clean(source): int(count)
                for source, count in (
                    subset[
                        subset[
                            "portability_eligible"
                        ]
                    ]
                    .groupby(
                        "candidate_source"
                    )
                    .size()
                    .items()
                )
            },
        }

    summary = {
        "step": "15C3",
        "script_version": SCRIPT_VERSION,
        "status": (
            "FAIL"
            if (
                missing_caches
                or processing_errors
            )
            else (
                "PASS_WITH_LD_EXCLUSIONS"
                if (
                    not final_comparison_frame.empty
                    and len(
                        final_comparison_frame
                    )
                    < len(manifest)
                )
                else "PASS"
            )
        ),
        "pruning_rule": {
            "ranking": (
                "EUR p ascending, then position, "
                "REF, ALT, variant ID"
            ),
            "criterion": (
                "candidate r2 strictly below "
                "threshold against every retained "
                "variant in both ancestry references"
            ),
            "thresholds": list(
                thresholds
            ),
            "primary_threshold": float(
                primary_threshold
            ),
            "minimum_retained_variants": int(
                minimum_retained
            ),
        },
        "input_counts": {
            "comparisons": int(
                len(manifest)
            ),
            "variant_request_rows": int(
                len(requests)
            ),
            "comparisons_by_source": {
                clean(source): int(count)
                for source, count in (
                    manifest.groupby(
                        "candidate_source"
                    ).size().items()
                )
            },
        },
        "threshold_results": (
            threshold_summaries
        ),
        "primary_final_comparisons": int(
            len(final_comparison_frame)
        ),
        "missing_or_stale_caches": (
            missing_caches
        ),
        "processing_errors": (
            processing_errors
        ),
        "outputs": {
            "variant_reference_resolution": str(
                resolution_path
            ),
            "variant_pruning_decisions": str(
                decisions_path
            ),
            "comparison_pruning_results": str(
                comparisons_path
            ),
            "retained_variant_tables": (
                retained_paths
            ),
            "primary_final_comparisons": str(
                final_comparison_path
            ),
        },
        "scientific_scope": (
            "Ancestry-paired LD pruning only; "
            "no portability slope estimated."
        ),
        "completed_at_utc": utc_now(),
    }
    summary_path = (
        output_dir
        / "15c3_ld_pruning_summary.json"
    )
    atomic_json_write(
        summary,
        summary_path,
    )

    return {
        "summary": summary,
        "summary_path": summary_path,
        "comparisons_frame": (
            comparisons_frame
        ),
    }


def main() -> int:
    args = parse_args()
    thresholds = parse_thresholds(
        args.thresholds
    )
    if not any(
        math.isclose(
            threshold,
            args.primary_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for threshold in thresholds
    ):
        raise SystemExit(
            "--primary-threshold must be included "
            "in --thresholds."
        )
    if args.minimum_retained < 1:
        raise SystemExit(
            "--minimum-retained must be positive."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.reference_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in [
        args.manifest,
        args.requests,
    ]:
        if not path.exists():
            raise SystemExit(
                f"Missing input: {path}"
            )

    manifest = pd.read_parquet(
        args.manifest
    )
    requests = pd.read_parquet(
        args.requests
    )

    require_columns(
        manifest,
        {
            "comparison_uid",
            "gene_trait_uid",
            "gene",
            "candidate_source",
            "candidate_trait_name",
            "comparison_population",
            "comparison_reference_population",
            "portability_ld_request_ready",
            "eur_variant_index",
            "comparison_variant_index",
            "eur_ld_matrix",
            "comparison_ld_matrix",
            "kgp_vcf_url",
            "kgp_vcf_index_url",
            "kgp_sample_panel_url",
        },
        "comparison manifest",
    )
    require_columns(
        requests,
        {
            "comparison_uid",
            "gene_trait_uid",
            "gene",
            "candidate_source",
            "candidate_trait_name",
            "comparison_population",
            "variant_id",
            "chrom",
            "pos",
            "ref",
            "alt",
            "eur_p",
            "eur_p_rank",
        },
        "variant requests",
    )

    manifest = manifest[
        manifest[
            "portability_ld_request_ready"
        ].map(as_bool)
    ].copy()
    requests = requests[
        requests["comparison_uid"].isin(
            manifest["comparison_uid"]
        )
    ].copy()

    if manifest[
        "comparison_uid"
    ].duplicated().any():
        raise SystemExit(
            "Comparison manifest contains duplicate "
            "comparison_uid values."
        )
    if set(
        manifest["candidate_source"]
    ) - {PANUKB, GBMI}:
        raise SystemExit(
            "Unexpected candidate sources: "
            f"{sorted(set(manifest['candidate_source']))}"
        )

    print("=" * 78)
    print(
        "STEP 15C3 — FULL ANCESTRY-PAIRED "
        "LD PRUNING"
    )
    print("=" * 78)
    print(
        "Comparisons:",
        len(manifest),
    )
    print(
        "Variant requests:",
        len(requests),
    )
    print(
        "Thresholds:",
        ", ".join(
            f"{value:.2f}"
            for value in thresholds
        ),
    )
    print(
        "Primary threshold:",
        f"{args.primary_threshold:.2f}",
    )
    print(
        "Minimum retained:",
        args.minimum_retained,
    )
    print(
        "Cache:",
        args.cache_dir,
    )
    print("=" * 78)

    if args.source in {
        "both",
        "panukb",
    }:
        process_panukb(
            manifest,
            requests,
            cache_dir=args.cache_dir,
            minimum_retained=(
                args.minimum_retained
            ),
            rebuild_cache=(
                args.rebuild_cache
            ),
            hail_master=args.hail_master,
            stop_after=args.stop_after,
        )

    if args.source in {
        "both",
        "gbmi",
    }:
        process_gbmi(
            manifest,
            requests,
            cache_dir=args.cache_dir,
            reference_dir=(
                args.reference_dir
            ),
            minimum_retained=(
                args.minimum_retained
            ),
            rebuild_cache=(
                args.rebuild_cache
            ),
            stop_after=args.stop_after,
        )

    aggregate = aggregate_outputs(
        manifest,
        requests,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        thresholds=thresholds,
        primary_threshold=(
            args.primary_threshold
        ),
        minimum_retained=(
            args.minimum_retained
        ),
    )
    summary = aggregate["summary"]

    print()
    print("=" * 78)
    print(
        "STEP 15C3 — FULL ANCESTRY-PAIRED "
        "LD PRUNING"
    )
    print("=" * 78)
    print("Status:", summary["status"])
    print(
        "Primary eligible comparisons:",
        summary[
            "primary_final_comparisons"
        ],
        "/",
        summary[
            "input_counts"
        ]["comparisons"],
    )
    for threshold, result in (
        summary[
            "threshold_results"
        ].items()
    ):
        print(
            f"r2 < {float(threshold):.2f}: "
            f"{result['n_portability_eligible']} "
            "eligible comparisons; "
            f"{result['total_variants_retained']} "
            "retained variants"
        )
    print(
        "Processing errors:",
        len(
            summary[
                "processing_errors"
            ]
        ),
    )
    print(
        "Missing or stale caches:",
        len(
            summary[
                "missing_or_stale_caches"
            ]
        ),
    )
    print(
        "Summary:",
        aggregate["summary_path"],
    )
    print(
        "No portability slope was estimated."
    )
    print("=" * 78)

    return (
        0
        if summary["status"] != "FAIL"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
