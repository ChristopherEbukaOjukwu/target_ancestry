#!/usr/bin/env python3
"""
Step 15C2B — Pan-UKB subset-LD pilot.

Uses the Step 15C1 frozen primary variant requests to select one
LD-ready Pan-UKB comparison, resolve every requested variant in the
EUR and comparison-population variant-index Hail Tables, extract the
same ordered subset from both public LD BlockMatrices, reconstruct the
symmetric matrices from the released triangular representation, and
write pilot diagnostics.

This is still a preflight. It does not perform greedy LD pruning or fit
a portability slope.

Required environment:
    export PYSPARK_SUBMIT_ARGS="--packages org.apache.hadoop:hadoop-aws:3.3.4 pyspark-shell"
    export SPARK_LOCAL_IP=127.0.0.1
    export SPARK_LOCAL_DIRS=/tmp/target_ancestry_spark

Default inputs:
    output/15c1_portability_comparison_manifest.parquet
    output/15c1_portability_variant_requests_primary.parquet

Outputs:
    output/15c2b_panukb_variant_resolution.csv
    output/15c2b_panukb_ld_raw_EUR.csv
    output/15c2b_panukb_ld_raw_<POP>.csv
    output/15c2b_panukb_ld_symmetric_EUR.csv
    output/15c2b_panukb_ld_symmetric_<POP>.csv
    output/15c2b_panukb_ld_r2_EUR.csv
    output/15c2b_panukb_ld_r2_<POP>.csv
    output/15c2b_panukb_subset_ld_pilot.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "15C2B.2"


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
        "--comparison-uid",
        default="",
        help="Optional exact Pan-UKB comparison override.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
    )
    parser.add_argument(
        "--minimum-variants",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--hail-master",
        default="local[2]",
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
    return str(value).strip()


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
    return chrom


def to_s3a(path: Any) -> str:
    text = clean(path)
    if text.startswith("s3://"):
        return "s3a://" + text[len("s3://") :]
    if text.startswith("s3a://"):
        return text
    raise RuntimeError(
        "Expected an S3 Pan-UKB path, observed: "
        f"{text}"
    )


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(
            f"{label} is missing required columns: {missing}"
        )


def choose_pilot(
    manifest: pd.DataFrame,
    requests: pd.DataFrame,
    comparison_uid: str,
) -> tuple[pd.Series, pd.DataFrame]:
    candidates = manifest[
        manifest["candidate_source"].eq("Pan-UKB")
        & manifest["portability_ld_request_ready"].map(as_bool)
    ].copy()

    counts = (
        requests[
            requests["candidate_source"].eq("Pan-UKB")
        ]
        .groupby("comparison_uid")
        .size()
        .rename("n_candidate_variants")
    )
    candidates = candidates.merge(
        counts,
        on="comparison_uid",
        how="inner",
        validate="one_to_one",
    )

    if comparison_uid:
        candidates = candidates[
            candidates["comparison_uid"].eq(comparison_uid)
        ]
        if len(candidates) != 1:
            raise SystemExit(
                "--comparison-uid matched "
                f"{len(candidates)} rows."
            )
    else:
        candidates = candidates.sort_values(
            [
                "n_candidate_variants",
                "candidate_trait_name",
                "gene",
                "comparison_population",
                "comparison_uid",
            ],
            kind="mergesort",
        ).head(1)

    if candidates.empty:
        raise SystemExit(
            "No primary LD-ready Pan-UKB comparison found."
        )

    pilot = candidates.iloc[0]
    selected = requests[
        requests["comparison_uid"].eq(
            pilot["comparison_uid"]
        )
    ].copy()
    selected = selected.sort_values(
        [
            "eur_p_rank",
            "eur_p",
            "pos",
            "ref",
            "alt",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    selected["request_order"] = np.arange(
        len(selected),
        dtype=int,
    )

    if selected.empty:
        raise SystemExit(
            "Selected comparison has no candidate variants."
        )

    return pilot, selected


def build_request_table(hl, selected: pd.DataFrame):
    request_pd = selected[
        [
            "variant_id",
            "chrom",
            "pos",
            "ref",
            "alt",
            "request_order",
        ]
    ].copy()
    request_pd["chrom"] = request_pd[
        "chrom"
    ].map(normalize_chrom)
    request_pd["pos"] = pd.to_numeric(
        request_pd["pos"],
        errors="raise",
    ).astype(int)
    request_pd["ref"] = request_pd[
        "ref"
    ].astype(str).str.upper()
    request_pd["alt"] = request_pd[
        "alt"
    ].astype(str).str.upper()

    request_ht = hl.Table.from_pandas(request_pd)
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


def resolve_indices(
    hl,
    request_ht,
    index_path: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    index_ht = hl.read_table(index_path)
    mapped = request_ht.annotate(
        panel_idx=index_ht[request_ht.key].idx
    )
    rows = mapped.select(
        "variant_id",
        "request_order",
        "panel_idx",
    ).collect()

    records = [
        {
            "variant_id": clean(row.variant_id),
            "request_order": int(row.request_order),
            "panel_idx": (
                None
                if row.panel_idx is None
                else int(row.panel_idx)
            ),
        }
        for row in rows
    ]
    frame = pd.DataFrame(records).sort_values(
        "request_order",
        kind="mergesort",
    )

    globals_value = hl.eval(
        index_ht.index_globals()
    )
    globals_dict = {
        field: globals_value[field]
        for field in globals_value
    }

    return frame, {
        "index_path": index_path,
        "globals": globals_dict,
        "n_requested": int(len(frame)),
        "n_resolved": int(
            frame["panel_idx"].notna().sum()
        ),
    }


def reconstruct_symmetric(
    raw: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise RuntimeError(
            f"Expected square matrix; observed {raw.shape}."
        )

    diagonal = np.diag(np.diag(raw))
    symmetric = raw + raw.T - diagonal

    asymmetry_before = float(
        np.nanmax(np.abs(raw - raw.T))
    )
    asymmetry_after = float(
        np.nanmax(
            np.abs(symmetric - symmetric.T)
        )
    )

    return symmetric, {
        "method": (
            "raw_plus_transpose_minus_diagonal"
        ),
        "maximum_asymmetry_raw": (
            asymmetry_before
        ),
        "maximum_asymmetry_symmetric": (
            asymmetry_after
        ),
    }


def matrix_diagnostics(
    matrix: np.ndarray,
) -> dict[str, Any]:
    diagonal = np.diag(matrix)
    finite = np.isfinite(matrix)
    return {
        "shape": list(matrix.shape),
        "all_entries_finite": bool(finite.all()),
        "n_finite_entries": int(finite.sum()),
        "n_total_entries": int(matrix.size),
        "symmetric": bool(
            np.allclose(
                matrix,
                matrix.T,
                atol=1e-10,
                equal_nan=True,
            )
        ),
        "diagonal_minimum": float(
            np.nanmin(diagonal)
        ),
        "diagonal_maximum": float(
            np.nanmax(diagonal)
        ),
        "matrix_minimum": float(
            np.nanmin(matrix)
        ),
        "matrix_maximum": float(
            np.nanmax(matrix)
        ),
    }


def write_matrix(
    matrix: np.ndarray,
    variant_ids: list[str],
    path: Path,
) -> None:
    pd.DataFrame(
        matrix,
        index=variant_ids,
        columns=variant_ids,
    ).to_csv(
        path,
        index=True,
        index_label="variant_id",
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in [args.manifest, args.requests]:
        if not path.exists():
            raise SystemExit(f"Missing input: {path}")

    import hail as hl
    import pyspark
    from hail.linalg import BlockMatrix

    manifest = pd.read_parquet(args.manifest)
    requests = pd.read_parquet(args.requests)

    require_columns(
        manifest,
        {
            "comparison_uid",
            "gene_trait_uid",
            "gene",
            "candidate_source",
            "candidate_trait_name",
            "comparison_population",
            "portability_ld_request_ready",
            "eur_variant_index",
            "comparison_variant_index",
            "eur_ld_matrix",
            "comparison_ld_matrix",
        },
        "comparison manifest",
    )
    require_columns(
        requests,
        {
            "comparison_uid",
            "candidate_source",
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

    pilot, selected = choose_pilot(
        manifest,
        requests,
        args.comparison_uid,
    )
    comparison_population = clean(
        pilot["comparison_population"]
    ).upper()

    print("Pilot comparison")
    print("  comparison_uid:", pilot["comparison_uid"])
    print("  gene:", pilot["gene"])
    print("  trait:", pilot["candidate_trait_name"])
    print("  ancestry:", comparison_population)
    print("  candidate variants:", len(selected))
    print("  Hail:", hl.__version__)
    print("  PySpark:", pyspark.__version__)

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
        master=args.hail_master,
        tmp_dir=(
            "file:///tmp/target_ancestry_hail"
        ),
        local_tmpdir=(
            "/tmp/target_ancestry_hail"
        ),
        log=(
            "/tmp/"
            "target_ancestry_15c2b_panukb.log"
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

    try:
        request_ht = build_request_table(
            hl,
            selected,
        )

        branch_specs = {
            "EUR": {
                "index_path": to_s3a(
                    pilot["eur_variant_index"]
                ),
                "matrix_path": to_s3a(
                    pilot["eur_ld_matrix"]
                ),
            },
            comparison_population: {
                "index_path": to_s3a(
                    pilot[
                        "comparison_variant_index"
                    ]
                ),
                "matrix_path": to_s3a(
                    pilot[
                        "comparison_ld_matrix"
                    ]
                ),
            },
        }

        resolutions = {}
        branch_metadata = {}

        for population, spec in branch_specs.items():
            print()
            print(
                f"Resolving {population} indices:"
            )
            print(" ", spec["index_path"])
            resolution, metadata = resolve_indices(
                hl,
                request_ht,
                spec["index_path"],
            )
            resolutions[population] = resolution
            branch_metadata[population] = {
                **metadata,
                "matrix_path": spec[
                    "matrix_path"
                ],
            }
            print(
                "  resolved:",
                metadata["n_resolved"],
                "/",
                metadata["n_requested"],
            )

        resolution_output = selected[
            [
                "variant_id",
                "chrom",
                "pos",
                "ref",
                "alt",
                "eur_p",
                "eur_p_rank",
                "request_order",
            ]
        ].copy()

        for population, resolution in (
            resolutions.items()
        ):
            mapping = dict(
                zip(
                    resolution["variant_id"],
                    resolution["panel_idx"],
                )
            )
            resolution_output[
                f"{population}_panel_idx"
            ] = resolution_output[
                "variant_id"
            ].map(mapping)

        required_index_columns = [
            "EUR_panel_idx",
            f"{comparison_population}_panel_idx",
        ]
        resolution_output["resolved_in_both"] = (
            resolution_output[
                required_index_columns
            ].notna().all(axis=1)
        )

        common = resolution_output[
            resolution_output[
                "resolved_in_both"
            ]
        ].sort_values(
            "request_order",
            kind="mergesort",
        )
        variant_ids = common[
            "variant_id"
        ].tolist()

        resolution_path = (
            args.output_dir
            / "15c2b_panukb_variant_resolution.csv"
        )
        resolution_output.to_csv(
            resolution_path,
            index=False,
        )

        if len(variant_ids) < args.minimum_variants:
            raise RuntimeError(
                "Only "
                f"{len(variant_ids)} variants resolved "
                "in both populations; minimum is "
                f"{args.minimum_variants}."
            )

        outputs = {
            "variant_resolution": str(
                resolution_path
            )
        }
        branch_results = {}

        for population, spec in branch_specs.items():
            index_column = (
                f"{population}_panel_idx"
            )
            indices = common[
                index_column
            ].astype(int).tolist()

            if len(set(indices)) != len(indices):
                raise RuntimeError(
                    f"{population} panel indices are not unique."
                )

            # Hail BlockMatrix.filter requires strictly increasing
            # row and column index lists. The scientific variant order
            # is EUR p-value rank, which does not generally match panel
            # index order. Extract in sorted panel-index order, then
            # permute the result back to the frozen scientific order.
            sorted_positions = np.argsort(
                np.asarray(indices),
                kind="stable",
            )
            sorted_indices = [
                indices[int(position)]
                for position in sorted_positions
            ]
            restore_positions = np.argsort(
                sorted_positions,
                kind="stable",
            )

            print()
            print(
                f"Extracting {population} "
                f"{len(indices)}x{len(indices)} subset:"
            )
            print(" ", spec["matrix_path"])
            print(
                "  panel indices sorted for Hail:",
                sorted_indices,
            )

            matrix = BlockMatrix.read(
                spec["matrix_path"]
            )
            raw_sorted = matrix.filter(
                sorted_indices,
                sorted_indices,
            ).to_numpy()
            raw = raw_sorted[
                np.ix_(
                    restore_positions,
                    restore_positions,
                )
            ]
            symmetric, reconstruction = (
                reconstruct_symmetric(raw)
            )
            r2 = np.square(symmetric)

            raw_path = (
                args.output_dir
                / (
                    "15c2b_panukb_ld_raw_"
                    f"{population}.csv"
                )
            )
            symmetric_path = (
                args.output_dir
                / (
                    "15c2b_panukb_ld_symmetric_"
                    f"{population}.csv"
                )
            )
            r2_path = (
                args.output_dir
                / (
                    "15c2b_panukb_ld_r2_"
                    f"{population}.csv"
                )
            )

            write_matrix(
                raw,
                variant_ids,
                raw_path,
            )
            write_matrix(
                symmetric,
                variant_ids,
                symmetric_path,
            )
            write_matrix(
                r2,
                variant_ids,
                r2_path,
            )

            diagnostics = matrix_diagnostics(
                symmetric
            )
            print(
                "  symmetric:",
                diagnostics["symmetric"],
            )
            print(
                "  finite:",
                diagnostics[
                    "all_entries_finite"
                ],
            )
            print(
                "  diagonal range:",
                diagnostics[
                    "diagonal_minimum"
                ],
                "to",
                diagnostics[
                    "diagonal_maximum"
                ],
            )

            branch_results[population] = {
                **branch_metadata[population],
                "indices_in_variant_order": (
                    indices
                ),
                "indices_sorted_for_hail": (
                    sorted_indices
                ),
                "sorted_positions": [
                    int(value)
                    for value in sorted_positions
                ],
                "restore_positions": [
                    int(value)
                    for value in restore_positions
                ],
                "raw_matrix_shape": list(
                    raw.shape
                ),
                "reconstruction": (
                    reconstruction
                ),
                "symmetric_diagnostics": (
                    diagnostics
                ),
                "outputs": {
                    "raw": str(raw_path),
                    "symmetric": str(
                        symmetric_path
                    ),
                    "r2": str(r2_path),
                },
            }
            outputs[
                f"{population}_raw"
            ] = str(raw_path)
            outputs[
                f"{population}_symmetric"
            ] = str(symmetric_path)
            outputs[
                f"{population}_r2"
            ] = str(r2_path)

    finally:
        hl.stop()

    pass_conditions = [
        len(variant_ids) >= args.minimum_variants,
        all(
            result[
                "symmetric_diagnostics"
            ]["all_entries_finite"]
            for result in branch_results.values()
        ),
        all(
            result[
                "symmetric_diagnostics"
            ]["symmetric"]
            for result in branch_results.values()
        ),
    ]
    status = (
        "PASS"
        if all(pass_conditions)
        else "FAIL"
    )

    summary = {
        "step": "15C2B",
        "script_version": SCRIPT_VERSION,
        "status": status,
        "comparison_uid": clean(
            pilot["comparison_uid"]
        ),
        "gene_trait_uid": clean(
            pilot["gene_trait_uid"]
        ),
        "gene": clean(pilot["gene"]),
        "trait": clean(
            pilot["candidate_trait_name"]
        ),
        "comparison_population": (
            comparison_population
        ),
        "n_requested_variants": int(
            len(selected)
        ),
        "n_resolved_in_both": int(
            len(variant_ids)
        ),
        "variant_order": variant_ids,
        "branches": branch_results,
        "outputs": outputs,
        "scientific_scope": (
            "Subset-LD extraction and triangular "
            "reconstruction only; no pruning or "
            "portability slope."
        ),
    }
    summary_path = (
        args.output_dir
        / "15c2b_panukb_subset_ld_pilot.json"
    )
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(
        "STEP 15C2B — PAN-UKB SUBSET-LD PILOT"
    )
    print("=" * 78)
    print("Status:", status)
    print(
        "Variants resolved in both:",
        len(variant_ids),
        "/",
        len(selected),
    )
    print("Summary:", summary_path)
    print(
        "No LD pruning or portability slope "
        "was calculated."
    )
    print("=" * 78)

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
