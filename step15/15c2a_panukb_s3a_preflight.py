#!/usr/bin/env python3
"""
Step 15C2A — Pan-UKB S3A LD preflight.

Validates one primary portability comparison by:
1. selecting the comparison with the smallest candidate set;
2. resolving one requested variant in EUR and comparison-population
   Pan-UKB variant-index Hail Tables;
3. opening both Pan-UKB LD BlockMatrices;
4. reading the variant's diagonal entry from each matrix.

No pruning or portability slope is calculated.

Run with:
    export PYSPARK_SUBMIT_ARGS="--packages org.apache.hadoop:hadoop-aws:3.3.4 pyspark-shell"
    python3 15c2a_panukb_s3a_preflight.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


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
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "output/15c2a_panukb_s3a_preflight.json"
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
    return str(value).strip()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return clean(value).casefold() in {"true", "1", "yes", "y"}


def to_s3a(path: str) -> str:
    path = clean(path)
    if path.startswith("s3://"):
        return "s3a://" + path[len("s3://"):]
    if path.startswith("s3a://"):
        return path
    raise RuntimeError(
        f"Expected an S3 Pan-UKB Hail path, observed: {path}"
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
                f"{len(candidates)} Pan-UKB comparisons."
            )
    else:
        candidates = candidates.sort_values(
            [
                "n_candidate_variants",
                "candidate_trait_name",
                "gene",
                "comparison_population",
                "comparison_uid",
            ]
        ).head(1)

    if candidates.empty:
        raise SystemExit(
            "No LD-ready primary Pan-UKB comparison found."
        )

    pilot = candidates.iloc[0]
    variants = requests[
        requests["comparison_uid"].eq(
            pilot["comparison_uid"]
        )
    ].sort_values(
        ["eur_p_rank", "eur_p", "pos", "ref", "alt"],
        kind="mergesort",
    )
    if variants.empty:
        raise SystemExit(
            "The selected comparison has no requested variants."
        )
    return pilot, variants


def struct_to_dict(value: Any) -> dict[str, Any]:
    try:
        return {
            field: value[field]
            for field in value
        }
    except Exception:
        return {"rendered": str(value)}


def main() -> int:
    args = parse_args()

    for path in [args.manifest, args.requests]:
        if not path.exists():
            raise SystemExit(f"Missing input: {path}")

    import hail as hl
    import pyspark
    from hail.linalg import BlockMatrix

    manifest = pd.read_parquet(args.manifest)
    requests = pd.read_parquet(args.requests)

    pilot, variants = choose_pilot(
        manifest,
        requests,
        args.comparison_uid,
    )
    variant = variants.iloc[0]
    comparison_population = clean(
        pilot["comparison_population"]
    ).upper()

    print("Pilot comparison")
    print("  comparison_uid:", pilot["comparison_uid"])
    print("  gene:", pilot["gene"])
    print("  trait:", pilot["candidate_trait_name"])
    print("  ancestry:", comparison_population)
    print("  candidate variants:", len(variants))
    print("  test variant:", variant["variant_id"])
    print("  Hail:", hl.__version__)
    print("  PySpark:", pyspark.__version__)

    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    os.environ.setdefault(
        "SPARK_LOCAL_DIRS",
        "/tmp/target_ancestry_spark",
    )
    Path("/tmp/target_ancestry_spark").mkdir(
        parents=True,
        exist_ok=True,
    )
    Path("/tmp/target_ancestry_hail").mkdir(
        parents=True,
        exist_ok=True,
    )

    hl.init(
        master="local[2]",
        tmp_dir="file:///tmp/target_ancestry_hail",
        local_tmpdir="/tmp/target_ancestry_hail",
        log="/tmp/target_ancestry_15c2a_panukb.log",
        spark_conf={
            "spark.hadoop.fs.s3a.impl":
                "org.apache.hadoop.fs.s3a.S3AFileSystem",
            "spark.hadoop.fs.s3a.aws.credentials.provider":
                "org.apache.hadoop.fs.s3a."
                "AnonymousAWSCredentialsProvider",
            "spark.hadoop.fs.s3a.endpoint":
                "s3.us-east-1.amazonaws.com",
            "spark.hadoop.fs.s3a.endpoint.region":
                "us-east-1",
            "spark.hadoop.fs.s3a.path.style.access":
                "false",
            "spark.hadoop.fs.s3a.connection.ssl.enabled":
                "true",
        },
        quiet=False,
    )
    hl.default_reference("GRCh37")

    results: dict[str, Any] = {}

    try:
        locus = hl.locus(
            clean(variant["chrom"]),
            int(variant["pos"]),
            reference_genome="GRCh37",
        )
        alleles = [
            clean(variant["ref"]).upper(),
            clean(variant["alt"]).upper(),
        ]

        branch_specs = {
            "EUR": {
                "index": to_s3a(
                    clean(pilot["eur_variant_index"])
                ),
                "matrix": to_s3a(
                    clean(pilot["eur_ld_matrix"])
                ),
            },
            comparison_population: {
                "index": to_s3a(
                    clean(
                        pilot["comparison_variant_index"]
                    )
                ),
                "matrix": to_s3a(
                    clean(
                        pilot["comparison_ld_matrix"]
                    )
                ),
            },
        }

        for population, spec in branch_specs.items():
            print()
            print(
                f"Opening {population} variant index:"
            )
            print(" ", spec["index"])

            index_ht = hl.read_table(spec["index"])
            matched_rows = (
                index_ht
                .filter(
                    (index_ht.locus.contig == clean(variant["chrom"]))
                    & (index_ht.locus.position == int(variant["pos"]))
                    & (index_ht.alleles[0] == alleles[0])
                    & (index_ht.alleles[1] == alleles[1])
                )
                .select("idx")
                .take(1)
            )
            index_row = matched_rows[0] if matched_rows else None

            if index_row is None:
                print("  variant not present")
                results[population] = {
                    "status": "VARIANT_NOT_FOUND",
                    "index_path": spec["index"],
                    "matrix_path": spec["matrix"],
                }
                continue

            idx = int(index_row.idx)
            globals_value = hl.eval(
                index_ht.index_globals()
            )

            print("  matrix index:", idx)
            print("  index globals:", globals_value)
            print(
                f"Opening {population} BlockMatrix:"
            )
            print(" ", spec["matrix"])

            matrix = BlockMatrix.read(spec["matrix"])
            diagonal = float(matrix[idx, idx])

            print("  shape:", matrix.shape)
            print("  block size:", matrix.block_size)
            print("  diagonal LD value:", diagonal)

            results[population] = {
                "status": "PASS",
                "index_path": spec["index"],
                "matrix_path": spec["matrix"],
                "matrix_index": idx,
                "matrix_shape": list(matrix.shape),
                "block_size": int(matrix.block_size),
                "diagonal_ld_value": diagonal,
                "index_globals": struct_to_dict(
                    globals_value
                ),
            }
    finally:
        hl.stop()

    branch_statuses = [
        result["status"]
        for result in results.values()
    ]
    status = (
        "PASS"
        if len(branch_statuses) == 2
        and all(
            branch_status == "PASS"
            for branch_status in branch_statuses
        )
        else "FAIL"
    )

    output = {
        "step": "15C2A",
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
        "n_candidate_variants": int(
            len(variants)
        ),
        "test_variant": clean(
            variant["variant_id"]
        ),
        "branches": results,
        "scope": (
            "S3A access, variant-index lookup, and "
            "single BlockMatrix diagonal only"
        ),
    }
    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("STEP 15C2A — PAN-UKB S3A PREFLIGHT")
    print("=" * 78)
    print("Status:", status)
    print("Output:", args.output)
    print(
        "No LD pruning or portability slope "
        "was calculated."
    )
    print("=" * 78)

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
