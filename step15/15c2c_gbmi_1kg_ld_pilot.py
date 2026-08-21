#!/usr/bin/env python3
"""
Step 15C2C — GBMI / 1000 Genomes GRCh38 LD pilot.

Selects one primary, nonmixed, LD-ready GBMI comparison from the frozen
Step 15C1 requests, reads the official 1000 Genomes sample panel, opens
the chromosome-level phased GRCh38 VCF remotely with pysam, resolves the
requested variants by exact CHR/POS/REF/ALT, and calculates EUR and
comparison-population dosage-correlation LD matrices.

This is a preflight only. It does not perform greedy LD pruning or fit a
portability slope.

Default inputs
--------------
output/15c1_portability_comparison_manifest.parquet
output/15c1_portability_variant_requests_primary.parquet

Outputs
-------
input/reference/1000G/integrated_call_samples_v3.20130502.ALL.panel
output/15c2c_gbmi_variant_resolution.csv
output/15c2c_gbmi_ld_r_EUR.csv
output/15c2c_gbmi_ld_r_<POP>.csv
output/15c2c_gbmi_ld_r2_EUR.csv
output/15c2c_gbmi_ld_r2_<POP>.csv
output/15c2c_gbmi_pairwise_n_EUR.csv
output/15c2c_gbmi_pairwise_n_<POP>.csv
output/15c2c_gbmi_1kg_ld_pilot.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pysam


SCRIPT_VERSION = "15C2C.1"


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
        help="Optional exact GBMI comparison override.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("input/reference/1000G"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
    )
    parser.add_argument(
        "--minimum-usable-variants",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--preferred-candidate-variants",
        type=int,
        default=5,
        help=(
            "Prefer the smallest pilot with at least this many candidates, "
            "then fall back to the smallest comparison with at least the "
            "minimum usable count."
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


def normalize_chrom(value: Any) -> str:
    chrom = clean(value)
    if chrom.casefold().startswith("chr"):
        chrom = chrom[3:]
    if chrom == "M":
        chrom = "MT"
    return chrom


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json_write(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"{label} missing required columns: {missing}")


def download_file(url: str, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        return {
            "url": url,
            "path": str(destination),
            "downloaded_now": False,
            "size_bytes": int(destination.stat().st_size),
            "sha256": sha256_file(destination),
        }

    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "target-ancestry-step15c2c/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
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


def choose_pilot(
    manifest: pd.DataFrame,
    requests: pd.DataFrame,
    comparison_uid: str,
    preferred_candidates: int,
    minimum_candidates: int,
) -> tuple[pd.Series, pd.DataFrame]:
    candidates = manifest[
        manifest["candidate_source"].eq("GBMI")
        & manifest["portability_ld_request_ready"].map(as_bool)
    ].copy()

    counts = (
        requests[requests["candidate_source"].eq("GBMI")]
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
                f"{len(candidates)} LD-ready GBMI comparisons."
            )
    else:
        preferred = candidates[
            candidates["n_candidate_variants"].ge(preferred_candidates)
        ]
        if preferred.empty:
            preferred = candidates[
                candidates["n_candidate_variants"].ge(minimum_candidates)
            ]

        candidates = preferred.sort_values(
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
        raise SystemExit("No usable primary GBMI LD pilot comparison found.")

    pilot = candidates.iloc[0]
    selected = requests[
        requests["comparison_uid"].eq(pilot["comparison_uid"])
    ].copy()
    selected = selected.sort_values(
        ["eur_p_rank", "eur_p", "pos", "ref", "alt"],
        kind="mergesort",
    ).reset_index(drop=True)
    selected["request_order"] = np.arange(len(selected), dtype=int)

    return pilot, selected


def read_sample_panel(path: Path) -> pd.DataFrame:
    panel = pd.read_csv(path, sep="\t", dtype=str)
    panel.columns = [
        re.sub(r"[^a-z0-9]+", "_", clean(column).casefold()).strip("_")
        for column in panel.columns
    ]

    aliases = {
        "sample": ["sample", "sample_id"],
        "population": ["pop", "population"],
        "super_population": ["super_pop", "super_population"],
    }
    resolved: dict[str, str] = {}
    for semantic, options in aliases.items():
        for option in options:
            if option in panel.columns:
                resolved[semantic] = option
                break

    missing = sorted(set(aliases) - set(resolved))
    if missing:
        raise RuntimeError(
            "1000 Genomes panel is missing required fields "
            f"{missing}; observed={panel.columns.tolist()}"
        )

    panel = panel.rename(
        columns={
            resolved["sample"]: "sample",
            resolved["population"]: "population",
            resolved["super_population"]: "super_population",
        }
    )
    for column in ["sample", "population", "super_population"]:
        panel[column] = panel[column].map(clean)
    return panel


def resolve_contig(vcf: pysam.VariantFile, chrom: str) -> str:
    target = normalize_chrom(chrom)
    contigs = list(vcf.header.contigs)
    for candidate in [target, f"chr{target}"]:
        if candidate in contigs:
            return candidate

    mapping = {normalize_chrom(contig): contig for contig in contigs}
    if target in mapping:
        return mapping[target]

    raise RuntimeError(
        f"Chromosome {target} is absent from VCF contigs."
    )


def genotype_dosages(record, samples: list[str]) -> np.ndarray:
    dosages = np.full(len(samples), np.nan, dtype=float)
    for index, sample in enumerate(samples):
        genotype = record.samples[sample].get("GT")
        if (
            genotype is None
            or len(genotype) != 2
            or genotype[0] is None
            or genotype[1] is None
        ):
            continue
        dosages[index] = float(genotype[0] + genotype[1])
    return dosages


def dosage_metrics(dosages: np.ndarray) -> dict[str, Any]:
    called = dosages[np.isfinite(dosages)]
    n_called = int(called.size)
    n_total = int(dosages.size)
    if n_called == 0:
        return {
            "n_total": n_total,
            "n_called": 0,
            "call_rate": 0.0,
            "alternate_allele_frequency": None,
            "minor_allele_frequency": None,
            "dosage_variance": None,
            "polymorphic": False,
        }

    alternate_af = float(np.mean(called) / 2.0)
    minor_af = float(min(alternate_af, 1.0 - alternate_af))
    variance = float(np.var(called, ddof=1)) if n_called > 1 else 0.0

    return {
        "n_total": n_total,
        "n_called": n_called,
        "call_rate": float(n_called / n_total),
        "alternate_allele_frequency": alternate_af,
        "minor_allele_frequency": minor_af,
        "dosage_variance": variance,
        "polymorphic": bool(n_called >= 3 and variance > 0.0),
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
        for second in range(first, n_variants):
            x = genotype_matrix[first]
            y = genotype_matrix[second]
            complete = np.isfinite(x) & np.isfinite(y)
            n_complete = int(complete.sum())
            pairwise_n[first, second] = n_complete
            pairwise_n[second, first] = n_complete

            if n_complete < 3:
                continue

            x_complete = x[complete]
            y_complete = y[complete]
            x_sd = float(np.std(x_complete, ddof=1))
            y_sd = float(np.std(y_complete, ddof=1))
            if x_sd == 0.0 or y_sd == 0.0:
                continue

            value = float(np.corrcoef(x_complete, y_complete)[0, 1])
            correlation[first, second] = value
            correlation[second, first] = value

    return correlation, pairwise_n


def matrix_diagnostics(matrix: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(matrix)
    diagonal = np.diag(matrix)
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
        "diagonal_minimum": (
            float(np.nanmin(diagonal)) if diagonal.size else None
        ),
        "diagonal_maximum": (
            float(np.nanmax(diagonal)) if diagonal.size else None
        ),
        "matrix_minimum": (
            float(np.nanmin(matrix)) if matrix.size else None
        ),
        "matrix_maximum": (
            float(np.nanmax(matrix)) if matrix.size else None
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
    ).to_csv(path, index=True, index_label="variant_id")


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
        return pysam.VariantFile(vcf_url)


def main() -> int:
    args = parse_args()
    args.reference_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for path in [args.manifest, args.requests]:
        if not path.exists():
            raise SystemExit(f"Missing input: {path}")

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
            "comparison_reference_population",
            "portability_ld_request_ready",
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
        args.preferred_candidate_variants,
        args.minimum_usable_variants,
    )

    comparison_population = clean(
        pilot["comparison_population"]
    ).upper()
    comparison_reference = clean(
        pilot["comparison_reference_population"]
    ).upper()
    vcf_url = clean(pilot["kgp_vcf_url"])
    index_url = clean(pilot["kgp_vcf_index_url"])
    panel_url = clean(pilot["kgp_sample_panel_url"])

    print("Pilot comparison")
    print("  comparison_uid:", pilot["comparison_uid"])
    print("  gene:", pilot["gene"])
    print("  trait:", pilot["candidate_trait_name"])
    print("  analysis ancestry:", comparison_population)
    print("  reference ancestry:", comparison_reference)
    print("  candidate variants:", len(selected))
    print("  chromosome:", normalize_chrom(selected["chrom"].iloc[0]))
    print("  VCF:", vcf_url)

    panel_path = (
        args.reference_dir
        / "integrated_call_samples_v3.20130502.ALL.panel"
    )
    panel_download = download_file(panel_url, panel_path)
    panel = read_sample_panel(panel_path)

    eur_panel_samples = panel.loc[
        panel["super_population"].eq("EUR"),
        "sample",
    ].tolist()
    comparison_panel_samples = panel.loc[
        panel["super_population"].eq(comparison_reference),
        "sample",
    ].tolist()

    if not eur_panel_samples:
        raise RuntimeError("No EUR samples found in the sample panel.")
    if not comparison_panel_samples:
        raise RuntimeError(
            "No samples found for reference population "
            f"{comparison_reference}."
        )

    print("  EUR panel samples:", len(eur_panel_samples))
    print(
        f"  {comparison_reference} panel samples:",
        len(comparison_panel_samples),
    )
    print()
    print("Opening remote indexed VCF...")

    vcf = open_remote_vcf(vcf_url, index_url)
    try:
        available = set(vcf.header.samples)
        eur_samples = [
            sample for sample in eur_panel_samples if sample in available
        ]
        comparison_samples = [
            sample
            for sample in comparison_panel_samples
            if sample in available
        ]

        if len(eur_samples) < 3:
            raise RuntimeError(
                "Fewer than 3 EUR samples are present in the VCF."
            )
        if len(comparison_samples) < 3:
            raise RuntimeError(
                f"Fewer than 3 {comparison_reference} samples "
                "are present in the VCF."
            )

        selected_samples = list(
            dict.fromkeys(eur_samples + comparison_samples)
        )
        vcf.subset_samples(selected_samples)

        chrom_values = {
            normalize_chrom(value) for value in selected["chrom"]
        }
        if len(chrom_values) != 1:
            raise RuntimeError(
                "Pilot comparison spans multiple chromosomes: "
                f"{sorted(chrom_values)}"
            )
        chrom = next(iter(chrom_values))
        contig = resolve_contig(vcf, chrom)

        positions = pd.to_numeric(
            selected["pos"],
            errors="raise",
        ).astype(int)
        start_zero_based = max(0, int(positions.min()) - 1)
        end_one_based = int(positions.max())

        request_lookup = {
            (
                normalize_chrom(row.chrom),
                int(row.pos),
                clean(row.ref).upper(),
                clean(row.alt).upper(),
            ): clean(row.variant_id)
            for row in selected.itertuples()
        }

        print(
            "Fetching region:",
            f"{contig}:{start_zero_based + 1}-{end_one_based}",
        )

        found: dict[str, dict[str, Any]] = {}
        records_scanned = 0

        for record in vcf.fetch(
            contig,
            start_zero_based,
            end_one_based,
        ):
            records_scanned += 1
            if not record.alts or len(record.alts) != 1:
                continue

            key = (
                normalize_chrom(record.contig),
                int(record.pos),
                clean(record.ref).upper(),
                clean(record.alts[0]).upper(),
            )
            variant_id = request_lookup.get(key)
            if not variant_id:
                continue

            found[variant_id] = {
                "vcf_id": clean(record.id),
                "eur_dosage": genotype_dosages(record, eur_samples),
                "comparison_dosage": genotype_dosages(
                    record,
                    comparison_samples,
                ),
            }
    finally:
        vcf.close()

    resolution = selected[
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
    resolution["resolved_in_vcf"] = resolution["variant_id"].isin(found)
    resolution["vcf_id"] = resolution["variant_id"].map(
        lambda variant_id: found.get(variant_id, {}).get("vcf_id", "")
    )

    eur_metrics: dict[str, dict[str, Any]] = {}
    comparison_metrics: dict[str, dict[str, Any]] = {}
    for variant_id, data in found.items():
        eur_metrics[variant_id] = dosage_metrics(data["eur_dosage"])
        comparison_metrics[variant_id] = dosage_metrics(
            data["comparison_dosage"]
        )

    for prefix, metric_map in [
        ("EUR", eur_metrics),
        (comparison_reference, comparison_metrics),
    ]:
        resolution[f"{prefix}_n_called"] = resolution["variant_id"].map(
            lambda variant_id: metric_map.get(variant_id, {}).get(
                "n_called"
            )
        )
        resolution[f"{prefix}_call_rate"] = resolution["variant_id"].map(
            lambda variant_id: metric_map.get(variant_id, {}).get(
                "call_rate"
            )
        )
        resolution[f"{prefix}_alternate_af"] = resolution[
            "variant_id"
        ].map(
            lambda variant_id: metric_map.get(variant_id, {}).get(
                "alternate_allele_frequency"
            )
        )
        resolution[f"{prefix}_minor_af"] = resolution["variant_id"].map(
            lambda variant_id: metric_map.get(variant_id, {}).get(
                "minor_allele_frequency"
            )
        )
        resolution[f"{prefix}_dosage_variance"] = resolution[
            "variant_id"
        ].map(
            lambda variant_id: metric_map.get(variant_id, {}).get(
                "dosage_variance"
            )
        )
        resolution[f"{prefix}_polymorphic"] = resolution["variant_id"].map(
            lambda variant_id: metric_map.get(variant_id, {}).get(
                "polymorphic",
                False,
            )
        )

    resolution["usable_in_both"] = (
        resolution["resolved_in_vcf"]
        & resolution["EUR_polymorphic"].fillna(False)
        & resolution[f"{comparison_reference}_polymorphic"].fillna(False)
    )

    usable = resolution[
        resolution["usable_in_both"]
    ].sort_values("request_order", kind="mergesort")
    variant_ids = usable["variant_id"].tolist()

    resolution_path = (
        args.output_dir / "15c2c_gbmi_variant_resolution.csv"
    )
    resolution.to_csv(resolution_path, index=False)

    if len(variant_ids) < args.minimum_usable_variants:
        summary = {
            "step": "15C2C",
            "script_version": SCRIPT_VERSION,
            "status": "FAIL",
            "failure_reason": (
                f"Only {len(variant_ids)} variants were exact, called, "
                "and polymorphic in both 1000 Genomes populations; "
                f"minimum={args.minimum_usable_variants}."
            ),
            "comparison_uid": clean(pilot["comparison_uid"]),
            "gene": clean(pilot["gene"]),
            "trait": clean(pilot["candidate_trait_name"]),
            "comparison_population": comparison_population,
            "comparison_reference_population": comparison_reference,
            "n_requested_variants": int(len(selected)),
            "n_resolved_exact": int(resolution["resolved_in_vcf"].sum()),
            "n_usable_in_both": int(len(variant_ids)),
            "records_scanned": int(records_scanned),
            "variant_resolution": str(resolution_path),
            "completed_at_utc": utc_now(),
        }
        summary_path = (
            args.output_dir / "15c2c_gbmi_1kg_ld_pilot.json"
        )
        atomic_json_write(summary, summary_path)
        print()
        print(summary["failure_reason"])
        print("Summary:", summary_path)
        return 1

    eur_genotypes = np.vstack(
        [found[variant_id]["eur_dosage"] for variant_id in variant_ids]
    )
    comparison_genotypes = np.vstack(
        [
            found[variant_id]["comparison_dosage"]
            for variant_id in variant_ids
        ]
    )

    eur_r, eur_pairwise_n = pairwise_correlation(eur_genotypes)
    comparison_r, comparison_pairwise_n = pairwise_correlation(
        comparison_genotypes
    )
    eur_r2 = np.square(eur_r)
    comparison_r2 = np.square(comparison_r)

    output_paths = {
        "variant_resolution": resolution_path,
        "EUR_r": args.output_dir / "15c2c_gbmi_ld_r_EUR.csv",
        f"{comparison_reference}_r": (
            args.output_dir
            / f"15c2c_gbmi_ld_r_{comparison_reference}.csv"
        ),
        "EUR_r2": args.output_dir / "15c2c_gbmi_ld_r2_EUR.csv",
        f"{comparison_reference}_r2": (
            args.output_dir
            / f"15c2c_gbmi_ld_r2_{comparison_reference}.csv"
        ),
        "EUR_pairwise_n": (
            args.output_dir / "15c2c_gbmi_pairwise_n_EUR.csv"
        ),
        f"{comparison_reference}_pairwise_n": (
            args.output_dir
            / f"15c2c_gbmi_pairwise_n_{comparison_reference}.csv"
        ),
    }

    write_matrix(eur_r, variant_ids, output_paths["EUR_r"])
    write_matrix(
        comparison_r,
        variant_ids,
        output_paths[f"{comparison_reference}_r"],
    )
    write_matrix(eur_r2, variant_ids, output_paths["EUR_r2"])
    write_matrix(
        comparison_r2,
        variant_ids,
        output_paths[f"{comparison_reference}_r2"],
    )
    write_matrix(
        eur_pairwise_n,
        variant_ids,
        output_paths["EUR_pairwise_n"],
    )
    write_matrix(
        comparison_pairwise_n,
        variant_ids,
        output_paths[f"{comparison_reference}_pairwise_n"],
    )

    eur_diagnostics = matrix_diagnostics(eur_r)
    comparison_diagnostics = matrix_diagnostics(comparison_r)

    status = (
        "PASS"
        if (
            eur_diagnostics["all_entries_finite"]
            and comparison_diagnostics["all_entries_finite"]
            and eur_diagnostics["symmetric"]
            and comparison_diagnostics["symmetric"]
        )
        else "FAIL"
    )

    summary = {
        "step": "15C2C",
        "script_version": SCRIPT_VERSION,
        "status": status,
        "comparison_uid": clean(pilot["comparison_uid"]),
        "gene_trait_uid": clean(pilot["gene_trait_uid"]),
        "gene": clean(pilot["gene"]),
        "trait": clean(pilot["candidate_trait_name"]),
        "comparison_population": comparison_population,
        "comparison_reference_population": comparison_reference,
        "vcf_url": vcf_url,
        "vcf_index_url": index_url,
        "sample_panel": panel_download,
        "records_scanned": int(records_scanned),
        "n_requested_variants": int(len(selected)),
        "n_resolved_exact": int(resolution["resolved_in_vcf"].sum()),
        "n_usable_in_both": int(len(variant_ids)),
        "variant_order": variant_ids,
        "n_eur_panel_samples": int(len(eur_panel_samples)),
        "n_comparison_panel_samples": int(
            len(comparison_panel_samples)
        ),
        "n_eur_vcf_samples": int(len(eur_samples)),
        "n_comparison_vcf_samples": int(len(comparison_samples)),
        "eur_r_diagnostics": eur_diagnostics,
        "comparison_r_diagnostics": comparison_diagnostics,
        "outputs": {
            key: str(value) for key, value in output_paths.items()
        },
        "scientific_scope": (
            "Exact variant resolution and population-specific dosage LD "
            "only; no greedy pruning or portability slope."
        ),
        "completed_at_utc": utc_now(),
    }
    summary_path = args.output_dir / "15c2c_gbmi_1kg_ld_pilot.json"
    atomic_json_write(summary, summary_path)

    print()
    print("=" * 78)
    print("STEP 15C2C — GBMI / 1000 GENOMES LD PILOT")
    print("=" * 78)
    print("Status:", status)
    print("Exact variants resolved:", int(resolution["resolved_in_vcf"].sum()))
    print(
        "Variants usable in both populations:",
        len(variant_ids),
        "/",
        len(selected),
    )
    print("Summary:", summary_path)
    print("No LD pruning or portability slope was calculated.")
    print("=" * 78)

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
