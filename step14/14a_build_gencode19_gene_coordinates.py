#!/usr/bin/env python3
"""
Step 14A: Build independent GRCh37 gene coordinates from native GENCODE v19.

Inputs
------
input/reference/gencode.v19.annotation.gtf.gz
output/13_panukb_direct_pairs_locked.parquet

Outputs
-------
output/14_gencode19_all_gene_coordinates.parquet
output/14_locked_gene_coordinates_grch37.parquet
output/14_locked_gene_coordinates_grch37.csv
output/14_locked_genes_unmatched.csv
output/14_gencode19_coordinate_summary.json

Coordinate convention
---------------------
GENCODE GTF positions are 1-based and inclusive. Outputs retain:
- start_1based
- end_1based
and add:
- start_0based = start_1based - 1
for tabix/BED-style interval construction.

When a gene symbol has multiple GENCODE gene records on the same chromosome,
the outer span is used and the number of source records is reported.
Symbols appearing on multiple chromosomes are not silently resolved.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


AUTOSOMES_SEX = {str(i) for i in range(1, 23)} | {"X", "Y"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--gtf",
        type=Path,
        default=Path("input/reference/gencode.v19.annotation.gtf.gz"),
    )
    p.add_argument(
        "--pairs",
        type=Path,
        default=Path("../step13/output/13_panukb_direct_pairs_locked.parquet"),
    )
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    return p.parse_args()


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_attributes(text: str) -> dict[str, str]:
    attrs = {}
    for item in text.strip().strip(";").split(";"):
        item = item.strip()
        if not item:
            continue
        match = re.match(r'(\S+)\s+"(.*)"$', item)
        if match:
            attrs[match.group(1)] = match.group(2)
    return attrs


def normalize_chrom(value: str) -> str:
    chrom = str(value)
    if chrom.startswith("chr"):
        chrom = chrom[3:]
    return chrom


def parse_gene_rows(gtf: Path) -> pd.DataFrame:
    rows = []
    with gzip.open(gtf, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                raise ValueError(
                    f"Malformed GTF line {line_number}: expected 9 fields, "
                    f"found {len(fields)}"
                )
            chrom, source, feature, start, end, score, strand, frame, attr_text = fields
            if feature != "gene":
                continue

            chrom = normalize_chrom(chrom)
            if chrom not in AUTOSOMES_SEX:
                continue

            attrs = parse_attributes(attr_text)
            gene_id = attrs.get("gene_id", "")
            gene_name = attrs.get("gene_name", "")
            gene_type = attrs.get("gene_type", attrs.get("gene_biotype", ""))
            gene_status = attrs.get("gene_status", "")

            if not gene_name:
                continue

            rows.append({
                "chrom": chrom,
                "start_1based": int(start),
                "end_1based": int(end),
                "strand": strand,
                "source": source,
                "gene_id": gene_id,
                "gene": gene_name,
                "gene_type": gene_type,
                "gene_status": gene_status,
            })

    if not rows:
        raise ValueError("No gene records parsed from GTF.")
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for path in [args.gtf, args.pairs]:
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")

    print(f"Parsing native GENCODE v19 GTF: {args.gtf}")
    genes = parse_gene_rows(args.gtf)
    print(f"  Gene records parsed: {len(genes):,}")
    print(f"  Unique symbols:      {genes['gene'].nunique():,}")

    # Detect symbols on multiple chromosomes before collapsing coordinates.
    chromosome_counts = (
        genes.groupby("gene")["chrom"].nunique().rename("n_chromosomes")
    )
    genes = genes.merge(chromosome_counts, on="gene", how="left")

    ambiguous = genes.loc[genes["n_chromosomes"] > 1].copy()

    unambiguous = genes.loc[genes["n_chromosomes"] == 1].copy()
    collapsed = (
        unambiguous.groupby(["gene", "chrom"], as_index=False)
        .agg(
            start_1based=("start_1based", "min"),
            end_1based=("end_1based", "max"),
            strand=("strand", lambda s: ",".join(sorted(set(s)))),
            gene_ids=("gene_id", lambda s: " || ".join(sorted(set(s)))),
            gene_types=("gene_type", lambda s: " || ".join(sorted(set(s)))),
            gene_statuses=("gene_status", lambda s: " || ".join(sorted(set(s)))),
            annotation_sources=("source", lambda s: " || ".join(sorted(set(s)))),
            n_gene_records=("gene_id", "size"),
        )
    )
    collapsed["start_0based"] = collapsed["start_1based"] - 1
    collapsed["coordinate_source"] = "GENCODE release 19"
    collapsed["genome_assembly"] = "GRCh37.p13"
    collapsed["coordinate_rule"] = (
        "Outer span of all GENCODE v19 gene records sharing the same symbol "
        "on one chromosome."
    )

    all_path = args.output_dir / "14_gencode19_all_gene_coordinates.parquet"
    collapsed.to_parquet(all_path, index=False)

    pairs = pd.read_parquet(args.pairs)
    if "gene" not in pairs.columns:
        raise SystemExit("Locked pair file lacks gene column.")

    locked_genes = (
        pairs[["gene"]]
        .drop_duplicates()
        .sort_values("gene")
        .reset_index(drop=True)
    )

    matched = locked_genes.merge(
        collapsed,
        on="gene",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = matched.loc[matched["_merge"] != "both", ["gene"]].copy()
    matched = matched.loc[matched["_merge"] == "both"].drop(columns="_merge")

    # Add counts of pairs and indications per gene.
    gene_usage = (
        pairs.groupby("gene")
        .agg(
            n_pairs=("ti_uid", "nunique"),
            n_indications=("indication_mesh_id", "nunique"),
            n_A=("pool", lambda s: int((s == "A").sum())),
            n_B=("pool", lambda s: int((s == "B").sum())),
        )
        .reset_index()
    )
    matched = matched.merge(gene_usage, on="gene", how="left", validate="one_to_one")

    locked_parquet = (
        args.output_dir / "14_locked_gene_coordinates_grch37.parquet"
    )
    locked_csv = args.output_dir / "14_locked_gene_coordinates_grch37.csv"
    unmatched_path = args.output_dir / "14_locked_genes_unmatched.csv"

    matched.to_parquet(locked_parquet, index=False)
    matched.to_csv(locked_csv, index=False)
    unmatched.to_csv(unmatched_path, index=False)

    ambiguous_locked = sorted(
        set(locked_genes["gene"]) & set(ambiguous["gene"])
    )

    summary = {
        "annotation": {
            "resource": "GENCODE",
            "release": 19,
            "assembly": "GRCh37.p13",
            "filename": args.gtf.name,
            "size_bytes": args.gtf.stat().st_size,
            "md5": digest(args.gtf, "md5"),
            "sha256": digest(args.gtf, "sha256"),
        },
        "gtf_gene_records": int(len(genes)),
        "gtf_unique_symbols": int(genes["gene"].nunique()),
        "gtf_unambiguous_collapsed_symbols": int(collapsed["gene"].nunique()),
        "locked_unique_genes": int(len(locked_genes)),
        "locked_genes_matched": int(len(matched)),
        "locked_genes_unmatched": int(len(unmatched)),
        "locked_multichromosome_symbols": ambiguous_locked,
        "coordinate_conventions": {
            "gtf_start_end": "1-based inclusive",
            "start_0based": "start_1based - 1",
            "planned_window": "Not applied in this step",
        },
        "outputs": {
            "all_coordinates": str(all_path),
            "locked_coordinates_parquet": str(locked_parquet),
            "locked_coordinates_csv": str(locked_csv),
            "unmatched": str(unmatched_path),
        },
    }

    summary_path = args.output_dir / "14_gencode19_coordinate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("=" * 76)
    print("STEP 14A GENCODE V19 GRCh37 COORDINATES")
    print("=" * 76)
    print(f"Locked genes:                  {len(locked_genes):,}")
    print(f"Matched unambiguously:         {len(matched):,}")
    print(f"Unmatched:                     {len(unmatched):,}")
    print(f"Multichromosome locked symbols:{len(ambiguous_locked):,}")
    if len(unmatched):
        print("\nUnmatched symbols:")
        print(unmatched.to_string(index=False))
    if ambiguous_locked:
        print("\nMultichromosome symbols requiring review:")
        print("\n".join(ambiguous_locked))
    print("\nOutputs:")
    print(f"  {locked_parquet}")
    print(f"  {locked_csv}")
    print(f"  {unmatched_path}")
    print(f"  {summary_path}")
    print("=" * 76)


if __name__ == "__main__":
    main()
