#!/usr/bin/env python3
"""
Step 15A2: Verify GBMI source files, reduce Pan-UKB to a clinical core,
and generate conservative indication-to-trait mapping candidates.

This is a candidate-generation step. It does NOT lock mappings and does NOT
calculate portability or colocalization.

Scientific inputs
-----------------
- output/15a_panukb_quantitative_catalog.csv
- output/15a_indication_mapping_review.csv
- Original Pan-UKB manifest-derived metadata already carried into 15A1
- Original public GBMI release bucket:
    https://gbmi-sumstats.s3.amazonaws.com/
  release token:
    GBMI_052021

Explicitly not used
-------------------
- Daniel's gbmi_map.json
- Daniel's pair_trait_map.parquet
- Daniel's portability or colocalization outputs

Outputs
-------
- output/15a2_panukb_core_traits.csv
- output/15a2_gbmi_verified_files.csv
- output/15a2_gbmi_verified_endpoints.csv
- output/15a2_mapping_candidates.csv
- output/15a2_unmapped_indications.csv
- output/15a2_mapping_summary.json
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


GBMI_BUCKET = "https://gbmi-sumstats.s3.amazonaws.com"
GBMI_RELEASE = "052021"
GBMI_SOURCE_POPS = ["AFR", "AMR", "EAS", "EUR", "SAS", "MID"]
POP_HARMONIZATION = {
    "AFR": "AFR",
    "AMR": "AMR",
    "EAS": "EAS",
    "EUR": "EUR",
    "SAS": "CSA",
    "CSA": "CSA",
    "MID": "MID",
}

GBMI_ENDPOINTS = {
    "AAA": {
        "trait_name": "Abdominal aortic aneurysm",
        "endpoint_type": "disease",
        "aliases": ["AAA"],
    },
    "AcApp": {
        "trait_name": "Acute appendicitis",
        "endpoint_type": "disease",
        "aliases": ["AcApp"],
    },
    "Asthma": {
        "trait_name": "Asthma",
        "endpoint_type": "disease",
        "aliases": ["Asthma"],
    },
    "Appendectomy": {
        "trait_name": "Appendectomy",
        "endpoint_type": "procedure",
        "aliases": ["Appendectomy", "Appy", "App"],
    },
    "COPD": {
        "trait_name": "Chronic obstructive pulmonary disease",
        "endpoint_type": "disease",
        "aliases": ["COPD"],
    },
    "Gout": {
        "trait_name": "Gout",
        "endpoint_type": "disease",
        "aliases": ["Gout"],
    },
    "HCM": {
        "trait_name": "Hypertrophic cardiomyopathy",
        "endpoint_type": "disease",
        "aliases": ["HCM"],
    },
    "HF": {
        "trait_name": "Heart failure",
        "endpoint_type": "disease",
        "aliases": ["HF"],
    },
    "IPF": {
        "trait_name": "Idiopathic pulmonary fibrosis",
        "endpoint_type": "disease",
        "aliases": ["IPF"],
    },
    "POAG": {
        "trait_name": "Primary open-angle glaucoma",
        "endpoint_type": "disease",
        "aliases": ["POAG"],
    },
    "Stroke": {
        "trait_name": "Stroke",
        "endpoint_type": "disease",
        "aliases": ["Stroke"],
    },
    "ThC": {
        "trait_name": "Thyroid cancer",
        "endpoint_type": "disease",
        "aliases": ["ThC"],
    },
    "UtC": {
        "trait_name": "Uterine cancer",
        "endpoint_type": "disease",
        "aliases": ["UtC"],
    },
    "VTE": {
        "trait_name": "Venous thromboembolism",
        "endpoint_type": "disease",
        "aliases": ["VTE"],
    },
}

# Explicit Pan-UKB phenocodes retained as canonical clinical/biological traits.
# A code is retained only if it is present in the independently generated 15A1
# catalog and has EUR plus at least one QC-passing non-EUR population.
PANUKB_EXPLICIT_CORE: dict[str, tuple[str, str, str]] = {
    # Blood biochemistry
    "30620": ("hepatic", "alanine_aminotransferase", "primary"),
    "30600": ("hepatic", "albumin", "primary"),
    "30610": ("hepatic_bone", "alkaline_phosphatase", "primary"),
    "30630": ("lipids", "apolipoprotein_a", "primary"),
    "30640": ("lipids", "apolipoprotein_b", "primary"),
    "30710": ("inflammation", "c_reactive_protein", "primary"),
    "30680": ("mineral", "calcium", "primary"),
    "30690": ("lipids", "total_cholesterol", "primary"),
    "30700": ("renal", "serum_creatinine", "primary"),
    "30720": ("renal", "cystatin_c", "primary"),
    "eGFR": ("renal", "egfr_creatinine", "primary"),
    "eGFRcreacys": ("renal", "egfr_cystatin_c", "secondary"),
    "eGFRcys": (
        "renal",
        "egfr_creatinine_cystatin_c",
        "secondary",
    ),
    "30730": ("hepatic", "gamma_glutamyltransferase", "primary"),
    "30750": ("glycemia", "hba1c", "primary"),
    "30760": ("lipids", "hdl_cholesterol", "primary"),
    "30770": ("growth_endocrine", "igf1", "primary"),
    "30780": ("lipids", "ldl_cholesterol", "primary"),
    "30810": ("mineral", "phosphate", "primary"),
    "30830": ("endocrine", "shbg", "secondary"),
    "30860": ("protein", "total_protein", "primary"),
    "30870": ("lipids", "triglycerides", "primary"),
    "30670": ("renal", "urea", "secondary"),
    "30890": ("mineral", "vitamin_d", "primary"),

    # Basic hematology: counts and standard indices, not duplicate percentages
    "30000": ("hematology", "white_blood_cell_count", "primary"),
    "30010": ("hematology", "red_blood_cell_count", "primary"),
    "30020": ("hematology", "haemoglobin", "primary"),
    "30030": ("hematology", "haematocrit", "primary"),
    "30040": ("hematology", "mean_corpuscular_volume", "primary"),
    "30050": ("hematology", "mean_corpuscular_haemoglobin", "secondary"),
    "30060": (
        "hematology",
        "mean_corpuscular_haemoglobin_concentration",
        "secondary",
    ),
    "30070": ("hematology", "red_cell_distribution_width", "primary"),
    "30080": ("hematology", "platelet_count", "primary"),
    "30100": ("hematology", "mean_platelet_volume", "secondary"),
    "30110": ("hematology", "platelet_distribution_width", "secondary"),
    "30120": ("hematology", "lymphocyte_count", "primary"),
    "30130": ("hematology", "monocyte_count", "primary"),
    "30140": ("hematology", "neutrophil_count", "primary"),
    "30150": ("hematology_allergy", "eosinophil_count", "primary"),
    "30250": ("hematology", "reticulocyte_count", "primary"),
    "30280": ("hematology", "immature_reticulocyte_fraction", "secondary"),

    # Urine assays
    "30510": ("renal", "urine_creatinine", "secondary"),
    "30520": ("renal_electrolyte", "urine_potassium", "secondary"),
    "30530": ("renal_electrolyte", "urine_sodium", "secondary"),

    # Canonical body-size measures
    "21001": ("anthropometry", "body_mass_index", "primary"),
    "48": ("anthropometry", "waist_circumference", "primary"),
    "49": ("anthropometry", "hip_circumference", "secondary"),
    "50": ("anthropometry", "standing_height", "primary"),
    "21002": ("anthropometry", "weight", "secondary"),
}

# Dynamically retain clinical traits whose exact phenocode may be easier to
# identify from the official category/name than to hard-code.
DYNAMIC_CORE_RULES = [
    {
        "domain": "blood_pressure",
        "canonical_group": "systolic_blood_pressure",
        "category_regex": r"blood pressure",
        "name_regex": r"\bsystolic blood pressure\b",
        "exclude_regex": r"pulse rate|manual.*second|second.*manual",
        "priority": "primary",
    },
    {
        "domain": "blood_pressure",
        "canonical_group": "diastolic_blood_pressure",
        "category_regex": r"blood pressure",
        "name_regex": r"\bdiastolic blood pressure\b",
        "exclude_regex": r"pulse rate|manual.*second|second.*manual",
        "priority": "primary",
    },
    {
        "domain": "pulmonary",
        "canonical_group": "fev1",
        "category_regex": r"spirometry",
        "name_regex": r"forced expiratory volume|fev1",
        "exclude_regex": r"predicted|percentage|z score",
        "priority": "primary",
    },
    {
        "domain": "pulmonary",
        "canonical_group": "fvc",
        "category_regex": r"spirometry",
        "name_regex": r"forced vital capacity|\bfvc\b",
        "exclude_regex": r"predicted|percentage|z score",
        "priority": "primary",
    },
    {
        "domain": "pulmonary",
        "canonical_group": "peak_expiratory_flow",
        "category_regex": r"spirometry",
        "name_regex": r"peak expiratory flow|\bpef\b",
        "exclude_regex": r"predicted|percentage|z score",
        "priority": "secondary",
    },
    {
        "domain": "pulmonary",
        "canonical_group": "fev1_fvc_ratio",
        "category_regex": r"spirometry",
        "name_regex": r"fev1.*fvc|fvc.*fev1|ratio",
        "exclude_regex": r"predicted|percentage|z score",
        "priority": "primary",
    },
    {
        "domain": "bone",
        "canonical_group": "heel_bone_mineral_density",
        "category_regex": r"bone-densitometry|bone densitometry",
        "name_regex": r"bone mineral density|quantitative ultrasound index",
        "exclude_regex": r"left|right",
        "priority": "primary",
    },
    {
        "domain": "ophthalmic",
        "canonical_group": "intraocular_pressure",
        "category_regex": r"intraocular pressure|eye measures",
        "name_regex": r"intraocular pressure",
        "exclude_regex": r"left|right",
        "priority": "primary",
    },
    {
        "domain": "vascular",
        "canonical_group": "arterial_stiffness_index",
        "category_regex": r"arterial stiffness",
        "name_regex": r"arterial stiffness",
        "exclude_regex": r"",
        "priority": "secondary",
    },
]

# Conservative indication-to-Pan-UKB candidate rules. These produce review
# candidates only. They are not accepted mappings.
PANUKB_MAPPING_RULES = [
    # Lipids and glycemia
    (
        r"hypercholester|hyperlipid|hyperlipoproteinemia type ii|dyslipid",
        ["ldl_cholesterol", "apolipoprotein_b", "total_cholesterol"],
        "Direct quantitative readout",
        "Circulating lipid measure directly represents the indication construct.",
    ),
    (
        r"hypertriglycer",
        ["triglycerides"],
        "Direct quantitative readout",
        "Triglyceride concentration directly represents hypertriglyceridemia.",
    ),
    (
        r"diabetes mellitus(?:, type 2)?|type 2 diabetes",
        ["hba1c"],
        "Direct quantitative readout",
        "HbA1c is a standard quantitative measure of chronic glycemia.",
    ),
    (
        r"\bobesity\b",
        ["body_mass_index", "waist_circumference"],
        "Direct quantitative readout",
        "BMI and waist circumference directly measure adiposity.",
    ),
    (
        r"\bhypertension\b|high blood pressure",
        ["systolic_blood_pressure", "diastolic_blood_pressure"],
        "Direct quantitative readout",
        "Blood pressure is the defining quantitative phenotype.",
    ),

    # Hematologic indications
    (
        r"\bthrombocytopenia\b|\bthrombocytosis\b|platelet disorder",
        ["platelet_count"],
        "Direct quantitative readout",
        "Platelet count directly represents the indication.",
    ),
    (
        r"\banemia\b|anaemia",
        ["haemoglobin", "haematocrit", "mean_corpuscular_volume"],
        "Direct quantitative readout",
        "Standard red-cell measures characterize anemia and major subtypes.",
    ),
    (
        r"polycyth|erythrocyt",
        ["red_blood_cell_count", "haematocrit", "haemoglobin"],
        "Direct quantitative readout",
        "Red-cell concentration measures directly represent erythrocytosis.",
    ),
    (
        r"neutropenia|neutrophilia",
        ["neutrophil_count"],
        "Direct quantitative readout",
        "Neutrophil count directly represents the indication.",
    ),
    (
        r"lymphopenia|lymphocytopenia|lymphocytosis",
        ["lymphocyte_count"],
        "Direct quantitative readout",
        "Lymphocyte count directly represents the indication.",
    ),
    (
        r"eosinophilia",
        ["eosinophil_count"],
        "Direct quantitative readout",
        "Eosinophil count directly represents eosinophilia.",
    ),
    (
        r"leukopenia|leucopenia|leukocytosis|leucocytosis",
        ["white_blood_cell_count"],
        "Direct quantitative readout",
        "White blood cell count directly represents the indication.",
    ),
    (
        r"monocytosis|monocytopenia",
        ["monocyte_count"],
        "Direct quantitative readout",
        "Monocyte count directly represents the indication.",
    ),

    # Respiratory and allergic disease
    (
        r"\basthma\b",
        ["eosinophil_count", "fev1", "fvc", "fev1_fvc_ratio"],
        "Proxy biomarker",
        "Eosinophils and spirometry measure allergic inflammation or airway physiology, not asthma diagnosis itself.",
    ),
    (
        r"pulmonary disease, chronic obstructive|chronic obstructive pulmonary|\bcopd\b",
        ["fev1", "fvc", "fev1_fvc_ratio"],
        "Closely related phenotype",
        "Spirometry measures airflow limitation central to COPD but is not the disease endpoint.",
    ),
    (
        r"rhinitis, allergic|allergic rhinitis|dermatitis, atopic|atopic dermatitis",
        ["eosinophil_count"],
        "Proxy biomarker",
        "Eosinophil count is an allergic-inflammatory biomarker rather than the disease endpoint.",
    ),

    # Renal, hepatic, inflammatory, bone, and eye traits
    (
        r"kidney disease|renal insufficien|renal failure|nephropath|glomerul",
        [
            "egfr_creatinine",
            "egfr_creatinine_cystatin_c",
            "egfr_cystatin_c",
            "serum_creatinine",
            "cystatin_c",
        ],
        "Proxy biomarker",
        "eGFR and renal filtration biomarkers measure kidney function but are not the disease endpoint.",
    ),
    (
        r"liver disease|hepatic|cirrhos|cholestasis|cholestatic|fatty liver",
        [
            "alanine_aminotransferase",
            "gamma_glutamyltransferase",
            "alkaline_phosphatase",
            "albumin",
        ],
        "Proxy biomarker",
        "Liver enzymes and albumin measure hepatic injury or function rather than a specific liver diagnosis.",
    ),
    (
        r"osteoporosis|osteopenia",
        ["heel_bone_mineral_density"],
        "Direct quantitative readout",
        "Bone mineral density is the defining quantitative measure.",
    ),
    (
        r"glaucoma|ocular hypertension",
        ["intraocular_pressure"],
        "Closely related phenotype",
        "Intraocular pressure is a major quantitative risk phenotype but not identical to glaucoma diagnosis.",
    ),
    (
        r"vitamin d deficiency",
        ["vitamin_d"],
        "Direct quantitative readout",
        "Circulating vitamin D directly represents the deficiency.",
    ),
    (
        r"hypercalcemia|hypocalcemia",
        ["calcium"],
        "Direct quantitative readout",
        "Serum calcium directly represents the indication.",
    ),
    (
        r"hyperphosphatemia|hypophosphatemia",
        ["phosphate"],
        "Direct quantitative readout",
        "Serum phosphate directly represents the indication.",
    ),
    (
        r"hypoalbuminemia",
        ["albumin"],
        "Direct quantitative readout",
        "Serum albumin directly represents hypoalbuminemia.",
    ),
    (
        r"acromegaly|growth hormone",
        ["igf1"],
        "Closely related phenotype",
        "IGF-1 is a standard biochemical readout of growth-hormone activity.",
    ),

    # These are intentionally tagged as proxies and require explicit review.
    (
        r"arthritis, rheumatoid|crohn disease|colitis, ulcerative|inflammatory bowel|psoriasis|lupus erythematosus|autoimmune disease",
        ["c_reactive_protein"],
        "Proxy biomarker",
        "CRP is a nonspecific inflammatory biomarker and does not represent the disease endpoint.",
    ),
]

# Conservative exact/clinical-equivalence mappings to published GBMI endpoints.
GBMI_MAPPING_PATTERNS = {
    "AAA": r"abdominal aortic aneurysm|aortic aneurysm, abdominal",
    "AcApp": r"acute appendicitis|\bappendicitis\b",
    # Appendectomy is a procedure and is intentionally not mapped automatically.
    "Asthma": r"\basthma\b",
    "COPD": r"pulmonary disease, chronic obstructive|chronic obstructive pulmonary|\bcopd\b",
    "Gout": r"\bgout\b",
    "HCM": r"cardiomyopathy, hypertrophic|hypertrophic cardiomyopathy",
    "HF": r"\bheart failure\b",
    "IPF": r"idiopathic pulmonary fibrosis|pulmonary fibrosis, idiopathic",
    "POAG": r"primary open-angle glaucoma|glaucoma, open-angle",
    "Stroke": r"\bstroke\b|cerebrovascular accident",
    "ThC": r"thyroid neoplasms|thyroid cancer|carcinoma, thyroid",
    "UtC": r"uterine neoplasms|uterine cancer|endometrial neoplasms|endometrial cancer",
    "VTE": r"venous thromboembolism|venous thrombosis|pulmonary embolism",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--panukb-catalog",
        type=Path,
        default=Path("output/15a_panukb_quantitative_catalog.csv"),
    )
    p.add_argument(
        "--indications",
        type=Path,
        default=Path("output/15a_indication_mapping_review.csv"),
    )
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument(
        "--skip-gbmi-network",
        action="store_true",
        help="Create expected GBMI rows without network verification.",
    )
    p.add_argument(
        "--validate-gbmi-schema",
        action="store_true",
        help="Stream the beginning of verified GBMI files and inspect columns.",
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


def request_bytes(
    url: str,
    *,
    timeout: float,
    retries: int,
    headers: dict[str, str] | None = None,
) -> tuple[int | None, dict[str, str], bytes, str]:
    merged_headers = {
        "User-Agent": "target-ancestry-step15a2/1.0",
        "Accept": "*/*",
    }
    if headers:
        merged_headers.update(headers)

    last_error = ""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=merged_headers)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                status = getattr(response, "status", 200)
                response_headers = {
                    k.lower(): v for k, v in response.headers.items()
                }
                body = response.read()
                return status, response_headers, body, ""
        except urllib.error.HTTPError as exc:
            body = exc.read(1000)
            return (
                exc.code,
                {k.lower(): v for k, v in exc.headers.items()},
                body,
                f"HTTP {exc.code}",
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))

    return None, {}, b"", last_error


def list_s3_bucket(
    *,
    bucket_url: str,
    timeout: float,
    retries: int,
) -> tuple[list[dict[str, Any]], str]:
    objects: list[dict[str, Any]] = []
    token = ""
    seen_tokens: set[str] = set()

    while True:
        params = {"list-type": "2", "max-keys": "1000"}
        if token:
            params["continuation-token"] = token

        url = f"{bucket_url}/?{urllib.parse.urlencode(params)}"
        status, _, body, error = request_bytes(
            url,
            timeout=timeout,
            retries=retries,
        )
        if status != 200:
            return [], error or f"bucket listing returned status {status}"

        try:
            root = ET.fromstring(body)
        except ET.ParseError as exc:
            return [], f"could not parse S3 listing XML: {exc}"

        def local(tag: str) -> str:
            return tag.split("}", 1)[-1]

        for content in root.iter():
            if local(content.tag) != "Contents":
                continue
            values = {
                local(child.tag): clean(child.text)
                for child in list(content)
            }
            objects.append(
                {
                    "key": values.get("Key", ""),
                    "last_modified": values.get("LastModified", ""),
                    "etag": values.get("ETag", "").strip('"'),
                    "size_bytes": int(values.get("Size", "0") or 0),
                }
            )

        truncated = next(
            (
                clean(node.text).casefold() == "true"
                for node in root.iter()
                if local(node.tag) == "IsTruncated"
            ),
            False,
        )
        if not truncated:
            break

        next_token = next(
            (
                clean(node.text)
                for node in root.iter()
                if local(node.tag) == "NextContinuationToken"
            ),
            "",
        )
        if not next_token or next_token in seen_tokens:
            return objects, "S3 listing was truncated without a usable next token"

        seen_tokens.add(next_token)
        token = next_token

    return objects, ""


GBMI_FILENAME_RE = re.compile(
    r"^(?P<endpoint>[^/]+?)_"
    r"(?P<sex>[^_]+)_"
    r"(?P<pop>afr|amr|eas|eur|sas|mid)_"
    r"inv_var_meta_GBMI_052021_nbbkgt1\.txt\.gz$",
    re.IGNORECASE,
)


def endpoint_from_alias(raw: str) -> str:
    raw_cf = raw.casefold()
    for endpoint, info in GBMI_ENDPOINTS.items():
        for alias in info["aliases"]:
            if raw_cf == alias.casefold():
                return endpoint
    return ""


def parse_gbmi_listing(
    objects: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {obj["key"]: obj for obj in objects}
    rows = []

    for obj in objects:
        key = obj["key"]
        match = GBMI_FILENAME_RE.match(key)
        if not match:
            continue

        endpoint = endpoint_from_alias(match.group("endpoint"))
        if not endpoint:
            continue

        source_pop = match.group("pop").upper()
        data_url = f"{GBMI_BUCKET}/{urllib.parse.quote(key)}"
        index_key = key + ".tbi"
        index_obj = by_key.get(index_key)

        rows.append(
            {
                "endpoint_code": endpoint,
                "trait_name": GBMI_ENDPOINTS[endpoint]["trait_name"],
                "endpoint_type": GBMI_ENDPOINTS[endpoint][
                    "endpoint_type"
                ],
                "sex_stratum": match.group("sex"),
                "source_population": source_pop,
                "harmonized_population": POP_HARMONIZATION.get(
                    source_pop, source_pop
                ),
                "genome_build": "GRCh38",
                "release": f"GBMI_{GBMI_RELEASE}",
                "data_url": data_url,
                "index_url": f"{data_url}.tbi",
                "data_exists": True,
                "index_exists": index_obj is not None,
                "verification_method": "public_s3_bucket_listing",
                "http_status": 200,
                "size_bytes": obj["size_bytes"],
                "etag": obj["etag"],
                "last_modified": obj["last_modified"],
                "schema_status": "NOT_CHECKED",
                "schema_columns": "",
                "sample_size_status": "NOT_CHECKED",
                "verification_error": "",
            }
        )

    return rows


def candidate_gbmi_urls(endpoint: str, pop: str) -> Iterable[tuple[str, str]]:
    aliases = GBMI_ENDPOINTS[endpoint]["aliases"]
    sex_tokens = (
        ["Female", "Females", "Women", "Bothsex"]
        if endpoint == "UtC"
        else ["Bothsex"]
    )

    for alias in aliases:
        for sex in sex_tokens:
            key = (
                f"{alias}_{sex}_{pop.casefold()}_"
                f"inv_var_meta_GBMI_{GBMI_RELEASE}_nbbkgt1.txt.gz"
            )
            yield key, f"{GBMI_BUCKET}/{key}"


def probe_gbmi_objects(
    *,
    timeout: float,
    retries: int,
) -> list[dict[str, Any]]:
    rows = []

    for endpoint in GBMI_ENDPOINTS:
        for pop in GBMI_SOURCE_POPS:
            found: dict[str, Any] | None = None
            errors = []

            for key, url in candidate_gbmi_urls(endpoint, pop):
                status, headers, _, error = request_bytes(
                    url,
                    timeout=timeout,
                    retries=retries,
                    headers={"Range": "bytes=0-0"},
                )
                if status in {200, 206}:
                    index_status, _, _, index_error = request_bytes(
                        url + ".tbi",
                        timeout=timeout,
                        retries=retries,
                        headers={"Range": "bytes=0-0"},
                    )
                    found = {
                        "endpoint_code": endpoint,
                        "trait_name": GBMI_ENDPOINTS[endpoint][
                            "trait_name"
                        ],
                        "endpoint_type": GBMI_ENDPOINTS[endpoint][
                            "endpoint_type"
                        ],
                        "sex_stratum": key.split("_")[1],
                        "source_population": pop,
                        "harmonized_population": POP_HARMONIZATION.get(
                            pop, pop
                        ),
                        "genome_build": "GRCh38",
                        "release": f"GBMI_{GBMI_RELEASE}",
                        "data_url": url,
                        "index_url": url + ".tbi",
                        "data_exists": True,
                        "index_exists": index_status in {200, 206},
                        "verification_method": "targeted_http_range_probe",
                        "http_status": status,
                        "size_bytes": headers.get(
                            "content-length", ""
                        ),
                        "etag": headers.get("etag", "").strip('"'),
                        "last_modified": headers.get(
                            "last-modified", ""
                        ),
                        "schema_status": "NOT_CHECKED",
                        "schema_columns": "",
                        "sample_size_status": "NOT_CHECKED",
                        "verification_error": (
                            ""
                            if index_status in {200, 206}
                            else f"index: {index_error or index_status}"
                        ),
                    }
                    break

                if status not in {403, 404} or error:
                    errors.append(f"{key}: {error or status}")

            if found is not None:
                rows.append(found)
            elif errors:
                # Only preserve non-404 errors. Ordinary absent ancestry files
                # are represented at the endpoint-summary level.
                rows.append(
                    {
                        "endpoint_code": endpoint,
                        "trait_name": GBMI_ENDPOINTS[endpoint][
                            "trait_name"
                        ],
                        "endpoint_type": GBMI_ENDPOINTS[endpoint][
                            "endpoint_type"
                        ],
                        "sex_stratum": "",
                        "source_population": pop,
                        "harmonized_population": POP_HARMONIZATION.get(
                            pop, pop
                        ),
                        "genome_build": "GRCh38",
                        "release": f"GBMI_{GBMI_RELEASE}",
                        "data_url": "",
                        "index_url": "",
                        "data_exists": False,
                        "index_exists": False,
                        "verification_method": "targeted_http_range_probe",
                        "http_status": "",
                        "size_bytes": "",
                        "etag": "",
                        "last_modified": "",
                        "schema_status": "NOT_CHECKED",
                        "schema_columns": "",
                        "sample_size_status": "NOT_CHECKED",
                        "verification_error": " | ".join(errors),
                    }
                )

    return rows


def read_gzip_head(
    url: str,
    *,
    timeout: float,
    retries: int,
    max_lines: int = 5,
) -> tuple[list[str], str]:
    last_error = ""

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "target-ancestry-step15a2/1.0",
                    "Accept-Encoding": "identity",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                with gzip.GzipFile(fileobj=response) as gz:
                    lines = []
                    for _ in range(max_lines):
                        raw = gz.readline()
                        if not raw:
                            break
                        lines.append(
                            raw.decode("utf-8", errors="replace").rstrip(
                                "\r\n"
                            )
                        )
                    return lines, ""
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))

    return [], last_error


def inspect_schema(lines: list[str]) -> tuple[str, str, str]:
    if not lines:
        return "FAILED", "", "No decompressed lines returned"

    # Prefer a comment/header line containing beta/se/p columns. Otherwise use
    # the first line.
    header = lines[0].lstrip("#")
    for line in lines:
        candidate = line.lstrip("#")
        normalized = candidate.casefold()
        if (
            ("beta" in normalized or "effect" in normalized)
            and ("se" in normalized or "stderr" in normalized)
            and ("p" in normalized)
        ):
            header = candidate
            break

    columns = re.split(r"\t+|\s+", header.strip())
    columns = [c for c in columns if c]
    lower = [c.casefold() for c in columns]

    has_position = any(
        c in {"pos", "position", "bp"} or "position" in c for c in lower
    )
    has_beta = any("beta" in c or "effect" in c for c in lower)
    has_se = any(
        c in {"se", "stderr", "sebeta"}
        or "sebeta" in c
        or "standard_error" in c
        for c in lower
    )
    has_p = any(
        c in {"p", "pval", "pvalue", "p-value"}
        or c.endswith("_p")
        or "pval" in c
        or "pvalue" in c
        for c in lower
    )
    has_alleles = (
        any(c in {"ref", "a0", "nea", "other_allele"} for c in lower)
        and any(c in {"alt", "a1", "ea", "effect_allele"} for c in lower)
    )

    status = (
        "PASS"
        if has_position and has_beta and has_se and has_p
        else "REVIEW"
    )
    note = (
        ""
        if status == "PASS"
        else (
            "Expected position, beta/effect, SE, and p-value fields were "
            "not all recognized automatically."
        )
    )
    if not has_alleles:
        note = (
            note + " Allele columns were not recognized automatically."
        ).strip()

    sample_fields = [
        c
        for c in columns
        if c.casefold()
        in {
            "n",
            "n_total",
            "n_cases",
            "n_controls",
            "cases",
            "controls",
        }
        or c.casefold().startswith("n_")
    ]
    sample_status = (
        "VARIANT_LEVEL_SAMPLE_FIELDS:" + ",".join(sample_fields)
        if sample_fields
        else "NO_RECOGNIZED_SAMPLE_SIZE_FIELD"
    )

    return status, "\t".join(columns), note + (
        f" {sample_status}" if sample_status else ""
    )


def verify_gbmi(
    *,
    skip_network: bool,
    validate_schema: bool,
    timeout: float,
    retries: int,
) -> tuple[pd.DataFrame, str]:
    if skip_network:
        rows = []
        for endpoint in GBMI_ENDPOINTS:
            for pop in GBMI_SOURCE_POPS:
                key, url = next(candidate_gbmi_urls(endpoint, pop))
                rows.append(
                    {
                        "endpoint_code": endpoint,
                        "trait_name": GBMI_ENDPOINTS[endpoint][
                            "trait_name"
                        ],
                        "endpoint_type": GBMI_ENDPOINTS[endpoint][
                            "endpoint_type"
                        ],
                        "sex_stratum": key.split("_")[1],
                        "source_population": pop,
                        "harmonized_population": POP_HARMONIZATION.get(
                            pop, pop
                        ),
                        "genome_build": "GRCh38",
                        "release": f"GBMI_{GBMI_RELEASE}",
                        "data_url": url,
                        "index_url": url + ".tbi",
                        "data_exists": False,
                        "index_exists": False,
                        "verification_method": "network_skipped_expected_path",
                        "http_status": "",
                        "size_bytes": "",
                        "etag": "",
                        "last_modified": "",
                        "schema_status": "NOT_CHECKED",
                        "schema_columns": "",
                        "sample_size_status": "NOT_CHECKED",
                        "verification_error": "Network verification skipped",
                    }
                )
        return pd.DataFrame(rows), "network skipped"

    objects, listing_error = list_s3_bucket(
        bucket_url=GBMI_BUCKET,
        timeout=timeout,
        retries=retries,
    )

    if objects:
        rows = parse_gbmi_listing(objects)
        method_note = (
            f"public S3 listing parsed: {len(objects):,} objects"
        )
    else:
        rows = probe_gbmi_objects(timeout=timeout, retries=retries)
        method_note = (
            "S3 listing unavailable; used targeted HTTP probes. "
            f"Listing error: {listing_error}"
        )

    files = pd.DataFrame(rows)
    if files.empty:
        return files, method_note

    # Retain positive files and explicit network errors; ordinary absent
    # endpoint-population combinations do not need one row each.
    files = files[
        files["data_exists"].eq(True)
        | files["verification_error"].ne("")
    ].copy()

    if validate_schema:
        for idx, row in files[files["data_exists"].eq(True)].iterrows():
            lines, error = read_gzip_head(
                row["data_url"],
                timeout=timeout,
                retries=retries,
            )
            if error:
                files.at[idx, "schema_status"] = "FAILED"
                files.at[idx, "verification_error"] = (
                    clean(files.at[idx, "verification_error"])
                    + (" | " if clean(files.at[idx, "verification_error"]) else "")
                    + error
                )
                continue

            status, columns, note = inspect_schema(lines)
            files.at[idx, "schema_status"] = status
            files.at[idx, "schema_columns"] = columns
            files.at[idx, "sample_size_status"] = (
                next(
                    (
                        token
                        for token in note.split()
                        if token.startswith(
                            (
                                "VARIANT_LEVEL_SAMPLE_FIELDS:",
                                "NO_RECOGNIZED_SAMPLE_SIZE_FIELD",
                            )
                        )
                    ),
                    "REVIEW_HEADER",
                )
            )
            if note:
                files.at[idx, "verification_error"] = (
                    clean(files.at[idx, "verification_error"])
                    + (" | " if clean(files.at[idx, "verification_error"]) else "")
                    + note
                )

    return files.reset_index(drop=True), method_note


def build_gbmi_endpoint_summary(files: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for endpoint, info in GBMI_ENDPOINTS.items():
        subset = files[
            files["endpoint_code"].eq(endpoint)
            & files["data_exists"].eq(True)
        ].copy()

        source_pops = sorted(set(subset["source_population"]))
        harmonized_pops = sorted(set(subset["harmonized_population"]))
        non_eur = [p for p in harmonized_pops if p != "EUR"]
        eur_present = "EUR" in harmonized_pops
        eligible = eur_present and bool(non_eur)

        rows.append(
            {
                "trait_key": f"GBMI::{endpoint}",
                "endpoint_code": endpoint,
                "trait_name": info["trait_name"],
                "endpoint_type": info["endpoint_type"],
                "genome_build": "GRCh38",
                "release": f"GBMI_{GBMI_RELEASE}",
                "source_populations_verified": ",".join(source_pops),
                "harmonized_populations_verified": ",".join(
                    harmonized_pops
                ),
                "non_eur_populations_verified": ",".join(non_eur),
                "n_verified_population_files": len(harmonized_pops),
                "eur_file_verified": eur_present,
                "has_verified_non_eur": bool(non_eur),
                "eligible_for_cross_ancestry": eligible,
                "all_indexes_verified": bool(
                    len(subset) > 0 and subset["index_exists"].all()
                ),
                "schema_pass_files": int(
                    subset["schema_status"].eq("PASS").sum()
                ),
                "schema_review_or_unchecked_files": int(
                    (~subset["schema_status"].eq("PASS")).sum()
                ),
                "data_urls": " || ".join(
                    subset.sort_values("source_population")["data_url"]
                ),
                "verification_notes": (
                    ""
                    if len(subset)
                    else "No release file independently verified"
                ),
            }
        )

    return pd.DataFrame(rows)


def score_dynamic_candidate(row: pd.Series, rule: dict[str, str]) -> bool:
    category = norm(row.get("category", ""))
    name = norm(row.get("trait_name", ""))

    if not re.search(rule["category_regex"], category, re.I):
        return False
    if not re.search(rule["name_regex"], name, re.I):
        return False
    if rule["exclude_regex"] and re.search(
        rule["exclude_regex"], name, re.I
    ):
        return False
    return True


def preference_score(row: pd.Series) -> tuple[int, int, str]:
    name = norm(row["trait_name"])
    score = 0
    if "automated" in name:
        score += 30
    if "mean" in name or "average" in name:
        score += 20
    if "both" in name:
        score += 10
    if "left" in name or "right" in name:
        score -= 30
    if "manual" in name:
        score -= 10
    if "second" in name:
        score -= 5
    return (-score, len(name), name)


def build_panukb_core(catalog: pd.DataFrame) -> pd.DataFrame:
    required = {
        "trait_key",
        "phenocode",
        "trait_name",
        "category",
        "pops_pass_qc",
        "non_eur_pops_pass_qc",
        "genome_build",
        "remote_data_path",
        "remote_index_path",
    }
    missing = sorted(required - set(catalog.columns))
    if missing:
        raise SystemExit(
            f"15A1 Pan-UKB catalog missing columns: {missing}"
        )

    selected = []

    for _, row in catalog.iterrows():
        phenocode = clean(row["phenocode"])
        if phenocode in PANUKB_EXPLICIT_CORE:
            domain, group, priority = PANUKB_EXPLICIT_CORE[phenocode]
            out = row.to_dict()
            out.update(
                {
                    "clinical_domain": domain,
                    "canonical_group": group,
                    "core_priority": priority,
                    "selection_method": "explicit_phenocode",
                    "selection_reason": (
                        "Prespecified canonical clinical or biological "
                        "measurement."
                    ),
                }
            )
            selected.append(out)

    # Dynamic rules may return multiple versions of the same construct.
    # Pick one canonical row per group using deterministic preferences.
    dynamic_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for _, row in catalog.iterrows():
        for rule in DYNAMIC_CORE_RULES:
            if score_dynamic_candidate(row, rule):
                out = row.to_dict()
                out.update(
                    {
                        "clinical_domain": rule["domain"],
                        "canonical_group": rule["canonical_group"],
                        "core_priority": rule["priority"],
                        "selection_method": "prespecified_category_name_rule",
                        "selection_reason": (
                            "Selected from an official clinical measurement "
                            "category and collapsed to one canonical version."
                        ),
                    }
                )
                dynamic_by_group[rule["canonical_group"]].append(out)

    for group, candidates in dynamic_by_group.items():
        candidates.sort(key=lambda x: preference_score(pd.Series(x)))
        selected.append(candidates[0])

    core = pd.DataFrame(selected)
    if core.empty:
        raise SystemExit("No Pan-UKB core traits were selected.")

    core = (
        core.sort_values(
            [
                "clinical_domain",
                "canonical_group",
                "core_priority",
                "trait_name",
            ]
        )
        .drop_duplicates("trait_key")
        .reset_index(drop=True)
    )

    duplicates = core["canonical_group"].duplicated(False)
    if duplicates.any():
        dup = core.loc[
            duplicates,
            ["canonical_group", "trait_key", "trait_name"],
        ]
        raise SystemExit(
            "Canonical group duplicated after reduction:\n"
            + dup.to_string(index=False)
        )

    return core


def build_mapping_candidates(
    indications: pd.DataFrame,
    core: pd.DataFrame,
    gbmi_endpoints: pd.DataFrame,
) -> pd.DataFrame:
    core_by_group = {
        row["canonical_group"]: row
        for _, row in core.iterrows()
    }
    gbmi_by_code = {
        row["endpoint_code"]: row
        for _, row in gbmi_endpoints.iterrows()
    }

    rows = []

    for _, indication in indications.iterrows():
        indication_name = clean(indication["indication_mesh_term"])
        preferred = clean(indication.get("mesh2026_preferred_term", ""))
        searchable = f"{indication_name} {preferred}".strip()

        base = {
            "indication_mesh_id": indication["indication_mesh_id"],
            "indication_mesh_term": indication_name,
            "mesh2026_preferred_term": preferred,
            "n_pairs": int(indication["n_pairs"]),
            "n_genes": int(indication["n_genes"]),
            "n_A": int(indication["n_A"]),
            "n_B": int(indication["n_B"]),
        }

        # Pan-UKB clinical/biological candidates
        for (
            pattern,
            groups,
            tier,
            rationale,
        ) in PANUKB_MAPPING_RULES:
            if not re.search(pattern, searchable, re.I):
                continue

            for group in groups:
                trait = core_by_group.get(group)
                if trait is None:
                    continue

                rows.append(
                    {
                        **base,
                        "candidate_source": "Pan-UKB",
                        "candidate_trait_key": trait["trait_key"],
                        "candidate_trait_name": trait["trait_name"],
                        "phenotype_id": trait["phenotype_id"],
                        "genome_build": trait["genome_build"],
                        "mapping_tier": tier,
                        "mapping_rationale": rationale,
                        "clinical_domain": trait["clinical_domain"],
                        "canonical_group": group,
                        "pops_available": trait["pops_pass_qc"],
                        "non_eur_pops_available": trait[
                            "non_eur_pops_pass_qc"
                        ],
                        "source_verified": True,
                        "candidate_status": "REVIEW",
                        "reviewer_decision": "",
                        "reviewer_notes": "",
                    }
                )

        # GBMI direct disease candidates
        for endpoint, pattern in GBMI_MAPPING_PATTERNS.items():
            if not re.search(pattern, searchable, re.I):
                continue

            trait = gbmi_by_code.get(endpoint)
            if trait is None:
                continue

            rows.append(
                {
                    **base,
                    "candidate_source": "GBMI",
                    "candidate_trait_key": trait["trait_key"],
                    "candidate_trait_name": trait["trait_name"],
                    "phenotype_id": endpoint,
                    "genome_build": "GRCh38",
                    "mapping_tier": "Direct disease",
                    "mapping_rationale": (
                        "The GBMI endpoint is an exact or clinically "
                        "equivalent disease endpoint."
                    ),
                    "clinical_domain": "direct_disease",
                    "canonical_group": endpoint,
                    "pops_available": trait[
                        "harmonized_populations_verified"
                    ],
                    "non_eur_pops_available": trait[
                        "non_eur_populations_verified"
                    ],
                    "source_verified": bool(
                        trait["eligible_for_cross_ancestry"]
                    ),
                    "candidate_status": (
                        "REVIEW"
                        if trait["eligible_for_cross_ancestry"]
                        else "SOURCE_NOT_ELIGIBLE"
                    ),
                    "reviewer_decision": "",
                    "reviewer_notes": "",
                }
            )

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return candidates

    # Prevent accidental duplicate rules.
    candidates = candidates.drop_duplicates(
        [
            "indication_mesh_id",
            "candidate_trait_key",
            "mapping_tier",
        ]
    ).reset_index(drop=True)

    tier_order = {
        "Direct disease": 1,
        "Direct quantitative readout": 2,
        "Closely related phenotype": 3,
        "Proxy biomarker": 4,
    }
    candidates["_tier_order"] = candidates["mapping_tier"].map(
        tier_order
    )
    candidates = (
        candidates.sort_values(
            [
                "n_pairs",
                "indication_mesh_term",
                "_tier_order",
                "candidate_source",
                "candidate_trait_name",
            ],
            ascending=[False, True, True, True, True],
        )
        .drop(columns="_tier_order")
        .reset_index(drop=True)
    )

    return candidates


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for path in [args.panukb_catalog, args.indications]:
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")

    print(f"Loading Pan-UKB quantitative catalog: {args.panukb_catalog}")
    panukb_catalog = pd.read_csv(
        args.panukb_catalog,
        low_memory=False,
    )
    print(f"Loading indication review table: {args.indications}")
    indications = pd.read_csv(args.indications, low_memory=False)

    panukb_core = build_panukb_core(panukb_catalog)

    print("Verifying GBMI release files...")
    gbmi_files, gbmi_method_note = verify_gbmi(
        skip_network=args.skip_gbmi_network,
        validate_schema=args.validate_gbmi_schema,
        timeout=args.timeout,
        retries=args.retries,
    )
    gbmi_endpoints = build_gbmi_endpoint_summary(gbmi_files)

    candidates = build_mapping_candidates(
        indications,
        panukb_core,
        gbmi_endpoints,
    )

    mapped_ids = (
        set(candidates["indication_mesh_id"])
        if not candidates.empty
        else set()
    )
    unmapped = indications[
        ~indications["indication_mesh_id"].isin(mapped_ids)
    ].copy()
    unmapped["unmapped_reason"] = (
        "No prespecified high-confidence clinical, biomarker, or "
        "verified GBMI endpoint rule matched this indication."
    )

    paths = {
        "panukb_core": (
            args.output_dir / "15a2_panukb_core_traits.csv"
        ),
        "gbmi_files": (
            args.output_dir / "15a2_gbmi_verified_files.csv"
        ),
        "gbmi_endpoints": (
            args.output_dir / "15a2_gbmi_verified_endpoints.csv"
        ),
        "mapping_candidates": (
            args.output_dir / "15a2_mapping_candidates.csv"
        ),
        "unmapped_indications": (
            args.output_dir / "15a2_unmapped_indications.csv"
        ),
    }

    panukb_core.to_csv(paths["panukb_core"], index=False)
    gbmi_files.to_csv(paths["gbmi_files"], index=False)
    gbmi_endpoints.to_csv(paths["gbmi_endpoints"], index=False)
    candidates.to_csv(paths["mapping_candidates"], index=False)
    unmapped.to_csv(paths["unmapped_indications"], index=False)

    summary = {
        "step": "15A2",
        "automatic_mapping_acceptance": False,
        "panukb": {
            "genome_build": "GRCh37",
            "n_quantitative_or_biomarker_traits_input": int(
                len(panukb_catalog)
            ),
            "n_canonical_core_traits": int(len(panukb_core)),
            "n_primary_core_traits": int(
                panukb_core["core_priority"].eq("primary").sum()
            ),
            "n_secondary_core_traits": int(
                panukb_core["core_priority"].eq("secondary").sum()
            ),
            "core_by_domain": {
                str(k): int(v)
                for k, v in panukb_core[
                    "clinical_domain"
                ].value_counts().items()
            },
        },
        "gbmi": {
            "genome_build": "GRCh38",
            "release": f"GBMI_{GBMI_RELEASE}",
            "bucket": GBMI_BUCKET,
            "verification_note": gbmi_method_note,
            "n_published_endpoints": len(GBMI_ENDPOINTS),
            "n_verified_data_files": int(
                gbmi_files["data_exists"].eq(True).sum()
                if not gbmi_files.empty
                else 0
            ),
            "n_verified_endpoints": int(
                gbmi_endpoints["eur_file_verified"].sum()
            ),
            "n_cross_ancestry_eligible_endpoints": int(
                gbmi_endpoints[
                    "eligible_for_cross_ancestry"
                ].sum()
            ),
            "eligible_endpoint_codes": gbmi_endpoints.loc[
                gbmi_endpoints[
                    "eligible_for_cross_ancestry"
                ],
                "endpoint_code",
            ].tolist(),
            "source_population_SAS_harmonized_to": "CSA",
        },
        "mapping_workspace": {
            "n_indications": int(len(indications)),
            "n_indications_with_at_least_one_candidate": int(
                len(mapped_ids)
            ),
            "n_unmapped_indications": int(len(unmapped)),
            "n_candidate_rows": int(len(candidates)),
            "candidates_by_source": (
                {
                    str(k): int(v)
                    for k, v in candidates[
                        "candidate_source"
                    ].value_counts().items()
                }
                if not candidates.empty
                else {}
            ),
            "candidates_by_tier": (
                {
                    str(k): int(v)
                    for k, v in candidates[
                        "mapping_tier"
                    ].value_counts().items()
                }
                if not candidates.empty
                else {}
            ),
        },
        "important_limits": [
            "No indication-to-trait mapping is locked in Step 15A2.",
            "GBMI sample sizes are not inferred from file existence.",
            "A GBMI endpoint is eligible only if EUR and at least one non-EUR file are independently verified.",
            "Pan-UKB duplicate measurements are collapsed to one canonical trait per construct.",
            "Unsupported indications remain unmapped rather than receiving forced lexical matches.",
        ],
        "outputs": {k: str(v) for k, v in paths.items()},
    }

    summary_path = (
        args.output_dir / "15a2_mapping_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 78)
    print("STEP 15A2 — SOURCE VERIFICATION AND MAPPING CANDIDATES")
    print("=" * 78)
    print(
        f"Pan-UKB quantitative/biomarker input: "
        f"{len(panukb_catalog):,}"
    )
    print(
        f"Pan-UKB canonical clinical core:      "
        f"{len(panukb_core):,}"
    )
    print(
        f"GBMI verified data files:             "
        f"{summary['gbmi']['n_verified_data_files']:,}"
    )
    print(
        f"GBMI cross-ancestry endpoints:        "
        f"{summary['gbmi']['n_cross_ancestry_eligible_endpoints']:,}"
    )
    print(
        f"Indications with candidates:          "
        f"{len(mapped_ids):,}/{len(indications):,}"
    )
    print(
        f"Candidate mapping rows:               "
        f"{len(candidates):,}"
    )
    print(
        f"Unmapped indications:                 "
        f"{len(unmapped):,}"
    )
    print()
    print(f"GBMI verification: {gbmi_method_note}")
    print()
    print("No mapping was accepted automatically.")
    print("Outputs:")
    for path in paths.values():
        print(f"  {path}")
    print(f"  {summary_path}")
    print("=" * 78)

    # A nonzero exit would make ordinary source sparsity look like a program
    # failure. Instead, the summary records unresolved verification. The next
    # step must not lock an unverified source.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
