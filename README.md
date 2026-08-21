# How and When Ancestry Matters in Genetic Evidence for Drug Targets

This repository contains the full analysis pipeline for the study **“How and When Ancestry Matters in Genetic Evidence for Drug Targets.”**

The repository is organized as a sequential workflow (`step1` through `step15`). The early steps construct the genetically supported drug-target cohort and ancestry representation variables; the middle steps fit the launch/approval analyses and robustness models; and the final steps evaluate direct-disease feasibility, cross-ancestry effect portability, and colocalization.

---

## 1. Repository structure

```text
target_ancestry/
├── genetic_support/        # Minikel et al. source-data snapshot used by Steps 1–3/9
├── step1/                  # Build genetically supported clinical-stage target–indication pairs
├── step2/                  # Link target–indication pairs to GWAS studies
├── step3/                  # Attach GWAS ancestry information
├── step4/                  # Build pair-level ancestry trails
├── step5/                  # Study-volume control analyses
├── step6/                  # Robustness analyses
├── step7/                  # Within-gene comparison
├── step8/                  # MeSH mapping and approval regressions
├── step9/                  # Cohort-selection diagnostics
├── step10/                 # Within-indication conditional models
├── step11/                 # Selection-weighted regression
├── step12/                 # Practical-significance / predicted-probability analyses
├── step13/                 # Direct-disease phenotype mapping feasibility
├── step14/                 # Direct-disease SNP feasibility
├── step15/                 # Portability, LD processing, colocalization, and integration
└── figures/
    ├── publication_v5_code/
    └── publication_v5/
```

Most scripts use paths relative to their own step directory. **Run each script from the directory in which the script is located unless stated otherwise.**

---

## 2. Software environment

The analysis was developed in Linux/WSL using **Python 3.12**.

### Create the Python environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Main Python libraries

The analysis uses:

- `numpy`
- `pandas`
- `pyarrow`
- `scipy`
- `statsmodels`
- `scikit-learn`
- `patsy`
- `pysam`
- `matplotlib`
- `hail`
- `pyspark`

---

## 3. Required input data

Most small/frozen inputs needed for reproduction are included directly in this repository. Large GWAS and LD resources are accessed remotely by the scripts rather than stored in GitHub.

| Resource | Local location | Included? | Used by |
|---|---|---:|---|
| Minikel drug-target/genetic-support data (`merge2.tsv.gz`) | `genetic_support/data/merge2.tsv.gz` | Yes | Steps 1, 2, 9 |
| Frozen GWAS Catalog ancestry table | `genetic_support/data/gwas_catalog-ancestry_r2022-02-02.tsv` | Yes | Step 3 |
| MeSH 2026 descriptors | `step8/input/desc2026.gz` | Yes | Steps 8–9 |
| MeSH 2026 descriptors used for phenotype mapping | `step13/input/manifests/desc2026.gz` | Yes | Steps 13 and 15A |
| Pan-UKB phenotype manifest | `step13/input/manifests/phenotype_manifest.tsv.bgz` | Yes | Steps 13 and 15A |
| GENCODE v19 annotation, GRCh37 | `step14/input/reference/gencode.v19.annotation.gtf.gz` | Yes | Steps 14 and 15B |
| GENCODE v38 annotation, GRCh38 | `step15/input/reference/gencode.v38.annotation.gtf.gz` | Yes | Step 15B |
| Pan-UKB summary statistics | remote paths stored in the Pan-UKB manifest | Remote | Steps 14B, 15B |
| GBMI summary statistics | `https://gbmi-sumstats.s3.amazonaws.com/` (`GBMI_052021`) | Remote | Step 15A2 onward |
| Pan-UKB ancestry-specific LD | public Pan-UKB S3 Hail resources | Remote | Step 15C |
| 1000 Genomes phased GRCh38 VCFs | official 1000 Genomes FTP/HTTPS release | Remote | Step 15C |

The Minikel source files were originally obtained from:

https://github.com/ericminikel/genetic_support

The source snapshot used when this repository was assembled corresponded to commit:

```text
b84cf0580b7df1af14b2c72068ed42b5aa0c52a4
```

Because the required source files are already included in `genetic_support/data/`, cloning the external repository is **not required** to run this pipeline.

### Internet access

Internet access is required for the parts of Steps 14–15 that query or download regional data from:

- Pan-UK Biobank
- GBMI
- 1000 Genomes
- public Pan-UKB LD resources

The full GWAS summary-statistic files do not need to be downloaded manually. The scripts retrieve the required regional data using the remote URLs stored or constructed by the pipeline.

---

# 4. Reproducing the analysis

Clone the repository and enter it:

```bash
git clone https://github.com/ChristopherEbukaOjukwu/target_ancestry.git
cd target_ancestry
```

Activate the environment:

```bash
source .venv/bin/activate
```

If starting on a new machine, create the environment first as described above.

---

## Step 1 — Build the genetically supported clinical-stage cohort

**Script**

```text
step1/01_fetch_pairs.py
```

**Main input**

```text
genetic_support/data/merge2.tsv.gz
```

The script restricts the Minikel target–indication data to genetically supported pairs (`comb_norm >= 0.8`) and defines:

- **Pool A:** launched targets
- **Pool B:** Phase I–III targets that had not launched

Run:

```bash
cd step1
python3 01_fetch_pairs.py
cd ..
```

**Main output**

```text
step1/output/01_pairs.parquet
```

Expected clinical-stage genetically supported cohort:

```text
1,354 target–indication pairs
```

---

## Step 2 — Link target–indication pairs to supporting GWAS studies

**Script**

```text
step2/02_pair_to_studies.py
```

**Inputs**

```text
step1/output/01_pairs.parquet
genetic_support/data/merge2.tsv.gz
```

The script extracts parseable GWAS study identifiers from the genetic-support records, including GWAS Catalog, Neale UK Biobank, FinnGen, and SAIGE identifiers.

Run:

```bash
cd step2
python3 02_pair_to_studies.py
cd ..
```

**Main output**

```text
step2/output/02_pair_studies.parquet
```

Expected number of clinical-stage pairs linked to parseable GWAS studies:

```text
797
```

---

## Step 3 — Attach study-level ancestry information

**Script**

```text
step3/03_study_ancestry.py
```

**Inputs**

```text
step2/output/02_pair_studies.parquet
genetic_support/data/gwas_catalog-ancestry_r2022-02-02.tsv
```

The script joins GWAS Catalog ancestry records and adds the known European ancestry designation for the supported Neale, FinnGen, and SAIGE sources used in this analysis.

Run:

```bash
cd step3
python3 03_study_ancestry.py
cd ..
```

**Main output**

```text
step3/output/03_study_ancestry.parquet
```

This is the study-by-ancestry table used to construct the ancestry landscape.

---

## Step 4 — Build target–indication ancestry trails

**Script**

```text
step4/04_build_trails.py
```

**Inputs**

```text
step1/output/01_pairs.parquet
step2/output/02_pair_studies.parquet
step3/output/03_study_ancestry.parquet
```

For each target–indication pair, the script constructs:

- ancestries represented in initial evidence
- ancestries represented in replication evidence
- all represented ancestries
- number of represented ancestries
- number of supporting studies
- replication indicators

Run:

```bash
cd step4
python3 04_build_trails.py
cd ..
```

**Main output**

```text
step4/output/04_trails.parquet
```

Expected final ancestry-analysis cohort:

```text
790 pairs
157 Pool A (launched)
633 Pool B (Phase I–III, not launched)
383 unique genes
```

This file is the main pair-level dataset used by most later analyses.

---

## Step 5 — Control for study volume

**Script**

```text
step5/05_volume_control.py
```

**Input**

```text
step4/output/04_trails.parquet
```

The analysis compares ancestry breadth between Pools A and B within study-volume strata.

Run:

```bash
cd step5
python3 05_volume_control.py
cd ..
```

Outputs are written to:

```text
step5/output/
```

---

## Step 6 — Robustness analyses

**Script**

```text
step6/06_robustness.py
```

**Input**

```text
step4/output/04_trails.parquet
```

This step evaluates robustness of the ancestry patterns, including:

- removal of highly diverse pharmacogenomic targets
- initial versus replication ancestry representation
- collapsing target–indication pairs to the gene level

Run:

```bash
cd step6
python3 06_robustness.py
cd ..
```

The script writes its summary results in the `step6/` directory, including:

```text
step6/06_robustness_result.txt
```

---

## Step 7 — Within-gene comparison

**Script**

```text
step7/07_within_gene.py
```

**Input**

```text
step4/output/04_trails.parquet
```

This is a descriptive within-target analysis restricted to genes represented in both Pool A and Pool B. It asks whether launched and non-launched indications for the same target differ in ancestry breadth.

Run:

```bash
cd step7
python3 07_within_gene.py
cd ..
```

Outputs are written to:

```text
step7/output/
```

Expected checkpoint:

```text
57 genes occur in both pools
32 genes are informative after excluding identical ancestry trails
```

---

## Step 8 — MeSH disease mapping and approval regression

Step 8 has two parts.

### Step 8A — MeSH mapping

**Script**

```text
step8/08_mesh_disease_mapping.py
```

**Inputs**

```text
step4/output/04_trails.parquet
step8/input/desc2026.gz
```

Run:

```bash
cd step8
python3 08_mesh_disease_mapping.py
```

The script maps indications to MeSH categories and writes the mapping tables to:

```text
step8/output/
```

Important outputs include:

```text
step8/output/08_pair_mesh_categories.parquet
step8/output/08_mesh_category_counts.csv
```

### Step 8B — Approval models

**Script**

```text
step8/08_approval_regression.py
```

**Inputs**

```text
../step4/output/04_trails.parquet
../step2/output/02_pair_studies.parquet
output/08_pair_mesh_categories.parquet
output/08_mesh_category_counts.csv
```

Run from `step8/`:

```bash
python3 08_approval_regression.py
cd ..
```

**Main outputs**

```text
step8/output/08_model_dataset.parquet
step8/output/08_approval_regression.parquet
step8/output/08_approval_regression_coefficients.parquet
step8/output/08_model_diagnostics.csv
step8/output/08_approval_regression.json
```

The models progressively adjust the ancestry-breadth association for study volume, earliest genetic evidence year, and MeSH indication categories.

Expected model dataset:

```text
N = 790
Pool A = 157
Pool B = 633
```

---

## Step 9 — Cohort-selection diagnostics

**Script**

```text
step9/09_cohort_selection.py
```

**Inputs**

```text
step1/output/01_pairs.parquet
step2/output/02_pair_studies.parquet
step4/output/04_trails.parquet
genetic_support/data/merge2.tsv.gz
step8/input/desc2026.gz
```

Run:

```bash
cd step9
python3 09_cohort_selection.py
cd ..
```

Outputs are written to:

```text
step9/output/
```

This step reconstructs the attrition from the broader genetically supported clinical-stage cohort to the final ancestry-analysis sample and checks whether filtering disproportionately removes Pool A or Pool B pairs.

Expected checkpoint:

```text
Clinical-stage supported pairs: 1,354
GWAS-linked pairs:              797
Final ancestry cohort:          790
```

---

## Step 10 — Within-indication conditional analysis

**Script**

```text
step10/10_within_indication.py
```

**Inputs**

```text
step4/output/04_trails.parquet
step2/output/02_pair_studies.parquet
```

The primary analysis uses conditional logistic regression stratified by exact MeSH indication. Fixed-effect logistic models with gene-clustered standard errors are also produced as sensitivity analyses.

Run:

```bash
cd step10
python3 10_within_indication.py
cd ..
```

**Outputs**

```text
step10/output/10_within_indication_dataset.parquet
step10/output/10_indication_counts.csv
step10/output/10_within_indication_differences.csv
step10/output/10_within_indication_regression.csv
step10/output/10_within_indication_coefficients.csv
step10/output/10_within_indication_diagnostics.csv
step10/output/10_within_indication.json
```

Expected checkpoint:

```text
55 mixed Pool A/Pool B indications
513 analysis pairs
116 Pool A
397 Pool B
```

---

## Step 11 — Selection-weighted regression

**Script**

```text
step11/11_selection_weighted_regression.py
```

**Inputs**

```text
step9/output/09_cohort_selection_pairs.parquet
step8/output/08_model_dataset.parquet
step8/output/08_mesh_category_counts.csv
step8/input/desc2026.gz
```

This step evaluates whether selection into the final ancestry-analysis cohort materially changes the approval-model result.

Run:

```bash
cd step11
python3 11_selection_weighted_regression.py
cd ..
```

Outputs are written to:

```text
step11/output/
```

This step requires `scikit-learn` in addition to the main statistical libraries.

---

## Step 12 — Practical significance and predicted probabilities

**Script**

```text
step12/12_practical_significance_power.py
```

**Inputs**

```text
step8/output/08_model_dataset.parquet
step8/output/08_mesh_category_counts.csv
```

This step converts the regression results into practical-effect summaries, including standardized predicted launch probabilities across ancestry breadth.

Run:

```bash
cd step12
python3 12_practical_significance_power.py
cd ..
```

Outputs are written to:

```text
step12/output/
```

---

# 5. Direct-disease feasibility branch

Steps 13–14 test whether the portability analysis could be restricted to GWAS of the therapeutic disease itself. This branch became too sparse for a meaningful launch comparison, but it is retained as a reproducible feasibility analysis.

---

## Step 13 — Pan-UKB direct-disease phenotype mapping

Required frozen inputs are already stored under:

```text
step13/input/manifests/
```

including:

```text
desc2026.gz
phenotype_manifest.tsv.bgz
```

Run the Step 13 scripts in order:

```bash
cd step13

python3 13a_panukb_feasibility.py
python3 13b_panukb_mapping_shortlist.py
python3 13c_mesh_direct_mapping_candidates.py
python3 13d_conservative_direct_candidates.py
python3 13e_lock_panukb_direct_mappings.py

cd ..
```

### What the substeps do

- **13A:** checks the Pan-UKB phenotype manifest, requires QC passage in at least two populations, and produces lexical indication-to-phenotype candidates.
- **13B:** converts the candidate set into a compact review table.
- **13C:** uses official MeSH names and entry terms to generate direct-disease ICD-10/PheCode candidates.
- **13D:** applies a conservative lexical screen so weak forced matches are removed.
- **13E:** locks the manually adjudicated direct-disease mappings.

The final adjudication decisions are encoded directly in `13e_lock_panukb_direct_mappings.py`, so the locked set can be regenerated deterministically from the candidate tables.

Key outputs include:

```text
step13/output/13_panukb_direct_mapping_locked.csv
step13/output/13_panukb_direct_pairs_locked.parquet
```

---

## Step 14 — Direct-disease SNP feasibility

### Step 14A — GENCODE v19 coordinates

**Script**

```text
step14/14a_build_gencode19_gene_coordinates.py
```

**Inputs**

```text
step14/input/reference/gencode.v19.annotation.gtf.gz
step13/output/13_panukb_direct_pairs_locked.parquet
```

Run:

```bash
cd step14
python3 14a_build_gencode19_gene_coordinates.py
```

### Step 14B — Pan-UKB regional SNP feasibility

**Script**

```text
step14/14b_panukb_snp_feasibility.py
```

First validate one remote schema:

```bash
python3 14b_panukb_snp_feasibility.py --mode validate
```

Then run the full count:

```bash
python3 14b_panukb_snp_feasibility.py --mode count
cd ..
```

Step 14B queries the Pan-UKB summary statistics remotely and stores downloaded tabix indexes under:

```text
step14/input/panukb_tabix_indexes/
```

The legacy file:

```text
step14/14b_panukb_snp_feasibility_original.py
```

is retained for provenance and should **not** be used for the final reproduction.

---

# 6. Step 15 — Cross-ancestry portability and colocalization

Step 15 is the mechanistic cross-ancestry branch. It uses ancestry-specific GWAS from Pan-UKB and GBMI, performs ancestry-paired LD pruning, estimates effect-size portability, runs colocalization, and integrates the results.

The primary analysis unit is a **target–trait–source unit**, because ancestry-specific GWAS of the therapeutic disease itself were often unavailable or insufficiently powered.

All commands below are run from:

```bash
cd step15
```

---

## Step 15A — Build and lock trait mappings

Run:

```bash
python3 15a_build_trait_mapping_workspace.py
python3 15a2_verify_sources_and_build_candidates.py
python3 15a3_lock_trait_mappings.py
```

### 15A1

Builds the phenotype-mapping workspace from:

```text
../step4/output/04_trails.parquet
../step13/input/manifests/phenotype_manifest.tsv.bgz
../step13/input/manifests/desc2026.gz
```

No mapping is automatically accepted.

### 15A2

Independently verifies GBMI source files from the public `GBMI_052021` release and builds conservative Pan-UKB/GBMI candidate mappings.

### 15A3

Locks the prespecified scientific mapping decisions. The accepted primary/sensitivity decisions are encoded in the script itself.

Key outputs:

```text
output/15_trait_mapping_locked.csv
output/15_trait_mapping_primary.csv
output/15_trait_mapping_sensitivity.csv
output/15_pair_trait_mapping_locked.parquet
output/15_gene_trait_units_locked.parquet
```

---

## Step 15B — Extract, harmonize, QC, and freeze analysis universes

Run:

```bash
python3 15b1_build_dual_build_extraction_manifest.py
python3 15b2_validate_regional_schemas.py
python3 15b3_extract_all_regions.py
python3 15b4_harmonize_and_qc.py
python3 15b5_freeze_analysis_universes.py
```

### 15B1 — Extraction manifest

Uses:

```text
output/15_gene_trait_units_locked.parquet
output/15a_trait_catalog.csv
output/15a2_gbmi_verified_files.csv
../step14/input/reference/gencode.v19.annotation.gtf.gz
input/reference/gencode.v38.annotation.gtf.gz
```

Coordinates are kept in each source's native build:

- Pan-UKB: GRCh37 / GENCODE v19
- GBMI: GRCh38 / GENCODE v38

The extraction region is the gene body ±100 kb.

### 15B2 — Remote-schema validation

Downloads only the necessary tabix indexes and validates representative Pan-UKB and GBMI regional queries.

Requires:

```text
pysam
```

Indexes are stored under:

```text
input/tabix_indexes/15b2/
```

### 15B3 — Regional extraction

Extracts all locked regional datasets and caches them under:

```text
intermediate/15b3/Pan-UKB/
intermediate/15b3/GBMI/
```

Downloaded tabix indexes are stored under:

```text
input/tabix_indexes/15b3/
```

### 15B4 — Harmonization and QC

Produces one harmonized EUR-versus-non-EUR regional dataset per comparison under:

```text
intermediate/15b4/comparisons/
```

Primary QC includes:

- biallelic A/C/G/T SNPs
- finite beta and SE in both ancestries
- SE > 0
- MAF >= 1% in both ancestries
- Pan-UKB `low_confidence == false`
- exact allele matching for the primary GBMI analysis
- at least 3 shared QC-qualified EUR genome-wide-significant variants for pre-LD portability eligibility
- at least 50 shared QC-qualified variants for colocalization input eligibility

### 15B5 — Freeze analysis universes

Creates the locked portability and colocalization comparison/unit tables used downstream.

Important outputs include:

```text
output/15_portability_comparisons_primary_locked.parquet
output/15_portability_comparisons_sensitivity_locked.parquet
output/15_coloc_comparisons_primary_locked.parquet
output/15_coloc_comparisons_powered_locked.parquet
output/15b5_unit_analysis_universe.parquet
```

---

## Step 15C — LD reference selection and ancestry-paired LD pruning

### 15C1 — Freeze LD requests

Run:

```bash
python3 15c1_freeze_ld_requests.py
```

LD policy:

- **Pan-UKB / GRCh37:** ancestry-specific Pan-UKB in-sample LD Hail BlockMatrices
- **GBMI / GRCh38:** ancestry-matched phased 1000 Genomes GRCh38 genotypes

Primary pruning threshold:

```text
r² < 0.10 in BOTH EUR and the comparison-ancestry LD reference
```

Sensitivity thresholds:

```text
r² < 0.01
r² < 0.20
```

At least 3 retained variants are required.

### Pan-UKB Hail/Spark environment

Before running the Pan-UKB LD scripts, set:

```bash
export PYSPARK_SUBMIT_ARGS="--packages org.apache.hadoop:hadoop-aws:3.3.4 pyspark-shell"
export SPARK_LOCAL_IP=127.0.0.1
export SPARK_LOCAL_DIRS=/tmp/target_ancestry_spark
```

A Java runtime compatible with the installed Hail/PySpark environment is required.

### 15C2 — LD preflight checks

These scripts validate the two LD branches before the full run:

```bash
python3 15c2a_panukb_s3a_preflight.py
python3 15c2b_panukb_subset_ld_pilot.py
python3 15c2c_gbmi_1kg_ld_pilot.py
```

- **15C2A:** verifies Pan-UKB Hail S3 access.
- **15C2B:** verifies extraction of ancestry-specific Pan-UKB LD subsets.
- **15C2C:** verifies exact variant resolution and LD calculation using 1000 Genomes GRCh38 for a GBMI comparison.

The 1000 Genomes sample panel is downloaded automatically into:

```text
input/reference/1000G/
```

### 15C3 — Full LD pruning

Run:

```bash
python3 15c3_run_full_ld_pruning.py
```

The script is resumable and caches LD resolution/results under:

```text
output/15c3_cache/
```

Primary outputs include:

```text
output/15c3_variant_reference_resolution.parquet
output/15c3_variant_pruning_decisions.parquet
output/15c3_comparison_pruning_results.parquet
output/15c3_portability_variants_primary_retained.parquet
output/15c3_portability_comparisons_primary_final.parquet
output/15c3_ld_pruning_summary.json
```

---

## Step 15D — Estimate cross-ancestry effect portability

Run:

```bash
python3 15d_estimate_portability.py
```

For every comparison that passes LD pruning, the script estimates the through-origin slope of non-European effects on European effects using:

1. naive slope
2. FIQT winner's-curse-corrected slope
3. Deming/errors-in-variables slope

Comparison-level confidence intervals are obtained by seeded bootstrap.

Main outputs:

```text
output/15d_primary_comparison_portability.parquet
output/15d_primary_unit_portability.parquet
output/15d_variant_effects_primary.parquet
output/15d_portability_summary.json
```

Expected primary portability checkpoint:

```text
104 EUR–non-EUR comparisons
50 target–trait–source units
```

---

## Step 15E — Cross-ancestry colocalization

### Input preflight

Run:

```bash
python3 15e0_coloc_input_preflight.py
```

This checks the locked colocalization universe and the actual harmonized-file schema.

### Colocalization

Run:

```bash
python3 15e_run_colocalization.py
```

The analysis implements coloc-style approximate Bayes factors in Python using:

```text
p1 = 1e-4
p2 = 1e-4
primary p12 = 1e-5
p12 sensitivity = 1e-6, 1e-5, 1e-4
```

Main outputs:

```text
output/15e_coloc_comparison_results.parquet
output/15e_coloc_unit_results.parquet
output/15e_coloc_variant_posteriors_primary.parquet
output/15e_coloc_headline_powered_units.parquet
```

### Prior-stability classification

Run:

```bash
python3 15e1_audit_coloc_prior_stability.py
python3 15e2_finalize_coloc_stability.py
```

The final classification distinguishes:

- `ROBUST_SHARED`
- `ROBUST_DISTINCT`
- `PRIOR_SENSITIVE`
- ancestry-discordant units where applicable

Expected powered colocalization checkpoint:

```text
17 ancestry comparisons
14 target–trait–source units
```

---

## Step 15F — Integrate portability and colocalization

Run:

```bash
python3 15f_integrate_mechanistic_results.py
```

This creates final unit- and comparison-level datasets combining the portability and colocalization branches.

Main outputs:

```text
output/15f_mechanistic_units.parquet
output/15f_mechanistic_comparisons.parquet
output/15f_portability_pool_summary.parquet
output/15f_portability_pool_contrasts.parquet
output/15f_coloc_pool_summary.parquet
output/15f_mechanistic_summary.json
output/15f_results_notes.md
```

The script contains explicit expected-count checks for:

```text
Primary mechanistic units:       213
Primary mechanistic comparisons: 538
Portability units:                50
Portability comparisons:         104
Powered coloc units:              14
Powered coloc comparisons:        17
```

---

## Step 15G — Portability robustness and sample-size analyses

Run:

```bash
python3 15g_portability_robustness.py
python3 15g2_sample_size_audit.py
python3 15g3_sample_balance_models_fixed.py
```

### 15G

Produces:

- precision-aware portability summaries
- variant-count sensitivity analyses
- colocalization-stratified portability
- random-effects unit-level meta-analysis
- locus-level heterogeneity and forest-plot data

### 15G2

Audits EUR and non-EUR sample sizes and derives effective sample size for case-control traits:

```text
Neff = 4 / (1/Ncase + 1/Ncontrol)
```

### 15G3

Tests whether portability is associated with the non-EUR/EUR effective-sample-size ratio after adjustment for comparison ancestry.

Key outputs include:

```text
output/15g2_sample_size_audit.csv
output/15g2_sample_size_summary.csv
output/15g2_portability_sample_size_sensitivity.csv
output/15g3_sample_balance_models.csv
output/15g3_sample_balance_descriptives.csv
output/15g3_sample_balance_summary.txt
```

After Step 15 is complete:

```bash
cd ..
```

---

# 7. Reproduce the manuscript figures

The publication figure scripts are in:

```text
figures/publication_v5_code/
```

The main figures are:

1. cohort and ancestry landscape
2. study volume and evidence stage
3. adjusted and matched ancestry–launch analyses
4. cross-ancestry portability
5. cross-ancestry colocalization

From the repository root:

```bash
mkdir -p figures/publication_v5

python3 -u figures/publication_v5_code/make_all.py \
  2>&1 | tee figures/publication_v5/make_all.log
```

Outputs are written to:

```text
figures/publication_v5/main/
figures/publication_v5/supplement/
figures/publication_v5/tables/
```

Figures are generated as PNG, PDF, and SVG.

---

# 8. Key reproducibility checkpoints

A successful full reproduction should recover the following principal counts/results, allowing only ordinary floating-point rounding differences.

## Cohort construction

```text
Genetically supported target–indication pairs: 2,166
Clinical-stage genetically supported pairs:    1,354
Linked to parseable GWAS studies:                 797
Complete ancestry information:                    790
Pool A, launched:                                 157
Pool B, Phase I–III/not launched:                 633
```

## Pair-level ancestry representation

```text
European ancestry represented: 749 / 790 pairs (94.8%)
```

## Approval models

Approximate checkpoints:

```text
Unadjusted OR per additional ancestry: ~1.13
Study-volume-adjusted OR:              ~1.03
Fully adjusted ancestry OR:            ~1.02
```

## Cross-ancestry portability

```text
Primary portability comparisons: 104
Primary portability units:        50
```

Random-effects FIQT summary:

```text
~0.684
```

## Colocalization

```text
Powered ancestry comparisons: 17
Powered target–trait units:    14
```

---

# 9. Notes on reproducibility

### Included outputs

The repository contains derived outputs from the original analysis. These are retained so that a reproducer can compare regenerated files against the analysis used for the manuscript.

### Remote resources

The large Pan-UKB, GBMI, Pan-UKB LD, and 1000 Genomes resources are not copied into this GitHub repository. The relevant scripts contain or derive the remote paths and retrieve only the data needed for the analysis.

### Manual scientific decisions

The phenotype-mapping stages intentionally do not accept lexical matches automatically. The final manually adjudicated decisions used in the analysis are encoded in:

```text
step13/13e_lock_panukb_direct_mappings.py
step15/15a3_lock_trait_mappings.py
```

This makes the final locked mapping reproducible even though the original scientific review was manual.

### Resumable Step 15 analyses

Several Step 15 scripts cache remote extractions, LD calculations, portability estimates, or colocalization outputs. Re-running the same commands will generally reuse valid caches. The scripts expose `--force` or `--rebuild-cache` options where a complete regeneration is required.

---

# 10. Minimal full run order

For reference, the analysis order is:

```text
Step 1
  ↓
Step 2
  ↓
Step 3
  ↓
Step 4
  ↓
Steps 5–7
  ↓
Step 8A → Step 8B
  ↓
Steps 9–12
  ↓
Steps 13A–13E          direct-disease feasibility
  ↓
Steps 14A–14B          direct-disease SNP feasibility

Step 15A1 → 15A2 → 15A3
  ↓
15B1 → 15B2 → 15B3 → 15B4 → 15B5
  ↓
15C1 → 15C2A/B/C → 15C3
  ↓
15D
  ↓
15E0 → 15E → 15E1 → 15E2
  ↓
15F
  ↓
15G → 15G2 → 15G3
  ↓
Publication figures
```

---

## Citation

If using this repository, please cite the accompanying manuscript and the original data resources described above.
