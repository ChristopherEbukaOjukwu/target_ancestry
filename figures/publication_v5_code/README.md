# target_ancestry publication figures v5

This version merges the previous Figures 3 and 4 and renumbers the remaining
main figures.

## Main figures

1. Cohort and ancestry landscape
2. Study volume and evidence stage
3. Adjusted and matched ancestry–launch analyses
4. Cross-ancestry portability
5. Cross-ancestry colocalization

M1–M4 and W1–W3 are restored in Figure 3. Their definitions are provided in
`CAPTIONS.md`.

The previous colocalization summary panel has been replaced. Figure 5a now
shows every powered gene–trait unit, its final stability category, and its
launch pool.

## Install

From the repository root:

```bash
mkdir -p figures/publication_v5_code

unzip -o \
  target_ancestry_publication_v5.zip \
  -d figures/publication_v5_code

python3 -m py_compile figures/publication_v5_code/*.py
```

## Run

```bash
mkdir -p figures/publication_v5

python3 -u figures/publication_v5_code/make_all.py \
  2>&1 | tee figures/publication_v5/make_all.log
```

## Output

```text
figures/publication_v5/main/
figures/publication_v5/supplement/
figures/publication_v5/tables/
```

Every figure is saved as 600-dpi PNG, vector PDF, and editable SVG.
