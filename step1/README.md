## Step 1 — Define the analysis cohort

### Dataset

`merge2.tsv.gz` from Minikel et al.:

https://github.com/ericminikel/genetic_support

### Why this dataset

`merge2.tsv.gz` already links three things needed for this study:

- drug target–indication pairs
- genetic associations supporting each pair
- clinical development stage

It also contains `comb_norm`, Minikel et al.'s trait–indication similarity score. Following their published definition, a target–indication pair is considered genetically supported if at least one association has:

```text
comb_norm >= 0.8
