Dataset
merge2.tsv.gz from Minikel's repo (github.com/ericminikel/genetic_support).

Why this dataset
It's the table their entire 2024 Nature paper was built on. Three properties matter:
It has both halves needed pre-joined. Each row is a target-indication pair joined to one of its candidate genetic associations. So the pipeline side (which gene targets which disease, what phase it reached) and the evidence side (which GWAS/OMIM associations exist for that gene) are already connected. You don't have to build that join yourself, and you inherit a join Minikel defended in peer review.
It carries the trait-similarity score. The column comb_norm measures how close each candidate association's trait is to the drug's indication. Minikel's rule for "this pair has genetic support" is comb_norm ≥ 0.8. Reusing their threshold means your supported pool is comparable to theirs — which matters because the differentiation argument for your paper is "Minikel built this pool and modeled approval without ancestry; I add ancestry." If you redefined "supported," that argument weakens.
It has the outcome column you need. ccat gives each pair's highest phase reached, with the values Launched / Phase I / II / III / Preclinical. That's how you split into approved vs. tried-but-not-approved.
Why not the other files
pp.tsv is pipeline-only — no genetic evidence attached.
assoc.tsv is evidence-only — no pipeline.
indic.tsv has indication-level metadata but not pair-level support.
merge2 is the one that has all three things stitched together.
Why not skip Minikel and use raw GWAS Catalog + a drug pipeline DB
Two reasons. First, you'd be rebuilding a pipeline that's already been published and peer-reviewed — months of work to reproduce what's downloadable. Second, you'd then have a slightly different supported pool from Minikel's, and any difference in your results vs. the field's existing numbers would be ambiguous: is it your ancestry layer, or your different pool? Using their pool isolates your contribution.
What Step 1 actually does
Take merge2, and produce one row per target-indication pair with three things attached: whether it's genetically supported, which pool it belongs to, and the launch year (if any).
The logic:

Supported? A pair is supported if any of its association rows has comb_norm ≥ 0.8. Most association rows in merge2 are candidates the algorithm considered; the ≥ 0.8 filter is what counts as real support.
Pool A (approved): ccat == "Launched" AND supported.
Pool B (tried but not approved): ccat in {Phase I, II, III} AND supported.
Excluded: Preclinical (never entered trials, so they don't belong to either pool), and any unsupported pair.

Output: a clean table of supported pairs labeled A or B.
Why this matters for the whole project
Step 1 produces the universe. Every later step — attaching trial dates (Step 2), pulling each pair's underlying studies (Step 3), getting ancestry (Step 4), building the before/after trails (Step 5) — operates on the pairs in this table. So the integrity of the whole project depends on this step being defensible. That's why "use Minikel's pool, exact threshold, log the decision" matters more than the code does.
What we'll know after Step 1
Two headcount numbers: how big is Pool A, how big is Pool B. Those numbers are the first real signal from real data. They decide whether the design we sketched holds, narrows, or needs to change.
