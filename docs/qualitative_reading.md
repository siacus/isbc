# Qualitative inspection

The numeric recipes regenerate selections, keywords and inspection dossiers.
They do not call an LLM service. Interpretations embedded in report-writing
scripts are frozen readings made during preparation of the paper; re-running
those scripts republishes those readings, it does not constitute an independent
replication of semantic judgments.

PHEME `content_analysis/prepare.py` defines the reproducible reading sample and
complete Charlie Hebdo decile dossier. `write_report.py` records the paper's
interpretations and verifies that cited document IDs belong to the stated groups.
All 286 Charlie Hebdo decile-tail documents were read in the original analysis.
Other sampling seeds and group rules are recorded by each recipe.

A suitable reading request is: summarize recurring topics and communicative
styles in each supplied document group; distinguish shared content from
contrasts; support each interpretation with supplied document IDs; report
counterexamples and uncertainty; do not infer causal effects or invent texts.
This is a reusable prompt template, not a verbatim record of every reading
conversation. The supplementary material describes the reading tasks used in
the paper. Manual inspection can replace the LLM-assisted summaries.
