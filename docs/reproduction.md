# Reproduction guide

## Inputs and execution order

1. Prepare the source datasets using the included downloaders. Match the input
   hashes and row counts in the training configurations. AG News uses all
   127,600 documents from the source train and test collections.
2. Restore evaluation splits with `python reproduce.py restore-splits`.
3. Train and encode the required configurations. The trainer extracts and caches
   frozen embeddings and validates existing caches. Configuration files specify
   the batch sizes, loss weights, seeds and optimization settings.
4. Run the numerical workflows listed by `python reproduce.py list`. Inspect
   prerequisites with `python reproduce.py run WORKFLOW` before execution.
5. Compare regenerated results with the reference outputs and manuscript.

The width/activity experiments use six Qwen configurations: 8/4, 8/6, 16/6,
32/6, 32/8 and 32/16. The capacity analyses require comparison manifests,
including the common ID split and raw embedding references. The comparison
tools generate these intermediate artifacts; they are not prebundled.

The ablation analysis uses eighteen configurations covering six objective
settings and seeds 1234, 2025 and 31415. A zero activity coefficient disables
the adaptive activity controller. The reference and variants use the same
corpus and evaluation split.

The cross-corpus workflow uses Qwen hard codes and the annotation specifications
in `reproduction/cross_corpus_targets.json`.

## Mixed-domain sample

The Mixed fitting corpus contains 111,000 documents.
`reproduction/mixed_membership.csv` records IDs, source IDs, source, type and text
hashes in fitting order. After preparing the source collections, run:

```sh
python reproduce.py run mixed-rebuild --execute
```

`analysis/rebuild_mixed.py` reconstructs this fixed sample using ID and text-hash
matching and checks the resulting file hashes. The frozen profile manifest
records the representation used for the additional-document analysis. A new
random sample or a new encoder fit does not preserve those exact bit identities.

## PHEME and timing

`pheme-extract` reads engagement counters from archived source tweets.
`pheme-log` fits OLS and variance-split random forests to training-standardized
log(1+count), followed by descriptive trees. R package versions are recorded in
`reproduction/reference/pheme/sessionInfo.txt`. The portable thread default is
one; set `ISBC_R_THREADS` to override it.

`pheme-tails` uses all documents and inclusive empirical quantile thresholds,
retaining ties. It is an exploratory analysis, separate from held-out prediction.

The timing workflow measures head/decoder optimization at 16/6 using cached
embeddings. Embedding extraction is outside the timing measurement. The paper's
measurements use an Apple M2 Max with 96 GB unified memory.

## Fitted representations and figures

Retraining can change hard codes across software and hardware versions, even
with the same seed. Numerical and qualitative examples referring to specific
bits or profiles require the corresponding fitted representation. The included
AG News and PHEME public bundles provide those hard codes and IDs, plus PHEME
annotations. Source content is joined locally using the
[dashboard preparation guide](dashboard_data.md). Check them with:

```sh
python analysis/export_dashboard_data.py --verify
```

The complete checkpoint and encoding archive is not yet bundled. Configurations
record the input hashes needed to identify the fitted experiments.

`analysis/plot_conjunction_profiles.py` projects the AG News conjunctions using
Hamming distance, 15 neighbors, minimum distance 0.1, random initialization and
seed 1234. Install `requirements-figures.txt` for these exports. Exact figure
layout and annotation placement have not yet been visually validated against
the manuscript. The model diagram is an authored static asset.

The PHEME tail recipes export `conditional_bit_examples.pdf` and
`keyword_tail_contrasts.pdf`, corresponding to the manuscript assets
`pheme_tail_conditional_bits.pdf` and `pheme_tail_keyword_contrasts.pdf`.
Some recipes write results alongside their scripts in `recipes/` or `runs/`.

## Validation and provenance

`reproduction/paper_inventory.json` identifies the manuscript snapshot.
`reproduction/source_manifest.json` records source provenance, and
`reproduction/release_files.json` records package file hashes.

Checks completed for this release include:

- Python parsing and workflow/configuration path checks.
- Unit tests for decoder initialization, embedding-cache resume, positive-PMI
  graph communities, reconstruction/reference calculations and the dashboard.
- Agreement of all twenty PHEME OLS specification/outcome predictions with the
  saved results to within 6e-15, and exact agreement of bootstrap indices.
- Reconstruction of the Mixed fitting tables with matching file hashes.
- Dashboard bundle checksums and agreement with source texts, codes and annotations.

The full training and analysis suite has not been rerun in a clean environment,
and random forests have not been refitted during packaging. End-to-end validation,
the complete checkpoint archive, complete table-to-command mapping and final
figure comparisons remain pending. Component checks do not establish numerical
reproduction of every reported result.
