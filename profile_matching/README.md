# Profile-to-profile matching across embedding models

## Question

Each embedding model learns its own binary factors, and bit indices are run-local. If the factorial representation is invariant to the embedding model, the same documents should receive corresponding *profiles* (patterns of active bits) in different models, whatever the bit numbering. This directory tests that directly on documents, without fixing a single global bit permutation in advance and without using any category label to decide the match.

## Method

Run `profile_matching.py` (reads `factor_collection_comparison.json` of a run collection and the encoded CSVs it points to; documents are aligned by `id`, which is identical across models).

The main `compare_factor_run_collection.py` command now runs this analysis automatically
when all runs share the same bit width and embeds its complete Markdown, tables, plots, and downloadable
CSV artifacts in the general comparison report. Use `--skip_profile_matching` only when
an intentionally lightweight report is needed. Existing reports can be refreshed
with `backfill_profile_matching.py`.

Profile identity keys support K=32 and larger codes. Bit offsets are encoded as `0`--`F`
inside repeated 16-bit blocks, `|` separates blocks, `-` preserves an empty block, and
`*` separates models. Thus `bit_01 + bit_17` is `1|1`; `bit_01 + bit_18` is `1|2`.
The block separator is required for an unambiguous reverse mapping.

1. **Preparation.** Bits that are constant (always on or always off) in a model are removed. All models remain eligible: diffuse profile distributions are part of the result rather than a reason to exclude a backbone.
2. **Start.** For each reference model, take the 20 most frequent observed profiles and the 10 least frequent observed profiles (`--top-reference-profiles 20`, `--tail-reference-profiles 10`), removing overlap when a codebook has fewer than 30 observed profiles. The rank rule is independent of corpus size and replaces the former 1% threshold.
3. **Chain.** Restrict to the documents with that profile. Among the other models, find the most frequent profile within those documents (no constraint on the number of active bits) and take the model for which it is most frequent. Restrict the documents to those that also have this profile and repeat with the remaining models until all are matched.
4. **Record.** A match is shown as the list of active bits of each model, in fixed model order, as bit indices (`0` = `bit_00`), e.g. `qwen 1,4,5,9,13,15`, `nvidia 0,3,6,8,11,13`, and so on. Internally the script keys each match with the reversible blockwise hexadecimal encoding described above only to detect identical matches; outputs show the bit lists. Per step the script also records the share of the current sample covered by the chosen profile and its lift (that share divided by the profile's overall frequency in that model), and the number of documents left. Matches whose final sample has fewer than 30 documents are marked weak.
5. **Tabulation.** With `--all-references` the chain is started from every model; a match reached from different starting models is a *reciprocal* match. A second table counts, for each reference bit and model, which bit appears most often in the matched profiles. Starting from a single model, whole matches cannot repeat (each begins with a different reference profile), so reciprocity across starting models is the informative tabulation.
6. **Semantics.** For each reference profile the script reports the most frequent labels among its documents for every taxonomy (label column) in the labels file, with the label's share of the whole dataset for comparison. Semicolon-separated fields are detected as multi-label taxonomies and expanded before counting: a document such as `flood; coastal erosion` contributes once to `flood` and once to `coastal erosion`. Percentages are document frequencies and may therefore sum to more than 100%. The labels play no role in the matching; they only describe it.

7. **Invariance index.** For each reference profile and mutually exclusive taxonomy, `coherence` = 1 - mean Jensen-Shannon divergence (bits) between the category distributions of the documents having each matched model's profile, and `information` = mean KL divergence (bits) of those distributions from the overall category distribution. Multi-label taxonomies are not passed to this categorical test. Two nulls (1,000 draws each, `--null-draws`): **swap** replaces the non-reference models' profiles with another profile of the same model of similar size; **chain** shuffles the document alignment of each non-reference model independently and reruns the whole chain, so the selection of the most frequent profile is part of the null. Results in `invariance.csv`, `invariance.png` and the "Invariance index" section of `output.md`.

8. **Multi-label graph.** When the report's primary taxonomy is multi-label, every per-profile graph retains the model-to-label flows and adds a horizontal histogram on the right. At most the ten most frequent individual labels in the initial reference subgroup are shown. On the same label order, paired bars compare the final matched intersection with the excluded residual documents. Exact initial, final, and residual frequencies for every observed label are stored in `profile_multilabel_distributions.csv`.

The per-profile frequency distributions of all models (including the excluded ones) are plotted in `profile_distribution.png`.

## Running

```bash
# Refresh selected integrated reports from the repository root:
./latent/bin/python backfill_profile_matching.py \
    --report runs/20ng_embedding_comparison/all_fixed_balance_label_coarse/factor_collection_comparison.json \
    --report runs/airline_embedding_comparison/all_fixed_balance_airline/factor_collection_comparison.json \
    --null-draws 1000

# Refresh every existing fixed-width collection report from the repository root:
./latent/bin/python backfill_profile_matching.py --null-draws 1000
```

The canonical outputs are written inside each collection as
`runs/<dataset>_embedding_comparison/<comparison>/profile_matching/` and embedded in
that collection's `factor_collection_comparison.md`. Standalone dataset output folders
under this source directory are not retained.

Outputs per dataset: `output.md` (tables and, with `--graphs`, one graph per reference profile in `graphs/`: models to categories, or models to individual labels plus final/residual histograms for a multi-label primary taxonomy), `matches.csv`, `profile_categories.csv` (top three labels per taxonomy, for the reference and final documents), `profile_multilabel_distributions.csv` when multi-label taxonomies are detected, `bit_pairs.csv`, `invariance.csv`, `invariance.png`, `profile_distribution.png`, and `manifest.json`.

## Results

**20 Newsgroups.** Used: qwen, nvidia, harrier, gemma (minilm excluded, top profile 2.0%). 15 reference profiles; 55 distinct matches, 14 of them reached from at least two starting models and 4 from all four . Only 4 of the 15 matches end with fewer than 30 documents. The reference profiles are nearly pure in the coarse category (mostly 91–100%): the two largest are rec (`rec.sport.hockey` 55% / `rec.sport.baseball` 45%) and talk (`soc.religion.christian` 48% / `alt.atheism` 25%). The 2.6% profile that is `misc` 46% and `comp` 42% (mostly `misc.forsale` and hardware groups) sits between two coarse categories.

**Airline tweets.** Used: qwen, harrier, gemma (minilm excluded at 1.1%, nvidia at 4.1%). 21 reference profiles; 45 distinct matches, only 5 reached from at least two starting models; 7 of 21 matches end with fewer than 30 documents. Quotes and lifts per step are lower than in 20ng. The profiles are nearly pure in the airline (96–100% in most), but not in sentiment (e.g. the 12.7% profile is United 100% but negative 61%, against 63% overall).

## Reading the results with care

- **Evidence of invariance is partial and taxonomy-specific.** Where profiles match across models, they match on the taxonomy they separate well: topic in 20ng, airline name in the tweets. In the tweets the airline is usually announced by a mention, so purity there is close to expected and says little about deeper content such as sentiment.
- **Reciprocal counts remain descriptive.** The swap and chain nulls test semantic
  coherence of each matched profile, not the number of reciprocal strings. Reciprocity
  therefore remains a descriptive stability check rather than a calibrated test.
- **Greedy and order dependent.** The model chosen at each step is the one with the largest share in the current sample; with lift as the criterion the matching order, and possibly the matches, would change. This was not tried.
- **Small final samples.** After several restrictions some matches rest on a handful of documents (marked weak); strong matches are those with high shares at every step.
- **Bit-pair table is thin.** It rests on 15 (20ng) or 21 (airline) reference profiles, so each bit appears in only a few of them; a fraction of 1.00 over five profiles is not proof.
- **Model exclusion is a threshold choice.** The integrated analysis starts at 5% but
  lowers that threshold exactly to Qwen's observed top-profile share when needed. This can admit additional
  dispersed models, so the final-document counts and weak-match flags remain essential.
