# ISBC dashboard: design and implementation

Status: implemented locally (2026-10-30). Python Dash + Plotly interface,
with NumPy/SciPy/scikit-learn/UMAP numerical dependencies in an isolated environment.
This document records the design. The user-facing guide is [README.md](README.md);
the application renders that same file in its Help panel. For file ownership,
setup, verification evidence and continuation steps, see [HANDOVER.md](HANDOVER.md).

Outcome-colour extension (2026-10-31): a separate selector overrides label colour
without changing selection or projection. Discrete mode reuses profile category
counts and tie handling. Continuous mode uses sampled-member medians, shows means
and valid/missing counts in hover, and fixes limits to full-corpus document values.
Both negative and positive values select symmetric limits about zero and a
diverging scale; otherwise the scale is sequential. Zero-only and other constant
outcomes use a noted padded display interval. Nonfinite/nonnumeric values are
missing for continuous summaries. JSON includes the scale and summaries; PDF
captures the live plot. `--annotations` prepopulates the optional join source,
including local annotation tables. A portable PHEME/Qwen bundle with engagement
is included under `dashboard/data/pheme_qwen/`, alongside the AG News bundle.
Continuous colour optionally transforms the original median and corpus bounds by
`sign(x) * log1p(abs(x))`; hover and exported means/medians stay in original units,
as do colour-bar tick labels. The transform is recorded separately. Hover lists
only active coordinates, each at 100% within the complete profile, in numeric order.

## Purpose

Explore the documents associated with individual learned bits and combinations of
bits. Read texts directly, inspect their binary-space positions, and compare
label distributions where annotations are available. Labels are descriptive
overlays, not inputs to the projection or the bit selection.

Strict scope: embeddings must never enter this dashboard. Do not discover, load,
compute, project or offer them as an alternative view. Use hard binary codes only
for geometry; continuous activation probabilities are not projection inputs either.

## Interface

- A local path picker accepts the runs root or an individual model run. If a root
  contains multiple runs, show a dataset/model/checkpoint selector.
- Build the bit selector from the available `bit_*` columns, rather than assuming
  that every model has 16 bits.
- Default selection means **all selected bits active**, with other bits unrestricted.
  No selected bits means all documents in this mode. An **Unselected bits** toggle
  also supports exact-profile matching: selected bits are one, all others zero.
  With no selected bits, exact mode matches only the all-zero profile. Apply the
  mode to sampling, label panels and export, and show the rule explicitly.
- Show the matching-document count and its percentage of the corpus.
- Plot a random sample of up to 1,000 matching documents by default, with a
  configurable sample size and user-editable integer seed.
- Each change to the bit selection triggers sampling from the new matching subset.
  The same run, selection, matching mode, seed and requested size must reproduce the same sample,
  independent of click order or previous selections. Sort eligible document IDs
  canonically and use a fresh deterministic random generator for each request.
  Sample without replacement; show every matching document when fewer are available.
- Provide a rotatable, zoomable and pannable 3D point cloud, with a reset-view
  control and a 2D option. Aggregate identical complete profiles into one point,
  sized by their sampled document count. Clicking opens a navigable list of those
  documents, with complete text, ID, active bits and labels. Do not add jitter that
  suggests differences between identical codes.
- Below the bit selector, provide a scrollable panel for label distributions,
  one card per selected annotation field. Sort categories by decreasing frequency.
- Show a clear empty state for zero matches. One or two matches should remain
  readable without attempting an unsuitable projection.

Colour overlays: choose active-bit count or an annotation field. A complete-profile
point is coloured by its most frequent observed label among sampled members;
tied maxima and wholly missing annotations have distinct colours. Hover shows the
composition and observed/missing denominators. Multi-label fields use document-level
prevalence. Palettes use the corpus-wide category vocabulary so filtering does not
reassign colours. Colouring never alters the sample, geometry or camera revision.
Export includes the colour rule and per-profile composition. Label panels continue
to use all matching documents, independently of the plotted sample.

## Projections

Every view projects the **complete hard binary codes**, filtered by the selected
bits. Selection is a document filter, not a restriction to the selected coordinates.
Identical codes must receive identical coordinates. Sample documents first, then
group identical profiles for display. Report both document and unique-profile counts.

Separate sampling from projection. The displayed documents change when the user
changes the selected bits; the coordinate system may remain fixed or be recomputed:

1. **Global context:** fit a projection once for the run and reuse it. Common
   profiles keep the same coordinates and axis ranges across selections. PCA
   is computed using profile counts as document-frequency weights.
2. **Selected sample (default, with PCA):** fit a projection to the current sampled documents. This
   emphasizes variation inside the selection, but coordinates cannot be compared
   directly across different selections.

Offer PCA on binary coordinates, classical MDS with Hamming or Jaccard distances,
and UMAP with Hamming or Jaccard distances. Hamming counts coordinate mismatches
(display whether normalized by K); Jaccard compares active-bit sets. Define
Jaccard distance between two empty sets as zero. Do not standardize bit columns
silently. PCA uses the document-frequency-weighted covariance of unique profiles, equivalent
to fitting all document rows. MDS/UMAP fit unique profiles so duplicate codes are
never artificially separated.
State this weighting difference: PCA reflects document frequency, while unique-profile
MDS/UMAP geometry treats profiles equally. Point size always reflects sampled counts.
For global MDS/UMAP, guard the number of unique profiles and report any fixed
reference-sample approximation explicitly; never silently substitute a local fit.

Projection choice and scope must be visible. For classical MDS, expose the distance
definition and report negative eigenvalues when distances are not Euclidean.
Classical MDS with Euclidean distances reproduces PCA sample geometry up to rigid
transformations, so it should not be presented as independent geometric evidence.
For UMAP, expose its seed, distance metric and core neighbourhood parameters.
Do not claim that projected clusters establish semantic categories.

Cache projections by unique-profile bytes/shape, document frequencies for PCA,
and method parameters. Dataset loading is cached by path, modification time and
size. Local/global scope determines which rows are supplied to the projection. The interface uses the selected seed for both sampling and UMAP.
A fixed seed targets reproducibility within the recorded software
environment; retain dependency versions and settings for reproducible exports.
Exact MDS requires a sample-size guard because it constructs pairwise distances.

## Label statistics

Compute statistics from **all matching documents**, not only the plotted sample.
Keep the plot sample count and the statistical population count visibly distinct.

Below the plot, document-count tables list every category for selected annotation
fields and the colour field: matching documents, percentage among observed matching
annotations, sampled documents and whole-corpus documents. Include missing and total
rows. Minority categories within a profile keep their own document counts regardless
of the point's colour. The tables follow the exact/unrestricted matching rule and
sample seed, and are included in exports. Multi-label counts can overlap.

For each categorical annotation, show counts and percentages among matching
documents with an observed applicable annotation. Also show missing/inapplicable
counts and the denominator. Include a corpus-reference percentage with the same
annotation-validity rule, so corpus imbalance is visible.

Mixed contains source-specific annotations: an unavailable annotation must not
be interpreted as a negative label. True multi-label fields require per-label
prevalence, whose percentages need not sum to 100%. Numeric outcomes should not
be silently treated as categorical labels. Let the user select annotation fields;
exclude IDs, text, bit/probability columns and provenance keys from automatic
categorical suggestions unless explicitly configured.

## Existing data contracts verified locally

Inspected AG News and Mixed Qwen runs on 2026-10-30:

- `encoded_*_best.csv`: `id`, `text`, `n_active`, `bit_00` etc., `prob_00` etc.
- `metadata.json`: model, code width, checkpoint and source paths. Ignore embedding
  cache references. Ignore `prob_*` columns when loading the binary representation.
- Prepared training inputs (`*_ready.csv`) contain only `id,text`, not labels.
- AG News annotations: `data/agnews/agnews_all.csv`.
- Mixed annotations: `data/mixed/mixed_labels.csv`; the richer corpus is also
  available as `data/mixed/mixed.parquet`.
- Collection reports can identify annotation files, for example
  `runs/agnews_embedding_comparison/all_fixed_balance_label_name/factor_collection_comparison.json`.

Align encoded documents and annotations by ID, never by assumed row order.
Validate ID uniqueness, code width, bit values and provenance. Report
unmatched IDs. Preserve ID strings, including leading zeros. If IDs are not
globally unique, require a verified composite key rather than guessing.

Resolve existing manifest paths first, then compatible paths relative to the run
and repository. Old absolute paths may be stale after relocation. If several
candidate files remain, show them for selection rather than silently picking one.
Allow a UI override for annotations and, if necessary, texts.
Ignore incomplete `.partial` files.

A run path alone is sufficient only when the required artifacts and referenced
annotations are accessible. A shareable report bundle without document-level data
is not sufficient. Missing labels should disable only label panels. No embedding
file is required for any functionality, including the spatial view.

## Local application and delivery

Provide a self-contained application folder with pinned dependencies and a documented
single launch command. The UI framework is Dash, with Plotly for the interactive map. The path
picker browses the filesystem of the machine running the application. Operate
locally and render document text as plain text, not executable HTML. No retraining,
embedding extraction, or external text service is needed.

Read run artifacts without modifying them. Derived projections and loaded tables
are cached in server memory only; restarting the application clears them. Export selected document IDs, selection
rule, sample seed, projection settings and annotation summaries. Source texts are
included only in the user-authorized AG News and PHEME portable bundles.

## Presentation and PDF reporting

`appearance.py` applies dark or white Plotly styling without changing traces,
coordinates, sample membership or the camera revision. The page uses a matching
CSS theme, saved as a local browser preference. `plot-state` retains the current
3D camera and 2D axis ranges, resetting them when the projection revision changes.
A colour or theme change preserves the revision. Global filtering also preserves it.
The PDF implementation captures the live Plotly view directly and does not consume
the legacy `plot-state` store or `apply_viewport` helper.

`assets/export.js` captures a clone of the actual dashboard DOM when Export PDF is
clicked. It copies input values, checked states and open/closed details explicitly,
and replaces the live WebGL plot with `Plotly.toImage` output. This retains the
current camera, zoom, theme and hidden-trace state without refitting. The copy expands
scrollable label panels, tables and the selected document. The source page remains
unchanged. Source details and Help are included only if open; menu popovers and hover
tooltips are excluded. The reader includes the currently selected document and its
annotations, not every document in the sample.

The self-contained snapshot is transferred to the local Dash server through a Store.
`report.py` sanitizes executable/embedded markup, applies a restrictive CSP, and uses
an isolated headless Chrome/Chromium process through Choreographer to print the
snapshot to PDF. Scripts and external HTTP/HTTPS/file requests are disabled. No live
app URL, source table or embedding is opened by the renderer. Fonts and Unicode text
use the browser's normal local rendering. The plot is a PNG; other content is native
PDF text. PDF width matches the captured viewport, with proportional portrait pages
and normal page breaks. This avoids a different responsive layout during export.

The export preserves the chosen dark/white theme. White mode is recommended for
printing. It preserves collapsed sections and hidden legend traces rather than
silently adding content the user chose to hide. Scrollable contents are expanded so
all current label cards, table rows and the selected text can be read. The browser
print fallback remains separate and is excluded from snapshot styles because it
hides interface controls. The PDF download is named `isbc-dashboard.pdf`, with a persistent download link
after generation. `pyproject.toml` declares Choreographer as a direct dependency.

The document-count table is also an interpretation safeguard: profile colours must
not be counted as documents. The AG News five-bit example has 2,432 matching documents,
147 of them Business, across 116 complete profiles. With seed 1234 and sample size
1,000, 59 Business documents are sampled. Exact matching instead yields 35 documents,
only four Business. The guide explains these distinct populations explicitly.

## Verification checks

- Verify ID alignment on deliberately reordered fixtures.
- Verify conjunction semantics and dynamic bit counts, including empty selections.
- Verify identical samples for identical settings regardless of interaction order.
- Verify label percentages and missing-label denominators against direct counts.
- Verify profile clicks retrieve the corresponding sampled documents.
- Verify useful behaviour without labels and for tiny subsets or few unique profiles.
- Verify no embedding files are opened or requested by any application path.
- Verify identical profiles remain a single point with correct document counts.
- Verify README content is accessible through the application's Help control.

## Limits and verification

- Classical MDS is limited to 2,500 unique profiles. For larger global analyses,
  choose PCA/UMAP or switch to a smaller local sample; no silent approximation.
- Sample size is configurable up to 10,000 documents in the interface.
- Fewer than four profiles use classical MDS instead of UMAP, with a visible note.
- Label cards show the 20 most frequent categories; document tables and JSON include
  all categories. PDF captures the current cards and open tables, preserving collapsed sections.
- Automatic source discovery uses `metadata.json` and nearby source-table naming
  conventions. Collection-report traversal is not implemented; use the annotation
  override when a source cannot be discovered. Ambiguous suggestions are shown in
  the source selector; the first is suggested visibly before Load analysis.
- CSV codes are required. Optional CSV/Parquet source tables join on unique `id`.
- Numerical annotations are available for explicit selection. Continuous outcome
  colour overlays are supported; separate outcome-distribution plots and automated
  semantic summaries remain outside this release.
- `test_dashboard.py` checks ID alignment, conjunctions, reproducible sampling,
  full-code grouping, global coordinate stability, PCA equivalence, label
  denominators, multi-label counts, small subsets, invalid inputs and no embedding reads.

## Maintenance status

See [HANDOVER.md](HANDOVER.md) for the 20-test baseline, actual PDF verification,
reference AG News counts and known limits. Python 3.12/macOS ARM64 was tested;
Python 3.11-3.13 is declared. `requirements.txt` is a generated, hashed export of
`uv.lock` for pip users. Git contains the application and a portable full AG News/Qwen bundle under
`dashboard/data/agnews_qwen/` and a full PHEME/Qwen bundle with engagement under
`dashboard/data/pheme_qwen/`. Other standard encoded run CSVs and source labels
may be ignored and must be supplied separately. The portable bundle embeds text
and labels alongside the hard bits; original source paths in its manifest are
provenance only, not runtime dependencies.
