# ISBC Explorer

ISBC Explorer lets readers inspect the document groups selected by the binary
representations studied in the accompanying paper. It displays saved codes,
texts, annotations and outcomes. It does not train a model or compute embeddings.

## Start with a published example

First follow [Dashboard data preparation](../docs/dashboard_data.md). It provides
commands to obtain AG News or the PHEME archive and join texts to the published
fitted codes locally. The public repository contains codes, IDs and PHEME
annotations, not source texts or platform counters; see the
[data-license notes](../docs/data_licenses.md).

After preparation, launch either example from the repository root:

```sh
uv run --project dashboard python dashboard/app.py --runs dashboard/local_data/agnews_qwen
uv run --project dashboard python dashboard/app.py --runs dashboard/local_data/pheme_qwen
```

Python 3.11–3.13 is supported. No model weights, training or GPU are needed.
Open **http://127.0.0.1:8050** and click **Load analysis**. Leave the optional
annotation source empty because the locally reconstructed CSV includes it.
Keep the terminal running; press **Ctrl+C** to stop. Append `--port 8051` to use
another port.

Both examples use sixteen binary coordinates from a fitted Qwen-based ISBC
encoder with target activity six. PHEME's locally extracted engagement values
are archived source-tweet counters, not current X/Twitter measurements.
The public code-only folders can also be opened directly, but text reading and
outcome displays require local preparation.

## Bits, conjunctions and profiles

A **bit** is one learned binary coordinate. Selecting required active bits chooses
documents in which those bits are all active. With the other coordinates left
unrestricted, this is a **conjunction**, potentially containing many profiles.
Selecting the exact-code option additionally requires every unselected bit to
be inactive.

A **profile** is the complete pattern across all sixteen coordinates. Documents
with the same profile share a plotted point. Click a point to inspect its texts
and annotations. A bit number identifies a coordinate of the selected fitted
model; it is not a semantic label and cannot be transferred between encoders by
number alone.

## Read the display

- **Selection and sampling:** distinguish the number of documents matching the
  bit selection from the documents used in the displayed sample. Adjust sample
  size and seed when needed. The count tables identify the relevant populations.
- **Projection:** select the method, distance and scope. UMAP and MDS give each
  distinct profile equal weight; PCA uses document-frequency weights. Two-dimensional
  proximity is a visualization of the selected distance, not a new category label.
- **Marker size:** represents the number of displayed documents sharing a profile.
  It does not show the number of active bits.
- **Category colours:** summarize annotations within a profile. The most frequent
  category determines its colour; ties and missing annotations have separate
  colours. A coloured point can contain documents from several categories.
- **Outcome colours:** select a numeric field and Continuous mode to display
  profile medians. Logarithmic colour scaling changes colour spacing; the values
  and summaries remain in their original units.

Changing colours does not refit the binary representation. External annotations
help describe the selected texts; they were not used to fit ISBC.

## Explore the AG News examples

Select bits **00, 05, 06, 10, 12**, leaving other bits unrestricted. In the included
encoding this selects **2,432 documents in 116 complete profiles**: 2,275 Sci/Tech,
147 Business and 10 World documents. Read texts from different profiles to examine
the technology-market conjunction discussed in the paper.

For the international political-economy example, select **03, 07, 08, 11, 15**.
This selects **1,568 documents in 108 profiles**, including 22 Business documents.
A smaller displayed sample need not reproduce the full-selection category counts.

## Explore PHEME engagement

Set the sample size to **5802** to include the full supplied corpus. Choose
`retweet_count` or `favorite_count` under **Colour by outcome**, select
**Continuous**, and optionally use the logarithmic colour scale. Click profiles
to read their texts and compare their event and rumour annotations.

The conjunction **05, 11**, with other coordinates unrestricted, is the Charlie
Hebdo subset discussed in the paper. Inspect the presence or absence of bit **02**
within this context to explore the document distinction analyzed there. The
dashboard supports inspection; its colour display is not a regression estimate
or evidence of a causal effect of a bit on engagement.

## Use your own encoded data

```sh
uv run --project dashboard python dashboard/app.py --runs /path/to/encoded_documents.csv
```

The filename should begin with `encoded_`. Required columns are a unique `id`
and hard binary columns named `bit_00`, `bit_01`, and so on. Include `text` for
reading documents. Annotations and outcomes may be columns in the same CSV or
provided through a separate annotation file joined by `id`. Probabilities alone
are not valid hard codes.

The original source files and model checkpoints are not required to open a
portable bundle. To regenerate the supplied format from saved experiment outputs,
see `analysis/prepare_dashboard_data.py` and the repository's
[data guide](../docs/data.md).

## Export and installation alternatives

Use **Export PDF** to save the current view. Chrome or Chromium must be installed
locally for PDF export. Choose the white appearance for a print-friendly view.

The dashboard dependencies are separate from the training environment and locked
in `uv.lock`. Without uv, create a Python 3.12 environment and install the hashed
requirements:

```sh
python3.12 -m venv dashboard/.venv
dashboard/.venv/bin/python -m pip install --require-hashes -r dashboard/requirements.txt
dashboard/.venv/bin/python dashboard/app.py --runs dashboard/local_data/agnews_qwen
```

For implementation details, see [DESIGN.md](DESIGN.md). For automated checks:

```sh
uv run --project dashboard python -m unittest dashboard.test_dashboard -v
```
