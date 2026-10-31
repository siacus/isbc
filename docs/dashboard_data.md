# Prepare data for ISBC Explorer

The repository provides the **fitted binary codes** for AG News and PHEME. Source
texts are downloaded separately and joined locally. No training, model weights,
embedding extraction, API key for X, or GPU is required. Read the
[data-license notes](data_licenses.md) before obtaining source content.

Install [uv](https://docs.astral.sh/uv/) and run these commands from the repository
root. The dashboard uses Python 3.11–3.13. `uv` installs the required packages.

## AG News: download, prepare, open

```sh
uv run --project dashboard --with datasets python download_agnews.py --output_dir data/agnews
uv run --project dashboard python analysis/prepare_dashboard_data.py --dataset agnews
uv run --project dashboard python dashboard/app.py --runs dashboard/local_data/agnews_qwen
```

The first command downloads all 120,000 training and 7,600 test records from
`fancyzhx/ag_news` and prepares `data/agnews/agnews_all.csv`. Do not supply
`--max_per_class` for this example. The second command joins text and category
labels to the published 16-bit codes, retaining all 127,600 fitted document IDs.

## PHEME: download, prepare, open

```sh
uv run --project dashboard python download_pheme.py --output_dir data/pheme
uv run --project dashboard python analysis/prepare_dashboard_data.py --dataset pheme
uv run --project dashboard python dashboard/app.py --runs dashboard/local_data/pheme_qwen
```

The first command obtains the versioned Figshare archive, verifies its published
MD5 and prepares the five events used in the paper. The second joins the 5,802
source-tweet texts to the codes and annotations, and extracts the archived
engagement/account fields directly from that archive. It does not query today's
X API. The original archive stays in `data/pheme/PHEME_veracity.tar.bz2`.

If you already have that archive, the download step can instead use:

```sh
uv run --project dashboard python download_pheme.py --output_dir data/pheme --archive_path /path/to/PHEME_veracity.tar.bz2
```

## Open and inspect

Run one dashboard at a time, open **http://127.0.0.1:8050**, and click
**Load analysis**. Leave the optional annotation source empty: the reconstructed
CSV contains the text and annotations. PHEME also contains the archived outcomes.
To run a second instance, append `--port 8051` to its dashboard command.

Generated files are:

- `dashboard/local_data/agnews_qwen/encoded_agnews_qwen.csv`
- `dashboard/local_data/pheme_qwen/encoded_pheme_qwen.csv`

Each includes `id`, `text`, the sixteen `bit_XX` columns and the applicable
annotations. Local PHEME data also include `retweet_count`, `favorite_count` and
account covariates. Local data and downloaded source archives are excluded from
Git. They are not covered by the code's MIT license.

## Checks and reruns

```sh
uv run --project dashboard python analysis/export_dashboard_data.py --verify
```

This verifies the public code files and text-checksum tables. Local preparation
then applies the fitted pipeline's whitespace normalization (collapse runs of
whitespace and strip outer whitespace), and checks unique IDs, corpus size,
every text's SHA-256 and PHEME annotations.
It fails before writing an output if the downloaded texts do not match the fitted
corpus. A changed upstream release or a different sampling option must be resolved
rather than silently attaching codes to different texts.

Rerunning preparation leaves an identical output unchanged and refuses to replace
a different file. You can choose another `--output-root` for a separate local copy.
If the prepared source collections are elsewhere, use `--source-root /path/to/root`,
where that root contains `data/agnews/` or `data/pheme/`.

Without downloading source content, you can open `dashboard/data/agnews_qwen` or
`dashboard/data/pheme_qwen` to inspect code geometry. Text reading, AG News category
colours and PHEME outcome displays require the local preparation above.
