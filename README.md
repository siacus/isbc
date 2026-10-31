# ISBC: Interpretable Sparse Binary Coding of Pretrained Text Embeddings

This repository accompanies the paper by Stefano M. Iacus and Haodong Qi.
It provides the ISBC model, data preparation scripts, experimental configurations,
analysis code, and an interactive application for inspecting the learned codes.

ISBC maps frozen document embeddings to a small sparse binary representation
without using category labels or outcome supervision. An affine decoder
reconstructs the embeddings. Each bit selects a document subset; combinations of
active bits select overlapping subsets. A **profile** is the complete pattern of
active and inactive bits and assigns each document to one exact group.

There are two ways to use this repository:

- **Explore the results:** open the included AG News or PHEME data in ISBC Explorer.
  This requires no model download, training or GPU.
- **Reproduce or extend the analyses:** prepare the source datasets, train or load
  encoders, and run the documented experiments.

## Explore the included results

The public bundles contain fitted codes and document IDs; PHEME also includes
CC BY 4.0 annotations. Source texts and archived platform counters are downloaded
and joined locally. Follow [Dashboard data preparation](docs/dashboard_data.md)
for the complete download → preparation → launch commands. No training or GPU is
needed. See [data licenses and attribution](docs/data_licenses.md).

After preparing the data, launch either example:

```sh
uv run --project dashboard python dashboard/app.py --runs dashboard/local_data/agnews_qwen
uv run --project dashboard python dashboard/app.py --runs dashboard/local_data/pheme_qwen
```

Run one command at a time. Open **http://127.0.0.1:8050** and click **Load analysis**.

| Dataset | Documents | Representation | Fields joined locally |
| --- | ---: | --- | --- |
| AG News | 127,600 | 16 Qwen-based ISBC bits, target activity 6 | Text and four news categories |
| PHEME | 5,802 | 16 Qwen-based ISBC bits, target activity 6 | Text, archived engagement/account fields |

See the [Explorer guide](dashboard/README.md) for filters, projections, category
colours and outcome displays. Each data directory includes provenance and
checksums. Bit numbers refer to these fitted encoders; they do not have a shared
meaning across datasets or independently trained models.

## Repository map

| Location | Contents |
| --- | --- |
| `binary_factors_train.py` | ISBC training, encoding and inspection |
| `MODEL_SPEC.md` | Architecture, reconstruction objective and regularization |
| `download_*.py` | Downloading, cleaning and preparing source datasets and annotations |
| `build_mixed_dataset.py` | Construction of a new Mixed sample |
| `analysis/rebuild_mixed.py` | Reconstruction of the Mixed fitting sample used in the paper using recorded membership and text hashes |
| `analysis/` | Portable data preparation, PHEME model components and figure/export helpers |
| `configs/training/` | 39 configurations recovered from fitted checkpoints, including ablations and width/activity settings |
| `reproduce.py` | List experiments, inspect commands, restore splits and launch workflows |
| `analyze_*.py`, `compare_*.py`, `report_*.py`, `plot_*.py` | Numerical analyses, comparisons and figures |
| `recipes/` | Selected numerical procedures for particular results and supplementary analyses |
| `runs/` | Selected experiment-specific scripts; generated results also use this directory |
| `profile_matching/` | Matching and comparison of complete profiles |
| `dashboard/` | ISBC Explorer application and its separate dependency lock |
| `dashboard/data/` | Public fitted codes, IDs, text checksums and PHEME annotations |
| `dashboard/local_data/` | Locally reconstructed text-bearing dashboard CSVs (Git-ignored) |
| `reproduction/` | Original split IDs, Mixed membership, provenance, figure-source mapping and validation records |
| `tests/` | Numerical and reproducibility checks |
| `docs/` | Data instructions, workflow notes, scope and qualitative-reading procedure |

Generated caches, model weights and other downloaded datasets are excluded from
version control. The public code bundles are included; source text and platform counters are not.

## Prepare data and run ISBC

Use a separate Python environment for training and numerical analyses:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python reproduce.py verify
python reproduce.py list
```

`requirements.txt` records the experiment environment, including the Transformers
revision. The dashboard has its own environment and can be used independently.

For the full AG News reference experiment:

```sh
python reproduce.py run download-agnews --execute
python reproduce.py restore-splits
python reproduce.py train agnews_qwen_all_fixed_balance --execute
python reproduce.py encode agnews_qwen_all_fixed_balance --execute
python reproduce.py run agnews-structure --execute
```

The saved configuration uses Apple MPS and BF16 backbone inference. Use
`--device cuda` or `--device cpu`, and an appropriate `--precision`, for other
hardware. Changing hardware or software can change the learned bit identities.
Without `--execute`, the training, encoding and workflow commands are printed
for inspection. The wrapper refuses to replace an existing fitted checkpoint.

The [data guide](docs/data.md) lists the download/preparation script for each
corpus. These scripts prepare IDs, texts and label tables; PHEME's archived counts
are extracted separately by `analysis/prepare_pheme_engagement.py`.
`analysis/export_dashboard_data.py` exports the public code-only bundles.
`analysis/prepare_dashboard_data.py` joins locally obtained source content for
the dashboard, as described in the [preparation guide](docs/dashboard_data.md). To check the included bundles without downloading
source data:

```sh
python analysis/export_dashboard_data.py --verify
```

## Reproduce the paper analyses

| Analysis | Workflow or entry point |
| --- | --- |
| AG News bit associations and document subsets | `agnews-structure` |
| Comparison with embedding clusters | `agnews-clustering` |
| Code width, activity and geometry | `agnews-capacity`, `agnews-geometry` |
| Learning-objective ablations | `agnews-ablation`, after training the recorded variants |
| Correspondence across embedding backbones | `agnews-backbones`, `backbone-communities` |
| Alternative encoders, PCA and ITQ | `external-setup`, `external-encoders`, `external-densities`, `itq-controls` |
| Other annotated corpora | `cross-corpus` |
| Mixed fitting sample and new-document analysis | `mixed-rebuild`, `mixed-new-documents-*` |
| PHEME prediction on log outcomes | `pheme-extract`, `pheme-log` |
| PHEME engagement tails and content | `pheme-tails` |
| AG News optimizer timing | `scaling` |

For example, `python reproduce.py run pheme-log` displays its commands and
prerequisites. Consult the [reproduction guide](docs/reproduction.md) for execution
order and intermediate artifacts. The [scope document](docs/scope.md) identifies
which analyses are included, and the [reading procedure](docs/qualitative_reading.md)
describes the qualitative component.

This is a release candidate. Component tests, all twenty PHEME OLS specifications,
and recovery of the original Mixed data have been checked. A full clean-install
reproduction has not yet been completed. The fitted checkpoints and the complete
encoding archive are not bundled; the included dashboard codes support direct
inspection of the two supplied fitted representations.

## Tests

```sh
python -m unittest discover -s tests -v
uv run --project dashboard python -m unittest dashboard.test_dashboard -v
```

The tests use synthetic inputs; bundle verification uses the included data.

## Citation and licenses

Citation metadata are in [CITATION.cff](CITATION.cff).
Original code is distributed under the [MIT license](LICENSE). Source datasets,
pretrained models and third-party implementations retain their own licenses and
terms; see [THIRD_PARTY.md](THIRD_PARTY.md), [data-license notes](docs/data_licenses.md)
and the per-bundle `DATA_LICENSE.md` notices.
