# Data and exact membership

Run downloaders explicitly from the repository root. Each downloader contains
its source URL, label definitions and preparation rules. Data are written under
`data/`, which is not committed. AG News and PHEME public code
bundles are included separately under `dashboard/data/`. Inspect `--help` for
selection/output options.

| Corpus | Preparation script |
| --- | --- |
| AG News | `download_agnews.py` (all 127,600 documents for the paper) |
| 20 Newsgroups | `download_20newsgroups.py` |
| Airline | `download_airline.py` |
| GoEmotions | `download_goemotions.py` |
| TweetEval sentiment/emotion/hate/offensive | `download_tweeteval.py` |
| FakeNewsNet | `download_fakenewsnet.py` |
| PHEME five events | `download_pheme.py` |
| Climate Migration | `download_climmig.py`, `download_climmig_label_binary.py` |
| IRA, Mixed component only | `download_ira_io.py` |
| Mixed | `build_mixed_dataset.py` and recorded membership below |

Upstream revisions and availability can change. The training configurations
record the original input and embedding-cache hashes; compare these before
claiming exact reproduction. IDs, row order and texts all matter.

The Mixed fitting corpus contains 111,000 documents.
`reproduction/mixed_membership.csv` records every fitted ID, source ID, source,
type and text hash in fitting order. Run `python reproduce.py run mixed-rebuild
--execute` after preparing the sources. The reconstruction uses ID and text-hash
matching and verifies the complete output hashes. Frozen profiles are recorded
separately; a new sample or encoder fit may change the bit identities.

`reproduction/splits/` preserves label/outcome split IDs. `python reproduce.py
restore-splits` restores them at their expected paths and refuses to overwrite
different files. Encoder fitting is transductive for most experiments; the
Mixed extension encodes documents excluded from fitting with a frozen model.
PHEME engagement counters are extracted from the archived source tweets using
`analysis/prepare_pheme_engagement.py`, not fetched from present-day X.

## Dashboard data and licenses

The public bundles provide fitted codes, document IDs and text-checksum tables;
PHEME includes CC BY 4.0 annotations with attribution. Source texts, AG News labels
and archived platform counters are obtained locally. Follow
[Dashboard data preparation](dashboard_data.md) for download, joining, verification
and launch commands. See [data licenses and attribution](data_licenses.md).

`analysis/export_dashboard_data.py` exports or verifies the public format.
`analysis/prepare_dashboard_data.py` creates the complete local dashboard CSVs
without fitting a model. Local outputs are excluded from version control.
