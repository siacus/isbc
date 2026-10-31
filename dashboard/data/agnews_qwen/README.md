# AG News: fitted codes for ISBC Explorer

This public bundle contains **127,600 document IDs** and the **16 fitted binary
coordinates**, with target activity 6 and seed 1234. It contains no source texts
or platform counters. AG News category labels are joined locally with the text.

For text reading and the full paper examples, follow
[download and preparation instructions](../../../docs/dashboard_data.md).
Then open the locally reconstructed data:

```sh
uv run --project dashboard python dashboard/app.py --runs dashboard/local_data/agnews_qwen
```

The public code-only folder can be opened without source downloads for geometry
inspection. No model weights, GPU, re-encoding or retraining are required.

## Files

- `encoded_agnews_qwen.csv`: IDs, hard codes.
- `text_hashes.csv`: ID-to-SHA-256 mapping for verification of local texts.
- `metadata.json`: fitted encoder identifiers.
- `manifest.json`: public file checksums and source provenance.
- `DATA_LICENSE.md`: source-specific attribution and license scope.

See [Data licenses](../../../docs/data_licenses.md) before obtaining source content.

## Explore the paper example

Select bits **00, 05, 06, 10, 12**, leaving other bits unrestricted:

- 2,432 matching documents: Sci_Tech 2,275; Business 147; World 10; Sports 0.
- 116 distinct complete profiles across all matching documents.
- With sample size 1,000 and seed 1234: 75 plotted profiles and 59 Business documents
  in the sample. Dominant point colour is not a document count.
- Requiring every unselected bit off instead gives 35 documents in one exact profile,
  including four Business documents.

These counts refer to this specific fitted run. Bit IDs are not interchangeable
between independently fitted encoders. Nothing in this bundle re-estimates a model.
