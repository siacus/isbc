# PHEME: fitted codes for ISBC Explorer

This public bundle contains **5,802 document IDs** and the **16 fitted binary
coordinates**, with target activity 6 and seed 1234. It contains no source texts
or platform counters. PHEME annotations are included under CC BY 4.0.

For text reading and the full paper examples, follow
[download and preparation instructions](../../../docs/dashboard_data.md).
Then open the locally reconstructed data:

```sh
uv run --project dashboard python dashboard/app.py --runs dashboard/local_data/pheme_qwen
```

The public code-only folder can be opened without source downloads for geometry
inspection. No model weights, GPU, re-encoding or retraining are required.

## Files

- `encoded_pheme_qwen.csv`: IDs, hard codes and source annotations.
- `text_hashes.csv`: ID-to-SHA-256 mapping for verification of local texts.
- `metadata.json`: fitted encoder identifiers.
- `manifest.json`: public file checksums and source provenance.
- `DATA_LICENSE.md`: source-specific attribution and license scope.

See [Data licenses](../../../docs/data_licenses.md) before obtaining source content.
