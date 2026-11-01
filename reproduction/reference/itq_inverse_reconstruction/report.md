# ITQ reconstruction on AG News

All 127,600 Qwen embeddings; the existing 16-bit PCA/ITQ fit is held fixed.

| Readout | Inverse NMSE | Inverse R2 | Affine OLS R2 |
|---|---:|---:|---:|
| ITQ | 0.966097 | 0.033903 | 0.171789 |
| ITQ_top6 | 0.969714 | 0.030286 | 0.151918 |

The fixed inverse uses signed codes (2B-1), reverses the learned rotation,
projects back through the PCA directions and restores the PCA mean.
Top-6 changes only the hard code. No decoder coefficients are optimized
for either inverse-transform score. OLS is the separate common affine reference.
The calculations reproduce all saved rotated scores and check the decomposition
of inverse-transform error into PCA residual error and quantization error.
