# External code and data

The MIT license applies to original ISBC code. Downloaded datasets, pretrained
models and third-party software retain their own licenses and access terms.
The public dashboard bundles contain fitted ISBC codes and IDs, with PHEME
annotations under CC BY 4.0. AG News has no identified standard dataset license;
PHEME's CC grant covers annotations, not tweet content. Source texts, AG News
labels and platform counters are therefore obtained locally, not redistributed
in these bundles. See [Data licenses and attribution](docs/data_licenses.md),
the per-bundle `DATA_LICENSE.md` files and [preparation instructions](docs/dashboard_data.md).
No pretrained model weights are included.

The external-encoder adapter loads the authors' implementations, pinned by
`prepare_external_method_packages.sh`:

| Method | Repository | Revision |
| --- | --- | --- |
| HypotheSAEs | https://github.com/rmovva/HypotheSAEs | `706d8979c71befe8e3151f51a8d4dbf87d4d7e41` |
| BinarySentEmb | https://github.com/Linear95/BinarySentEmb | `2aaaae8062ea68fdce700c5765822a09d525eaf1` |

The setup script downloads these checkouts on request into `.external_methods/`,
which is ignored by Git. Their own license files remain with the checkouts.
The adaptations here target sixteen-coordinate document codes and are not
claims to reproduce the original papers' experimental settings.
