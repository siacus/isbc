# Data licenses and attribution

Checked against the identified source releases on 31 October 2026.
The repository's MIT license covers original ISBC software, not third-party
texts, annotations, social-media metadata or pretrained model weights.

## Public files and local source content

The public dashboard files contain document identifiers and the fitted ISBC
binary codes. PHEME additionally includes the source annotations described below.
Separate SHA-256 tables allow locally obtained texts to be checked without
publishing those texts. AG News category labels, all source texts, and PHEME
engagement/account fields are obtained and joined locally. The completed CSVs
are written to `dashboard/local_data/`, which is excluded from Git.

The ISBC authors grant permission under the repository's MIT license for their
original numerical code outputs (the `bit_00` through `bit_15` columns), to the
extent they hold rights in those outputs. This does not grant rights in the
underlying source content, identifiers or third-party annotations.

## AG News

Source: [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news).
The classification benchmark was constructed by Xiang Zhang from Antonio Gulli's
AG news corpus and described by Xiang Zhang, Junbo Zhao and Yann LeCun,
*Character-level Convolutional Networks for Text Classification* (2015).

The Hugging Face dataset card marks the license as **unknown**. Its description
refers to research and other non-commercial uses; it does not identify a standard
open-data license or establish permission for us to redistribute the collected
news text. We therefore do not assign MIT, CC BY or another license to that text
and do not include it or its category table in the public dashboard bundle.
See the [upstream card](https://huggingface.co/datasets/fancyzhx/ag_news/blob/main/README.md)
and the linked original corpus page for source terms.

The public bundle contains 127,600 fitted codes, document IDs and text checksums.
The local preparation joins the source text and four category labels by ID and
rejects texts whose hashes differ from those used in the fitted experiment.
Downloading the source is a separate action by the user, subject to the source's
terms; availability for download is not a general redistribution license.

## PHEME

Elena Kochkina, Maria Liakata and Arkaitz Zubiaga (2018).
*PHEME dataset for Rumour Detection and Veracity Classification*, version 1.
[DOI: 10.6084/m9.figshare.6392078.v1](https://doi.org/10.6084/m9.figshare.6392078.v1).
[Version-specific source metadata](https://api.figshare.com/v2/articles/6392078/versions/1).

The source metadata lists **CC BY 4.0**, and its description explicitly limits
that grant to the **annotations**, reserving tweet-content rights separately.
We apply this attribution to the distributed `label`, `veracity`, `event` and
`label_fine` annotations, not to tweet texts or archived platform counters.

License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
([legal code](https://creativecommons.org/licenses/by/4.0/legalcode.en)).
Reuse of the licensed annotations requires attribution, a license link and an
indication of modifications. This release selects the five events used in the
paper (5,802 source tweets), converts annotations to a flat CSV and adds the
combined `label_fine` field and independently learned ISBC codes. It does not
imply endorsement by the PHEME creators. Event names identify the source folders.

Tweet texts, creation timestamps, engagement counters and account statistics are
not redistributed in the public dashboard bundle. The local preparation reads
them from the versioned PHEME archive. The annotations' CC BY license must not be
interpreted as permission covering those platform fields. Applicable source,
platform and privacy conditions remain separate; see the
[X developer policy](https://docs.x.com/developer-terms/policy).

## Other corpora and models

Download scripts for other corpora identify their sources; their presence does
not relicense the downloaded material. Those raw collections and pretrained
model weights are not bundled here. Consult their source terms before use or
redistribution. The instructions for preparing the two dashboard examples are
in [Dashboard data preparation](dashboard_data.md).
