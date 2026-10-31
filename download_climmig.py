#!/usr/bin/env python3
"""
Download the "Climate Migration Literature" coded dataset and prepare it for
binary_factors_train.py.

Source: HaodongQi/deep_literature_review_migration on GitHub
        https://github.com/HaodongQi/deep_literature_review_migration
~5.5k papers' title/abstract, each hand-coded on 6 questions (Q1-Q6). Raw
source columns are Q1..Q6 integer codes; key-values.csv maps each
(question, value) pair to a human-readable label. Some questions allow
multiple comma-separated codes per document (e.g. a paper can cite several
climate hazards), so label columns here may contain several values joined
by "; ".

Label columns produced in climmig_labels.csv (renamed from Q1-Q6):
    about_migration   (Q1) : no / yes
    migration_drivers (Q2) : other / climate / environment / economic /
                             conflict / political / social / family
    sentiment         (Q3) : neutral / negative / positive
    climate_hazards   (Q4) : none / algal blooms / drought / heat wave / ...
                             (43 hazard types, multi-label)
    discipline        (Q5) : other / social science / natural science /
                             arts and humanities / multidiscipline and
                             interdiscipline
    quant_methods     (Q6) : no / yes

Output files in --output_dir:
    climmig_ready.csv   id, text
    climmig_labels.csv  id, about_migration, migration_drivers, sentiment,
                        climate_hazards, discipline, quant_methods
    climmig_stats.txt   value counts per label column

Usage:
    python download_climmig.py --output_dir data/climmig

Then train:
    python binary_factors_train.py \\
        --input data/climmig/climmig_ready.csv \\
        --output_dir runs/climmig_sae \\
        --k_factors 16 --target_active 6 --epochs 5

Analysis (run once per label column you care about):
    bash run_analysis.sh \\
        --encoded  runs/climmig_sae/encoded_climmig_ready_best.csv \\
        --labels   data/climmig/climmig_labels.csv \\
        --outdir   runs/climmig_sae/analysis \\
        --label_col about_migration
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ALL_CODED_URL = (
    "https://raw.githubusercontent.com/"
    "HaodongQi/deep_literature_review_migration/main/analysis/allcoded.csv"
)
KEY_VALUES_URL = (
    "https://raw.githubusercontent.com/"
    "HaodongQi/deep_literature_review_migration/main/analysis/key-values.csv"
)

QUESTION_COLS = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
LABEL_MAP = {
    "Q1": "about_migration",
    "Q2": "migration_drivers",
    "Q3": "sentiment",
    "Q4": "climate_hazards",
    "Q5": "discipline",
    "Q6": "quant_methods",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Download the climate migration literature dataset and "
                     "prepare it for binary_factors_train.py"
    )
    p.add_argument("--output_dir", type=str, default="data/climmig")
    p.add_argument("--min_text_len", type=int, default=20,
                   help="Drop documents whose abstract is shorter than this")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Downloading allcoded.csv + key-values.csv from GitHub ...")
    coded = pd.read_csv(ALL_CODED_URL, sep="\t", dtype=str).fillna("")
    coded = coded.drop_duplicates(subset="id")
    key_value = pd.read_csv(KEY_VALUES_URL, dtype=str).fillna("")
    print(f"Raw rows: {len(coded):,}")

    # Blank code -> explicit "none" (only meaningful for Q2-Q5; Q1/Q6 are binary)
    for col in ["Q2", "Q3", "Q4", "Q5"]:
        coded.loc[coded[col] == "", col] = "none"

    # Wide (one column per question) -> long (one row per document x question)
    long = coded.melt(
        id_vars=["id", "title", "abstract", "doi"],
        value_vars=QUESTION_COLS,
        var_name="question",
        value_name="value",
    )

    # Some questions allow multiple comma-separated codes per document
    long["value"] = long["value"].astype(str).str.split(",")
    long = long.explode("value")
    long["value"] = long["value"].str.strip()

    # Join coded values to their human-readable labels
    long = long.merge(key_value, on=["question", "value"], how="left")
    long.loc[long["value"] == "none", "key"] = "none"
    long["question"] = long["question"].map(LABEL_MAP)

    # Long -> wide: one row per document, one column per (renamed) question,
    # multi-label answers joined with "; "
    decoded = (
        long.groupby(["id", "title", "abstract", "question"], dropna=False)["key"]
        .apply(lambda x: "; ".join(x.dropna()))
        .reset_index()
        .pivot(index=["id", "title", "abstract"], columns="question", values="key")
        .reset_index()
    )
    decoded.columns.name = None
    decoded = decoded.rename(columns={"abstract": "text"})

    decoded["text"] = decoded["text"].astype(str).str.strip()
    decoded = decoded[decoded["text"].str.len() >= args.min_text_len].reset_index(drop=True)
    print(f"After min_text_len filter: {len(decoded):,} rows")

    label_cols = list(LABEL_MAP.values())
    ready_path  = out / "climmig_ready.csv"
    labels_path = out / "climmig_labels.csv"

    decoded[["id", "text"]].to_csv(ready_path, index=False)
    decoded[["id"] + label_cols].to_csv(labels_path, index=False)

    lines = [f"Total documents: {len(decoded):,}", ""]
    for col in label_cols:
        lines += [col, "-" * 40]
        for val, cnt in decoded[col].value_counts().head(20).items():
            lines.append(f"  {val:<30} {cnt:>6,}  ({cnt/len(decoded)*100:.1f}%)")
        lines.append("")
    stats_text = "\n".join(lines)
    print("\n" + stats_text)
    (out / "climmig_stats.txt").write_text(stats_text + "\n")

    print(f"\nSaved: {ready_path}")
    print(f"       {labels_path}")
    print(f"       {out / 'climmig_stats.txt'}")
    print(f"""
Next:
    python binary_factors_train.py \\
        --input {ready_path} \\
        --output_dir runs/climmig_sae \\
        --k_factors 16 --target_active 6 --epochs 5

    bash run_analysis.sh \\
        --encoded  runs/climmig_sae/encoded_climmig_ready_best.csv \\
        --labels   {labels_path} \\
        --outdir   runs/climmig_sae/analysis \\
        --label_col about_migration
""")


if __name__ == "__main__":
    main()
