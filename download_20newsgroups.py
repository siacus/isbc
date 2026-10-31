#!/usr/bin/env python3
"""
Download 20 Newsgroups and prepare it for binary_factors_train.py.

Source: sklearn.datasets.fetch_20newsgroups (no API needed)

Schema:
    text       : post body (headers optionally stripped)
    label      : int 0-19
    label_name : e.g. "comp.graphics", "rec.sport.hockey"

Also provides a coarse 6-class grouping (label_coarse):
    comp    : comp.graphics, comp.os.ms-windows.misc, comp.sys.ibm.pc.hardware,
              comp.sys.mac.hardware, comp.windows.x
    rec     : rec.autos, rec.motorcycles, rec.sport.baseball, rec.sport.hockey
    sci     : sci.crypt, sci.electronics, sci.med, sci.space
    misc    : misc.forsale
    talk    : talk.politics.guns, talk.politics.mideast, talk.politics.misc,
              talk.religion.misc, alt.atheism, soc.religion.christian
    (no 'other' — all 20 groups are assigned)

Output files in --output_dir:
    20ng_all_ready.csv     id, text (all source partitions; SAE input)
    20ng_all.csv           labels and source_partition provenance
    20ng_stats.txt         dataset statistics

Usage:
    pip install scikit-learn

    python download_20newsgroups.py --output_dir data/20ng

    # Strip email headers/footers (recommended for cleaner text):
    python download_20newsgroups.py --output_dir data/20ng --strip_headers

    # Subsample:
    python download_20newsgroups.py --output_dir data/20ng --max_per_class 500

Then train the unsupervised SAE on every available document:
    python binary_factors_train.py \\
        --input data/20ng/20ng_all_ready.csv \\
        --output_dir runs/20ng_sae \\
        --model_name Qwen/Qwen3-Embedding-0.6B \\
        --pooling_mode last_token \\
        --objective sparse_ae \\
        --k_factors 16 --target_active 6 --epochs 10
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

COARSE_MAP = {
    "comp.graphics":              "comp",
    "comp.os.ms-windows.misc":    "comp",
    "comp.sys.ibm.pc.hardware":   "comp",
    "comp.sys.mac.hardware":      "comp",
    "comp.windows.x":             "comp",
    "rec.autos":                  "rec",
    "rec.motorcycles":            "rec",
    "rec.sport.baseball":         "rec",
    "rec.sport.hockey":           "rec",
    "sci.crypt":                  "sci",
    "sci.electronics":            "sci",
    "sci.med":                    "sci",
    "sci.space":                  "sci",
    "misc.forsale":               "misc",
    "talk.politics.guns":         "talk",
    "talk.politics.mideast":      "talk",
    "talk.politics.misc":         "talk",
    "talk.religion.misc":         "talk",
    "alt.atheism":                "talk",
    "soc.religion.christian":     "talk",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--strip_headers", action="store_true", default=False,
                   help="Strip email headers, footers, and quotes (recommended)")
    p.add_argument("--max_per_class", type=int, default=None)
    p.add_argument("--min_text_len",  type=int, default=20)
    return p.parse_args()


def fetch_split(subset: str, strip_headers: bool) -> pd.DataFrame:
    from sklearn.datasets import fetch_20newsgroups
    remove = ("headers", "footers", "quotes") if strip_headers else ()
    bunch = fetch_20newsgroups(subset=subset, remove=remove,
                               shuffle=False, random_state=42)
    df = pd.DataFrame({
        "text":       bunch.data,
        "label":      bunch.target,
        "label_name": [bunch.target_names[i] for i in bunch.target],
    })
    return df


def process(df: pd.DataFrame, max_per_class: int | None,
            min_text_len: int, id_offset: int):
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= min_text_len].reset_index(drop=True)
    if max_per_class is not None:
        df = (df.groupby("label", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), max_per_class),
                                          random_state=42))
                .reset_index(drop=True))
    df.insert(0, "id", range(id_offset, id_offset + len(df)))
    df["label_coarse"] = df["label_name"].map(COARSE_MAP)
    ready = df[["id", "text"]].copy()
    full  = df[["id", "text", "label", "label_name", "label_coarse"]].copy()
    return ready, full


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Fetching 20 Newsgroups (train) ...")
    train_raw = fetch_split("train", args.strip_headers)
    print("Fetching 20 Newsgroups (test) ...")
    test_raw  = fetch_split("test",  args.strip_headers)

    train_ready, train_full = process(train_raw, args.max_per_class,
                                       args.min_text_len, 0)
    test_ready,  test_full  = process(test_raw,  args.max_per_class,
                                       args.min_text_len, len(train_ready))

    for name, ready, full in [("train", train_ready, train_full),
                               ("test",  test_ready,  test_full)]:
        dist = full["label_name"].value_counts().sort_index().to_dict()
        print(f"\n{name}: {len(ready):,} posts")
        coarse = full["label_coarse"].value_counts().to_dict()
        print(f"  Coarse: {coarse}")

    all_ready = pd.concat([train_ready, test_ready], ignore_index=True)
    all_full = pd.concat(
        [
            train_full.assign(source_partition="official_train"),
            test_full.assign(source_partition="official_test"),
        ],
        ignore_index=True,
    )
    all_ready.to_csv(out / "20ng_all_ready.csv", index=False)
    all_full.to_csv(out / "20ng_all.csv", index=False)
    for legacy_name in (
        "20ng_train_ready.csv",
        "20ng_test_ready.csv",
        "20ng_train.csv",
        "20ng_test.csv",
    ):
        (out / legacy_name).unlink(missing_ok=True)

    lines = [f"Total train: {len(train_full):,}",
             f"Total test : {len(test_full):,}",
             f"Strip headers: {args.strip_headers}", ""]
    for cat in sorted(train_full["label_name"].unique()):
        n = (train_full["label_name"] == cat).sum()
        lines.append(f"  {cat:<35} {n:>6,}")
    (out / "20ng_stats.txt").write_text("\n".join(lines) + "\n")

    print(f"\nSaved to: {out}")
    print(f"""
Next steps:
  bash run_analysis.sh \\
    --encoded  runs/20ng_sae/encoded_20ng_all_ready_best.csv \\
    --labels   {out}/20ng_all.csv \\
    --outdir   runs/20ng_sae/analysis_fine \\
    --label_col label_name        # 20 classes

  # or coarse (5 classes):
    --label_col label_coarse
""")


if __name__ == "__main__":
    main()
