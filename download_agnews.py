#!/usr/bin/env python3
"""
Download AG News from HuggingFace and prepare it for binary_factors_train.py.

Schema on fancyzhx/ag_news:
    text  : already merged title + description
    label : 0=World, 1=Sports, 2=Business, 3=Sci/Tech

Output files written to --output_dir:
    agnews_all_ready.csv     -> id, text (all selected source partitions; SAE input)
    agnews_all.csv           -> labels plus source_partition provenance

Usage
-----
pip install datasets

# Full dataset (~120k train, 7.6k test):
python download_agnews.py --output_dir data/agnews

# Stratified subsample (e.g. 5000 per class):
python download_agnews.py --output_dir data/agnews --max_per_class 5000

Then train the unsupervised SAE on every available document:
python binary_factors_train.py \
    --input data/agnews/agnews_all_ready.csv \
    --output_dir runs/agnews \
    --k_factors 16 \
    --target_active 6 \
    --epochs 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from datasets import Dataset, load_dataset
except ImportError:
    print(
        "[ERROR] The 'datasets' package is required.\n"
        "Install it with:  pip install datasets"
    )
    sys.exit(1)

import pandas as pd

LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci_Tech"}


def load_agnews(cache_dir: str | None):
    """Load the canonical dataset, falling back to complete local Arrow caches.

    The fallback keeps dataset preparation reproducible when the source is already
    cached but the machine is temporarily offline.
    """
    try:
        return load_dataset("fancyzhx/ag_news", cache_dir=cache_dir)
    except (ConnectionError, FileNotFoundError) as error:
        cache_root = (
            Path(cache_dir)
            if cache_dir is not None
            else Path.home() / ".cache" / "huggingface" / "datasets"
        )
        candidates = sorted(
            cache_root.glob("ag_news/default/*/*/ag_news-train.arrow")
        )
        if not candidates:
            raise error
        parent = candidates[-1].parent
        arrow_paths = {
            split: parent / f"ag_news-{split}.arrow" for split in ("train", "test")
        }
        if not all(path.is_file() for path in arrow_paths.values()):
            raise error
        print(f"Using complete local AG News Arrow cache at {parent}")
        return {split: Dataset.from_file(str(path)) for split, path in arrow_paths.items()}


def process_split(
    hf_split,
    split_name: str,
    max_per_class: int | None,
    id_offset: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        ready_df : id, text only           (for training script)
        full_df  : id, text, label, label_name  (for evaluation)
    """
    df = hf_split.to_pandas()

    # Validate schema
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Unexpected schema. Got columns: {list(df.columns)}. "
            "Expected: text, label"
        )

    # Drop empty texts
    df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)

    # Stratified subsample if requested
    if max_per_class is not None:
        df = (
            df.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), max_per_class), random_state=42))
            .reset_index(drop=True)
        )

    # Assign stable unique integer ids
    df.insert(0, "id", range(id_offset, id_offset + len(df)))
    df["label_name"] = df["label"].map(LABEL_NAMES)

    ready_df = df[["id", "text"]].copy()
    full_df  = df[["id", "text", "label", "label_name"]].copy()

    return ready_df, full_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download AG News and prepare it for binary_factors_train.py"
    )
    parser.add_argument("--output_dir", type=str, default="data/agnews")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "test"],
        default=["train", "test"],
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=None,
        help="Stratified sample: max examples per class. Default: use all.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="HuggingFace cache directory.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading fancyzhx/ag_news from HuggingFace ...")
    dataset = load_agnews(args.cache_dir)

    id_offset = 0
    ready_frames = []
    full_frames = []
    for split_name in args.splits:
        if split_name not in dataset:
            print(f"[WARN] Split '{split_name}' not found, skipping.")
            continue

        n = len(dataset[split_name])
        print(f"\nProcessing '{split_name}' ({n} rows) ...")

        ready_df, full_df = process_split(
            hf_split=dataset[split_name],
            split_name=split_name,
            max_per_class=args.max_per_class,
            id_offset=id_offset,
        )
        id_offset += len(ready_df)

        dist = full_df["label_name"].value_counts().sort_index().to_dict()
        print(f"  Class distribution : {dist}")
        print(f"  Total rows         : {len(ready_df)}")

        ready_frames.append(ready_df)
        full_frames.append(
            full_df.assign(source_partition=f"official_{split_name}")
        )

    if not ready_frames:
        raise SystemExit("No requested dataset splits were available")
    pd.concat(ready_frames, ignore_index=True).to_csv(
        out_dir / "agnews_all_ready.csv", index=False
    )
    pd.concat(full_frames, ignore_index=True).to_csv(
        out_dir / "agnews_all.csv", index=False
    )
    for legacy_name in (
        "agnews_train_ready.csv",
        "agnews_test_ready.csv",
        "agnews_train.csv",
        "agnews_test.csv",
    ):
        (out_dir / legacy_name).unlink(missing_ok=True)

    print("\nDone. Example usage:")
    print(
        f"  python binary_factors_train.py \\\n"
        f"    --input {out_dir}/agnews_all_ready.csv \\\n"
        f"    --output_dir runs/agnews \\\n"
        f"    --k_factors 16 \\\n"
        f"    --target_active 6 \\\n"
        f"    --epochs 3"
    )


if __name__ == "__main__":
    main()
