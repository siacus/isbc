#!/usr/bin/env python3
"""Render one encoded run's hard-bit activation rates by a categorical label."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoded", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--label-col", required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--title", default="P(bit=1 | category)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    encoded = pd.read_csv(args.encoded, dtype={"id": str})
    labels = pd.read_csv(args.labels, dtype={"id": str})[["id", args.label_col]]
    bits = sorted(
        (column for column in encoded.columns if column.startswith("bit_")),
        key=lambda column: int(column.removeprefix("bit_")),
    )
    if not bits:
        raise ValueError("Encoded input contains no hard bit columns")
    frame = encoded[["id", *bits]].merge(labels, on="id", how="inner", validate="one_to_one")
    if len(frame) != len(encoded):
        raise ValueError("Labels do not align one-to-one with the encoded documents")
    matrix = frame.groupby(args.label_col, sort=True)[bits].mean().T
    matrix.index.name = "bit"
    matrix.to_csv(args.output_csv)

    figure, axis = plt.subplots(figsize=(7.0, 7.5), constrained_layout=True)
    image = axis.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axis.set_title(args.title)
    axis.set_xlabel("AG News category")
    axis.set_ylabel("Qwen run-local bit")
    axis.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=35, ha="right")
    axis.set_yticks(range(matrix.shape[0]), matrix.index)
    # White squares identify the category/categories with the largest activation
    # rate for each bit; they do not claim bit alignment across runs.
    for bit_index, (_, row) in enumerate(matrix.iterrows()):
        maximum = row.max()
        for class_index, value in enumerate(row):
            if value == maximum:
                axis.scatter(class_index, bit_index, marker="s", s=105, facecolors="none", edgecolors="white", linewidths=1)
    figure.colorbar(image, ax=axis, label="Conditional hard-bit activation rate")
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_png, dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
