#!/usr/bin/env python3
"""Plot the prevalence of the 20 most common hard profiles in AG News 16/6.

The figure is deliberately rank-based: binary identifiers are arbitrary within a
trained SAE, while the distribution of documents across complete profiles is
not.  Each panel therefore shows ranks 1--20 on the x-axis and the percentage
of the complete AG News corpus assigned to that ranked profile on the y-axis.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "runs/agnews_embedding_comparison/all_fixed_balance_label_name"
MODEL_RUNS = {
    "Qwen": "qwen",
    "MiniLM": "minilm",
    "NVIDIA": "nvidia",
    "Harrier": "harrier",
    "EmbeddingGemma 2": "embeddinggemma2",
}


def bit_columns(columns: pd.Index) -> list[str]:
    selected = [column for column in columns if column.startswith("bit_")]
    return sorted(selected, key=lambda column: int(column.removeprefix("bit_")))


def profile_table(model: str, run_dir: Path) -> pd.DataFrame:
    candidates = sorted(run_dir.glob("encoded_agnews_all_ready_*.csv"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one encoded AG News CSV under {run_dir}")
    frame = pd.read_csv(candidates[0])
    bits = bit_columns(frame.columns)
    if not bits:
        raise ValueError(f"No bit columns in {candidates[0]}")
    # The prefix prevents spreadsheet software from stripping meaningful leading
    # zeroes when a profile is exported to CSV.
    profiles = "b" + frame[bits].astype(int).astype(str).agg("".join, axis=1)
    counts = profiles.value_counts().rename_axis("profile").reset_index(name="documents")
    counts["rank"] = range(1, len(counts) + 1)
    counts["percent"] = 100.0 * counts["documents"] / len(frame)
    counts["model"] = model
    counts["corpus_documents"] = len(frame)
    return counts[["model", "rank", "profile", "documents", "percent", "corpus_documents"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--run-suffix",
        default="fixed_balance",
        help="Suffix after agnews_<backbone>_all_, e.g. k8_target4_fixed_balance.",
    )
    parser.add_argument("--title-prefix", default="AG News 16/6")
    parser.add_argument(
        "--expected-documents",
        type=int,
        default=None,
        help="Include only encoded runs with this corpus size (useful during a rerun).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = []
    runs = {
        model: ROOT / f"runs/agnews_{tag}_all_{args.run_suffix}"
        for model, tag in MODEL_RUNS.items()
    }
    for model, run_dir in runs.items():
        table = profile_table(model, run_dir)
        documents = int(table["corpus_documents"].iat[0])
        if args.expected_documents is not None and documents != args.expected_documents:
            print(f"Skipping {model}: {documents:,} rows, expected {args.expected_documents:,}")
            continue
        tables.append(table)
    if not tables:
        raise RuntimeError("No encoded runs matched the requested corpus size")
    all_profiles = pd.concat(tables, ignore_index=True)
    top20 = all_profiles.loc[all_profiles["rank"] <= 20].copy()
    args.output.mkdir(parents=True, exist_ok=True)
    top20.to_csv(args.output / "profile_prevalence_top20.csv", index=False)
    summary = (
        all_profiles.groupby("model", sort=False)
        .agg(
            observed_profiles=("rank", "max"),
            corpus_documents=("corpus_documents", "first"),
        )
        .join(
            top20.groupby("model", sort=False)["documents"]
            .sum()
            .rename("top20_documents")
        )
        .reset_index()
    )
    summary["top20_percent"] = 100.0 * summary["top20_documents"] / summary["corpus_documents"]
    summary.to_csv(args.output / "profile_prevalence_summary.csv", index=False)

    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]
    for (model, _), color in zip(runs.items(), colors):
        subset = top20.loc[top20["model"].eq(model)].sort_values("rank")
        if subset.empty:
            continue
        fig, axis = plt.subplots(figsize=(9.5, 4.8))
        axis.bar(subset["rank"], subset["percent"], color=color, width=0.82)
        axis.set_title(f"{args.title_prefix}: 20 most common hard profiles ({model})")
        axis.set_xticks([1, 5, 10, 15, 20])
        axis.set_xlabel("Profile rank")
        axis.set_ylabel("Documents (% of sample)")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        fig.tight_layout()
        slug = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
        fig.savefig(args.output / f"profile_prevalence_top20_{slug}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
