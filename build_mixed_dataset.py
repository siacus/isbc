#!/usr/bin/env python3
"""Build a reproducible, model-ready mixture of the maintained text datasets.

The output keeps dataset-specific labels in separate columns.  Sampling is without
replacement and uses an independent deterministic RNG stream per source, so changing
one source's requested count does not change which rows are sampled from the others.
After concatenation, a separate seeded permutation assigns unique IDs from 1..N; rows
are then sorted by that ID, which reproducibly interleaves all sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


UNKNOWN = "unknown"


@dataclass(frozen=True)
class SourceSpec:
    key: str
    ready_path: str
    labels_path: str
    source: str
    data_type: str
    default_count: int
    labels: tuple[tuple[str, str], ...]


SOURCE_SPECS = (
    SourceSpec(
        "tweeteval_sentiment",
        "tweeteval/sentiment/sentiment_all_ready.csv",
        "tweeteval/sentiment/sentiment_all.csv",
        "tweeteval",
        "tweet",
        4_000,
        (("label_name", "sentiment"),),
    ),
    SourceSpec(
        "tweeteval_emotion",
        "tweeteval/emotion/emotion_all_ready.csv",
        "tweeteval/emotion/emotion_all.csv",
        "tweeteval",
        "tweet",
        4_000,
        (("label_name", "emotion"),),
    ),
    SourceSpec(
        "tweeteval_hate",
        "tweeteval/hate/hate_all_ready.csv",
        "tweeteval/hate/hate_all.csv",
        "tweeteval",
        "tweet",
        4_000,
        (("label_name", "hate"),),
    ),
    SourceSpec(
        "tweeteval_offensive",
        "tweeteval/offensive/offensive_all_ready.csv",
        "tweeteval/offensive/offensive_all.csv",
        "tweeteval",
        "tweet",
        4_000,
        (("label_name", "offensive"),),
    ),
    SourceSpec(
        "pheme",
        "pheme/pheme_ready.csv",
        "pheme/pheme_labels.csv",
        "pheme",
        "tweet",
        5_000,
        (
            ("label", "pheme_label"),
            ("veracity", "pheme_veracity"),
            ("event", "pheme_event"),
        ),
    ),
    SourceSpec(
        "climmig",
        "climmig/climmig_ready.csv",
        "climmig/climmig_labels.csv",
        "climmig",
        "abstract",
        5_000,
        (
            ("about_migration", "climmig_migration"),
            ("discipline", "climmig_discipline"),
            ("sentiment", "climmig_sentiment"),
        ),
    ),
    SourceSpec(
        "agnews",
        "agnews/agnews_all_ready.csv",
        "agnews/agnews_all.csv",
        "agnews",
        "news",
        10_000,
        (("label_name", "agnews_label"),),
    ),
    SourceSpec(
        "20ng",
        "20ng/20ng_all_ready.csv",
        "20ng/20ng_all.csv",
        "20ng",
        "post",
        15_000,
        (
            ("label_name", "newsgroups_label"),
            ("label_coarse", "newsgroups_label_coarse"),
        ),
    ),
    SourceSpec(
        "fakenewsnet",
        "fakenews/fakenews_ready.csv",
        "fakenews/fakenews_labels.csv",
        "fakenewsnet",
        "news",
        15_000,
        (
            ("label", "fakenewsnet_label"),
            ("source", "fakenewsnet_source"),
        ),
    ),
    SourceSpec(
        "ira_io",
        "ira/ira_ready.csv",
        "ira/ira_labels.csv",
        "ira_io",
        "tweet",
        20_000,
        (
            ("account_category", "ira_account_category"),
            ("account_type", "ira_account_type"),
            ("language", "ira_language"),
        ),
    ),
    SourceSpec(
        "airline",
        "airline/airline_ready.csv",
        "airline/airline_labels.csv",
        "airline",
        "tweet",
        10_000,
        (
            ("airline_sentiment", "airline_sentiment"),
            ("airline", "airline_name"),
        ),
    ),
    SourceSpec(
        "goemotions",
        "goemotions/goemotions_ready.csv",
        "goemotions/goemotions_labels.csv",
        "goemotions",
        "comment",
        15_000,
        (
            ("label_primary", "goemotions_label_primary"),
            ("label_grouped", "goemotions_label_grouped"),
            ("labels_all", "goemotions_labels_all"),
        ),
    ),
)


def parse_count(value: str) -> tuple[str, int]:
    try:
        key, raw_count = value.split("=", 1)
        count = int(raw_count)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("counts must use SOURCE=N") from exc
    valid = {spec.key for spec in SOURCE_SPECS}
    if key not in valid:
        raise argparse.ArgumentTypeError(
            f"unknown source {key!r}; choose one of: {', '.join(sorted(valid))}"
        )
    if count < 0:
        raise argparse.ArgumentTypeError("source counts cannot be negative")
    return key, count


def parse_args() -> argparse.Namespace:
    defaults = ", ".join(f"{spec.key}={spec.default_count}" for spec in SOURCE_SPECS)
    parser = argparse.ArgumentParser(
        description="Create the seeded mixed text corpus and its wide label table."
    )
    parser.add_argument("--data_dir", default="data", help="Root of prepared source data.")
    parser.add_argument("--output_dir", default="data/mixed")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--count",
        action="append",
        default=[],
        type=parse_count,
        metavar="SOURCE=N",
        help=(
            "Override one source count; repeat as needed. Unspecified sources retain "
            f"their defaults: {defaults}"
        ),
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derived_seed(seed: int, stream: str) -> int:
    payload = f"latent-mixed-v1:{seed}:{stream}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def read_source(spec: SourceSpec, data_dir: Path) -> tuple[pd.DataFrame, dict]:
    ready_path = data_dir / spec.ready_path
    labels_path = data_dir / spec.labels_path
    for path in (ready_path, labels_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing prepared source file: {path}")

    ready = pd.read_csv(ready_path, dtype=str, keep_default_na=False)
    labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
    if not {"id", "text"}.issubset(ready.columns):
        raise ValueError(f"{ready_path} must contain id,text")
    required_labels = {"id", *(original for original, _ in spec.labels)}
    missing_labels = required_labels.difference(labels.columns)
    if missing_labels:
        raise ValueError(f"{labels_path} is missing columns: {sorted(missing_labels)}")
    if ready["id"].duplicated().any():
        raise ValueError(f"Duplicate IDs in {ready_path}")
    if labels["id"].duplicated().any():
        raise ValueError(f"Duplicate IDs in {labels_path}")

    selected_labels = labels[["id", *(original for original, _ in spec.labels)]].copy()
    frame = ready[["id", "text"]].merge(
        selected_labels, on="id", how="left", validate="one_to_one", indicator=True
    )
    missing_ids = int((frame["_merge"] != "both").sum())
    if missing_ids:
        raise ValueError(f"{spec.key}: {missing_ids} text rows have no label row")
    frame = frame.drop(columns="_merge").rename(columns={"id": "source_id"})
    frame = frame.rename(columns=dict(spec.labels))
    provenance = {
        "ready_path": spec.ready_path,
        "ready_sha256": file_sha256(ready_path),
        "labels_path": spec.labels_path,
        "labels_sha256": file_sha256(labels_path),
        "available_rows": len(frame),
    }
    return frame, provenance


def atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".partial")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)


def atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".partial")
    frame.to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
    temporary.replace(destination)


def atomic_text(text: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(destination)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = {spec.key: spec.default_count for spec in SOURCE_SPECS}
    for key, count in args.count:
        counts[key] = count

    label_columns = [output for spec in SOURCE_SPECS for _, output in spec.labels]
    if len(label_columns) != len(set(label_columns)):
        raise RuntimeError("Output label column names must be unique")

    pieces: list[pd.DataFrame] = []
    source_metadata: dict[str, dict] = {}
    for spec in SOURCE_SPECS:
        frame, provenance = read_source(spec, data_dir)
        requested = counts[spec.key]
        if requested > len(frame):
            raise ValueError(
                f"{spec.key}: requested {requested:,}, only {len(frame):,} available"
            )
        sample_seed = derived_seed(args.seed, f"sample:{spec.key}")
        rng = np.random.default_rng(sample_seed)
        selected_positions = np.sort(rng.choice(len(frame), size=requested, replace=False))
        sampled = frame.iloc[selected_positions].copy().reset_index(drop=True)
        sampled.insert(2, "source", spec.source)
        sampled.insert(3, "type", spec.data_type)
        for label_column in label_columns:
            if label_column not in sampled.columns:
                sampled[label_column] = UNKNOWN
        for _, output_column in spec.labels:
            sampled[output_column] = (
                sampled[output_column].astype(str).str.strip().replace("", UNKNOWN)
            )
        sampled = sampled[["source_id", "text", "source", "type", *label_columns]]
        pieces.append(sampled)
        source_metadata[spec.key] = {
            **provenance,
            "source": spec.source,
            "type": spec.data_type,
            "requested_rows": requested,
            "sampled_rows": len(sampled),
            "sample_seed": sample_seed,
            "label_columns": [output for _, output in spec.labels],
        }

    mixed = pd.concat(pieces, ignore_index=True)
    total_rows = len(mixed)
    id_seed = derived_seed(args.seed, "mixed_id")
    id_rng = np.random.default_rng(id_seed)
    mixed.insert(0, "id", id_rng.permutation(np.arange(1, total_rows + 1)))
    mixed = mixed.sort_values("id", kind="stable").reset_index(drop=True)

    if mixed["id"].tolist() != list(range(1, total_rows + 1)):
        raise RuntimeError("Mixed IDs are not the complete unique range 1..N")
    if mixed["source"].eq("").any() or mixed["type"].eq("").any():
        raise RuntimeError("Every row must have source and type")
    if (mixed[label_columns] == UNKNOWN).all(axis=1).any():
        raise RuntimeError("Every row must instantiate at least one source-specific label")

    ready = mixed[["id", "text"]]
    labels = mixed.drop(columns="text")
    all_path = output_dir / "mixed_all.csv"
    ready_path = output_dir / "mixed_ready.csv"
    labels_path = output_dir / "mixed_labels.csv"
    parquet_path = output_dir / "mixed.parquet"
    metadata_path = output_dir / "mixed_metadata.json"
    stats_path = output_dir / "mixed_stats.txt"
    atomic_csv(mixed, all_path)
    atomic_csv(ready, ready_path)
    atomic_csv(labels, labels_path)
    atomic_parquet(mixed, parquet_path)

    source_counts = mixed["source"].value_counts().sort_index().to_dict()
    type_counts = mixed["type"].value_counts().sort_index().to_dict()
    populated_labels = {
        column: int(mixed[column].ne(UNKNOWN).sum()) for column in label_columns
    }
    metadata = {
        "schema_version": 1,
        "description": "Seeded mixed corpus with source-specific wide labels",
        "seed": args.seed,
        "sampling": "without replacement; independent SHA-256-derived NumPy RNG stream per source",
        "id_assignment": "seeded permutation of 1..N followed by ascending ID sort",
        "unknown_value": UNKNOWN,
        "rows": total_rows,
        "id_min": 1,
        "id_max": total_rows,
        "id_unique": bool(mixed["id"].is_unique),
        "id_seed": id_seed,
        "source_counts": source_counts,
        "type_counts": type_counts,
        "label_columns": label_columns,
        "populated_label_counts": populated_labels,
        "sources": source_metadata,
        "outputs": {
            "all": {"path": all_path.name, "sha256": file_sha256(all_path)},
            "ready": {"path": ready_path.name, "sha256": file_sha256(ready_path)},
            "labels": {"path": labels_path.name, "sha256": file_sha256(labels_path)},
            "parquet": {
                "path": parquet_path.name,
                "sha256": file_sha256(parquet_path),
                "compression": "zstd",
            },
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    atomic_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", metadata_path)

    lines = [
        "Mixed dataset",
        f"seed: {args.seed}",
        f"rows: {total_rows}",
        "",
        "Sampling sources:",
    ]
    lines.extend(
        f"  {key}: {details['sampled_rows']} / {details['available_rows']}"
        for key, details in source_metadata.items()
    )
    lines.extend(["", "Output source values:"])
    lines.extend(f"  {key}: {value}" for key, value in source_counts.items())
    lines.extend(["", "Types:"])
    lines.extend(f"  {key}: {value}" for key, value in type_counts.items())
    lines.extend(["", "Populated label cells:"])
    lines.extend(f"  {key}: {value}" for key, value in populated_labels.items())
    atomic_text("\n".join(lines) + "\n", stats_path)

    print(f"Wrote {total_rows:,} rows to {output_dir}")
    print(f"Model input: {ready_path}")
    print(f"Labels:     {labels_path}")
    print(f"Combined:   {all_path}")
    print(f"Parquet:    {parquet_path}")


if __name__ == "__main__":
    main()
