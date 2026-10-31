#!/usr/bin/env python3
"""
Download the official PHEME archive from Figshare and prepare it for
binary_factors_train.py.

Source: Kochkina, Liakata & Zubiaga (2018),
        PHEME dataset for Rumour Detection and Veracity Classification
        https://doi.org/10.6084/m9.figshare.6392078.v1

By default this script keeps the five events used by the established PHEME-5
pipeline in this repository:
    charliehebdo, ferguson, germanwings-crash, ottawashooting, sydneysiege

Labels:
    label       : rumour / non-rumour
    veracity    : true / false / unverified for rumours; not_applicable otherwise
    event       : breaking-news event
    label_fine  : non-rumour / rumour_true / rumour_false / rumour_unverified

Output files in --output_dir:
    pheme_ready.csv     id, text
    pheme_labels.csv    id, label, veracity, event, label_fine
    pheme_stats.txt

The downloaded archive is cached in --output_dir so subsequent runs are local.
Pass --archive_path to use an existing local copy instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

import pandas as pd


EVENTS = [
    "charliehebdo",
    "ferguson",
    "germanwings-crash",
    "ottawashooting",
    "sydneysiege",
]

FIGSHARE_ARTICLE_URL = (
    "https://figshare.com/articles/dataset/"
    "PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078"
)
ARCHIVE_URL = "https://ndownloader.figshare.com/files/11767817"
ARCHIVE_NAME = "PHEME_veracity.tar.bz2"
ARCHIVE_MD5 = "11530d4c0c7127fc78bbc1e46f2498f8"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--event",
        default="all",
        choices=["all", *EVENTS],
        help="One PHEME-5 event, or 'all' for all five (default).",
    )
    parser.add_argument("--min_text_len", type=int, default=5)
    parser.add_argument(
        "--archive_path",
        help="Use an existing local PHEME_veracity.tar.bz2 instead of downloading.",
    )
    return parser.parse_args()


def file_md5(path: Path) -> str:
    digest = hashlib.md5()  # nosec B324 - checksum published by the data provider
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading the official PHEME archive from Figshare:\n  {ARCHIVE_URL}")
    try:
        with urllib.request.urlopen(ARCHIVE_URL) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
        partial.replace(destination)
    finally:
        if partial.exists():
            partial.unlink()


def resolve_archive(args, output_dir: Path) -> Path:
    if args.archive_path:
        archive = Path(args.archive_path).expanduser().resolve()
        if not archive.is_file():
            raise FileNotFoundError(f"Local archive not found: {archive}")
    else:
        archive = output_dir / ARCHIVE_NAME
        if not archive.exists():
            download_archive(archive)
        else:
            print(f"Reusing cached archive: {archive}")

    actual_md5 = file_md5(archive)
    if actual_md5 != ARCHIVE_MD5:
        raise ValueError(
            f"Archive checksum mismatch for {archive}: expected {ARCHIVE_MD5}, "
            f"got {actual_md5}"
        )
    print(f"Archive MD5 verified: {actual_md5}")
    return archive


def event_from_directory(directory: str) -> str:
    suffix = "-all-rnr-threads"
    return directory[: -len(suffix)] if directory.endswith(suffix) else directory


def convert_veracity(annotation: dict) -> str:
    """Apply the conversion supplied alongside the official Figshare archive."""
    misinformation = annotation.get("misinformation")
    truth = annotation.get("true")
    if misinformation is None:
        return "unverified"
    misinformation = int(misinformation)
    if truth is None:
        return "false" if misinformation == 1 else "unverified"
    truth = int(truth)
    if misinformation == 0 and truth == 1:
        return "true"
    if misinformation == 1 and truth == 0:
        return "false"
    if misinformation == 0 and truth == 0:
        return "unverified"
    raise ValueError(
        "Invalid PHEME annotation: misinformation and true cannot both equal 1"
    )


def build_label_fine(label: str, veracity: str) -> str:
    if label == "non-rumour":
        return "non-rumour"
    return f"rumour_{veracity}"


def load_pheme_rows(archive: Path, selected_events: set[str]) -> list[dict]:
    rows: list[dict] = []
    annotations: dict[tuple[str, str, str], dict] = {}
    skipped_invalid_tweets = 0

    # Figshare publishes the file as ``.tar.bz2`` but its bytes are gzip.  Use
    # signature-based detection rather than trusting the historical extension.
    with tarfile.open(archive, mode="r:*") as bundle:
        members = bundle.getmembers()

        for member in members:
            parts = PurePosixPath(member.name).parts
            if len(parts) != 5 or parts[-1] != "annotation.json":
                continue
            event = event_from_directory(parts[1])
            kind, thread_id = parts[2], parts[3]
            if event not in selected_events or kind not in {"rumours", "non-rumours"}:
                continue
            extracted = bundle.extractfile(member)
            if extracted is not None:
                annotations[(event, kind, thread_id)] = json.load(extracted)

        for member in members:
            parts = PurePosixPath(member.name).parts
            if (
                len(parts) != 6
                or parts[4] not in {"source-tweet", "source-tweets"}
                or not parts[-1].endswith(".json")
            ):
                continue
            event = event_from_directory(parts[1])
            kind, thread_id = parts[2], parts[3]
            if event not in selected_events or kind not in {"rumours", "non-rumours"}:
                continue

            extracted = bundle.extractfile(member)
            if extracted is None:
                continue
            try:
                tweet = json.load(extracted)
            except (json.JSONDecodeError, UnicodeDecodeError):
                skipped_invalid_tweets += 1
                continue
            text = str(tweet.get("full_text") or tweet.get("text") or "").strip()
            label = "rumour" if kind == "rumours" else "non-rumour"
            if label == "rumour":
                annotation = annotations.get((event, kind, thread_id), {})
                veracity = convert_veracity(annotation)
            else:
                veracity = "not_applicable"

            rows.append(
                {
                    "id": str(tweet.get("id_str") or tweet.get("id") or thread_id),
                    "text": text,
                    "label": label,
                    "veracity": veracity,
                    "event": event,
                    "label_fine": build_label_fine(label, veracity),
                }
            )

    if skipped_invalid_tweets:
        print(
            f"Skipped {skipped_invalid_tweets} empty or invalid source-tweet JSON file(s)."
        )
    return rows


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = resolve_archive(args, output_dir)

    selected_events = set(EVENTS if args.event == "all" else [args.event])
    rows = load_pheme_rows(archive, selected_events)
    combined = pd.DataFrame(rows)
    if combined.empty:
        raise ValueError(f"No PHEME rows found for event selection: {args.event}")

    combined = combined[combined["text"].str.len() >= args.min_text_len].copy()
    combined = combined.drop_duplicates(subset="id").sort_values(
        ["event", "id"], kind="stable"
    )
    combined = combined.reset_index(drop=True)

    ready_path = output_dir / "pheme_ready.csv"
    labels_path = output_dir / "pheme_labels.csv"
    combined[["id", "text"]].to_csv(ready_path, index=False)
    combined[["id", "label", "veracity", "event", "label_fine"]].to_csv(
        labels_path, index=False
    )

    lines = [
        f"Source: {FIGSHARE_ARTICLE_URL}",
        f"Archive MD5: {ARCHIVE_MD5}",
        f"Events: {', '.join(sorted(selected_events))}",
        f"Total: {len(combined):,}",
        "",
    ]
    for column in ["label", "veracity", "event", "label_fine"]:
        lines += [column, "-" * 40]
        for value, count in combined[column].value_counts().items():
            lines.append(
                f"  {value:<30} {count:>6,}  ({count / len(combined) * 100:.1f}%)"
            )
        lines.append("")
    stats_text = "\n".join(lines)
    print("\n" + stats_text)
    (output_dir / "pheme_stats.txt").write_text(stats_text + "\n")

    print(f"Saved: {ready_path}")
    print(f"       {labels_path}")


if __name__ == "__main__":
    main()
