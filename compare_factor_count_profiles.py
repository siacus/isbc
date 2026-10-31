#!/usr/bin/env python3
"""Compare profile matching across two factor-count configurations, backbone by backbone."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def weighted(rows: list[dict], key: str, weight: str = "start_share") -> float | None:
    den = sum(float(row[weight]) for row in rows)
    if not den:
        return None
    return sum(float(row[key]) * float(row[weight]) for row in rows) / den


def fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def load_codes(path: str) -> tuple[list[str], list[str]]:
    with Path(path).open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        id_col = header.index("id")
        bit_cols = [i for i, name in enumerate(header) if name.startswith("bit_")]
        ids, codes = [], []
        for row in reader:
            ids.append(row[id_col])
            codes.append("".join(row[i] for i in bit_cols))
    return ids, codes


def entropy(counts: Counter, n: int) -> float:
    return -sum((value / n) * math.log2(value / n) for value in counts.values())


def partition_metrics(left_codes: list[str], right_codes: list[str]) -> dict:
    n = len(left_codes)
    left_counts = Counter(left_codes)
    right_counts = Counter(right_codes)
    joint = Counter(zip(left_codes, right_codes))
    right_by_left = defaultdict(Counter)
    left_by_right = defaultdict(Counter)
    for (left_code, right_code), count in joint.items():
        right_by_left[left_code][right_code] = count
        left_by_right[right_code][left_code] = count
    h_left = entropy(left_counts, n)
    h_right = entropy(right_counts, n)
    h_joint = entropy(joint, n)
    mutual_information = h_left + h_right - h_joint
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
        left_codes, right_codes
    )
    return {
        "documents": n,
        "left_unique_profiles": len(left_counts),
        "right_unique_profiles": len(right_counts),
        "left_entropy_bits": h_left,
        "right_entropy_bits": h_right,
        "mutual_information_bits": mutual_information,
        "right_given_left_entropy_bits": h_joint - h_left,
        "left_given_right_entropy_bits": h_joint - h_right,
        "right_from_left_modal_accuracy": sum(max(x.values()) for x in right_by_left.values()) / n,
        "left_from_right_modal_accuracy": sum(max(x.values()) for x in left_by_right.values()) / n,
        "normalized_mutual_information": normalized_mutual_info_score(left_codes, right_codes),
        "adjusted_mutual_information": adjusted_mutual_info_score(left_codes, right_codes),
        "adjusted_rand_index": adjusted_rand_score(left_codes, right_codes),
        "right_homogeneity_wrt_left": homogeneity,
        "right_completeness_wrt_left": completeness,
        "v_measure": v_measure,
    }


ap = argparse.ArgumentParser()
ap.add_argument("--left", required=True, type=Path, help="K=16 collection JSON")
ap.add_argument("--right", required=True, type=Path, help="K=32 collection JSON")
ap.add_argument("--left-name", default="k16_target6")
ap.add_argument("--right-name", default="k32_target6")
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--min-share", type=float, default=0.01)
ap.add_argument("--null-draws", type=int, default=1000)
ap.add_argument("--graphs", action="store_true")
args = ap.parse_args()

left = load_json(args.left)
right = load_json(args.right)
models = left["run_order"]
if models != right["run_order"]:
    raise SystemExit("run orders differ")
if left["label_column"] != right["label_column"]:
    raise SystemExit("label columns differ")
if left["labels"]["sha256"] != right["labels"]["sha256"]:
    raise SystemExit("label files differ")
if left["n_aligned_rows"] != right["n_aligned_rows"]:
    raise SystemExit("aligned row counts differ")
left_split = left.get("predictive_split", {})
right_split = right.get("predictive_split", {})
left_split_sha = left_split.get("assignment", {}).get("sha256")
right_split_sha = right_split.get("assignment", {}).get("sha256")
if left_split_sha != right_split_sha:
    raise SystemExit("predictive split hashes differ")

args.out.mkdir(parents=True, exist_ok=True)
script = Path(__file__).resolve().parent / "profile_matching" / "profile_matching.py"
summary_rows = []
partition_rows = []

for model in models:
    model_out = args.out / model
    model_out.mkdir(parents=True, exist_ok=True)
    comparison = {
        "schema_version": 1,
        "comparison": f"{model}: {args.left_name} vs {args.right_name}",
        "label_column": left["label_column"],
        "n_aligned_rows": left["n_aligned_rows"],
        "labels": left["labels"],
        "predictive_split": left_split,
        "run_order": [args.left_name, args.right_name],
        "runs": {
            args.left_name: {"encoded": left["runs"][model]["encoded"]},
            args.right_name: {"encoded": right["runs"][model]["encoded"]},
        },
        "source_collections": {
            args.left_name: str(args.left.resolve()),
            args.right_name: str(args.right.resolve()),
        },
        "backbone": model,
    }
    (model_out / "factor_collection_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n"
    )
    command = [
        sys.executable,
        str(script),
        "--run-dir", str(model_out),
        "--out", str(model_out),
        "--reference", args.left_name,
        "--all-references",
        "--min-share", str(args.min_share),
        "--min-top-share", "0",
        "--null-draws", str(args.null_draws),
    ]
    if args.graphs:
        command.append("--graphs")
    subprocess.run(command, check=True)

    left_ids, left_codes = load_codes(left["runs"][model]["encoded"]["path"])
    right_ids, right_codes = load_codes(right["runs"][model]["encoded"]["path"])
    if left_ids != right_ids:
        raise SystemExit(f"document ids differ for {model}")
    partition_rows.append({"backbone": model, **partition_metrics(left_codes, right_codes)})

    with (model_out / "matches.csv").open(newline="") as handle:
        matches = list(csv.DictReader(handle))
    with (model_out / "invariance.csv").open(newline="") as handle:
        invariance = list(csv.DictReader(handle))

    for reference, other in (
        (args.left_name, args.right_name),
        (args.right_name, args.left_name),
    ):
        selected = [row for row in matches if row["reference"] == reference]
        inv = [
            row for row in invariance
            if row["taxonomy"] == left["label_column"]
            and any(row["string"] == match.get("string") for match in selected)
        ]
        # Older matches.csv intentionally does not repeat the internal string. Its order is
        # identical to invariance.csv within each reference; select the corresponding block.
        if selected and not inv:
            if reference == args.left_name:
                inv = invariance[: len(selected)]
            else:
                inv = invariance[-len(selected):]
        total_start = sum(int(row["start_docs"]) for row in selected)
        total_final = sum(int(row["final_docs"]) for row in selected)
        lifts = [
            float(row["step_lifts"].split(";")[0])
            for row in selected if row["step_lifts"]
        ]
        summary_rows.append({
            "backbone": model,
            "reference": reference,
            "matched_configuration": other,
            "reference_profiles": len(selected),
            "reference_coverage": total_start / left["n_aligned_rows"],
            "exact_modal_retention": total_final / total_start if total_start else None,
            "mean_lift": sum(lifts) / len(lifts) if lifts else None,
            "weighted_coherence": weighted(inv, "coherence"),
            "weighted_information_bits": weighted(inv, "information"),
            "chain_significant_profiles": sum(float(row["chain_p"]) < 0.05 for row in inv),
            "invariance_profiles": len(inv),
        })

fields = list(summary_rows[0])
with (args.out / "profile_matching_summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(summary_rows)

partition_fields = list(partition_rows[0])
with (args.out / "profile_partition_metrics.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=partition_fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(partition_rows)

forward = [row for row in summary_rows if row["reference"] == args.left_name]
lines = [
    "# AG News profile matching: 16/6 versus 32/6\n",
    "## TL;DR\n",
    "This is a within-backbone comparison: every row compares the codes assigned to the same "
    f"{left['n_aligned_rows']:,} AG News documents by `{args.left_name}` and `{args.right_name}`. "
    "A frequent reference profile is matched to the most frequent whole profile in the other "
    "configuration among exactly those documents. Bit numbers are local to each run.\n",
    "The global table is the principal comparison. It treats every distinct full bit pattern as "
    "a profile category, so it is invariant to the profile names and includes all documents.\n",
    "| Backbone | Profiles K16 / K32 | NMI | AMI | ARI | K16 recoverable from K32 | K32 recoverable from K16 | H(K16\u007cK32) | H(K32\u007cK16) |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in partition_rows:
    lines.append(
        f"| {row['backbone']} | {row['left_unique_profiles']} / {row['right_unique_profiles']} | "
        f"{row['normalized_mutual_information']:.3f} | "
        f"{row['adjusted_mutual_information']:.3f} | {row['adjusted_rand_index']:.3f} | "
        f"{row['left_from_right_modal_accuracy']:.1%} | "
        f"{row['right_from_left_modal_accuracy']:.1%} | "
        f"{row['left_given_right_entropy_bits']:.2f} | "
        f"{row['right_given_left_entropy_bits']:.2f} |"
    )
partition_by_model = {row["backbone"]: row for row in partition_rows}
best_adjusted = max(partition_rows, key=lambda row: row["adjusted_mutual_information"])
qwen_metrics = partition_by_model.get("qwen")
minilm_metrics = partition_by_model.get("minilm")
harrier_metrics = partition_by_model.get("harrier")
gemma_metrics = partition_by_model.get("gemma")
lines += ["\n### Interpretation\n"]
lines.append(
    f"**{best_adjusted['backbone'].upper()} preserves the profile structure most faithfully.** "
    f"It has the highest adjusted mutual information ({best_adjusted['adjusted_mutual_information']:.3f}) "
    f"and adjusted Rand index ({best_adjusted['adjusted_rand_index']:.3f}), with comparatively "
    f"symmetric modal recovery ({best_adjusted['left_from_right_modal_accuracy']:.1%} from K32 to K16; "
    f"{best_adjusted['right_from_left_modal_accuracy']:.1%} in the reverse direction).\n"
)
if qwen_metrics:
    lines.append(
        f"**Qwen behaves mainly as a refinement when capacity increases.** A K32 profile recovers its "
        f"modal K16 parent for {qwen_metrics['left_from_right_modal_accuracy']:.1%} of documents, but a "
        f"K16 profile predicts one exact K32 child for only "
        f"{qwen_metrics['right_from_left_modal_accuracy']:.1%}. The K32 representation therefore splits "
        "many K16 profiles into several finer patterns rather than reproducing them one-to-one.\n"
    )
if harrier_metrics and gemma_metrics:
    lines.append(
        f"**Harrier and Gemma retain partial but not one-to-one structure.** Their AMI values are "
        f"{harrier_metrics['adjusted_mutual_information']:.3f} and "
        f"{gemma_metrics['adjusted_mutual_information']:.3f}; the lower reverse modal accuracies "
        f"({harrier_metrics['right_from_left_modal_accuracy']:.1%} and "
        f"{gemma_metrics['right_from_left_modal_accuracy']:.1%}) show that K32 again redistributes "
        "documents over a larger set of profiles.\n"
    )
if minilm_metrics:
    lines.append(
        f"**MiniLM's raw NMI is misleadingly high.** It produces "
        f"{minilm_metrics['left_unique_profiles']:,} K16 and "
        f"{minilm_metrics['right_unique_profiles']:,} K32 profiles for only "
        f"{minilm_metrics['documents']:,} documents. Once agreement expected from this extreme number "
        f"of rare profiles is removed, AMI falls to "
        f"{minilm_metrics['adjusted_mutual_information']:.3f} and ARI to "
        f"{minilm_metrics['adjusted_rand_index']:.3f}; it should not be interpreted as the strongest "
        "cross-capacity match.\n"
    )
lines += [
    "\n`K16 recoverable from K32` assigns each K32 profile to its most common K16 profile and "
    "reports document accuracy; the reverse column swaps the direction. If K32 were a clean "
    "refinement of K16, the first value would be high while the reverse value could be lower. "
    "The two conditional entropies express the remaining uncertainty in bits (lower is better).\n",
    "NMI can look artificially high when almost every document has a rare or unique profile. "
    "AMI corrects for agreement expected by chance given the very large number of profiles; ARI "
    "is an even stricter pairwise agreement measure. These adjusted measures should therefore be "
    "preferred for MiniLM.\n",
    "### Frequent-profile detail\n",
    f"The next table applies the existing profile-matching procedure to K16 profiles covering at "
    f"least {args.min_share:.0%} of documents.\n",
    "| Backbone | K=16 profiles >= threshold | Coverage | Exact modal retention in K=32 | Mean lift | Category coherence | Significant vs chain null |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for row in forward:
    lines.append(
        f"| {row['backbone']} | {row['reference_profiles']} | {row['reference_coverage']:.1%} | "
        f"{row['exact_modal_retention']:.1%} | {fmt(row['mean_lift'], 1)} | "
        f"{fmt(row['weighted_coherence'])} | {row['chain_significant_profiles']}/{row['invariance_profiles']} |"
    )
lines += [
    "\n`Exact modal retention` is the share of documents covered by the reported K=16 profiles "
    "that also carry their profile's most frequent exact K=32 code. This is deliberately strict: "
    "one differing active bit means a different profile. `Mean lift` compares that conditional "
    "match with the K=32 profile's overall frequency. `Category coherence` tests whether the two "
    "matched profiles select documents with the same AG News label composition; it does not require "
    "identical document membership.\n",
    "## Per-backbone reports\n",
]
for model in models:
    lines += [
        f"### {model}\n",
        f"[Complete profile-matching report]({model}/output.md) · "
        f"[matches CSV]({model}/matches.csv) · [bit-pair CSV]({model}/bit_pairs.csv)\n",
        f"![{model} profile matching]({model}/graphs/contact_sheet.png)\n" if args.graphs else "",
    ]
lines += [
    "## Provenance\n",
    f"- Left collection: `{args.left.resolve()}`",
    f"- Right collection: `{args.right.resolve()}`",
    f"- Label file SHA-256: `{left['labels']['sha256']}`",
    f"- Predictive split SHA-256: `{left_split_sha}`",
    f"- Frequent-profile threshold: {args.min_share:.2%}",
    f"- Null draws per test: {args.null_draws}",
]
(args.out / "README.md").write_text("\n".join(line for line in lines if line != "") + "\n")
