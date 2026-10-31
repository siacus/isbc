#!/usr/bin/env python3
"""Compare Qwen profile partitions across selected AG News code capacities.

This is a thin, reproducible wrapper around ``profile_matching.py``.  It never
fits an encoder: it reads the three already encoded full-corpus AG News runs,
creates one two-run matching input for each capacity pair, and records global
partition agreement on every document.  The detailed reports retain the
rank-based top-20 plus tail-10 profile selection used elsewhere in the project.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
)


ROOT = Path(__file__).resolve().parent
SOURCE_COLLECTION = ROOT / "runs/agnews_embedding_comparison/all_fixed_balance_label_name/factor_collection_comparison.json"
CONFIGURATIONS = {
    "k8_target6": ROOT / "runs/agnews_qwen_all_k8_target6_fixed_balance/encoded_agnews_all_ready_best.csv",
    "k16_target6": ROOT / "runs/agnews_qwen_all_fixed_balance/encoded_agnews_all_ready_best.csv",
    "k32_target6": ROOT / "runs/agnews_qwen_all_k32_target6_fixed_balance/encoded_agnews_all_ready_best.csv",
}
RUN_DIRECTORIES = {
    "k8_target6": ROOT / "runs/agnews_qwen_all_k8_target6_fixed_balance",
    "k16_target6": ROOT / "runs/agnews_qwen_all_fixed_balance",
    "k32_target6": ROOT / "runs/agnews_qwen_all_k32_target6_fixed_balance",
}
PAIRS = (("k8_target6", "k16_target6"), ("k16_target6", "k32_target6"), ("k8_target6", "k32_target6"))
TRIPLET = ("k8_target6", "k16_target6", "k32_target6")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_codes(path: Path) -> tuple[list[str], list[str]]:
    with path.open(newline="") as handle:
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


def partition_metrics(left: list[str], right: list[str]) -> dict[str, float | int]:
    n = len(left)
    left_counts, right_counts = Counter(left), Counter(right)
    joint = Counter(zip(left, right))
    right_by_left: dict[str, Counter] = defaultdict(Counter)
    left_by_right: dict[str, Counter] = defaultdict(Counter)
    for (left_code, right_code), count in joint.items():
        right_by_left[left_code][right_code] = count
        left_by_right[right_code][left_code] = count
    h_left, h_right, h_joint = entropy(left_counts, n), entropy(right_counts, n), entropy(joint, n)
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(left, right)
    return {
        "documents": n,
        "left_unique_profiles": len(left_counts),
        "right_unique_profiles": len(right_counts),
        "left_entropy_bits": h_left,
        "right_entropy_bits": h_right,
        "mutual_information_bits": h_left + h_right - h_joint,
        "right_given_left_entropy_bits": h_joint - h_left,
        "left_given_right_entropy_bits": h_joint - h_right,
        "left_from_right_modal_accuracy": sum(max(x.values()) for x in left_by_right.values()) / n,
        "right_from_left_modal_accuracy": sum(max(x.values()) for x in right_by_left.values()) / n,
        "normalized_mutual_information": normalized_mutual_info_score(left, right),
        "adjusted_mutual_information": adjusted_mutual_info_score(left, right),
        "adjusted_rand_index": adjusted_rand_score(left, right),
        "right_homogeneity_wrt_left": homogeneity,
        "right_completeness_wrt_left": completeness,
        "v_measure": v_measure,
    }


def three_way_metrics(codes: dict[str, list[str]]) -> dict[str, float | int | str]:
    """Compute joint, rather than pairwise, profile-partition diagnostics."""
    names = list(TRIPLET)
    n = len(codes[names[0]])
    if any(len(codes[name]) != n for name in names):
        raise ValueError("Three-way codes have different document counts")
    rows = list(zip(*(codes[name] for name in names)))
    joint = Counter(rows)
    marginal = {name: Counter(codes[name]) for name in names}
    entropies = {name: entropy(marginal[name], n) for name in names}
    joint_entropy = entropy(joint, n)
    result: dict[str, float | int | str] = {
        "documents": n,
        "joint_unique_profile_triples": len(joint),
        "joint_entropy_bits": joint_entropy,
        "total_correlation_bits": sum(entropies.values()) - joint_entropy,
    }
    for index, name in enumerate(names):
        other = [candidate for candidate in names if candidate != name]
        pairs = [tuple(row[j] for j in range(3) if j != index) for row in rows]
        pair_counts = Counter(pairs)
        conditional = joint_entropy - entropy(pair_counts, n)
        value_by_pair: dict[tuple[str, str], Counter] = defaultdict(Counter)
        for value, pair in zip(codes[name], pairs):
            value_by_pair[pair][value] += 1
        result[f"{name}_unique_profiles"] = len(marginal[name])
        result[f"{name}_entropy_bits"] = entropies[name]
        result[f"{name}_given_other_two_entropy_bits"] = conditional
        result[f"{name}_recoverable_from_other_two_modal_accuracy"] = (
            sum(max(values.values()) for values in value_by_pair.values()) / n
        )
        result[f"{name}_other_pair"] = "+".join(other)
    return result


def profile_token(active_bits: str, width: int) -> str:
    """Compact, reversible active-bit token for a Markdown profile table."""
    active = [] if not active_bits else [int(value) for value in active_bits.split(",")]
    blocks = [[] for _ in range((width + 15) // 16)]
    digits = "0123456789ABCDEF"
    for bit in active:
        blocks[bit // 16].append(digits[bit % 16])
    return "|".join("".join(block) or "-" for block in blocks)


def write_three_way_profile_sankey(
    out: Path, codes: dict[str, list[str]], order_names: tuple[str, str, str],
    stem: str, top_n: int | dict[str, int] = 20, label_limit: int = 20,
) -> tuple[Path, Path]:
    """Draw an aggregated alluvial/Sankey view of the three profile assignments.

    Every document contributes to both transitions.  The 20 most prevalent
    profiles are shown in each encoder and all remaining profiles are pooled as
    ``Other`` only for display.  No labels enter this figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.path import Path as MplPath

    names = list(order_names)
    n = len(codes[names[0]])
    display_limit = {
        name: (top_n[name] if isinstance(top_n, dict) else top_n) for name in names
    }
    displayed: dict[str, list[str]] = {
        name: [value for value, _count in Counter(codes[name]).most_common(display_limit[name])]
        for name in names
    }
    groups = {
        name: [value if value in set(displayed[name]) else "__other__" for value in codes[name]]
        for name in names
    }
    node_rows: list[dict[str, str | int | float]] = []
    order: dict[str, list[str]] = {}
    for name in names:
        counts = Counter(groups[name])
        order[name] = [value for value in displayed[name] if counts[value]]
        if counts["__other__"]:
            order[name].append("__other__")
        for rank, value in enumerate(order[name], 1):
            node_rows.append({
                "configuration": name,
                "display_rank": rank if value != "__other__" else "other",
                "profile_code": value if value != "__other__" else "Other profiles pooled for display",
                "documents": counts[value],
                "share": counts[value] / n,
            })
    positions: dict[tuple[str, str], tuple[float, float]] = {}
    gap = 0.006
    for name in names:
        counts = Counter(groups[name])
        usable = 1.0 - gap * (len(order[name]) - 1)
        scale = usable / n
        y = 1.0
        for value in order[name]:
            height = counts[value] * scale
            y -= height
            positions[(name, value)] = (y, height)
            y -= gap
    palette = plt.get_cmap("tab20")
    colors = {
        (name, value): palette(index % 20)
        for name in names
        for index, value in enumerate(order[name])
    }
    fig, ax = plt.subplots(figsize=(16, 15), constrained_layout=True)
    x = {names[0]: 0.0, names[1]: 0.5, names[2]: 1.0}
    sources: dict[tuple[str, str, str], float] = defaultdict(float)
    for transition, (left, right) in enumerate(zip(names, names[1:]), 1):
        flow = Counter(zip(groups[left], groups[right]))
        source_offset = defaultdict(float)
        target_offset = defaultdict(float)
        scale = (1.0 - gap * (len(order[left]) - 1)) / n
        for (source, target), count in sorted(flow.items(), key=lambda item: -item[1]):
            y0, _ = positions[(left, source)]
            y1, _ = positions[(right, target)]
            height = count * scale
            start = y0 + source_offset[source]
            end = y1 + target_offset[target]
            source_offset[source] += height
            target_offset[target] += height
            mid0 = x[left] + 0.18
            mid1 = x[right] - 0.18
            vertices = [
                (x[left], start), (mid0, start), (mid1, end), (x[right], end),
                (x[right], end + height), (mid1, end + height),
                (mid0, start + height), (x[left], start + height), (x[left], start),
            ]
            codes_path = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                          MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                          MplPath.CLOSEPOLY]
            ax.add_patch(PathPatch(MplPath(vertices, codes_path), facecolor=colors[(left, source)],
                                   edgecolor="none", alpha=0.30, zorder=1))
            sources[(f"{left}_to_{right}", source, target)] += count
    for name in names:
        for value in order[name]:
            y, height = positions[(name, value)]
            ax.add_patch(Rectangle((x[name] - 0.013, y), 0.026, height,
                                   facecolor=colors[(name, value)], edgecolor="white",
                                   linewidth=0.35, zorder=3))
            rank_index = order[name].index(value) + 1
            rank = (
                "Other" if value == "__other__" else
                (f"P{rank_index:02d}" if rank_index <= label_limit else "")
            )
            align = "right" if name == names[0] else "left"
            offset = -0.02 if name == names[0] else 0.02
            ax.text(x[name] + offset, y + height / 2, rank, ha=align, va="center", fontsize=7)
        ax.text(x[name], 1.045, name.replace("_target", "/").replace("k", "K="),
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    limit_text = ", ".join(f"{name} top {display_limit[name]}" for name in names)
    ax.text(0.5, -0.065,
            f"Each ribbon aggregates documents; labelled P nodes are top profiles ({limit_text}). "
            f"All remaining profiles are pooled as Other. N={n:,}.",
            ha="center", va="top", fontsize=8, color="#444444")
    ax.set_xlim(-0.19, 1.19)
    ax.set_ylim(-0.10, 1.08)
    ax.axis("off")
    png = out / f"{stem}.png"
    pdf = out / f"{stem}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    with (out / f"{stem}_nodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(node_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(node_rows)
    flow_rows = [
        {"transition": transition, "source_profile": source, "target_profile": target,
         "documents": count, "share": count / n}
        for (transition, source, target), count in sources.items()
    ]
    with (out / f"{stem}_flows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flow_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(flow_rows)
    return png, pdf


def read_pair_summary(path: Path) -> list[dict[str, float | int | str]]:
    """Reuse verified pairwise diagnostics when regenerating only the triplet outputs."""
    integer_fields = {"documents", "left_unique_profiles", "right_unique_profiles"}
    text_fields = {"left_configuration", "right_configuration"}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, float | int | str]] = []
        for row in reader:
            rows.append({
                key: value if key in text_fields else (int(value) if key in integer_fields else float(value))
                for key, value in row.items()
            })
    if len(rows) != len(PAIRS):
        raise RuntimeError(f"Expected {len(PAIRS)} existing pairwise summaries in {path}")
    return rows


def write_three_way_markdown_table(out: Path) -> Path:
    """Make a compact rendering-safe table for the three-way greedy matches."""
    widths = {"k8_target6": 8, "k16_target6": 16, "k32_target6": 32}
    with (out / "matches.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["reference"]].append(row)
    lines = [
        "# Three-way profile matches: compact table",
        "",
        "This table is a compact rendering-safe counterpart to `output.md`. Each row is one greedy document-restriction chain over all three encoders. `P` is a reversible active-bit token: hexadecimal characters are zero-based bit indices within a 16-bit block; `|` separates blocks. It is an identifier, not a learned shared bit numbering.",
        "",
    ]
    for reference in TRIPLET:
        lines.extend([
            f"## Starting from {reference}",
            "",
            "| Group | Rank | Start share | 8/6 P | 16/6 P | 32/6 P | Final docs |",
            "|---|---:|---:|---|---|---|---:|",
        ])
        for row in grouped[reference]:
            entries = dict(item.split("=", 1) for item in row["profiles"].split("; "))
            tokens = {
                name: profile_token(entries[name], widths[name])
                for name in TRIPLET
            }
            lines.append(
                f"| {row['selection_group']} | {row['selection_rank']} | "
                f"{float(row['start_share']):.2%} | `{tokens['k8_target6']}` | "
                f"`{tokens['k16_target6']}` | `{tokens['k32_target6']}` | {row['final_docs']} |"
            )
        lines.append("")
    path = out / "three_way_profile_matches.md"
    path.write_text("\n".join(lines).rstrip() + "\n")
    return path


def _pmi_phi(left: np.ndarray, right: np.ndarray) -> tuple[int, float, float, float, float]:
    """Return joint count, joint share, PMI base 2, phi, and conditional lift."""
    n = len(left)
    joint_count = int(np.count_nonzero(left & right))
    p_left = float(left.mean())
    p_right = float(right.mean())
    joint = joint_count / n
    if not joint or not p_left or not p_right:
        return joint_count, joint, float("nan"), float("nan"), float("nan")
    pmi = math.log2(joint / (p_left * p_right))
    denominator = math.sqrt(p_left * (1 - p_left) * p_right * (1 - p_right))
    phi = (joint - p_left * p_right) / denominator if denominator else float("nan")
    lift = (joint / p_left) / p_right
    return joint_count, joint, pmi, phi, lift


def _load_communities(name: str) -> dict[int, int]:
    path = RUN_DIRECTORIES[name] / "bit_structure/bit_community_nodes.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            int(row["bit"].split("_")[1]): int(row["community_resolution_1"])
            for row in reader
        }


def analyze_three_way_bit_communities(out: Path, codes: dict[str, list[str]]) -> None:
    """Relate Leiden communities and bits across 8/6, 16/6, and 32/6 without labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.path import Path as MplPath

    names = list(TRIPLET)
    matrices = {
        name: np.asarray([[char == "1" for char in code] for code in codes[name]], dtype=bool)
        for name in names
    }
    communities = {name: _load_communities(name) for name in names}
    n = len(matrices[names[0]])
    bit_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    pair_metrics: dict[tuple[str, int, str, int], tuple[float, float]] = {}
    for source_index, source in enumerate(names):
        for target in names[source_index + 1:]:
            for source_bit in range(matrices[source].shape[1]):
                for target_bit in range(matrices[target].shape[1]):
                    joint_count, joint_share, pmi, phi, lift = _pmi_phi(
                        matrices[source][:, source_bit], matrices[target][:, target_bit]
                    )
                    pair_metrics[(source, source_bit, target, target_bit)] = (pmi, phi)
                    edge_rows.append({
                        "source_configuration": source,
                        "source_bit": f"bit_{source_bit:02d}",
                        "source_community": communities[source].get(source_bit),
                        "target_configuration": target,
                        "target_bit": f"bit_{target_bit:02d}",
                        "target_community": communities[target].get(target_bit),
                        "joint_activation_count": joint_count,
                        "joint_activation_share": joint_share,
                        "pmi_bits": pmi,
                        "phi_correlation": phi,
                        "activation_lift": lift,
                        "positive_pmi_edge": bool(np.isfinite(pmi) and pmi > 0),
                    })
    for name in names:
        prevalence = matrices[name].mean(axis=0)
        for bit, share in enumerate(prevalence):
            bit_rows.append({
                "configuration": name,
                "bit": f"bit_{bit:02d}",
                "community_resolution_1": communities[name].get(bit),
                "activation_prevalence": share,
            })

    # Community conjunctions are the documents that activate every bit in a
    # 8/6 Leiden community.  They capture the documented within-run structure
    # before asking how it is expressed at wider code capacities.
    by_community: dict[int, list[int]] = defaultdict(list)
    for bit, community in communities["k8_target6"].items():
        by_community[community].append(bit)
    community_rows: list[dict[str, object]] = []
    for community, bits in sorted(by_community.items()):
        mask = matrices["k8_target6"][:, bits].all(axis=1)
        count = int(mask.sum())
        for target in ("k16_target6", "k32_target6"):
            for target_bit in range(matrices[target].shape[1]):
                joint_count, joint_share, pmi, phi, lift = _pmi_phi(mask, matrices[target][:, target_bit])
                community_rows.append({
                    "reference_configuration": "k8_target6",
                    "reference_community": community,
                    "required_bits": ";".join(f"bit_{bit:02d}" for bit in bits),
                    "community_conjunction_documents": count,
                    "community_conjunction_share": count / n,
                    "target_configuration": target,
                    "target_bit": f"bit_{target_bit:02d}",
                    "target_community": communities[target].get(target_bit),
                    "joint_activation_count": joint_count,
                    "joint_activation_share": joint_share,
                    "pmi_bits": pmi,
                    "phi_correlation": phi,
                    "activation_lift": lift,
                })

    # A triadic core is a closed positive-PMI triangle, ranked conservatively by
    # its weakest phi association so one strong pair cannot dominate the score.
    triads: list[dict[str, object]] = []
    for bit8 in range(matrices["k8_target6"].shape[1]):
        for bit16 in range(matrices["k16_target6"].shape[1]):
            pmi_816, phi_816 = pair_metrics[("k8_target6", bit8, "k16_target6", bit16)]
            if not (np.isfinite(pmi_816) and pmi_816 > 0 and np.isfinite(phi_816) and phi_816 > 0):
                continue
            for bit32 in range(matrices["k32_target6"].shape[1]):
                pmi_832, phi_832 = pair_metrics[("k8_target6", bit8, "k32_target6", bit32)]
                pmi_1632, phi_1632 = pair_metrics[("k16_target6", bit16, "k32_target6", bit32)]
                values = (pmi_832, phi_832, pmi_1632, phi_1632)
                if not all(np.isfinite(value) and value > 0 for value in values):
                    continue
                all_pmi = (pmi_816, pmi_832, pmi_1632)
                all_phi = (phi_816, phi_832, phi_1632)
                triads.append({
                    "k8_bit": f"bit_{bit8:02d}", "k8_community": communities["k8_target6"].get(bit8),
                    "k16_bit": f"bit_{bit16:02d}", "k16_community": communities["k16_target6"].get(bit16),
                    "k32_bit": f"bit_{bit32:02d}", "k32_community": communities["k32_target6"].get(bit32),
                    "minimum_phi": min(all_phi), "mean_phi": float(np.mean(all_phi)),
                    "minimum_pmi_bits": min(all_pmi), "mean_pmi_bits": float(np.mean(all_pmi)),
                })
    triads.sort(key=lambda row: (row["minimum_phi"], row["mean_phi"], row["minimum_pmi_bits"]), reverse=True)
    for rank, row in enumerate(triads, 1):
        row["triad_rank"] = rank

    def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    write_csv(out / "three_way_bit_nodes.csv", bit_rows)
    write_csv(out / "three_way_cross_capacity_bit_edges.csv", edge_rows)
    write_csv(out / "three_way_k8_community_associations.csv", community_rows)
    if triads:
        write_csv(out / "three_way_bit_core_triads.csv", triads)

    # Tripartite graph: preserve every positive-PMI edge, with transparent
    # ribbons, and colour node outlines by the Leiden community inside its own
    # encoder.  Community identifiers are local and are not equated by colour.
    fig, ax = plt.subplots(figsize=(17, 11), constrained_layout=True)
    x = {"k8_target6": 0, "k16_target6": 1, "k32_target6": 2}
    cmap = plt.get_cmap("tab20")
    positions: dict[tuple[str, int], tuple[float, float]] = {}
    for name in names:
        widths = matrices[name].shape[1]
        sorted_bits = sorted(range(widths), key=lambda bit: (communities[name].get(bit, 999), -matrices[name][:, bit].mean()))
        for rank, bit in enumerate(sorted_bits):
            positions[(name, bit)] = (x[name], 1 - (rank + 1) / (widths + 1))
    positive = [row for row in edge_rows if row["positive_pmi_edge"]]
    max_pmi = max((float(row["pmi_bits"]) for row in positive), default=1.0)
    for row in positive:
        source_bit = int(str(row["source_bit"]).split("_")[1])
        target_bit = int(str(row["target_bit"]).split("_")[1])
        x0, y0 = positions[(str(row["source_configuration"]), source_bit)]
        x1, y1 = positions[(str(row["target_configuration"]), target_bit)]
        alpha = 0.06 + 0.36 * float(row["pmi_bits"]) / max_pmi
        ax.plot([x0, x1], [y0, y1], color="#5e5e5e", linewidth=0.25 + 1.2 * float(row["pmi_bits"]) / max_pmi,
                alpha=alpha, zorder=1)
    for name in names:
        for bit in range(matrices[name].shape[1]):
            px, py = positions[(name, bit)]
            community = communities[name].get(bit, -1)
            ax.scatter(px, py, s=85 + 240 * matrices[name][:, bit].mean(), color="white",
                       edgecolor=cmap(community % 20), linewidth=2.2, zorder=3)
            ax.text(px + (-0.04 if name == names[0] else 0.04), py, f"b{bit}",
                    ha="right" if name == names[0] else "left", va="center", fontsize=8)
        ax.text(x[name], 1.04, name.replace("k", "K=").replace("_target", "/"),
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.3, 2.3); ax.set_ylim(-0.04, 1.10); ax.axis("off")
    ax.set_title("Cross-capacity bit associations: all positive-PMI edges", loc="left", fontsize=14)
    ax.text(1, -0.02, "Node border colour = Leiden community within that encoder only; edge width/opacity = PMI (base 2).",
            ha="center", va="top", fontsize=8, color="#444444")
    fig.savefig(out / "three_way_cross_capacity_bit_network.png", dpi=220)
    fig.savefig(out / "three_way_cross_capacity_bit_network.pdf")
    plt.close(fig)

    # A sparse companion view of the strongest closed triangles makes the
    # recurring cross-capacity core visible without hiding it among all edges.
    selected_triads = triads[:20]
    if selected_triads:
        core_nodes: dict[str, set[int]] = {name: set() for name in names}
        core_edges: dict[tuple[str, int, str, int], float] = {}
        for row in selected_triads:
            members = [
                ("k8_target6", int(str(row["k8_bit"]).split("_")[1])),
                ("k16_target6", int(str(row["k16_bit"]).split("_")[1])),
                ("k32_target6", int(str(row["k32_bit"]).split("_")[1])),
            ]
            for name, bit in members:
                core_nodes[name].add(bit)
            for left, right in ((members[0], members[1]), (members[0], members[2]), (members[1], members[2])):
                if x[left[0]] > x[right[0]]:
                    left, right = right, left
                edge = (left[0], left[1], right[0], right[1])
                core_edges[edge] = max(core_edges.get(edge, 0.0), float(row["minimum_phi"]))
        fig, ax = plt.subplots(figsize=(13, 7.5), constrained_layout=True)
        core_positions: dict[tuple[str, int], tuple[float, float]] = {}
        for name in names:
            ordered = sorted(core_nodes[name], key=lambda bit: (communities[name].get(bit, 999), bit))
            for rank, bit in enumerate(ordered):
                core_positions[(name, bit)] = (x[name], 1 - (rank + 1) / (len(ordered) + 1))
        maximum = max(core_edges.values())
        for (source, source_bit, target, target_bit), weight in core_edges.items():
            x0, y0 = core_positions[(source, source_bit)]
            x1, y1 = core_positions[(target, target_bit)]
            ax.plot([x0, x1], [y0, y1], color="#4f4f4f", linewidth=1.0 + 4.0 * weight / maximum,
                    alpha=0.62, zorder=1)
        for name in names:
            for bit in sorted(core_nodes[name]):
                px, py = core_positions[(name, bit)]
                community = communities[name].get(bit, -1)
                ax.scatter(px, py, s=190, color="white", edgecolor=cmap(community % 20), linewidth=3, zorder=3)
                ax.text(px + (-0.045 if name == names[0] else 0.045), py,
                        f"b{bit} (C{community})", ha="right" if name == names[0] else "left",
                        va="center", fontsize=9)
            ax.text(x[name], 1.05, name.replace("k", "K=").replace("_target", "/"),
                    ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.set_xlim(-0.35, 2.35); ax.set_ylim(-0.05, 1.11); ax.axis("off")
        ax.set_title("Strongest closed cross-capacity bit cores", loc="left", fontsize=14)
        ax.text(1, -0.02, "Edges are induced by the top 20 closed triads and scaled by their weakest φ association.",
                ha="center", va="top", fontsize=8, color="#444444")
        fig.savefig(out / "three_way_bit_core_triads.png", dpi=220)
        fig.savefig(out / "three_way_bit_core_triads.pdf")
        plt.close(fig)

        # Sankey/alluvial counterpart.  Every bit remains a displayed node,
        # including those with no selected core triad; ribbons represent the
        # top-triad score rather than a partition of documents.
        all_bits = {name: list(range(matrices[name].shape[1])) for name in names}
        all_positions: dict[tuple[str, int], tuple[float, float]] = {}
        node_weight: dict[tuple[str, int], float] = defaultdict(float)
        triad_weight = {int(row["triad_rank"]): float(row["minimum_phi"]) for row in selected_triads}
        for row in selected_triads:
            weight = triad_weight[int(row["triad_rank"])]
            for name, key in (("k8_target6", "k8_bit"), ("k16_target6", "k16_bit"), ("k32_target6", "k32_bit")):
                node_weight[(name, int(str(row[key]).split("_")[1]))] += weight
        core_order: dict[str, list[int]] = {
            name: sorted(all_bits[name], key=lambda bit: (communities[name].get(bit, 999), bit))
            for name in names
        }
        gap = 0.007
        display_weight: dict[tuple[str, int], float] = {}
        for name in names:
            # Give non-core nodes a small visible height so communities cannot
            # disappear merely because they have no selected triad.
            minimum = max(sum(node_weight[(name, bit)] for bit in all_bits[name]) * 0.004, 0.002)
            for bit in all_bits[name]:
                display_weight[(name, bit)] = max(node_weight[(name, bit)], minimum)
            scale = (1 - gap * (len(core_order[name]) - 1)) / sum(display_weight[(name, bit)] for bit in core_order[name])
            y = 1.0
            for bit in core_order[name]:
                height = display_weight[(name, bit)] * scale
                y -= height
                all_positions[(name, bit)] = (y, height)
                y -= gap
        fig, ax = plt.subplots(figsize=(17, 13), constrained_layout=True)
        offsets = defaultdict(float)
        for row in selected_triads:
            weight = triad_weight[int(row["triad_rank"])]
            sequence = [
                ("k8_target6", int(str(row["k8_bit"]).split("_")[1])),
                ("k16_target6", int(str(row["k16_bit"]).split("_")[1])),
                ("k32_target6", int(str(row["k32_bit"]).split("_")[1])),
            ]
            colour = cmap(communities[sequence[0][0]].get(sequence[0][1], 0) % 20)
            for left, right in zip(sequence, sequence[1:]):
                y0, _ = all_positions[left]
                y1, _ = all_positions[right]
                source_scale = all_positions[left][1] / display_weight[left]
                target_scale = all_positions[right][1] / display_weight[right]
                height = min(weight * source_scale, weight * target_scale)
                start = y0 + offsets[(left, "out")]
                end = y1 + offsets[(right, "in")]
                offsets[(left, "out")] += height
                offsets[(right, "in")] += height
                x0, x1 = x[left[0]], x[right[0]]
                vertices = [
                    (x0, start), (x0 + 0.18, start), (x1 - 0.18, end), (x1, end),
                    (x1, end + height), (x1 - 0.18, end + height),
                    (x0 + 0.18, start + height), (x0, start + height), (x0, start),
                ]
                commands = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                            MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                            MplPath.CLOSEPOLY]
                ax.add_patch(PathPatch(MplPath(vertices, commands), facecolor=colour,
                                       edgecolor="none", alpha=0.42, zorder=1))
        for name in names:
            for bit in core_order[name]:
                node_y, height = all_positions[(name, bit)]
                community = communities[name].get(bit, -1)
                ax.add_patch(Rectangle((x[name] - 0.013, node_y), 0.026, height,
                                       facecolor="white", edgecolor=cmap(community % 20), linewidth=1.7, zorder=3))
                label = f"b{bit} C{community}"
                ax.text(x[name] + (-0.02 if name == names[0] else 0.02), node_y + height / 2, label,
                        ha="right" if name == names[0] else "left", va="center", fontsize=7)
            ax.text(x[name], 1.045, name.replace("k", "K=").replace("_target", "/"),
                    ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.set_xlim(-0.24, 2.24); ax.set_ylim(-0.06, 1.09); ax.axis("off")
        ax.set_title("Strongest closed cross-capacity bit cores: Sankey representation", loc="left", fontsize=14)
        ax.text(1, -0.02, "All bits are shown, including nodes without a selected core; ribbons represent top-triad minimum φ scores, not document counts.",
                ha="center", va="top", fontsize=8, color="#444444")
        fig.savefig(out / "three_way_bit_core_triads_sankey.png", dpi=220)
        fig.savefig(out / "three_way_bit_core_triads_sankey.pdf")
        plt.close(fig)

    lines = [
        "# Three-way bit-community analysis",
        "",
        "## Scope",
        "",
        "This label-free analysis uses the same 127,600 documents and the existing positive-PMI Leiden communities inside each Qwen code. It does not assume that bit indices or Leiden community numbers have a common meaning across widths.",
        "",
        "## Cross-capacity bit network",
        "",
        "![Cross-capacity bit network](three_way_cross_capacity_bit_network.png)",
        "",
        "Every edge with finite positive PMI is drawn. PMI is association relative to independence, not a significance test; absent edges do not prove independence. Node-border colours denote the Leiden community *within the same configuration* and are intentionally not a cross-configuration identity claim.",
        "",
        "## Strongest closed cross-capacity cores",
        "",
        "![Strongest closed cross-capacity bit cores](three_way_bit_core_triads.png)",
        "",
        "This sparse graph is induced by the 20 highest-ranked closed triads below. It makes repeated cross-capacity motifs visible; its edge weight is the weakest φ association within the originating triad. It is evidence of recurring co-activation structure, not proof that the three bits have identical semantics.",
        "",
        "![Strongest closed cross-capacity bit cores: Sankey](three_way_bit_core_triads_sankey.png)",
        "",
        "This alluvial counterpart retains all bit nodes, including communities with no selected top-20 triad. Its ribbons represent triad-association strength rather than a partition of documents, so a node can have a small visible bar without a ribbon.",
        "",
        "",
        "## 8/6 community conjunctions in wider codes",
        "",
        "For each 8/6 Leiden community, the focal documents activate every bit in that community jointly. The accompanying CSV tests how often each 16/6 or 32/6 bit activates in those same documents, reporting PMI, φ, and lift over its corpus prevalence. This is the direct document-level answer to how a low-capacity community is expressed at wider capacities.",
        "",
        "| 8/6 community | Required bits | Documents | Top 16/6 bit by φ | φ | Top 32/6 bit by φ | φ |",
        "|---:|---|---:|---|---:|---|---:|",
    ]
    for community, bits in sorted(by_community.items()):
        subset = [row for row in community_rows if row["reference_community"] == community]
        count = int(subset[0]["community_conjunction_documents"])
        best16 = max((row for row in subset if row["target_configuration"] == "k16_target6" and np.isfinite(row["phi_correlation"])), key=lambda row: float(row["phi_correlation"]), default=None)
        best32 = max((row for row in subset if row["target_configuration"] == "k32_target6" and np.isfinite(row["phi_correlation"])), key=lambda row: float(row["phi_correlation"]), default=None)
        lines.append(
            f"| {community} | " + ", ".join(f"bit_{bit:02d}" for bit in bits) + f" | {count:,} | "
            + (f"{best16['target_bit']} | {float(best16['phi_correlation']):.3f}" if best16 else "n/a | n/a")
            + " | " + (f"{best32['target_bit']} | {float(best32['phi_correlation']):.3f}" if best32 else "n/a | n/a") + " |"
        )
    lines += [
        "",
        "## Closed triadic cores",
        "",
        "A closed triadic core is one 8/6, one 16/6, and one 32/6 bit for which all three pairwise PMI and φ values are strictly positive. Rows are ranked by the weakest φ edge, so a triad cannot rank highly solely through one unusually strong pair. It indicates a robust co-activation motif across capacities, not identity of three features.",
        "",
        "| Rank | 8/6 bit (community) | 16/6 bit (community) | 32/6 bit (community) | Minimum φ | Mean φ | Minimum PMI |",
        "|---:|---|---|---|---:|---:|---:|",
    ]
    for row in triads[:20]:
        lines.append(
            f"| {row['triad_rank']} | {row['k8_bit']} (C{row['k8_community']}) | "
            f"{row['k16_bit']} (C{row['k16_community']}) | {row['k32_bit']} (C{row['k32_community']}) | "
            f"{float(row['minimum_phi']):.3f} | {float(row['mean_phi']):.3f} | {float(row['minimum_pmi_bits']):.3f} |"
        )
    lines += [
        "",
        "## Artifacts",
        "",
        "- `three_way_bit_nodes.csv`: bit prevalence and local Leiden membership.",
        "- `three_way_cross_capacity_bit_edges.csv`: every cross-capacity bit pair, including joint counts, PMI, φ, and lift.",
        "- `three_way_k8_community_associations.csv`: document-level community-conjunction associations with 16/6 and 32/6 bits.",
        "- `three_way_bit_core_triads.csv`: all positive closed triads and their ranking.",
    ]
    (out / "three_way_bit_communities.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "runs/agnews_embedding_comparison/qwen_capacity_profile_matching",
    )
    parser.add_argument("--null-draws", type=int, default=1000)
    parser.add_argument("--graphs", action="store_true")
    parser.add_argument(
        "--triplet-only", action="store_true",
        help="retain existing pairwise reports and regenerate only the three-way outputs",
    )
    parser.add_argument(
        "--reuse-triplet-matches", action="store_true",
        help="reuse an existing completed three-way matching table while regenerating figures and summaries",
    )
    args = parser.parse_args()

    source = json.loads(SOURCE_COLLECTION.read_text())
    for name, path in CONFIGURATIONS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing encoded Qwen run for {name}: {path}")
    args.out.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []

    matcher = ROOT / "profile_matching/profile_matching.py"
    if args.triplet_only:
        summary = read_pair_summary(args.out / "profile_partition_metrics.csv")
    else:
        for left_name, right_name in PAIRS:
            out = args.out / f"{left_name}_vs_{right_name}"
            out.mkdir(parents=True, exist_ok=True)
            comparison = {
            "schema_version": 1,
            "comparison": f"Qwen AG News profiles: {left_name} vs {right_name}",
            "label_column": source["label_column"],
            "n_aligned_rows": source["n_aligned_rows"],
            "labels": source["labels"],
            "predictive_split": source["predictive_split"],
            "run_order": [left_name, right_name],
            "runs": {
                name: {"encoded": {"path": str(path), "sha256": sha256(path)}}
                for name, path in ((left_name, CONFIGURATIONS[left_name]), (right_name, CONFIGURATIONS[right_name]))
            },
            "source_runs": {name: str(path.parent) for name, path in ((left_name, CONFIGURATIONS[left_name]), (right_name, CONFIGURATIONS[right_name]))},
            }
            (out / "factor_collection_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
            command = [
            sys.executable, str(matcher), "--run-dir", str(out), "--out", str(out),
            "--reference", left_name, "--all-references", "--null-draws", str(args.null_draws),
            "--top-reference-profiles", "20", "--tail-reference-profiles", "10",
            ]
            if args.graphs:
                command.append("--graphs")
            subprocess.run(command, check=True)

            left_ids, left_codes = load_codes(CONFIGURATIONS[left_name])
            right_ids, right_codes = load_codes(CONFIGURATIONS[right_name])
            if left_ids != right_ids:
                raise RuntimeError(f"Document IDs differ: {left_name} versus {right_name}")
            row = {"left_configuration": left_name, "right_configuration": right_name, **partition_metrics(left_codes, right_codes)}
            summary.append(row)

    # Principal output: one greedy matching chain across all three capacities.
    triplet_name = "_vs_".join(TRIPLET)
    triplet_out = args.out / triplet_name
    triplet_out.mkdir(parents=True, exist_ok=True)
    triplet_comparison = {
        "schema_version": 1,
        "comparison": "Qwen AG News three-way profiles: 8/6 vs 16/6 vs 32/6",
        "label_column": source["label_column"],
        "n_aligned_rows": source["n_aligned_rows"],
        "labels": source["labels"],
        "predictive_split": source["predictive_split"],
        "run_order": list(TRIPLET),
        "runs": {
            name: {"encoded": {"path": str(CONFIGURATIONS[name]), "sha256": sha256(CONFIGURATIONS[name])}}
            for name in TRIPLET
        },
        "source_runs": {name: str(CONFIGURATIONS[name].parent) for name in TRIPLET},
    }
    (triplet_out / "factor_collection_comparison.json").write_text(
        json.dumps(triplet_comparison, indent=2) + "\n"
    )
    command = [
        sys.executable, str(matcher), "--run-dir", str(triplet_out), "--out", str(triplet_out),
        "--reference", TRIPLET[0], "--all-references", "--null-draws", str(args.null_draws),
        "--top-reference-profiles", "20", "--tail-reference-profiles", "10",
    ]
    if args.graphs:
        command.append("--graphs")
    if args.reuse_triplet_matches:
        if not (triplet_out / "matches.csv").is_file():
            raise FileNotFoundError("--reuse-triplet-matches requires an existing three-way matches.csv")
    else:
        subprocess.run(command, check=True)
    triplet_codes: dict[str, list[str]] = {}
    triplet_ids: list[str] | None = None
    for name in TRIPLET:
        ids, code_values = load_codes(CONFIGURATIONS[name])
        if triplet_ids is None:
            triplet_ids = ids
        elif ids != triplet_ids:
            raise RuntimeError(f"Document IDs differ in three-way match: {name}")
        triplet_codes[name] = code_values
    detailed_profile_limits = {
        "k8_target6": 50,
        "k16_target6": 200,
        "k32_target6": 500,
    }
    detailed_profile_coverage = {
        name: sum(count for _profile, count in Counter(triplet_codes[name]).most_common(detailed_profile_limits[name])) / len(triplet_codes[name])
        for name in TRIPLET
    }
    triple = three_way_metrics(triplet_codes)
    with (args.out / "three_way_partition_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(triple), lineterminator="\n")
        writer.writeheader()
        writer.writerow(triple)
    write_three_way_profile_sankey(
        args.out, triplet_codes, TRIPLET, "three_way_profile_sankey", top_n=20
    )
    write_three_way_profile_sankey(
        args.out, triplet_codes, tuple(reversed(TRIPLET)), "three_way_profile_sankey_32_to_8", top_n=20
    )
    write_three_way_profile_sankey(
        args.out, triplet_codes, TRIPLET, "three_way_profile_sankey_detailed",
        top_n=detailed_profile_limits, label_limit=20,
    )
    write_three_way_profile_sankey(
        args.out, triplet_codes, tuple(reversed(TRIPLET)), "three_way_profile_sankey_detailed_32_to_8",
        top_n=detailed_profile_limits, label_limit=20,
    )
    write_three_way_markdown_table(triplet_out)
    analyze_three_way_bit_communities(args.out, triplet_codes)

    fields = list(summary[0])
    with (args.out / "profile_partition_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)

    lines = [
        "# Qwen AG News profile matching across capacities\n",
        "## Scope\n",
        "The same 127,600 AG News documents are encoded by three Qwen sparse autoencoders: 8/6, 16/6, and 32/6. Bit numbers are local to a fitted encoder; the global comparison therefore treats each whole binary profile as an arbitrary category.\n",
        "## Three-way profile matching\n",
        "The main output is one three-model chain, not three independent pairwise matches. Starting from a selected profile in one configuration, the procedure restricts the document set to its modal partner in a second configuration, then restricts again to its modal partner in the third. The linked report repeats this from 8/6, 16/6, and 32/6.\n",
        f"![Three-way profile Sankey](three_way_profile_sankey.png)\n",
        "The Sankey aggregates every document. Each column displays its 20 most prevalent complete profiles plus `Other`; ribbons show document counts through 8/6, 16/6, and 32/6. It is label-free. The node and flow tables preserve the precise numbers.\n",
        f"- [Three-way matching report]({triplet_name}/output.md)\n",
        f"- [Compact three-way profile table]({triplet_name}/three_way_profile_matches.md)\n",
        "- [Sankey nodes](three_way_profile_sankey_nodes.csv) · [Sankey flows](three_way_profile_sankey_flows.csv)\n",
        "\n### Detailed profile Sankeys\n",
        "![Detailed three-way profile Sankey](three_way_profile_sankey_detailed.png)\n",
        "![Detailed three-way profile Sankey, 32/6 to 8/6](three_way_profile_sankey_detailed_32_to_8.png)\n",
        f"These retain the top 50 profiles for 8/6 ({detailed_profile_coverage['k8_target6']:.1%} of documents), top 200 for 16/6 ({detailed_profile_coverage['k16_target6']:.1%}), and top 500 for 32/6 ({detailed_profile_coverage['k32_target6']:.1%}). Only P01–P20 are labelled to retain readable figures; the complete profile identifiers and counts remain in the node CSVs. The persistent `Other` mass at 32/6 is an observed diffuse profile distribution, not a plotting omission.\n",
        "\n### Reverse direction\n",
        "![Three-way profile Sankey, 32/6 to 8/6](three_way_profile_sankey_32_to_8.png)\n",
        "This is a separately ordered alluvial display, `32/6 → 16/6 → 8/6`, over the same documents; it is retained alongside the increasing-capacity direction.\n",
        "\n### Cross-capacity bit communities\n",
        "[Open the label-free three-way bit-community analysis](three_way_bit_communities.md). It begins with 8/6 Leiden-community conjunctions, traces their document sets to 16/6 and 32/6 bits, and reports closed three-bit co-activation cores.\n",
        "| Three-way statistic | Value |\n",
        "|---|---:|\n",
        f"| Distinct profile triples | {triple['joint_unique_profile_triples']:,} |\n",
        f"| Joint profile entropy | {triple['joint_entropy_bits']:.2f} bits |\n",
        f"| Total correlation | {triple['total_correlation_bits']:.2f} bits |\n",
        f"| 8/6 recoverable from 16/6 + 32/6 | {triple['k8_target6_recoverable_from_other_two_modal_accuracy']:.1%} |\n",
        f"| 16/6 recoverable from 8/6 + 32/6 | {triple['k16_target6_recoverable_from_other_two_modal_accuracy']:.1%} |\n",
        f"| 32/6 recoverable from 8/6 + 16/6 | {triple['k32_target6_recoverable_from_other_two_modal_accuracy']:.1%} |\n",
        "`Total correlation` is the non-negative reduction in profile uncertainty from observing all three assignments jointly rather than independently. `Recoverable` uses the modal profile given the ordered pair of profiles in the other two encoders.\n",
        "## Global partition agreement\n",
        "| Pair | Profiles (left / right) | NMI | AMI | ARI | Left recoverable from right | Right recoverable from left | H(left | right) | H(right | left) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['left_configuration']} vs {row['right_configuration']} | "
            f"{row['left_unique_profiles']:,} / {row['right_unique_profiles']:,} | "
            f"{row['normalized_mutual_information']:.3f} | {row['adjusted_mutual_information']:.3f} | "
            f"{row['adjusted_rand_index']:.3f} | {row['left_from_right_modal_accuracy']:.1%} | "
            f"{row['right_from_left_modal_accuracy']:.1%} | {row['left_given_right_entropy_bits']:.2f} | "
            f"{row['right_given_left_entropy_bits']:.2f} |"
        )
    lines += [
        "\nNMI measures normalized mutual information; AMI removes agreement expected from the observed number and sizes of profiles; ARI is a stricter pairwise partition agreement. Larger values indicate greater agreement. The two modal-recovery columns map every profile in one configuration to its most frequent partner in the other and report document-level recovery. Conditional entropy is residual uncertainty in bits, so lower values are better.\n",
        "## Pairwise detail retained as diagnostics\n",
        "Each linked report starts from the 20 most frequent and 10 least frequent observed profiles of each configuration. It then finds the modal counterpart profile among exactly the same documents, reports the label composition only after matching, and supplies profile, bit-pair, invariance, and optional graph artifacts. Those selected profiles are illustrative; the global table above always uses all documents and all profiles.\n",
    ]
    for left_name, right_name in PAIRS:
        rel = f"{left_name}_vs_{right_name}"
        lines.append(f"- [{left_name} vs {right_name}]({rel}/output.md)\n")
    lines += [
        "## Reproduction\n",
        "```bash\nsource .venv/bin/activate\npython compare_qwen_profile_capacities.py --graphs\n```\n",
        "The script only reads existing encoded files. It does not calculate frozen embeddings or train an encoder.\n",
    ]
    # Several strings above intentionally carry paragraph-spacing newlines.
    # Remove only empty rows *inside* a pipe table: CommonMark otherwise ends
    # the table after its header in renderers such as VS Code's preview.
    rendered = "\n".join(lines)
    raw_lines = rendered.splitlines()
    cleaned = []
    for index, line in enumerate(raw_lines):
        previous = raw_lines[index - 1] if index else ""
        following = raw_lines[index + 1] if index + 1 < len(raw_lines) else ""
        if not line.strip() and previous.lstrip().startswith("|") and following.lstrip().startswith("|"):
            continue
        cleaned.append(line)
    (args.out / "README.md").write_text("\n".join(cleaned).rstrip() + "\n")


if __name__ == "__main__":
    main()
