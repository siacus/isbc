"""Greedy profile-to-profile matching across embedding models, document by document.

For a reference model, take its 20 most frequent observed binary profiles and its 10 least
frequent observed profiles (or all profiles when fewer exist). Restrict to the documents
with each selected profile. Among the other models, find the most frequent profile within
those documents (no constraint on the number of active bits), take the model whose profile
is most frequent, restrict the documents to those that also have it, and repeat with the
remaining models until all are matched.

Each match is shown as the list of active bit indices of each model (e.g. 1,4,5,9,13,15 = bit_01, bit_04,
...). Internally a match is keyed by a reversible blockwise hexadecimal string. Within each
16-bit block, bit offsets 0..15 map to 0..F; `|` separates blocks and `-` marks an empty
block. Models remain separated by `*`, e.g. `13E|1B5*2BC|4D*...*` for 32-bit codes.
Bits that are constant (always on / always off) in a model are removed first.

For each match, the prevailing category among the documents of the reference profile and among the
documents left at the end of the chain is reported, next to its overall share. Also plots the
frequency distribution of every model's profiles.

Outputs (in --out): matches.csv, output.md, bit_pairs.csv, profile_distribution.png,
invariance.csv, invariance.png, profile_categories.csv, and manifest.json.
All models remain eligible: rank-based selection is well-defined even when profile
frequencies are diffuse.
Usage: python3 profile_matching.py [--run-dir DIR] [--reference qwen]
                                   [--top-reference-profiles 20] [--tail-reference-profiles 10]
                                   [--all-references] [--out .]
"""
import argparse
import csv
import json
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

HEX = "0123456789ABCDEF"
csv.field_size_limit(sys.maxsize)

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", default="../runs/20ng_embedding_comparison/all_fixed_balance_label_coarse")
ap.add_argument("--reference", default="qwen")
ap.add_argument("--fallback-reference", action="store_true",
                help="retained for CLI compatibility; rank selection never excludes a reference model")
ap.add_argument("--adaptive-reference-threshold", action="store_true",
                help="retained for CLI compatibility; rank selection does not use share thresholds")
ap.add_argument("--min-share", type=float, default=None,
                help="deprecated compatibility option; ignored by the default rank-based selection")
ap.add_argument("--max-reference-profiles", type=int, default=None,
                help="deprecated compatibility option; use --top-reference-profiles")
ap.add_argument("--min-top-share", type=float, default=None,
                help="deprecated compatibility option; rank selection never excludes models")
ap.add_argument("--top-reference-profiles", type=int, default=20,
                help="number of most frequent observed profiles retained per reference (default: 20)")
ap.add_argument("--tail-reference-profiles", type=int, default=10,
                help="number of least frequent observed profiles retained per reference (default: 10)")
ap.add_argument("--min-docs", type=int, default=30, help="flag matches whose final sample is smaller")
ap.add_argument("--all-references", action="store_true", help="also start from every other model")
ap.add_argument("--graphs", action="store_true", help="draw one model -> category graph per reference profile")
ap.add_argument("--second-taxonomy", default=None,
                help="with --graphs, add a third column: this taxonomy among the reference documents of each category")
ap.add_argument("--additional-label-summary", action="store_true",
                help="plot the prevalent observed class of every additional taxonomy by reference profile")
ap.add_argument("--missing-label-value", action="append", default=[],
                help="value treated as unavailable in the additional-label summary; repeat as needed")
ap.add_argument("--additional-labels-per-plot", type=int, default=8,
                help="maximum taxonomy columns per additional-label prevalence plot (default: 8)")
ap.add_argument("--multilabel-separator", default=";",
                help="separator used to auto-detect and split multi-label taxonomies (default: ';')")
ap.add_argument("--multilabel-top-k", type=int, default=10,
                help="maximum number of labels shown in multi-label profile histograms (default: 10)")
ap.add_argument("--null-draws", type=int, default=1000,
                help="draws of the profile-swap null for the invariance index (0 = skip)")
ap.add_argument("--out", default=".")
args = ap.parse_args()

report = json.load(open(Path(args.run_dir) / "factor_collection_comparison.json"))
models = report["run_order"]

# ---- load codes (documents aligned by id) -------------------------------------------------------
ids_ref, raw = None, {}
for m in models:
    with open(report["runs"][m]["encoded"]["path"]) as f:
        r = csv.reader(f)
        h = next(r)
        bit_cols = [i for i, c in enumerate(h) if c.startswith("bit_")]
        idc = h.index("id")
        ids, rows = [], []
        for row in r:
            ids.append(row[idc])
            rows.append([row[i] for i in bit_cols])
    if ids_ref is None:
        ids_ref = ids
    assert ids == ids_ref, f"document ids of {m} differ from {models[0]}"
    raw[m] = rows
N = len(ids_ref)

# ---- document categories: every taxonomy (label column) of the labels file, joined on id -----------
label_col = report["label_column"]
with open(report["labels"]["path"]) as f:
    r = csv.reader(f)
    h = next(r)
    rows_lab = {row[h.index("id")]: row for row in r}
SKIP = {"id", "source_id", "text", "source_partition"}
taxonomies = [c for c in h if c not in SKIP and not (c == "label" and "label_name" in h)]
if label_col not in taxonomies:
    taxonomies.append(label_col)
tax_cat = {t: [rows_lab[i][h.index(t)] for i in ids_ref] for t in taxonomies}
multilabel_taxonomies = [
    t for t in taxonomies
    if any(args.multilabel_separator in value for value in tax_cat[t])
]


def split_taxonomy_value(value, taxonomy):
    """Return the distinct labels carried by one document for a taxonomy."""
    if taxonomy not in multilabel_taxonomies:
        return [value]
    labels = [part.strip() for part in value.split(args.multilabel_separator) if part.strip()]
    return list(dict.fromkeys(labels)) or ["(missing)"]


tax_tokens = {
    t: [split_taxonomy_value(value, t) for value in values]
    for t, values in tax_cat.items()
}
tax_base = {
    t: Counter(label for labels in tax_tokens[t] for label in labels)
    for t in taxonomies
}
doc_cat = tax_cat[label_col]
cat_base = tax_base[label_col]


def category_profile(docs, tax, k=3):
    """Top-k document-level labels of `tax`, each as (label, share in docs, overall share)."""
    c = Counter(label for i in docs for label in tax_tokens[tax][i])
    return [(cat, v / len(docs), tax_base[tax][cat] / N) for cat, v in c.most_common(k)]


def top_category(docs):
    cat, share, base = category_profile(docs, label_col, 1)[0]
    return {"category": cat, "share": share, "base": base}


# ---- drop constant bits (always on / always off) ------------------------------------------------
kept, dropped = {}, {}
for m in models:
    k = len(raw[m][0])
    ones = [sum(row[i] == "1" for row in raw[m]) for i in range(k)]
    kept[m] = [i for i in range(k) if 0 < ones[i] < N]
    dropped[m] = [i for i in range(k) if i not in kept[m]]
codes = {m: ["".join(row[i] for i in kept[m]) for row in raw[m]] for m in models}


def binseq(m, profile):
    """Active bits of a profile as comma-separated bit indices (0 = bit_00)."""
    return ",".join(str(kept[m][j]) for j, c in enumerate(profile) if c == "1")


def active_indices(m, profile):
    """Original zero-based bit indices active in a kept-bit profile."""
    return [kept[m][j] for j, c in enumerate(profile) if c == "1"]


def profile_key(m, profile):
    """Reversible hexadecimal active-set key, grouped into 16-bit blocks."""
    n_blocks = max(1, (len(raw[m][0]) + 15) // 16)
    blocks = [[] for _ in range(n_blocks)]
    for bit in active_indices(m, profile):
        blocks[bit // 16].append(HEX[bit % 16])
    return "|".join("".join(block) or "-" for block in blocks)


marg = {m: Counter(codes[m]) for m in models}
marg_all = dict(marg)  # kept for the distribution plot, before models are excluded
top_share = {m: marg[m].most_common(1)[0][1] / N for m in models}
requested_reference = args.reference
excluded = []
if args.top_reference_profiles < 1:
    sys.exit("--top-reference-profiles must be at least 1")
if args.tail_reference_profiles < 0:
    sys.exit("--tail-reference-profiles cannot be negative")


def model_cat(m, profile):
    """Documents with this profile in model m: count and label counts (run's label column)."""
    docs = [i for i, c in enumerate(codes[m]) if c == profile]
    return {
        "n": len(docs),
        "cats": Counter(label for i in docs for label in tax_tokens[label_col][i]),
    }


def match_from(ref, profile, selection_group, selection_rank):
    docs = [i for i, c in enumerate(codes[ref]) if c == profile]
    n_start = len(docs)
    ref_docs = list(docs)
    cat_start = top_category(docs)
    tax_start = {t: category_profile(docs, t) for t in taxonomies}
    matched = {ref: profile}
    steps = []
    remaining = [m for m in models if m != ref]
    while remaining:
        best = None
        for m in remaining:
            c = Counter(codes[m][i] for i in docs)
            prof, cnt = c.most_common(1)[0]
            share = cnt / len(docs)
            lift = share / (marg[m][prof] / N)
            key = (share, lift)
            if best is None or key > best[0]:
                best = (key, m, prof, cnt)
        (share, lift), m, prof, cnt = best
        docs = [i for i in docs if codes[m][i] == prof]
        matched[m] = prof
        steps.append({"model": m, "share": share, "lift": lift, "docs_left": len(docs)})
        remaining.remove(m)
    string = "".join(profile_key(m, matched[m]) + "*" for m in models)
    return {"reference": ref, "string": string,
            "selection_group": selection_group, "selection_rank": selection_rank,
            "profiles": {m: binseq(m, matched[m]) for m in models}, "start_docs": n_start, "final_docs": len(docs),
            "order": [s["model"] for s in steps], "steps": steps, "matched": matched,
            "cat_start": cat_start, "cat_final": top_category(docs) if docs else None,
            "ref_docs": ref_docs, "final_doc_indices": list(docs),
            "model_cats": {m: model_cat(m, matched[m]) for m in models},
            "tax_start": tax_start, "tax_final": {t: category_profile(docs, t) for t in taxonomies} if docs else None}


refs = models if args.all_references else [args.reference]
results = []
for ref in refs:
    ranked_desc = marg[ref].most_common()
    top = ranked_desc[:args.top_reference_profiles]
    selected = [("top", rank, prof) for rank, (prof, _cnt) in enumerate(top, 1)]
    selected_keys = {prof for prof, _cnt in top}
    ranked_asc = sorted(marg[ref].items(), key=lambda item: (item[1], item[0]))
    tail = [item for item in ranked_asc if item[0] not in selected_keys][:args.tail_reference_profiles]
    selected.extend(("tail", rank, prof) for rank, (prof, _cnt) in enumerate(tail, 1))
    for selection_group, selection_rank, prof in selected:
        results.append(match_from(ref, prof, selection_group, selection_rank))

# ---- write matches.csv ---------------------------------------------------------------------------
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
with open(out / "matches.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["reference", "selection_group", "selection_rank", "profiles", "start_docs", "start_share", "final_docs", "weak", "order",
                "step_shares", "step_lifts", "start_top_category", "start_top_share",
                "start_category_base", "final_top_category", "final_top_share", "final_category_base"])
    for r in results:
        w.writerow([r["reference"], r["selection_group"], r["selection_rank"], "; ".join(f"{m}={p}" for m, p in r["profiles"].items()), r["start_docs"], f"{r['start_docs'] / N:.4f}", r["final_docs"],
                    int(r["final_docs"] < args.min_docs), ">".join(r["order"]),
                    ";".join(f"{s['share']:.2f}" for s in r["steps"]),
                    ";".join(f"{s['lift']:.1f}" for s in r["steps"]),
                    r["cat_start"]["category"], f"{r['cat_start']['share']:.3f}", f"{r['cat_start']['base']:.3f}",
                    r["cat_final"]["category"], f"{r['cat_final']['share']:.3f}", f"{r['cat_final']['base']:.3f}"])

with open(out / "profile_categories.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["reference", "selection_group", "selection_rank", "profiles", "sample", "taxonomy", "rank", "category", "share", "overall_share"])
    for r in results:
        for sample, key in (("reference_docs", "tax_start"), ("final_docs", "tax_final")):
            for t, lst in (r[key] or {}).items():
                for rank, (cat, share, base) in enumerate(lst, 1):
                    w.writerow([r["reference"], r["selection_group"], r["selection_rank"], "; ".join(f"{m}={p}" for m, p in r["profiles"].items()), sample, t, rank, cat, f"{share:.3f}", f"{base:.3f}"])


def label_share(docs, taxonomy, label):
    """Share of documents containing `label`; multi-label shares need not sum to one."""
    if not docs:
        return None
    return sum(label in tax_tokens[taxonomy][i] for i in docs) / len(docs)


multilabel_rows = []
for profile_rank, r in enumerate(
    [row for row in results if row["reference"] == args.reference], 1
):
    start_docs = r["ref_docs"]
    final_docs = r["final_doc_indices"]
    final_set = set(final_docs)
    residual_docs = [i for i in start_docs if i not in final_set]
    for taxonomy in multilabel_taxonomies:
        counts = Counter(label for i in start_docs for label in tax_tokens[taxonomy][i])
        ranked_labels = sorted(counts, key=lambda label: (-counts[label], label))
        for label_rank, label in enumerate(ranked_labels, 1):
            multilabel_rows.append({
                "profile_rank": profile_rank,
                "reference": r["reference"],
                "profiles": "; ".join(f"{m}={p}" for m, p in r["profiles"].items()),
                "taxonomy": taxonomy,
                "label_rank": label_rank,
                "label": label,
                "start_docs": len(start_docs),
                "final_docs": len(final_docs),
                "residual_docs": len(residual_docs),
                "start_share": label_share(start_docs, taxonomy, label),
                "final_share": label_share(final_docs, taxonomy, label),
                "residual_share": label_share(residual_docs, taxonomy, label),
            })
if multilabel_rows:
    with open(out / "profile_multilabel_distributions.csv", "w", newline="") as f:
        fields = list(multilabel_rows[0])
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for row in multilabel_rows:
            w.writerow({
                key: ("" if value is None else f"{value:.6f}" if key.endswith("_share") else value)
                for key, value in row.items()
            })

# ---- tabulation 1: reciprocity (same string reached from different starting models) ---------------
by_string = defaultdict(list)
for r in results:
    by_string[r["string"]].append(r["reference"])

# ---- tabulation 2: bit pairs, reference bit x model bit, over matches from --reference ------------
pair_rows = []
ref_results = [r for r in results if r["reference"] == args.reference]

# ---- optional additional-label prevalence summary ----------------------------------------------
missing_label_values = {value.strip().casefold() for value in args.missing_label_value}
additional_taxonomies = [
    taxonomy
    for taxonomy in taxonomies
    if taxonomy not in {label_col, args.second_taxonomy}
]


def observed_taxonomy_summary(docs, taxonomy):
    """Dominant class conditional on this taxonomy being observed in a document."""
    observed = []
    for index in docs:
        values = [
            value
            for value in tax_tokens[taxonomy][index]
            if value.strip().casefold() not in missing_label_values
        ]
        if values:
            observed.append(values)
    if not observed:
        return None
    counts = Counter(value for values in observed for value in values)
    dominant, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return {
        "dominant_class": dominant,
        "dominant_share": count / len(observed),
        "observed_docs": len(observed),
    }


additional_overall = {
    taxonomy: observed_taxonomy_summary(range(N), taxonomy)
    for taxonomy in additional_taxonomies
}
additional_taxonomies = [
    taxonomy for taxonomy in additional_taxonomies if additional_overall[taxonomy]
]
additional_label_rows = []
if args.additional_label_summary:
    for profile_rank, result in enumerate(ref_results, 1):
        for taxonomy in additional_taxonomies:
            summary = observed_taxonomy_summary(result["ref_docs"], taxonomy)
            overall = additional_overall[taxonomy]
            additional_label_rows.append({
                "profile_rank": profile_rank,
                "reference": result["reference"],
                "start_docs": result["start_docs"],
                "start_share": result["start_docs"] / N,
                "taxonomy": taxonomy,
                "dominant_class": summary["dominant_class"] if summary else "",
                "dominant_share_observed": summary["dominant_share"] if summary else None,
                "observed_docs": summary["observed_docs"] if summary else 0,
                "observed_coverage": (
                    summary["observed_docs"] / result["start_docs"] if summary else 0.0
                ),
                "overall_dominant_class": overall["dominant_class"],
                "overall_dominant_share_observed": overall["dominant_share"],
                "overall_observed_docs": overall["observed_docs"],
            })
    with open(out / "profile_additional_label_prevalence.csv", "w", newline="") as f:
        fields = list(additional_label_rows[0]) if additional_label_rows else []
        if fields:
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for row in additional_label_rows:
                writer.writerow({
                    key: (
                        "" if value is None else
                        f"{value:.6f}" if isinstance(value, float) else value
                    )
                    for key, value in row.items()
                })

for m in models:
    if m == args.reference:
        continue
    cnt_i = Counter()
    cnt_ij = Counter()
    cnt_j = Counter()
    for r in ref_results:
        P = active_indices(args.reference, r["matched"][args.reference])
        X = active_indices(m, r["matched"][m])
        for i in P:
            cnt_i[i] += 1
        for j in X:
            cnt_j[j] += 1
        for i in P:
            for j in X:
                cnt_ij[(i, j)] += 1
    n = len(ref_results)
    for i in sorted(cnt_i):
        ranked = sorted(((cnt_ij[(i, j)] / cnt_i[i], j) for j in sorted(cnt_j)
                         if cnt_ij[(i, j)]), reverse=True)
        top = ranked[:2]
        # A profile chain can have an empty final intersection.  Such a chain
        # contains no observed co-activation for this reference bit; preserve
        # that fact as an explicit missing partner rather than indexing an
        # empty ranking and aborting the full directional analysis.
        if not top:
            pair_rows.append({"model": m, "ref_bit": i, "profiles_with_bit": cnt_i[i],
                              "top_partner": "", "top_fraction": 0.0,
                              "top_partner_baseline": 0.0,
                              "second_partner": "", "second_fraction": 0.0})
            continue
        pair_rows.append({"model": m, "ref_bit": i, "profiles_with_bit": cnt_i[i],
                          "top_partner": top[0][1], "top_fraction": top[0][0],
                          "top_partner_baseline": cnt_j[top[0][1]] / n,
                          "second_partner": top[1][1] if len(top) > 1 else "",
                          "second_fraction": top[1][0] if len(top) > 1 else 0.0})
with open(out / "bit_pairs.csv", "w", newline="") as f:
    pair_fields = ["model", "ref_bit", "profiles_with_bit", "top_partner", "top_fraction",
                   "top_partner_baseline", "second_partner", "second_fraction"]
    w = csv.DictWriter(f, fieldnames=pair_fields, lineterminator="\n")
    w.writeheader()
    w.writerows(pair_rows)

# ---- profile frequency distributions (all models, before exclusion) ----------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
plt.rcParams.update({"font.size": 9, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
for k, m in enumerate(report["run_order"]):
    freqs = sorted((v / N for v in marg_all[m].values()), reverse=True)
    ls = "--" if m in excluded else "-"
    a1.plot(range(1, len(freqs) + 1), [f * 100 for f in freqs], ls, color=PAL[k], lw=2,
            label=f"{m}{' (excluded)' if m in excluded else ''}")
    cum, tot = [], 0.0
    for f in freqs:
        tot += f
        cum.append(tot * 100)
    a2.plot(range(1, len(freqs) + 1), cum, ls, color=PAL[k], lw=2)
a1.axvline(args.top_reference_profiles, color="#8a8984", lw=0.8, ls=":")
a1.text(args.top_reference_profiles * 1.05, 0.9, f"top {args.top_reference_profiles}",
        transform=a1.get_xaxis_transform(), color="#52514e", fontsize=8, va="top")
a1.set_xscale("log"); a1.set_yscale("log")
a1.set_xlabel("profile rank"); a1.set_ylabel("% of documents")
a1.set_title("Profile frequency by rank", loc="left", fontsize=10)
a1.legend(frameon=False, fontsize=8)
a2.set_xscale("log")
a2.set_xlabel("number of most frequent profiles"); a2.set_ylabel("cumulative % of documents")
a2.set_title("Coverage by the most frequent profiles", loc="left", fontsize=10)
for a in (a1, a2):
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    a.grid(color="#e6e5e0", lw=0.6)
    a.set_axisbelow(True)
fig.tight_layout()
fig.savefig(out / "profile_distribution.png", dpi=200)
plt.close(fig)

# ---- invariance index per profile, with a profile-swap null ----------------------------------------------
import numpy as np

additional_label_plot_files = []
if args.additional_label_summary and additional_label_rows:
    lookup = {
        (row["profile_rank"], row["taxonomy"]): row
        for row in additional_label_rows
    }
    group_size = max(1, args.additional_labels_per_plot)
    for group_index, start in enumerate(
        range(0, len(additional_taxonomies), group_size), 1
    ):
        group = additional_taxonomies[start : start + group_size]
        values = np.full((len(ref_results), len(group)), np.nan, dtype=float)
        for row_index in range(len(ref_results)):
            for column_index, taxonomy in enumerate(group):
                row = lookup[(row_index + 1, taxonomy)]
                if row["dominant_share_observed"] is not None:
                    values[row_index, column_index] = row["dominant_share_observed"]

        width = max(8.0, 2.35 * len(group))
        height = max(5.0, 0.72 * len(ref_results) + 2.8)
        fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
        cmap = plt.get_cmap("YlGnBu").copy()
        cmap.set_bad("#efeee9")
        image = ax.imshow(
            np.ma.masked_invalid(values), cmap=cmap, vmin=0.0, vmax=1.0,
            aspect="auto", interpolation="nearest",
        )
        xlabels = []
        for taxonomy in group:
            overall = additional_overall[taxonomy]
            overall_class = str(overall["dominant_class"])
            if len(overall_class) > 18:
                overall_class = overall_class[:17] + "…"
            xlabels.append(
                f"{taxonomy}\noverall: {overall_class} "
                f"{overall['dominant_share']:.0%}"
            )
        ylabels = [
            f"P{rank:02d} ({result['start_docs'] / N:.1%}; n={result['start_docs']:,})"
            for rank, result in enumerate(ref_results, 1)
        ]
        ax.set_xticks(np.arange(len(group)), labels=xlabels, rotation=32, ha="left")
        ax.xaxis.tick_top()
        ax.set_yticks(np.arange(len(ref_results)), labels=ylabels)
        ax.tick_params(axis="both", labelsize=8)
        for row_index in range(len(ref_results)):
            for column_index, taxonomy in enumerate(group):
                row = lookup[(row_index + 1, taxonomy)]
                share = row["dominant_share_observed"]
                if share is None:
                    annotation = "—"
                    color = "#777570"
                else:
                    category = str(row["dominant_class"])
                    if len(category) > 18:
                        category = category[:17] + "…"
                    annotation = (
                        f"{textwrap.fill(category, width=12)}\n"
                        f"{share:.0%} · n={row['observed_docs']:,}"
                    )
                    color = "white" if share >= 0.58 else "#171714"
                ax.text(
                    column_index, row_index, annotation,
                    ha="center", va="center", fontsize=6.5, color=color,
                )
        ax.set_xticks(np.arange(-0.5, len(group), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(ref_results), 1), minor=True)
        ax.grid(which="minor", color="#fcfcfb", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_title(
            "Additional labels by reference profile\n"
            "cell = prevalent class, share among observed labels, and observed n; "
            "grey = unavailable",
            loc="left", fontsize=11, color="#52514e", pad=18,
        )
        colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
        colorbar.set_label("prevalent-class share among observed labels", fontsize=8)
        filename = f"additional_label_prevalence_{group_index:02d}.png"
        fig.savefig(out / filename, dpi=180)
        plt.close(fig)
        additional_label_plot_files.append(filename)

inv_rows = []
invariance_taxonomies = [t for t in taxonomies if t not in multilabel_taxonomies]
if args.null_draws > 0 and invariance_taxonomies:
    rng_inv = np.random.default_rng(0)
    by_profile = {m: defaultdict(list) for m in models}
    for m in models:
        for i, c in enumerate(codes[m]):
            by_profile[m][c].append(i)
    by_profile = {m: {p: np.array(v) for p, v in d_.items()} for m, d_ in by_profile.items()}
    tax_arr, tax_K, tax_base_p = {}, {}, {}
    for t in invariance_taxonomies:
        cats_t = sorted(tax_base[t])
        idx_t = {c: k for k, c in enumerate(cats_t)}
        tax_arr[t] = np.array([idx_t[c] for c in tax_cat[t]])
        tax_K[t] = len(cats_t)
        tax_base_p[t] = np.bincount(tax_arr[t], minlength=tax_K[t]) / N
    dist_cache = {}

    def dist_of(m, profile, t):
        key = (m, profile, t)
        if key not in dist_cache:
            docs_p = by_profile[m][profile]
            dist_cache[key] = np.bincount(tax_arr[t][docs_p], minlength=tax_K[t]) / len(docs_p)
        return dist_cache[key]

    def entropy(p):
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    def jsd(p, q):
        return entropy((p + q) / 2) - (entropy(p) + entropy(q)) / 2

    def coherence(dists):
        pairs = [(a, b) for k, a in enumerate(dists) for b in dists[k + 1:]]
        return 1 - float(np.mean([jsd(a, b) for a, b in pairs]))

    def kl_to_base(p, t):
        mask = p > 0
        return float((p[mask] * np.log2(p[mask] / tax_base_p[t][mask])).sum())

    ref_profile_models = [m for m in models if m != args.reference]

    # Chain null: shuffle the document alignment of every non-reference model independently, rerun the
    # whole chain (so the selection of the most frequent profile is part of the null), and keep the
    # profiles it picks; their category composition is then read on the real, unshuffled documents.
    profile_list = {m: list(by_profile[m]) for m in models}
    pid_of = {m: {p: k for k, p in enumerate(profile_list[m])} for m in models}
    ids_arr = {m: np.array([pid_of[m][c] for c in codes[m]]) for m in models}
    prof_n = {m: np.array([len(by_profile[m][p]) for p in profile_list[m]]) for m in models}
    ref_docs_arr = [np.array(r["ref_docs"]) for r in ref_results]
    chain_picks = [[] for _ in ref_results]  # per reference profile: list of {model: profile string}
    rng_chain = np.random.default_rng(1)
    for b in range(args.null_draws):
        ids_p = {args.reference: ids_arr[args.reference]}
        for m in ref_profile_models:
            ids_p[m] = ids_arr[m][rng_chain.permutation(N)]
        for k, docs0 in enumerate(ref_docs_arr):
            docs, remaining, pick = docs0, list(ref_profile_models), {}
            while remaining:
                best = None
                for m in remaining:
                    cnt = np.bincount(ids_p[m][docs], minlength=len(profile_list[m]))
                    pid = int(cnt.argmax())
                    share = cnt[pid] / len(docs)
                    key = (share, share / (prof_n[m][pid] / N))
                    if best is None or key > best[0]:
                        best = (key, m, pid)
                _, m, pid = best
                docs = docs[ids_p[m][docs] == pid]
                pick[m] = profile_list[m][pid]
                remaining.remove(m)
            chain_picks[k].append(pick)
    cand = {}
    for m in models:
        cand[m] = sorted(by_profile[m], key=lambda p: len(by_profile[m][p]))
    for rk, r in enumerate(ref_results):
        n_prof = {m: len(by_profile[m][r["matched"][m]]) for m in models}
        pools = {}
        for m in ref_profile_models:
            n0 = n_prof[m]
            pool = [p for p in cand[m] if p != r["matched"][m] and 0.5 * n0 <= len(by_profile[m][p]) <= 2 * n0]
            if len(pool) < 10:
                pool = sorted((p for p in cand[m] if p != r["matched"][m]),
                              key=lambda p: abs(len(by_profile[m][p]) - n0))[:10]
            pools[m] = pool
        for t in invariance_taxonomies:
            obs_d = [dist_of(m, r["matched"][m], t) for m in models]
            c_obs = coherence(obs_d)
            info = float(np.mean([kl_to_base(p, t) for p in obs_d]))
            # When a matched non-reference model has no alternative observed
            # profile, a profile-swap null is not identifiable.  Preserve the
            # observed and chain-null diagnostics, and mark swap quantities as
            # unavailable rather than silently resampling the observed profile.
            swap_available = all(len(pools[m]) > 0 for m in ref_profile_models)
            if swap_available:
                null = np.empty(args.null_draws)
                for b in range(args.null_draws):
                    ds = [dist_of(args.reference, r["matched"][args.reference], t)]
                    for m in ref_profile_models:
                        pool = pools[m]
                        ds.append(dist_of(m, pool[rng_inv.integers(len(pool))], t))
                    null[b] = coherence(ds)
                swap_mean = float(null.mean())
                swap_lo, swap_hi = (float(value) for value in np.quantile(null, [0.025, 0.975]))
                swap_p = float((1 + (null >= c_obs).sum()) / (1 + args.null_draws))
            else:
                swap_mean = swap_lo = swap_hi = swap_p = float("nan")
            null_c = np.array([coherence([dist_of(args.reference, r["matched"][args.reference], t)]
                                         + [dist_of(m, pk[m], t) for m in ref_profile_models])
                               for pk in chain_picks[rk]])
            inv_rows.append({"taxonomy": t, "selection_group": r["selection_group"],
                             "selection_rank": r["selection_rank"],
                             "start_share": r["start_docs"] / N, "coherence": c_obs,
                             "information": info, "swap_available": swap_available,
                             "swap_candidate_counts": ";".join(
                                 f"{m}:{len(pools[m])}" for m in ref_profile_models
                             ),
                             "null_draws": int(args.null_draws),
                             "swap_mean": swap_mean, "swap_lo": swap_lo,
                             "swap_hi": swap_hi, "swap_p": swap_p,
                             "chain_mean": float(null_c.mean()),
                             "chain_lo": float(np.quantile(null_c, 0.025)),
                             "chain_hi": float(np.quantile(null_c, 0.975)),
                             "chain_p": float((1 + (null_c >= c_obs).sum()) / (1 + args.null_draws)),
                             "string": r["string"]})
    with open(out / "invariance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(inv_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(inv_rows)

    # plot: observed coherence against the null interval, per taxonomy
    nt = len(invariance_taxonomies)
    fig, axes = plt.subplots(1, nt, figsize=(4.2 * nt, 0.34 * len(ref_results) + 1.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, invariance_taxonomies):
        rows_t = [x for x in inv_rows if x["taxonomy"] == t]
        for k, x in enumerate(rows_t):
            if x["swap_available"]:
                ax.plot([x["swap_lo"], x["swap_hi"]], [k - 0.17, k - 0.17], color="#c9c8c1", lw=4, solid_capstyle="round")
            ax.plot([x["chain_lo"], x["chain_hi"]], [k + 0.17, k + 0.17], color="#6f6e69", lw=4, solid_capstyle="round")
            ax.scatter([x["coherence"]], [k], s=28, zorder=3,
                       color="#2a78d6" if x["chain_p"] < 0.05 else "#e34948")
        ax.set_yticks(range(len(rows_t)))
        ax.set_yticklabels(
            [f"{x['selection_group'][0].upper()}{x['selection_rank']}: {x['start_share']:.1%}" for x in rows_t],
            fontsize=7,
        )
        ax.invert_yaxis()
        ax.set_title(t, loc="left", fontsize=9)
        ax.set_xlabel("coherence (1 - mean JSD)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="x", color="#e6e5e0", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("reference profile (share of documents)")
    fig.suptitle("Observed coherence (dot: blue if p < 0.05 against the chain null, red otherwise); "
                 "bars 2.5-97.5%: light grey = profile-swap null, dark grey = chain null", x=0.01, ha="left", fontsize=9, color="#52514e")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "invariance.png", dpi=170)
    plt.close(fig)

# ---- graphs: model profile -> category, one per reference profile --------------------------------------
if args.graphs:
    from matplotlib.path import Path as MPath
    gdir = out / "graphs"
    gdir.mkdir(exist_ok=True)
    PAL8 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    cats_sorted = sorted(cat_base)
    CATCOL = {c: PAL8[k % len(PAL8)] for k, c in enumerate(cats_sorted)}
    CATCOL["other"] = "#b8b7b0"
    MIN_SHARE = 0.10
    tax2 = args.second_taxonomy
    if tax2 and tax2 not in tax_cat:
        sys.exit(f"unknown taxonomy {tax2}; available: {taxonomies}")

    def bezier(ax, x0, y0, x1, y1, color, v):
        xm = (x0 + x1) / 2
        path = MPath([(x0, y0), (xm, y0), (xm, y1), (x1, y1)],
                     [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4])
        ax.add_patch(matplotlib.patches.PathPatch(path, fc="none", ec=color, lw=1 + 13 * v, alpha=0.75,
                                                  capstyle="butt"))

    def edge_label(ax, x, y, v, compact):
        if not compact and v >= 0.05:
            ax.text(x, y, f"{v:.0%}", fontsize=7, color="#52514e", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.12", fc="#fcfcfb", ec="none", alpha=0.85))

    def draw_multilabel_graph(ax, r, compact=False):
        """Profile-to-label flows plus final/residual document-frequency bars."""
        mc = r["model_cats"]
        start_docs = r["ref_docs"]
        final_docs = r["final_doc_indices"]
        final_set = set(final_docs)
        residual_docs = [i for i in start_docs if i not in final_set]
        start_counts = Counter(
            label for i in start_docs for label in tax_tokens[label_col][i]
        )
        shown = sorted(start_counts, key=lambda label: (-start_counts[label], label))[
            : args.multilabel_top_k
        ]
        mods = list(mc)
        ym = {m: 1 - (k + 0.5) / len(mods) for k, m in enumerate(mods)}
        yc = {c: 1 - (k + 0.5) / len(shown) for k, c in enumerate(shown)}
        x_model, x_label, x_bar = 0.18, 0.54, 0.82
        bar_scale = 0.28
        ms = 32 if compact else 65
        fs = 5.2 if compact else 7.5

        for m in mods:
            n = mc[m]["n"]
            for c in shown:
                v = mc[m]["cats"].get(c, 0) / n
                if v < 0.01:
                    continue
                bezier(ax, x_model, ym[m], x_label, yc[c], CATCOL[c], v)
        for m in mods:
            label = (
                f"{m}\n{binseq(m, r['matched'][m])}\nn={mc[m]['n']}"
                if not compact
                else m
            )
            ax.text(x_model - 0.01, ym[m], label, ha="right", va="center", fontsize=fs)
            ax.scatter([x_model], [ym[m]], s=ms, color="#52514e", zorder=3)

        bar_height = min(0.026, 0.22 / max(1, len(shown)))
        for c in shown:
            y = yc[c]
            final_share = label_share(final_docs, label_col, c) or 0.0
            residual_share = label_share(residual_docs, label_col, c) or 0.0
            ax.scatter([x_label], [y], s=ms, color=CATCOL[c], zorder=3)
            base = cat_base[c] / N
            text = c if compact else f"{c} ({base:.0%} overall)"
            ax.text(x_label + 0.015, y, text, ha="left", va="center", fontsize=fs)
            ax.add_patch(
                matplotlib.patches.Rectangle(
                    (x_bar, y + 0.006), bar_scale * final_share, bar_height,
                    color="#1baf7a", alpha=0.88,
                )
            )
            ax.add_patch(
                matplotlib.patches.Rectangle(
                    (x_bar, y - bar_height - 0.006), bar_scale * residual_share, bar_height,
                    color="#eb6834", alpha=0.82,
                )
            )
            if not compact:
                ax.text(x_bar + bar_scale * final_share + 0.008, y + 0.006 + bar_height / 2,
                        f"{final_share:.0%}", va="center", fontsize=6.5, color="#52514e")
                ax.text(x_bar + bar_scale * residual_share + 0.008, y - 0.006 - bar_height / 2,
                        f"{residual_share:.0%}", va="center", fontsize=6.5, color="#52514e")
        if not compact:
            ax.text(x_bar, 1.035, f"matched final (n={len(final_docs)})", color="#1baf7a",
                    fontsize=7, ha="left")
            ax.text(x_bar + 0.18, 1.035, f"residual (n={len(residual_docs)})", color="#eb6834",
                    fontsize=7, ha="left")
            ax.text(x_bar, -0.04,
                    "bars are % of documents containing each label; totals may exceed 100%",
                    fontsize=6.5, color="#52514e", ha="left")
        ax.set_xlim(-0.32, 1.25)
        ax.set_ylim(-0.07, 1.08)
        ax.axis("off")

    def draw_graph(ax, r, compact=False):
        if label_col in multilabel_taxonomies:
            draw_multilabel_graph(ax, r, compact=compact)
            return
        mc = r["model_cats"]
        shown = sorted({c for m in mc for c, v in mc[m]["cats"].items() if v / mc[m]["n"] >= MIN_SHARE},
                       key=lambda c: -sum(mc[m]["cats"].get(c, 0) / mc[m]["n"] for m in mc))
        has_other = any(sum(v for c, v in mc[m]["cats"].items() if c not in shown) / mc[m]["n"] >= 0.01 for m in mc)
        right = shown + (["other"] if has_other else [])
        mods = list(mc)
        xs = (0.20, 0.55, 0.90) if tax2 else (0.28, 0.72)
        ym = {m: 1 - (k + 0.5) / len(mods) for k, m in enumerate(mods)}
        yc = {c: 1 - (k + 0.5) / len(right) for k, c in enumerate(right)}
        ms = 40 if compact else 70
        fs = 6 if compact else 8
        for m in mods:
            n = mc[m]["n"]
            for c in right:
                v = (sum(x for k2, x in mc[m]["cats"].items() if k2 not in shown) if c == "other"
                     else mc[m]["cats"].get(c, 0)) / n
                if v < 0.01:
                    continue
                bezier(ax, xs[0], ym[m], xs[1], yc[c], CATCOL[c], v)
                edge_label(ax, xs[0] + (xs[1] - xs[0]) * 0.22, ym[m] + (yc[c] - ym[m]) * 0.12, v, compact)
        for m in mods:
            lab = f"{m}\n{binseq(m, r['matched'][m])}\nn={mc[m]['n']}" if not compact else m
            ax.text(xs[0] - 0.01, ym[m], lab, ha="right", va="center", fontsize=fs, color="#0b0b0b")
            ax.scatter([xs[0]], [ym[m]], s=ms, color="#52514e", zorder=3)
        # optional third column: second taxonomy, among the reference documents of each category
        right_x = xs[1]
        if tax2:
            ref_docs = r["ref_docs"]
            dist = {}
            for c in shown:
                docs_c = [i for i in ref_docs if doc_cat[i] == c]
                if docs_c:
                    cnt = Counter(label for i in docs_c for label in tax_tokens[tax2][i])
                    dist[c] = {k2: v / len(docs_c) for k2, v in cnt.items()}
            tb = tax_base[tax2]
            # fixed order by overall frequency, so the third column reads the same in every graph
            shown2 = sorted({k2 for d_ in dist.values() for k2, v in d_.items() if v >= MIN_SHARE},
                            key=lambda k2: -tb[k2])
            right2 = shown2 + (["other"] if any(sum(v for k2, v in d_.items() if k2 not in shown2) >= 0.01
                                                for d_ in dist.values()) else [])
            y2 = {k2: 1 - (k + 0.5) / len(right2) for k, k2 in enumerate(right2)}
            for c, d_ in dist.items():
                for k2 in right2:
                    v = (sum(x for q, x in d_.items() if q not in shown2) if k2 == "other" else d_.get(k2, 0))
                    if v < 0.01:
                        continue
                    bezier(ax, xs[1], yc[c], xs[2], y2[k2], CATCOL[c], v)
                    edge_label(ax, xs[1] + (xs[2] - xs[1]) * 0.22, yc[c] + (y2[k2] - yc[c]) * 0.12, v, compact)
            for k2 in right2:
                base = f" ({tb[k2] / N:.0%} overall)" if k2 != "other" else ""
                ax.text(xs[2] + 0.035, y2[k2], f"{k2}{base}" if not compact else k2, ha="left", va="center",
                        fontsize=fs, color="#0b0b0b")
                ax.scatter([xs[2]], [y2[k2]], s=ms, marker="s", color="#52514e", zorder=3)
        for c in right:
            base = f" ({cat_base[c] / N:.0%} overall)" if c != "other" else ""
            if tax2:
                ax.text(xs[1], yc[c] + 0.035, f"{c}{base}" if not compact else c,
                        ha="center", va="bottom", fontsize=fs, color="#0b0b0b")
            else:
                ax.text(xs[1] + 0.02, yc[c], f"{c}{base}" if not compact else c, ha="left", va="center",
                        fontsize=fs, color="#0b0b0b")
            ax.scatter([xs[1]], [yc[c]], s=ms, color=CATCOL[c], zorder=3)
        ax.set_xlim(-0.30, 1.55 if tax2 else 1.2)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ref_res = [r for r in results if r["reference"] == args.reference]
    if label_col in multilabel_taxonomies:
        sub = (
            f"up to {args.multilabel_top_k} {label_col} labels; bars compare matched final "
            "with residual documents"
        )
    else:
        sub = (f"documents of each {label_col} category are then split by {tax2}" if tax2 else
               "edge width = share of the model's profile documents in the category")
    for k, r in enumerate(ref_res, 1):
        fig_width = 12 if label_col in multilabel_taxonomies else (11 if tax2 else 8)
        fig, ax = plt.subplots(figsize=(fig_width, 1.6 + 0.55 * len(r["model_cats"])))
        draw_graph(ax, r)
        ax.set_title(f"{r['selection_group'].title()} profile {r['selection_rank']}: "
                     f"{r['start_docs'] / N:.1%} of documents  ({sub})",
                     loc="left", fontsize=9, color="#52514e")
        fig.tight_layout()
        fig.savefig(gdir / f"graph_{k:02d}.png", dpi=170)
        plt.close(fig)
    cols = 2 if label_col in multilabel_taxonomies else (3 if tax2 else 4)
    rows_n = -(-len(ref_res) // cols)
    cell_width = 6.2 if label_col in multilabel_taxonomies else (5.2 if tax2 else 3.6)
    fig, axes = plt.subplots(rows_n, cols, figsize=(cell_width * cols, 2.8 * rows_n))
    axes = np.atleast_1d(axes)
    for ax, (k, r) in zip(axes.flat, enumerate(ref_res, 1)):
        draw_graph(ax, r, compact=True)
        ax.set_title(f"{r['selection_group'][0].upper()}{r['selection_rank']}: "
                     f"{r['start_docs'] / N:.1%}", loc="left", fontsize=8)
    for ax in list(axes.flat)[len(ref_res):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(gdir / "contact_sheet.png", dpi=150)
    plt.close(fig)

# ---- output.md -----------------------------------------------------------------------------------
L = [f"# Profile-to-profile matching ({Path(args.run_dir).name})\n",
     f"{N:,} documents (ids identical across models), models in order: {', '.join(models)}. "
     f"Reference: `{args.reference}`" + (" (and every other model)" if args.all_references else "") +
     f"; selected reference profiles: {len(ref_results)} "
     f"({args.top_reference_profiles} most frequent plus {args.tail_reference_profiles} least frequent observed profiles, without overlap).\n",
     "Most frequent profile per model: " + ", ".join(f"{m} {top_share[m]:.1%}" for m in top_share) +
     ". No model is excluded on profile prevalence under rank-based selection.\n",
     "Constant bits removed before matching: " +
     (", ".join(f"{m}: {dropped[m]}" for m in models if dropped[m]) or "none") + ".\n",
     "## Profile frequency distributions\n",
     "![Profile frequency distributions](profile_distribution.png)\n",
     "## Method\n",
     "For each reference profile P: restrict to the documents with P; among the other models take the "
     "most frequent profile within those documents (no constraint on the number of active bits), pick the "
     "model whose profile is most frequent (share of the current sample), restrict the sample to documents "
     "that also have it, and repeat. Each match is shown as the active bits of each model "
     "(comma-separated bit indices, 0 = bit_00). `lift` is the share divided by "
     "the profile's overall frequency in that model. Matches whose final sample has fewer than "
     f"{args.min_docs} documents are marked weak.\n",
     "## Matches from the reference\n"]


def render(r):
    return "<br>".join(f"{m} `{p}`" for m, p in r["profiles"].items())


def fmt_cat(c):
    return f"{c['category']} {c['share']:.0%} (overall {c['base']:.0%})"

def append_match_table(title, rows):
    L.extend([
        f"### {title}\n",
        "| Rank | Start share | Docs | Matched profiles (active bits) | Match order | Share at each step | Lift at each step | Final docs | Top category, reference docs | Top category, final docs |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ])
    for r in rows:
        L.append(f"| {r['selection_rank']} | {r['start_docs'] / N:.1%} | {r['start_docs']} | {render(r)} | {' > '.join(r['order'])} | "
                 + " / ".join(f"{s['share']:.2f}" for s in r["steps"]) + " | "
                 + " / ".join(f"{s['lift']:.0f}" for s in r["steps"]) + f" | {r['final_docs']}"
                 + (" (weak)" if r["final_docs"] < args.min_docs else "")
                 + f" | {fmt_cat(r['cat_start'])} | {fmt_cat(r['cat_final'])} |")
    L.append("")


top_ref_results = [r for r in ref_results if r["selection_group"] == "top"]
tail_ref_results = [r for r in ref_results if r["selection_group"] == "tail"]
append_match_table(f"Top {len(top_ref_results)} profiles by frequency", top_ref_results)
append_match_table(f"Tail {len(tail_ref_results)} least frequent observed profiles", tail_ref_results)
if inv_rows:
    L += ["\n## Invariance index per profile, against a profile-swap null\n",
          "For each reference profile and taxonomy: **coherence** = 1 - mean Jensen-Shannon divergence (bits) "
          "between the category distributions of the documents having each matched model's profile "
          "(1 = identical composition); **information** = mean KL divergence (bits) of those distributions "
          "from the overall category distribution, i.e. how much the profile says about the category. "
          f"Two nulls, {args.null_draws} draws each. **Swap**: the profiles of the non-reference models are "
          "replaced by another profile of the same model with a similar number of documents, so each keeps "
          "its own information but the pairing is wrong. **Chain**: the document alignment of every "
          "non-reference model is shuffled independently and the whole chain is rerun, so the selection of "
          "the most frequent profile is part of the null; composition is read on the real documents. `p` is "
          "the share of null draws with coherence at least as high as the observed one (minimum "
          f"1/{args.null_draws + 1}).\n",
          "![Invariance index](invariance.png)\n"]
    for t in invariance_taxonomies:
        rows_t = [x for x in inv_rows if x["taxonomy"] == t]
        wts = np.array([x["start_share"] for x in rows_t])
        L += [f"### `{t}`\n",
              f"Share-weighted mean coherence {np.average([x['coherence'] for x in rows_t], weights=wts):.3f} "
              f"against swap null {np.average([x['swap_mean'] for x in rows_t], weights=wts):.3f} and chain null "
              f"{np.average([x['chain_mean'] for x in rows_t], weights=wts):.3f}; profiles with p < 0.05: "
              f"{sum(x['swap_p'] < 0.05 for x in rows_t)} of {len(rows_t)} (swap), "
              f"{sum(x['chain_p'] < 0.05 for x in rows_t)} of {len(rows_t)} (chain); share-weighted mean information "
              f"{np.average([x['information'] for x in rows_t], weights=wts):.2f} bits.\n"]

        def append_invariance_table(title, rows):
            L.extend([
                f"#### {title}\n",
                "| Rank | Start share | Coherence | Swap null mean (95% interval) | p (swap) | Chain null mean (95% interval) | p (chain) | Information (bits) |",
                "|---|---|---|---|---|---|---|---|",
            ])
            for x in rows:
                L.append(f"| {x['selection_rank']} | {x['start_share']:.1%} | {x['coherence']:.3f} | {x['swap_mean']:.3f} "
                         f"({x['swap_lo']:.3f}-{x['swap_hi']:.3f}) | {x['swap_p']:.3f} | {x['chain_mean']:.3f} "
                         f"({x['chain_lo']:.3f}-{x['chain_hi']:.3f}) | {x['chain_p']:.3f} | {x['information']:.2f} |")
            L.append("")

        append_invariance_table(
            f"Top {sum(x['selection_group'] == 'top' for x in rows_t)} profiles by frequency",
            [x for x in rows_t if x["selection_group"] == "top"],
        )
        append_invariance_table(
            f"Tail {sum(x['selection_group'] == 'tail' for x in rows_t)} least frequent observed profiles",
            [x for x in rows_t if x["selection_group"] == "tail"],
        )
if multilabel_taxonomies:
    L += [
        "\n## Multi-label taxonomy handling\n",
        "The following taxonomies contain semicolon-separated, non-mutually-exclusive "
        "labels and are expanded before profile summaries: "
        + ", ".join(f"`{t}`" for t in multilabel_taxonomies)
        + ". Every percentage is the share of documents containing that individual label; "
        "one document may contribute to several labels, so percentages may sum to more "
        "than 100%. They are not passed to the categorical Jensen-Shannon invariance test. "
        "For each reference profile, the downloadable table compares the initial subgroup, "
        "the final matched intersection, and the excluded residual subgroup.\n",
        "- [Complete multi-label profile distributions](profile_multilabel_distributions.csv)",
    ]
if args.graphs:
    if label_col in multilabel_taxonomies:
        graph_description = (
            f"One graph per reference profile. Left: flows from the matched model profiles "
            f"to at most the {args.multilabel_top_k} most frequent individual `{label_col}` labels "
            "in the initial reference subgroup. Because labels are non-mutually-exclusive, "
            "flow shares need not sum to 100%. Right: horizontal bars compare the percentage "
            "of final matched documents containing each label with the percentage among the "
            "excluded residual documents. Label selection and ordering are fixed from the "
            "initial subgroup so the two distributions are directly comparable.\n"
        )
    else:
        graph_description = (
            f"One graph per reference profile (taxonomy `{label_col}`"
            + (f", then `{args.second_taxonomy}`" if args.second_taxonomy else "")
            + "). Left: the model profiles of the match with their active bits and number "
            "of documents; right: the categories. Edge width and label are the share of "
            "the documents having that model's profile that fall in the category; categories "
            "below 10% in every model are grouped as `other`; the bracket on a category is "
            "its overall share in the dataset."
            + (f" Third column: the `{args.second_taxonomy}` split of the reference documents "
               f"in each category (edge width = share of that category's reference documents); "
               f"edges keep the colour of the `{label_col}` category they come from."
               if args.second_taxonomy else "")
            + "\n"
        )
    L += ["\n## Graphs: model profiles and categories\n",
          graph_description,
          "![All reference profiles](graphs/contact_sheet.png)\n"]
    for k, r in enumerate(ref_results, 1):
        L += [f"### {r['selection_group'].title()} profile {r['selection_rank']} ({r['start_docs'] / N:.1%} of documents, "
              f"{r['cat_start']['category']} {r['cat_start']['share']:.0%})\n",
              f"![Reference profile {k}](graphs/graph_{k:02d}.png)\n"]
if additional_label_plot_files:
    L += [
        "\n## Additional labels by reference profile\n",
        "Each cell reports the prevalent class of one additional label, its share "
        "**among documents for which that label is observed**, and the corresponding "
        "observed sample size (`n`). Values configured as unavailable ("
        + ", ".join(f"`{value}`" for value in args.missing_label_value)
        + ") are excluded before calculating the percentage. Grey cells have no "
        "observed value in that profile. The row label gives the full profile size, "
        "so the displayed `n` also shows how much of the profile is covered by each "
        "source-specific taxonomy. The column header gives the corpus-wide prevalent "
        "class among observed values as a baseline.\n",
    ]
    for plot_file in additional_label_plot_files:
        L.append(f"![Additional-label prevalence]({plot_file})\n")
L += ["\n## Category composition by taxonomy (reference documents)\n",
      "For each reference profile, the two most frequent labels of every taxonomy among "
      "its documents, with the label's share of the whole dataset in brackets. Semicolon-separated "
      "multi-label fields are expanded first, and their percentages may sum to more than 100%. "
      "A taxonomy only shows an association where the share is far above its overall share. Taxonomies: " +
      ", ".join(f"`{t}`" for t in taxonomies) + ". The top three labels for the reference and final "
      "documents are in `profile_categories.csv`.\n"]


def append_category_table(title, rows):
    L.extend([f"### {title}\n", "| Rank | Start share | " + " | ".join(taxonomies) + " |",
              "|---|---|" + "---|" * len(taxonomies)])
    for r in rows:
        cells = []
        for t in taxonomies:
            if args.additional_label_summary and t in additional_taxonomies:
                summary = observed_taxonomy_summary(r["ref_docs"], t)
                cells.append(
                    "unavailable"
                    if summary is None
                    else (
                        f"{summary['dominant_class']} {summary['dominant_share']:.0%} "
                        f"among observed (n={summary['observed_docs']:,})"
                    )
                )
            else:
                lst = r["tax_start"][t][:2]
                cells.append("; ".join(f"{c} {sh:.0%} ({b:.0%})" for c, sh, b in lst))
        L.append(f"| {r['selection_rank']} | {r['start_docs'] / N:.1%} | " + " | ".join(cells) + " |")
    L.append("")


append_category_table(f"Top {len(top_ref_results)} profiles by frequency", top_ref_results)
append_category_table(f"Tail {len(tail_ref_results)} least frequent observed profiles", tail_ref_results)
if args.all_references:
    rec = {s: v for s, v in by_string.items() if len(set(v)) > 1}
    L += ["\n## Reciprocity: the same string reached from different starting models\n",
          f"{len(by_string)} distinct strings in total; {len(rec)} reached from at least two starting models.\n"]
    if rec:
        first = {}
        for r in results:
            first.setdefault(r["string"], r)
        L += ["| Matched profiles (active bits) | Started from |", "|---|---|"]
        for s, v in sorted(rec.items(), key=lambda kv: -len(set(kv[1]))):
            L.append(f"| {render(first[s])} | {', '.join(sorted(set(v)))} |")
L += [f"\n## Bit pairs across profiles (reference `{args.reference}`)\n",
      "For each reference bit (index, 0 = bit_00) and model: among the reference profiles that contain the bit, the share of "
      "their matched profiles containing the top partner bit. `baseline` is how often that partner bit is "
      "active in the matched profiles overall, so a fraction well above its baseline is an association.\n",
      "| Model | Ref bit | Profiles | Top partner | Fraction | Baseline | 2nd partner | Fraction |",
      "|---|---|---|---|---|---|---|---|"]
for p in pair_rows:
    L.append(f"| {p['model']} | {p['ref_bit']} | {p['profiles_with_bit']} | {p['top_partner']} | "
             f"{p['top_fraction']:.2f} | {p['top_partner_baseline']:.2f} | {p['second_partner']} | "
             f"{p['second_fraction']:.2f} |")
L += ["\n## Downloadable profile-matching artifacts\n",
      "- [Complete match table](matches.csv)",
      "- [Category composition by profile and taxonomy](profile_categories.csv)",
      "- [Cross-model bit-pair frequencies](bit_pairs.csv)"]
if additional_label_rows:
    L += ["- [Additional-label prevalence by reference profile](profile_additional_label_prevalence.csv)"]
if multilabel_rows:
    L += ["- [Multi-label distributions for initial, final, and residual documents](profile_multilabel_distributions.csv)"]
if inv_rows:
    L += ["- [Per-profile invariance statistics](invariance.csv)"]
(out / "manifest.json").write_text(json.dumps({
    "schema_version": 4,
    "method": "greedy profile-to-profile document matching",
    "n_documents": N,
    "requested_reference": requested_reference,
    "reference": args.reference,
    "all_references": bool(args.all_references),
    "models_in_report": report["run_order"],
    "models_used": models,
    "excluded_models": excluded,
    "top_profile_share": top_share,
    "constant_bits_removed": dropped,
    "profile_key_encoding": {
        "description": "active zero-based bit indices encoded as hexadecimal offsets within 16-bit blocks",
        "block_separator": "|",
        "empty_block": "-",
        "model_separator": "*",
    },
    "selection_rule": "top_n_plus_tail_n_observed_profiles",
    "top_reference_profiles": args.top_reference_profiles,
    "tail_reference_profiles": args.tail_reference_profiles,
    "deprecated_min_reference_profile_share": args.min_share,
    "deprecated_max_reference_profiles": args.max_reference_profiles,
    "deprecated_min_model_top_profile_share": args.min_top_share,
    "weak_match_min_documents": args.min_docs,
    "null_draws": args.null_draws,
    "null_random_seeds": {
        "profile_swap": 0,
        "chain_permutation": 1,
    },
    "profile_swap_unavailable_policy": (
        "Report NaN when a matched model has no alternative observed profile; "
        "do not resample the observed partner as a pseudo-null."
    ),
    "reference_profile_count": len(ref_results),
    "distinct_match_strings": len(by_string),
    "reciprocal_match_strings": sum(len(set(v)) > 1 for v in by_string.values()),
    "taxonomies": taxonomies,
    "multilabel_taxonomies": multilabel_taxonomies,
    "multilabel_separator": args.multilabel_separator,
    "multilabel_top_k": args.multilabel_top_k,
    "additional_label_summary": bool(args.additional_label_summary),
    "additional_taxonomies": additional_taxonomies if args.additional_label_summary else [],
    "missing_label_values": args.missing_label_value,
    "invariance_taxonomies": invariance_taxonomies,
    "files": {
        "markdown": "output.md",
        "matches": "matches.csv",
        "profile_categories": "profile_categories.csv",
        "profile_multilabel_distributions": "profile_multilabel_distributions.csv" if multilabel_rows else None,
        "profile_additional_label_prevalence": (
            "profile_additional_label_prevalence.csv" if additional_label_rows else None
        ),
        "additional_label_prevalence_plots": additional_label_plot_files,
        "bit_pairs": "bit_pairs.csv",
        "profile_distribution_plot": "profile_distribution.png",
        "invariance": "invariance.csv" if inv_rows else None,
        "invariance_plot": "invariance.png" if inv_rows else None,
        "graphs_directory": "graphs" if args.graphs else None,
    },
}, indent=2) + "\n")
(out / "output.md").write_text("\n".join(L) + "\n")
print(len(ref_results), "reference profiles;", len(by_string), "distinct strings")
