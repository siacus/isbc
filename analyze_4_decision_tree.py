#!/usr/bin/env python3
"""
Step 4: Shallow decision tree over binary factors -> AG News category.

Fits a depth-limited decision tree on the K binary factor columns and prints:
  - the tree structure as readable text (which bits are load-bearing)
  - per-class precision / recall / F1
  - feature importances ranked

Usage:
    python analyze_4_decision_tree.py \
        --encoded  runs/agnews_sae/encoded_agnews_all_ready_best.csv \
        --labels   data/agnews/agnews_all.csv \
        --output   runs/agnews_sae/decision_tree.txt \
        [--max_depth 4]
"""

import argparse
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, confusion_matrix


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoded",   required=True)
    p.add_argument("--labels",    required=True)
    p.add_argument("--label_col", type=str, default="label_name",
                   help="Column in the labels CSV to use as category. "
                        "e.g. label_name, airline_sentiment, negativereason, airline")
    p.add_argument("--output",    required=True)
    p.add_argument("--max_depth", type=int, default=4)
    p.add_argument("--seed",      type=int, default=42,
                   help="Random seed used by the decision tree (default: 42)")
    return p.parse_args()


def load_data(args):
    enc = pd.read_csv(args.encoded)
    bit_cols = sorted([c for c in enc.columns if c.startswith("bit_")])
    lab = pd.read_csv(args.labels)
    label_col = args.label_col
    if label_col not in lab.columns:
        available = [c for c in lab.columns if c != "id"]
        raise ValueError(f"--label_col '{label_col}' not found. Available columns: {available}")
    lab["label_name"] = lab[label_col].astype(str).str.strip()
    df = enc.merge(lab[["id", "label_name"]], on="id", how="inner")
    return df, bit_cols


def tree_to_text(tree, feature_names, class_names):
    """Export tree as readable indented rules."""
    return export_text(tree, feature_names=feature_names, class_names=class_names)


def node_rules(tree, feature_names, class_names, X=None, y=None):
    """
    Walk the tree and emit one human-readable rule per leaf:
      IF bit_04 = 1 AND bit_05 = 1 AND bit_07 = 0  →  Sports  (n=583, purity=98%)
    """
    t = tree.tree_
    n_nodes = t.node_count
    children_left  = t.children_left
    children_right = t.children_right
    feature        = t.feature
    threshold      = t.threshold   # for binary features threshold is always 0.5
    values         = t.value       # shape (n_nodes, 1, n_classes)

    rules = []
    leaf_counts = {}
    if X is not None and y is not None:
        for node, label in zip(tree.apply(X), y):
            leaf_counts.setdefault(int(node), {}).setdefault(str(label), 0)
            leaf_counts[int(node)][str(label)] += 1

    def recurse(node, conditions):
        if children_left[node] == children_right[node]:  # leaf
            weighted_counts = values[node][0]
            pred_idx = weighted_counts.argmax()
            label = class_names[pred_idx]
            if node in leaf_counts:
                counts = {name: leaf_counts[node].get(str(name), 0) for name in class_names}
                total = sum(counts.values())
                purity = counts.get(str(label), 0) / max(total, 1) * 100
            else:
                # Fallback for callers that do not provide X/y. Newer sklearn
                # versions store normalized proportions in tree_.value.
                total = int(t.n_node_samples[node])
                proportions = weighted_counts / max(weighted_counts.sum(), 1e-12)
                counts = {
                    class_names[i]: int(round(proportions[i] * total))
                    for i in range(len(class_names))
                }
                purity = proportions[pred_idx] * 100
            cond_str = " AND ".join(conditions) if conditions else "(root)"
            rules.append({
                "conditions": cond_str,
                "prediction": label,
                "n": int(total),
                "purity": purity,
                "counts": counts,
            })
        else:
            fname = feature_names[feature[node]]
            # binary feature: left branch = 0 (<=0.5), right branch = 1 (>0.5)
            recurse(children_left[node],  conditions + [f"{fname}=0"])
            recurse(children_right[node], conditions + [f"{fname}=1"])

    recurse(0, [])
    return rules


def main():
    args = parse_args()
    df, bit_cols = load_data(args)
    categories = sorted(df["label_name"].unique())

    X = df[bit_cols].values.astype(np.int8)
    y = df["label_name"].values

    clf = DecisionTreeClassifier(
        max_depth=args.max_depth,
        criterion="gini",
        class_weight="balanced",
        random_state=args.seed,
    )
    clf.fit(X, y)
    y_pred = clf.predict(X)

    # ── Feature importances ───────────────────────────────────────────────────
    importances = pd.Series(clf.feature_importances_, index=bit_cols).sort_values(ascending=False)
    active_bits = importances[importances > 0]

    # ── Leaf rules ────────────────────────────────────────────────────────────
    rules = node_rules(clf, bit_cols, categories, X=X, y=y)
    rules_df = pd.DataFrame(rules).sort_values("purity", ascending=False)

    # ── Build output text ─────────────────────────────────────────────────────
    buf = StringIO()

    def p(*a, **kw):
        print(*a, **kw, file=buf)
        print(*a, **kw)   # also to stdout

    p("=" * 80)
    p(f"DECISION TREE  (max_depth={args.max_depth})")
    p("=" * 80)

    p("\n── Tree structure ───────────────────────────────────────────────────────")
    p(tree_to_text(clf, bit_cols, categories))

    p("\n── Leaf rules (sorted by purity) ────────────────────────────────────────")
    p(f"{'prediction':<12} {'n':>6} {'purity':>7}  conditions")
    p("-" * 80)
    for _, row in rules_df.iterrows():
        p(f"{row['prediction']:<12} {row['n']:>6,} {row['purity']:>6.1f}%  {row['conditions']}")

    p("\n── Classification report ─────────────────────────────────────────────────")
    p(classification_report(y, y_pred, target_names=categories))

    p("\n── Confusion matrix ──────────────────────────────────────────────────────")
    cm = confusion_matrix(y, y_pred, labels=categories)
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in categories],
                             columns=[f"pred_{c}" for c in categories])
    p(cm_df.to_string())

    p("\n── Feature importances (non-zero only) ───────────────────────────────────")
    p(f"{'bit':<10} {'importance':>12}  bar")
    p("-" * 50)
    max_imp = active_bits.iloc[0] if len(active_bits) else 1
    for bit, imp in active_bits.items():
        bar = "█" * int(imp / max_imp * 40)
        p(f"{bit:<10} {imp:>12.4f}  {bar}")
    p(f"\n  {len(active_bits)} of {len(bit_cols)} bits used by the tree")
    p(f"  {len(bit_cols) - len(active_bits)} bits are entirely redundant for category prediction")

    Path(args.output).write_text(buf.getvalue(), encoding="utf-8")
    print(f"\nSaved to: {args.output}")

    output_path = Path(args.output)
    importances_path = output_path.with_name(output_path.stem + "_feature_importances.csv")
    pd.DataFrame({
        "factor": bit_cols,
        "importance": [float(importances.get(bit, 0.0)) for bit in bit_cols],
    }).sort_values("importance", ascending=False).to_csv(importances_path, index=False)
    rules_path = output_path.with_name(output_path.stem + "_leaf_rules.csv")
    rules_df.to_csv(rules_path, index=False)
    print(f"Saved feature importances to: {importances_path}")
    print(f"Saved leaf rules to:          {rules_path}")


if __name__ == "__main__":
    main()
