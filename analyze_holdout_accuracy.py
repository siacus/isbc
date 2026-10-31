#!/usr/bin/env python3
"""
Held-out classification accuracy: how well do the binary factors predict the label
on data the classifier never saw?

Unlike analyze_4_decision_tree.py (which fits and evaluates a shallow, depth-limited
tree on the SAME data, purely to read off human-interpretable rules), this script
does a proper train/test split, fits one or more standard classifiers on the K
binary factor columns, and reports accuracy/precision/recall/F1 on the held-out test
set only. It answers a different question: not "which bits matter and why", but
"how much of the label can actually be recovered from this code, out of sample?".

Reports, per classifier:
  - test accuracy (headline number) vs. train accuracy (generalization gap)
  - majority-class baseline accuracy (context for whether the test accuracy is
    actually better than just guessing the most common category)
  - per-class precision/recall/F1 and confusion matrix on the test set
  - leave-one-factor-out ablation: retrain without each factor on the exact
    same split, then report the loss (or gain) in accuracy and macro-F1

Five classifiers are available:
  - logistic_regression : linear baseline over the K bits (L2-penalized)
  - elastic_net          : logistic regression with a combined L1+L2 penalty
                           (--elasticnet_l1_ratio, --elasticnet_C). The L1
                           component can zero out weakly-informative bits'
                           coefficients entirely — useful to check whether a
                           sparser linear model (using fewer of the K bits)
                           does about as well as the full L2 model above.
  - random_forest        : captures interactions between bits
  - knn                  : k-nearest-neighbors vote using a distance defined
                           directly on the bit patterns (--knn_metric). Gower
                           distance was requested for this, but Gower's
                           coefficient is a way to mix numeric/categorical/binary
                           columns into one distance — with nothing but K binary
                           columns here, it reduces exactly to one of two
                           standard binary distances, both built into sklearn:
                             'jaccard' (default) — ignores 0/0 co-absence, i.e.
                               two documents are close if the factors they DO
                               have active overlap. Matches the asymmetric
                               treatment analyze_8_jaccard.py already uses for
                               these sparse codes (see the Jaccard callout in
                               README.md).
                             'hamming' — symmetric matching, counts every
                               bit (both 0/0 and 1/1) as agreement. This is the
                               literal "Gower distance for all-binary data".

  - pattern_majority     : non-parametric exact-pattern lookup. It remains useful
                           as a full-model baseline, with unseen-pattern fallback
                           and tied-majority rates reported. Per-factor ablation is
                           deliberately not computed: removing a bit merges lookup
                           cells and changes coverage, confounding information with
                           smoothing.

Standalone — not part of run_analysis.sh, since it involves a random
train/test split (adds --test_size/--seed knobs and run-to-run variance) unlike
the deterministic analyze_1..9 steps.

Usage:
    python analyze_holdout_accuracy.py \
        --encoded   runs/agnews_sae/encoded_agnews_all_ready_best.csv \
        --labels    data/agnews/agnews_all.csv \
        --label_col label_name \
        --output    runs/agnews_sae/analysis/holdout_accuracy.txt \
        [--classifier all|logistic_regression|elastic_net|random_forest|knn|pattern_majority] \
        [--elasticnet_l1_ratio 0.5] [--elasticnet_C 1.0] \
        [--knn_neighbors 5] [--knn_metric jaccard|hamming] \
        [--test_size 0.2] [--seed 42] [--no_ablation]
"""

import argparse
import inspect
import warnings
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from analysis_reporting import (
    EXACT_MATCHING_ABLATION_NOTE,
    ablation_feedback,
    build_provenance,
    retained_performance_pct,
    write_json,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoded",   required=True)
    p.add_argument("--labels",    required=True)
    p.add_argument("--label_col", type=str, default="label_name",
                   help="Column in the labels CSV to use as category. "
                        "e.g. label_name, airline_sentiment, negativereason, airline")
    p.add_argument("--output",    required=True)
    p.add_argument("--classifier", type=str, default="all",
                   choices=["all", "logistic_regression", "elastic_net", "random_forest",
                            "knn", "pattern_majority"])
    p.add_argument("--elasticnet_l1_ratio", type=float, default=0.5,
                   help="0 = pure L2 (ridge), 1 = pure L1 (lasso), in between = mixed")
    p.add_argument("--elasticnet_C", type=float, default=1.0,
                   help="Inverse regularization strength (smaller = stronger penalty)")
    p.add_argument("--knn_neighbors", type=int, default=5)
    p.add_argument("--knn_metric", type=str, default="jaccard", choices=["jaccard", "hamming"],
                   help="Distance between bit patterns. 'jaccard' (default) ignores 0/0 "
                        "co-absence, matching analyze_8_jaccard.py's convention for sparse "
                        "codes. 'hamming' is symmetric matching — the literal Gower distance "
                        "for all-binary data.")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--ablation", dest="ablation", action="store_true", default=True,
                   help="Run leave-one-factor-out retraining ablations (default: on)")
    p.add_argument("--no_ablation", dest="ablation", action="store_false",
                   help="Skip leave-one-factor-out ablations")
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


class PatternMajorityClassifier:
    """Exact string-pattern majority lookup used only as a full-model baseline."""

    def __init__(self, random_state=None):
        self.random_state = random_state

    @staticmethod
    def _majority_with_tiebreak(labels, rng):
        counts = pd.Series(labels).value_counts()
        tied = counts[counts == counts.iloc[0]].index.to_numpy()
        if len(tied) == 1:
            return tied[0], False
        return rng.choice(tied), True

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        self.global_majority_, _ = self._majority_with_tiebreak(y, rng)
        table = {}
        for pattern, label in zip(self.pattern_strings(X), y):
            table.setdefault(pattern, []).append(label)
        self.table_ = {}
        self.tied_patterns_ = set()
        for pattern, labels in table.items():
            label, was_tied = self._majority_with_tiebreak(labels, rng)
            self.table_[pattern] = label
            if was_tied:
                self.tied_patterns_.add(pattern)
        return self

    def predict(self, X):
        return np.array([
            self.table_.get(pattern, self.global_majority_)
            for pattern in self.pattern_strings(X)
        ], dtype=object)

    def fallback_rate(self, X) -> float:
        """Fraction of rows whose exact pattern was never seen in training."""
        return float(np.mean([pattern not in self.table_ for pattern in self.pattern_strings(X)]))

    def tie_rate(self, X) -> float:
        """Rate of tied pattern-majority labels among test rows with a seen pattern."""
        seen = [pattern for pattern in self.pattern_strings(X) if pattern in self.table_]
        if not seen:
            return 0.0
        return float(np.mean([pattern in self.tied_patterns_ for pattern in seen]))

    @staticmethod
    def pattern_strings(X) -> list[str]:
        """Canonical left-to-right bit strings, e.g. ``001101``."""
        matrix = np.asarray(X, dtype=np.uint8)
        return ["".join(row.astype(str).tolist()) for row in matrix]


def build_classifier(name: str, args: argparse.Namespace):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed)
    if name == "elastic_net":
        # sklearn <1.8 requires penalty="elasticnet"; >=1.8 infers it from
        # l1_ratio and deprecates the penalty argument. Inspecting the installed
        # signature keeps the model genuinely elastic-net without either warning.
        kwargs = {
            "solver": "saga",
            "l1_ratio": args.elasticnet_l1_ratio,
            "C": args.elasticnet_C,
            "max_iter": 5000,
            "class_weight": "balanced",
            "random_state": args.seed,
        }
        penalty = inspect.signature(LogisticRegression).parameters.get("penalty")
        if penalty is not None and penalty.default != "deprecated":
            kwargs["penalty"] = "elasticnet"
        return LogisticRegression(**kwargs)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=args.seed, n_jobs=-1
        )
    if name == "knn":
        # algorithm="auto" picks a BallTree for jaccard/hamming on binary data, so this
        # does an efficient nearest-neighbor search rather than building the full O(n^2)
        # pairwise distance matrix that algorithm="brute" would.
        return KNeighborsClassifier(
            n_neighbors=args.knn_neighbors, metric=args.knn_metric, algorithm="auto", n_jobs=-1
        )
    if name == "pattern_majority":
        return PatternMajorityClassifier(random_state=args.seed)
    raise ValueError(f"Unknown classifier: {name}")


def evaluate_classifier(name, args, X_train, X_test, y_train, y_test, categories):
    """Fit one classifier and return comparable metrics for full/ablated runs."""
    clf = build_classifier(name, args)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The max_iter was reached.*",
            category=ConvergenceWarning,
        )
        clf.fit(X_train, y_train)
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    report = classification_report(
        y_test, test_pred, labels=categories, target_names=categories,
        output_dict=True, zero_division=0,
    )
    fallback_rate = None
    tie_rate = None
    if isinstance(clf, PatternMajorityClassifier):
        fallback_rate = clf.fallback_rate(X_test)
        tie_rate = clf.tie_rate(X_test)
    return {
        "train_accuracy": accuracy_score(y_train, train_pred),
        "test_accuracy": accuracy_score(y_test, test_pred),
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "fallback_rate": fallback_rate,
        "tie_rate": tie_rate,
        "report": report,
        "predictions": test_pred,
    }


def main():
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df, bit_cols = load_data(args)
    categories = sorted(df["label_name"].unique())

    # bool, not int8: sklearn's jaccard/hamming metrics treat boolean-typed input
    # as native, avoiding a DataConversionWarning it'd otherwise emit converting
    # int8 internally. Numerically identical either way for every classifier here.
    X = df[bit_cols].values.astype(bool)
    y = df["label_name"].to_numpy(dtype=object)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=y
        )
        stratification_used = True
    except ValueError as e:
        print(f"Warning: stratified split failed ({e}); falling back to a random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed
        )
        stratification_used = False

    majority_class = pd.Series(y_test).value_counts().idxmax()
    baseline_acc = (y_test == majority_class).mean()

    classifier_names = (
        ["logistic_regression", "elastic_net", "random_forest", "knn", "pattern_majority"]
        if args.classifier == "all" else [args.classifier]
    )

    buf = StringIO()

    def p(*a, **kw):
        print(*a, **kw, file=buf)
        print(*a, **kw)

    p("=" * 80)
    p("HELD-OUT CLASSIFICATION ACCURACY")
    p("=" * 80)
    p(f"Documents : {len(df):,}  (train={len(X_train):,}, test={len(X_test):,}, "
      f"test_size={args.test_size})")
    p(f"Factors   : {len(bit_cols)}")
    p(f"Categories: {len(categories)}  {categories}")
    p(f"Majority-class baseline (test set, always predict '{majority_class}'): "
      f"{baseline_acc*100:.1f}%")

    metrics_rows = []
    per_class_rows = []
    full_results = {}
    for name in classifier_names:
        result = evaluate_classifier(
            name, args, X_train, X_test, y_train, y_test, categories
        )
        full_results[name] = result
        train_acc = result["train_accuracy"]
        test_acc = result["test_accuracy"]
        y_pred = result["predictions"]
        report = result["report"]

        p(f"\n{'─'*80}")
        p(f"{name}")
        p(f"{'─'*80}")
        p(f"  Train accuracy: {train_acc*100:.1f}%")
        p(f"  Test accuracy:  {test_acc*100:.1f}%  "
          f"(vs. baseline {baseline_acc*100:.1f}%, "
          f"gap to train {abs(train_acc-test_acc)*100:.1f} pts)")

        fallback_rate = result["fallback_rate"]
        tie_rate = result["tie_rate"]
        if name == "pattern_majority":
            p(f"  Unseen-pattern fallback rate: {fallback_rate*100:.1f}% of test documents")
            p(f"  Tied-majority rate among seen test patterns: {tie_rate*100:.1f}%")
            p("  Per-factor ablation: not applicable; exact-profile cell merging changes "
              "coverage and smoothing.")

        p("\n  Classification report (test set):")
        p(classification_report(y_test, y_pred, labels=categories, target_names=categories,
                                 zero_division=0))

        cm = confusion_matrix(y_test, y_pred, labels=categories)
        cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in categories],
                                 columns=[f"pred_{c}" for c in categories])
        p("  Confusion matrix (test set):")
        p(cm_df.to_string())

        metrics_rows.append({
            "classifier": name,
            "train_accuracy": round(train_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "baseline_accuracy": round(baseline_acc, 4),
            "macro_f1": round(result["macro_f1"], 4),
            "weighted_f1": round(result["weighted_f1"], 4),
            "fallback_rate": round(fallback_rate, 4) if fallback_rate is not None else "",
            "tie_rate": round(tie_rate, 4) if tie_rate is not None else "",
        })

        for cat in categories:
            per_class_rows.append({
                "classifier": name,
                "class": cat,
                "precision": round(report[cat]["precision"], 4),
                "recall": round(report[cat]["recall"], 4),
                "f1": round(report[cat]["f1-score"], 4),
                "support": int(report[cat]["support"]),
            })

    ablation_rows = []
    if args.ablation:
        if len(bit_cols) < 2:
            p("\nAblation skipped: at least two factors are required.")
        else:
            p(f"\n{'='*80}")
            p("LEAVE-ONE-FACTOR-OUT ABLATION")
            p(f"{'='*80}")
            p("Each applicable classifier is retrained without one factor on the same split.")
            p("pattern_majority remains in the full-model comparison but its per-factor "
              "ablation is marked not applicable and is not computed.")
            p("Positive drop = performance was lost when the factor was removed.")
            p("Completeness = ablated macro-F1 / full macro-F1 × 100; values above 100% "
              "mean removal improved performance.")

            for omitted_idx, omitted_factor in enumerate(bit_cols):
                keep = [i for i in range(len(bit_cols)) if i != omitted_idx]
                for name in classifier_names:
                    if name == "pattern_majority":
                        ablation_rows.append({
                            "classifier": name,
                            "omitted_factor": omitted_factor,
                            "n_factors_remaining": len(keep),
                            "ablation_applicable": False,
                            "full_test_accuracy": "",
                            "ablated_test_accuracy": "",
                            "accuracy_drop": "",
                            "full_macro_f1": "",
                            "ablated_macro_f1": "",
                            "macro_f1_drop": "",
                            "full_weighted_f1": "",
                            "ablated_weighted_f1": "",
                            "weighted_f1_drop": "",
                            "accuracy_completeness_pct": "",
                            "macro_f1_completeness_pct": "",
                            "feedback": EXACT_MATCHING_ABLATION_NOTE,
                        })
                        continue
                    full = full_results[name]
                    ablated = evaluate_classifier(
                        name, args, X_train[:, keep], X_test[:, keep],
                        y_train, y_test, categories,
                    )
                    accuracy_completeness = retained_performance_pct(
                        full["test_accuracy"], ablated["test_accuracy"]
                    )
                    macro_f1_completeness = retained_performance_pct(
                        full["macro_f1"], ablated["macro_f1"]
                    )
                    row = {
                        "classifier": name,
                        "omitted_factor": omitted_factor,
                        "n_factors_remaining": len(keep),
                        "ablation_applicable": True,
                        "full_test_accuracy": round(full["test_accuracy"], 6),
                        "ablated_test_accuracy": round(ablated["test_accuracy"], 6),
                        "accuracy_drop": round(
                            full["test_accuracy"] - ablated["test_accuracy"], 6
                        ),
                        "full_macro_f1": round(full["macro_f1"], 6),
                        "ablated_macro_f1": round(ablated["macro_f1"], 6),
                        "macro_f1_drop": round(full["macro_f1"] - ablated["macro_f1"], 6),
                        "full_weighted_f1": round(full["weighted_f1"], 6),
                        "ablated_weighted_f1": round(ablated["weighted_f1"], 6),
                        "weighted_f1_drop": round(
                            full["weighted_f1"] - ablated["weighted_f1"], 6
                        ),
                        "accuracy_completeness_pct": (
                            round(accuracy_completeness, 3)
                            if accuracy_completeness is not None else ""
                        ),
                        "macro_f1_completeness_pct": (
                            round(macro_f1_completeness, 3)
                            if macro_f1_completeness is not None else ""
                        ),
                    }
                    row["feedback"] = ablation_feedback(
                        name, omitted_factor, full["macro_f1"], ablated["macro_f1"]
                    )
                    ablation_rows.append(row)

            ablation_df = pd.DataFrame(ablation_rows)
            for name in classifier_names:
                if name == "pattern_majority":
                    p(f"\n  {name}: not applicable")
                    p(f"    - {EXACT_MATCHING_ABLATION_NOTE}")
                    continue
                ranked = (
                    ablation_df[ablation_df["classifier"] == name]
                    .sort_values(["macro_f1_drop", "accuracy_drop"], ascending=False)
                    .head(10)
                )
                p(f"\n  {name}: largest performance drops")
                p(ranked[["omitted_factor", "accuracy_drop", "macro_f1_drop",
                          "macro_f1_completeness_pct"]]
                  .to_string(index=False))
                for feedback in ranked["feedback"]:
                    p(f"    - {feedback}")

    metrics_df = pd.DataFrame(metrics_rows)
    p(f"\n{'='*80}")
    p("COMPARISON ACROSS METHODS (test set)")
    p(f"{'='*80}")
    p(metrics_df.to_string(index=False))

    p(f"\n{'='*80}")
    p("PER-CLASS F1 COMPARISON (test set)")
    p(f"{'='*80}")
    per_class_df = pd.DataFrame(per_class_rows)
    f1_pivot = per_class_df.pivot(index="class", columns="classifier", values="f1")
    p(f1_pivot.to_string())

    Path(args.output).write_text(buf.getvalue(), encoding="utf-8")
    print(f"\nSaved report to: {args.output}")

    metrics_path = Path(args.output).with_name(Path(args.output).stem + "_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to: {metrics_path}")

    per_class_path = Path(args.output).with_name(Path(args.output).stem + "_per_class.csv")
    per_class_df.to_csv(per_class_path, index=False)
    print(f"Saved per-class metrics to: {per_class_path}")

    if ablation_rows:
        ablation_path = Path(args.output).with_name(Path(args.output).stem + "_ablation.csv")
        pd.DataFrame(ablation_rows).to_csv(ablation_path, index=False)
        print(f"Saved factor ablations to: {ablation_path}")

    metadata_path = Path(args.output).with_name(Path(args.output).stem + "_metadata.json")
    provenance = build_provenance(
        encoded=args.encoded,
        labels=args.labels,
        label_column=args.label_col,
        analysis_parameters={
            "classifier": args.classifier,
            "elasticnet_l1_ratio": args.elasticnet_l1_ratio,
            "elasticnet_C": args.elasticnet_C,
            "knn_neighbors": args.knn_neighbors,
            "knn_metric": args.knn_metric,
            "test_size": args.test_size,
            "ablation": args.ablation,
            "convergence_warning_policy": (
                "Suppress sklearn ConvergenceWarning messages stating that max_iter was "
                "reached; all other warning categories remain visible."
            ),
            "ablation_not_applicable": {
                "pattern_majority": EXACT_MATCHING_ABLATION_NOTE,
            },
            "ablation_completeness_definition": (
                "100 * ablated_macro_f1 / full_macro_f1; values above 100 mean "
                "factor removal improved macro-F1"
            ),
        },
        analysis_seeds={"holdout_split_and_classifiers": args.seed},
    )
    provenance["split"] = {
        "n_documents": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "stratification_requested": True,
        "stratification_used": stratification_used,
    }
    write_json(metadata_path, provenance)
    print(f"Saved holdout metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
