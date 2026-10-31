import pandas as pd
import json

# --- Load encoded test + labels ---
encoded = pd.read_csv("runs/agnews_sae/encoded_agnews_all_ready_best.csv")
labels  = pd.read_csv("data/agnews/agnews_all.csv")[["id", "label_name"]]
df = encoded.merge(labels, on="id")

print(f"Total docs: {len(df)}")
print(f"Active factors per doc (mean): {df['n_active'].mean():.2f}")
print()

# --- For each factor: label distribution when active ---
bit_cols = [c for c in df.columns if c.startswith("bit_")]
prob_cols = [c for c in df.columns if c.startswith("prob_")]
print("=== Probability statistics ===")
print(df[prob_cols].describe().T[["mean", "std", "min", "max"]])

print("=== Factor → Topic alignment ===")
for bit in bit_cols:
    active = df[df[bit] == 1]
    if len(active) == 0:
        print(f"{bit}: DEAD (never active)")
        continue
    dist = active["label_name"].value_counts(normalize=True)
    top_label = dist.index[0]
    top_pct   = dist.iloc[0] * 100
    print(f"{bit}  n={len(active):5d}  dominant={top_label:<12s} ({top_pct:.0f}%)")

print()

# --- How many factors are effectively dead? ---
dead = [b for b in bit_cols if df[b].sum() == 0]
print(f"Dead factors (never active): {len(dead)} / {len(bit_cols)}")

print()

# --- Load and print bit report top examples ---
with open("runs/agnews/bit_report_agnews_all_ready_best.json") as f:
    report = json.load(f)

print("=== Top example per factor ===")
for entry in report:
    top = entry["top_examples"][0] if entry["top_examples"] else None
    if top:
        print(f"bit {entry['bit']:02d}  mean_prob={entry['mean_prob']:.3f}  "
              f"top_text: {top['text'][:80]}")

