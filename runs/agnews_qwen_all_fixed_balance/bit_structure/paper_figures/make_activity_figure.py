"""Plot saved Qwen training activity; no encoder fitting or label use.

Run with python from the repository root. Outputs stay beside this script.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

out = Path(__file__).resolve().parent
run = out.parent.parent
source = run / 'training_active_bits_diagnostics.csv'
df = pd.read_csv(source).sort_values('global_step')
assert len(df) and df.global_step.is_unique
assert df.target_active.nunique() == 1
window = json.loads((run / 'training_convergence_diagnostics.json').read_text())['rolling_window']
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), gridspec_kw={'width_ratios': [1.5, 1]},
                         sharey=True, layout='constrained')
for ax, frame, title in [(axes[0], df, '(a) Complete training'),
                          (axes[1], df[df.global_step <= 1000], '(b) First 1,000 steps')]:
    ax.plot(frame.global_step, frame.hard_active_rolling, color='#176b91', lw=.8,
            label='Binary activations', rasterized=True)
    ax.plot(frame.global_step, frame.soft_active_rolling, color='#c65a20', lw=.8,
            alpha=.85, label='Sum of probabilities', rasterized=True)
    ax.axhline(df.target_active.iloc[0], color='black', ls='--', lw=1, label='Target = 6')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Optimizer step')
    ax.grid(axis='y', alpha=.18)
    ax.set_xlim(0, frame.global_step.max())
for step in df.groupby('epoch').global_step.min().iloc[1:]:
    axes[0].axvline(step, color='.55', ls=':', lw=.8)
axes[0].set_ylabel('Mean active bits per document')
axes[0].ticklabel_format(axis='x', style='sci', scilimits=(3, 3), useMathText=True)
axes[1].legend(fontsize=8, loc='lower right', frameon=False)
for ext in ['pdf', 'png']:
    fig.savefig(out / f'qwen_active_bits_training.{ext}', dpi=220)
plt.close(fig)
(out / 'qwen_active_bits_training.json').write_text(json.dumps({
    'source': str(source), 'rolling_window_steps': window, 'zoom_max_step': 1000,
    'rows': len(df), 'max_step': int(df.global_step.max()),
    'series': ['hard_active_rolling', 'soft_active_rolling'],
    'description': 'Saved trailing rolling means of training minibatch activity; not corpus checkpoint evaluation.',
}, indent=2) + '\n')
print('Created PDF and PNG from', len(df), 'saved steps; rolling window:', window)
