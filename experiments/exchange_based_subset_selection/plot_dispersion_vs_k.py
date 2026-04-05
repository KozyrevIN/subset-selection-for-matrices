"""
Plot (a): relative dispersion of frob-orthonormal metric across algorithms vs k.

For each k we compute the mean across all algorithms (median of trials per algo),
then the max and min over algorithms.  The shaded band shows
  (max - min) / mean
as a function of k, one line per experiment (matrix type).

Run from the exchange_based_subset_selection directory:
    python plot_dispersion_vs_k.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# ── paths ─────────────────────────────────────────────────────────────────────
import os
SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = Path(os.environ.get('RESULTS_DIR', SCRIPT_DIR / 'results'))
FIGURES_DIR = SCRIPT_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

# ── parameters ────────────────────────────────────────────────────────────────
METRIC = 'X_S_dag_X_frobenius_norm_inv'

# ── style ─────────────────────────────────────────────────────────────────────
CM         = 1 / 2.54
TEXT_WIDTH = 14.7 * CM

plt.rcParams.update({
    "text.usetex":     True,
    "font.family":     "serif",
    "font.size":       7,
    "axes.titlesize":  7,
    "legend.fontsize": 7,
    "axes.labelsize":  7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 0.8,
    "axes.linewidth":  0.5,
    "axes.edgecolor":  "gray",
})

colors = plt.cm.tab10.colors

# ── load data ─────────────────────────────────────────────────────────────────
with open(RESULTS_DIR / 'index.json') as fh:
    index = json.load(fh)

experiments = {}
for exp_name in index['experiments']:
    folder = RESULTS_DIR / exp_name.replace(' ', '_')
    with open(folder / 'config.json') as fh:
        cfg = json.load(fh)
    algo_data = {}
    for algo_cfg in cfg['algorithms']:
        display  = algo_cfg.get('display_name', algo_cfg['name'])
        csv_path = folder / (display.replace(' ', '_') + '.csv')
        if csv_path.exists():
            algo_data[display] = pd.read_csv(csv_path)
    experiments[exp_name] = {'config': cfg, 'data': algo_data}

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 0.55 * TEXT_WIDTH))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_axisbelow(True)
ax.tick_params(axis='both', which='major', length=2, width=0.5,
               color='gray', direction='in')
ax.set_xlabel(r'$k$')
ax.set_ylabel(r'$(\max - \min) \,/\, \mathrm{mean}$')
ax.set_yscale('log')

for ei, (exp_name, exp) in enumerate(experiments.items()):
    algo_data = exp['data']
    if not algo_data:
        continue

    m = exp['config']['matrix']['rows']

    # only k in [m, 3m]
    all_k = sorted(
        k for df in algo_data.values()
        for k in df['k'].unique()
        if m <= k <= 3 * m
    )

    dispersions = []
    for k in all_k:
        medians = []
        for df in algo_data.values():
            vals = df[df['k'] == k][METRIC].dropna().values
            if len(vals) > 0:
                medians.append(np.median(vals))
        if len(medians) < 2:
            dispersions.append(np.nan)
            continue
        mean = np.mean(medians)
        if mean == 0:
            dispersions.append(np.nan)
            continue
        dispersions.append((max(medians) - min(medians)) / mean)

    k_arr = np.array(all_k)
    d_arr = np.array(dispersions)
    color = colors[ei % len(colors)]
    ax.plot(k_arr, d_arr, color=color, label=exp_name, linewidth=0.9)

ax.legend(frameon=False)
fig.tight_layout()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
outpath   = FIGURES_DIR / f'dispersion_vs_k_{timestamp}.pdf'
fig.savefig(outpath, bbox_inches='tight')
print(f'Saved: {outpath}')
