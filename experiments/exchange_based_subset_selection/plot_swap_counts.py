"""
Plot (c): swap counts (log scale) vs k for exchange-based algorithms,
first 100 k values (k = 100..200), one subplot per experiment (matrix type).

Run from the exchange_based_subset_selection directory:
    python plot_swap_counts.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatter
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
# Only algorithms that perform swaps (swap_count >= 0 and not always -1)
EXCHANGE_ALGOS = {
    'Dominant-CPQR', 'Dominant-greedy', 'Dominant-advanced',
    'Dominant-split-CPQR', 'Dominant-split-greedy', 'Dominant-split-advanced',
}
# K range is determined per-experiment from matrix rows: [m, 3m]

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

# Collect exchange-algo names actually present in data, preserving order
first_data = list(experiments.values())[0]['data']
algo_names  = [n for n in first_data if n in EXCHANGE_ALGOS]
algo_color  = {n: colors[i % len(colors)] for i, n in enumerate(algo_names)}

# ── figure: one subplot per experiment ───────────────────────────────────────
n_exp = len(experiments)
fig, axes = plt.subplots(
    nrows=1, ncols=n_exp,
    figsize=(TEXT_WIDTH, 0.55 * TEXT_WIDTH),
    squeeze=False,
)
fig.subplots_adjust(left=0.10, right=0.99, top=0.88, bottom=0.13,
                    wspace=0.35)

for col, (exp_name, exp) in enumerate(experiments.items()):
    ax        = axes[0, col]
    algo_data = exp['data']
    m         = exp['config']['matrix']['rows']
    k_min, k_max = m, 3 * m

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', length=2, width=0.5,
                   color='gray', direction='in')
    ax.set_title(exp_name, fontsize=7)
    ax.set_xlabel(r'$k$')
    if col == 0:
        ax.set_ylabel('swap count')

    for name in algo_names:
        if name not in algo_data:
            continue
        df = algo_data[name]
        df = df[(df['k'] >= k_min) & (df['k'] <= k_max)].copy()
        if df.empty or 'swap_count' not in df.columns:
            continue
        # filter out -1 (algorithms that don't track swaps)
        df = df[df['swap_count'] >= 0]
        if df.empty:
            continue
        grouped = df.groupby('k')['swap_count']
        k_vals   = np.array(sorted(grouped.groups.keys()))
        medians  = np.array([np.median(grouped.get_group(k).values) for k in k_vals])
        q25      = np.array([np.percentile(grouped.get_group(k).values, 25) for k in k_vals])
        q75      = np.array([np.percentile(grouped.get_group(k).values, 75) for k in k_vals])
        color    = algo_color[name]
        ax.plot(k_vals, medians, color=color, label=name, linewidth=0.9)
        ax.fill_between(k_vals, np.maximum(q25, 0.5), q75,
                        color=color, alpha=0.15)

# ── shared legend ─────────────────────────────────────────────────────────────
handles = [plt.Line2D([0], [0], color=algo_color[n], linewidth=4,
                      alpha=0.6, label=n) for n in algo_names]
fig.legend(handles=handles, loc='upper center',
           bbox_to_anchor=(0.5, 0.01), ncols=3,
           frameon=False, fontsize=7)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
outpath   = FIGURES_DIR / f'swap_counts_{timestamp}.pdf'
fig.savefig(outpath, bbox_inches='tight')
print(f'Saved: {outpath}')
