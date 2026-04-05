"""
Algorithm comparison plot: one subplot per k, one violin per algorithm.
Mirrors the ROM-error plot style.

Set PLOT_MODE to choose what to plot:
  'frob_orthonormal' — 1/‖X_S† X‖_F  (orthonormal Frobenius metric)
  'volume'           — vol(X_S) / vol(X)  (volume ratio)

Run from the exchange_based_subset_selection directory:
    python plot_frob_orthonormal_comparison.py
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
SCRIPT_DIR   = Path(__file__).parent
RESULTS_DIR  = Path(os.environ.get('RESULTS_DIR', SCRIPT_DIR / 'results'))
FIGURES_DIR  = SCRIPT_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

# ── parameters ────────────────────────────────────────────────────────────────
# Set these to the 3 representative k values chosen from plot_dispersion_vs_k
PLOT_K     = [100, 104, 108]
PLOT_MODE  = 'frob_orthonormal'   # 'frob_orthonormal' or 'volume'
SQUARE     = False                # square the metric (frob_orthonormal only)

# Derived from PLOT_MODE
if PLOT_MODE == 'volume':
    METRIC     = 'volume_ratio'
    SQUARE     = False
elif PLOT_MODE == 'frob_orthonormal':
    METRIC     = 'X_S_dag_X_frobenius_norm_inv'
else:
    raise ValueError(f"Unknown PLOT_MODE: {PLOT_MODE!r}")

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

experiments = {}   # name → {algorithm_name: DataFrame}
for exp_name in index['experiments']:
    folder = RESULTS_DIR / exp_name.replace(' ', '_')
    with open(folder / 'config.json') as fh:
        cfg = json.load(fh)
    algo_data = {}
    for algo_cfg in cfg['algorithms']:
        display = algo_cfg.get('display_name', algo_cfg['name'])
        csv_path = folder / (display.replace(' ', '_') + '.csv')
        if csv_path.exists():
            algo_data[display] = pd.read_csv(csv_path)
    experiments[exp_name] = {'config': cfg, 'data': algo_data}

# Algorithm order / color assignment from first experiment
first_exp     = list(experiments.values())[0]
algorithm_names = list(first_exp['data'].keys())
n_algos       = len(algorithm_names)
algo_color    = {name: colors[i % len(colors)] for i, name in enumerate(algorithm_names)}

# ── helpers ───────────────────────────────────────────────────────────────────

def _style_ax(ax, n):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', length=2, width=0.5,
                   color='gray', direction='in')
    ax.set_xlim(-0.6, n - 0.4)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.get_offset_text().set_x(0.04)


def _violin(ax, xi, vals, color):
    if len(vals) < 2:
        ax.plot(xi, vals[0] if len(vals) else np.nan, 'o',
                color=color, markersize=3)
        return
    parts = ax.violinplot(vals, positions=[xi], widths=0.6,
                          showmeans=False, showmedians=True, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_alpha(0.35)
        pc.set_edgecolor(color)
        pc.set_linewidth(0.5)
    parts['cmedians'].set_color(color)
    parts['cmedians'].set_linewidth(0.8)


def get_vals(algo_df, k_val):
    rows = algo_df[algo_df['k'] == k_val][METRIC].dropna().values
    return rows ** 2 if SQUARE else rows



# ── figure: one row per experiment, one column per k ─────────────────────────
n_exp  = len(experiments)
n_k    = len(PLOT_K)
x_pos  = np.arange(n_algos)
# short labels: strip common prefix "Dominant" to save space
x_labels = [n.replace('Dominant-', 'D-') for n in algorithm_names]

fig, axes = plt.subplots(
    nrows=n_exp, ncols=n_k,
    figsize=(TEXT_WIDTH, n_exp * 0.42 * TEXT_WIDTH),
    squeeze=False,
)
fig.subplots_adjust(left=0.10, right=0.99, top=0.93, bottom=0.06,
                    wspace=0.40, hspace=0.25)

if PLOT_MODE == 'volume':
    ylabel_text = r'$\mathrm{vol}(X_\mathcal{S}) / \mathrm{vol}(X)$'
else:
    ylabel_text = (r'$1/\Vert X_\mathcal{S}^\dag X \Vert_F^2$'
                   if SQUARE else r'$1/\Vert X_\mathcal{S}^\dag X \Vert_F$')

for row, (exp_name, exp) in enumerate(experiments.items()):
    cfg       = exp['config']
    algo_data = exp['data']

    for col, k_val in enumerate(PLOT_K):
        ax = axes[row, col]

        # collect all values to determine y-limits
        all_vals = []
        for name in algorithm_names:
            if name in algo_data:
                v = get_vals(algo_data[name], k_val)
                all_vals.extend(v.tolist())

        for xi, name in enumerate(algorithm_names):
            if name not in algo_data:
                continue
            vals  = get_vals(algo_data[name], k_val)
            color = algo_color[name]
            _violin(ax, xi, vals, color)

        _style_ax(ax, n_algos)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([])

        if all_vals:
            span = max(all_vals) - min(all_vals)
            ax.set_ylim(bottom=min(all_vals) - span * 0.3,
                        top=max(all_vals) + span * 0.3)

        # column title (k value) — only on top row
        if row == 0:
            ax.set_title(rf'$k = {k_val}$')

        # y-label — only left column
        if col == 0:
            ax.set_ylabel(ylabel_text)

    # experiment name as a row label on the left
    fig.text(0.01, axes[row, 0].get_position().y0
             + axes[row, 0].get_position().height / 2,
             exp_name, va='center', ha='left', fontsize=7, rotation=90)

# ── legend ────────────────────────────────────────────────────────────────────
# Layout: 4 columns.
# Col 1-3: Dominant-CPQR/greedy/advanced then split-CPQR/greedy/advanced (2 rows)
# Col 4:   Frobenius removal (row 1), blank (row 2)
exchange_names = [n for n in algorithm_names if n != 'Frob-removal-orth']
other_names    = [n for n in algorithm_names if n == 'Frob-removal-orth']

def _handle(name):
    return plt.Line2D([0], [0], color=algo_color[name], linewidth=4,
                      alpha=0.35, label=name)

blank = plt.Line2D([0], [0], color='none', label='')

# matplotlib fills legends column-by-column with ncols=4, nrows=2:
# positions: col1=[0,1], col2=[2,3], col3=[4,5], col4=[6,7]
# Desired:
#   col1: D-CPQR, D-greedy
#   col2: D-advanced, D-split-CPQR
#   col3: D-split-greedy, D-split-advanced
#   col4: Frobenius removal, (blank)
legend_handles = (
    [_handle(exchange_names[0]), _handle(exchange_names[1]),   # col1
     _handle(exchange_names[2]), _handle(exchange_names[3]),   # col2
     _handle(exchange_names[4]), _handle(exchange_names[5])] + # col3
    [_handle(n) for n in other_names] + [blank]                # col4
)

fig.legend(handles=legend_handles, loc='upper center',
           bbox_to_anchor=(0.5, 0.01), ncols=4,
           frameon=False, fontsize=7)

# ── save ──────────────────────────────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
outpath   = FIGURES_DIR / f'{PLOT_MODE}_comparison_{timestamp}.pdf'
fig.savefig(outpath, bbox_inches='tight')
print(f'Saved: {outpath}')
