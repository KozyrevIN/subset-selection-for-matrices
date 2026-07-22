#!/usr/bin/env python3
"""
Plot script for the subset_selection_for_matrices_by_volume_sampling experiment.

Adapted from the spectral-norm experiment's plot.py. The deterministic and
randomized tester runs both write into the same results subfolder
(Superconductivity_dataset), so this script globs every CSV in the folder
rather than trusting a single config's algorithm list.

Layout (matrix has a regression target):
  - left  = 1 / ‖X_S† X‖_F   (Frobenius norm of the projected pseudo-inverse)
  - right = ‖X†‖_F / ‖X_S†‖_F (Frobenius norm ratio)

Usage (from repo root or from this directory):
    python experiments/subset_selection_for_matrices_by_volume_sampling/plot.py

Environment overrides:
    RESULTS_DIR  – path to the results directory  (default: <script_dir>/results)
    FIGURES_DIR  – path to save figures           (default: <script_dir>/figures)
"""

import os
import json
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = Path(os.environ.get('RESULTS_DIR', SCRIPT_DIR / 'results'))
FIGURES_DIR = Path(os.environ.get('FIGURES_DIR', SCRIPT_DIR / 'figures'))
FIGURES_DIR.mkdir(exist_ok=True)

# ── style ─────────────────────────────────────────────────────────────────────
CM         = 1 / 2.54
TEXT_WIDTH = 17 * CM

plt.rcParams.update({
    "text.usetex":     True,
    "font.family":     "serif",
    "font.size":       11,
    "axes.titlesize":  11,
    "legend.fontsize": 11,
    "axes.labelsize":  11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "lines.linewidth": 1.0,
    "axes.linewidth":  0.8,
    "axes.edgecolor":  "gray",
})

COLORS = plt.cm.tab10.colors

# Map experiment name → output file stem
OUT_STEMS = {
    'Superconductivity dataset': 'plot_superconductivity',
}

# Canonical algorithm order (by display_name) so colours are stable across runs.
CANONICAL = ['FDVS', 'RDVS', 'Frobenius selection', 'Frobenius removal',
             'Dominant', 'Dominant-split', 'VS', 'leverage scores',
             'random columns']

# Algorithms for which a theoretical Frobenius bound is meaningful.
BOUND_ALGOS = {'FDVS', 'RDVS', 'Frobenius selection', 'Dominant', 'Dominant-split'}

# Analytical bounds for the left subplot (1 / ‖X_S† X‖_F), as a function of
# (m, n, k). Each formula below bounds ‖X_S† X‖_F², so the value plotted on
# the 1/‖X_S† X‖_F axis is the inverse square root of the formula.
_ORTHONORMAL_BOUND_SQ = {
    'FDVS':                lambda m, n, k: m * (n - m + 1) / (k - m + 1),
    'RDVS':                lambda m, n, k: m * (n - m + 1) / (k - m + 1),
    'Dominant':            lambda m, n, k: m * (n - m + 1) / (k - m + 1),
    'Dominant-split':      lambda m, n, k: m * (n - m + 1) / (k - m + 1),
    'Frobenius selection': lambda m, n, k: (m ** 2 / k) * (n - m + 1),
}


def orthonormal_bound(name: str, m: int, n: int, k: np.ndarray) -> np.ndarray:
    """1/‖X_S† X‖_F bound for *name* at k, or None if not defined."""
    func = _ORTHONORMAL_BOUND_SQ.get(name)
    if func is None:
        return None
    return 1.0 / np.sqrt(func(m, n, k))


def infer_m_n(cfg: dict) -> tuple:
    """Infer (m, n) of the (auto-transposed) data matrix for a file-based
    experiment. m = smallest configured k (subset selection requires k >= m,
    and these experiments always start their k range at m). n = number of
    samples, read from the target file (one value per column of X)."""
    k_values = cfg.get('k_values', [])
    if not k_values:
        return None, None
    m = min(k_values)

    target_file = cfg.get('matrix', {}).get('target_file')
    if not target_file:
        return m, None
    target_path = SCRIPT_DIR / target_file
    if not target_path.exists():
        return m, None
    with open(target_path) as fh:
        n = sum(1 for line in fh if line.strip())
    return m, n


# ── helpers ───────────────────────────────────────────────────────────────────

def load_experiment(exp_name: str):
    folder = RESULTS_DIR / exp_name.replace(' ', '_')
    with open(folder / 'config.json') as fh:
        cfg = json.load(fh)

    # Collect all CSVs present in the folder (covers results from multiple
    # tester runs with different configs writing to the same subfolder).
    all_csvs = {p.stem.replace('_', ' '): p for p in folder.glob('*.csv')}

    ordered = CANONICAL + sorted(n for n in all_csvs if n not in CANONICAL)
    cfg['_all_algo_names'] = ordered

    data = {}
    for display, csv_path in all_csvs.items():
        data[display] = pd.read_csv(csv_path)
    return cfg, data


def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', length=2, width=0.5,
                   color='gray', direction='in')
    ax.tick_params(axis='both', which='minor', length=1, width=0.5,
                   color='gray', direction='in')


def plot_metric_subplot(ax, data, algo_names, metric_col, show_ylabel: bool,
                        ylabel: str, show_std: bool, show_bound: bool = True,
                        bound_algos: set = None, m: int = None, n: int = None):
    """Plot mean (± std) of *metric_col* on *ax* for every algorithm.
    If bound_algos is given, only draw bounds for algorithms in that set.
    When metric_col is 'X_S_dag_X_frobenius_norm_inv' and m, n are given,
    draws the per-algorithm analytical bound from orthonormal_bound() instead
    of the CSV 'frobenius_bound' column."""
    for idx, name in enumerate(algo_names):
        if name not in data:
            continue
        df    = data[name]
        color = COLORS[idx % len(COLORS)]

        df_filt = df[df[metric_col].notna()]
        if df_filt.empty:
            continue

        k_vals = np.sort(df_filt['k'].unique())
        grp    = df_filt.groupby('k')[metric_col]
        means  = grp.mean().reindex(k_vals).values

        ax.plot(k_vals, means, color=color, label=name)

        if show_std:
            stds = grp.std().reindex(k_vals).fillna(0).values
            ax.fill_between(k_vals, means - stds, means + stds,
                            color=color, alpha=0.25, linewidth=0)

        if (show_bound and metric_col == 'X_S_dag_X_frobenius_norm_inv'
                and m is not None and n is not None
                and (bound_algos is None or name in bound_algos)):
            bound = orthonormal_bound(name, m, n, k_vals)
            if bound is not None:
                dash_offsets = [0, 3, 6, 9, 12]
                offset = dash_offsets[idx % len(dash_offsets)]
                ax.plot(k_vals, bound, color=color, linewidth=1.2, alpha=0.8,
                        linestyle=(offset, (4, 4)))
        elif (show_bound and 'frobenius_bound' in df_filt.columns
                and metric_col == 'pinv_frobenius_norm_ratio'
                and (bound_algos is None or name in bound_algos)):
            bound = df_filt.groupby('k')['frobenius_bound'].mean().reindex(k_vals).values
            if bound.any():
                dash_offsets = [0, 3, 6, 9, 12]
                offset = dash_offsets[idx % len(dash_offsets)]
                ax.plot(k_vals, bound, color=color, linewidth=1.2, alpha=0.8,
                        linestyle=(offset, (4, 4)))

    style_ax(ax)
    ax.set_xlabel(r'$k$')
    if show_ylabel:
        ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.margins(x=0)


def has_nonzero_bound(data: dict) -> bool:
    for df in data.values():
        if 'frobenius_bound' in df.columns and (df['frobenius_bound'] != 0).any():
            return True
    return False


def make_legend(fig, algo_names, data, show_std: bool, show_bound: bool,
                ncols: int = 4):
    algo_handles = [
        plt.Line2D([0], [0], color=COLORS[i % len(COLORS)], linewidth=1.2,
                   label=name)
        for i, name in enumerate(algo_names)
        if name in data
    ]
    extra_handles = []
    if show_std:
        extra_handles.append(
            plt.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.25,
                           label='standard deviation')
        )
    if show_bound:
        extra_handles.append(
            plt.Line2D([0], [0], color='black', linestyle='--',
                       linewidth=1.2, label='theoretical bound')
        )

    n_extra = len(extra_handles)
    if n_extra > 0:
        total_before_extras = len(algo_handles)
        remainder = total_before_extras % ncols
        slots_left = (ncols - remainder) % ncols
        if slots_left >= n_extra:
            spacers = slots_left - n_extra
        else:
            spacers = slots_left + (ncols - n_extra)
        blank = plt.Line2D([0], [0], color='none', label='')
        handles = algo_handles + [blank] * spacers + extra_handles
    else:
        handles = algo_handles

    fig.legend(handles=handles, loc='upper center',
               bbox_to_anchor=(0.5, 0.0), ncols=ncols,
               frameon=False, handlelength=1.0)


def save_figure(fig, stem: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in [FIGURES_DIR / f'{stem}_{timestamp}.pdf',
                 FIGURES_DIR / f'{stem}.pdf']:
        fig.savefig(path, bbox_inches='tight')
        print(f'Saved: {path}')
    plt.close(fig)


def make_figure(exp_name, cfg, data, algo_names, out_stem):
    """Two subplots: left = 1/‖X_S† X‖_F, right = ‖X†‖_F/‖X_S†‖_F."""
    # Randomized algorithms have multiple rows per k; deterministic ones a
    # single row (std = 0, no visible band). Show the std machinery whenever
    # any algorithm has more than one trial at some k.
    show_std   = any((df.groupby('k').size() > 1).any() for df in data.values())
    show_bound = has_nonzero_bound(data)
    m, n       = infer_m_n(cfg)
    ylabel_left  = r'$1 \,/\, \Vert X_\mathcal{S}^\dag X \Vert_F$'
    ylabel_right = r'$\Vert X^\dag \Vert_F \,/\, \Vert X_\mathcal{S}^\dag \Vert_F$'

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(TEXT_WIDTH, 0.45 * TEXT_WIDTH), layout='constrained')

    plot_metric_subplot(ax_left, data, algo_names,
                        metric_col='X_S_dag_X_frobenius_norm_inv',
                        show_ylabel=True, ylabel=ylabel_left,
                        show_std=show_std, show_bound=show_bound,
                        bound_algos=BOUND_ALGOS, m=m, n=n)

    plot_metric_subplot(ax_right, data, algo_names,
                        metric_col='pinv_frobenius_norm_ratio',
                        show_ylabel=True, ylabel=ylabel_right,
                        show_std=show_std, show_bound=show_bound,
                        bound_algos=BOUND_ALGOS)

    # 9 algorithms + 2 extras → 5 columns keeps the legend to three rows.
    make_legend(fig, algo_names, data, show_std=show_std, show_bound=show_bound,
                ncols=5)
    save_figure(fig, out_stem)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    index_file = RESULTS_DIR / 'index.json'
    if not index_file.exists():
        raise FileNotFoundError(
            f"Results index not found: {index_file}\n"
            "Run the tester binary with the config(s) first."
        )

    with open(index_file) as fh:
        index = json.load(fh)

    for exp_name in index['experiments']:
        print(f'Loading: {exp_name}')
        cfg, data = load_experiment(exp_name)

        algo_names = cfg.get('_all_algo_names',
                             [a.get('display_name', a['name']) for a in cfg['algorithms']])
        stem       = OUT_STEMS.get(exp_name, exp_name.replace(' ', '_').lower())

        make_figure(exp_name, cfg, data, algo_names, stem)


if __name__ == '__main__':
    main()
