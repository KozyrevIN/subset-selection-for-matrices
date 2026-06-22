#!/usr/bin/env python3
"""
Plot script for the subset_selection_for_matrices_in_spectral_norm experiment.

Metric: ||X†||_2 / ||X_S†||_2  (larger is better).

Convention: if X_S is singular, ||X_S†||_2 = ∞, so the metric value is 0.

Layout per experiment:
  - Experiments without regression target: left = zoomed k range, right = full k range (spectral norm ratio)
  - Experiments with regression target:    left = spectral norm ratio (full k), right = regression MSE (full k)

Usage (from repo root or from this directory):
    python experiments/subset_selection_for_matrices_in_spectral_norm/plot.py

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
    'Matrices with orthonormal rows':       'plot_orthonormal',
    'Incidence matrices of a random graph': 'plot_weighted_graph',
    'Abalone dataset':                      'plot_abalone',
}

ABALONE_FIGURE = {'Abalone dataset'}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_experiment(exp_name: str):
    folder = RESULTS_DIR / exp_name.replace(' ', '_')
    with open(folder / 'config.json') as fh:
        cfg = json.load(fh)

    # Collect all CSVs present in the folder (covers results from multiple
    # tester runs with different configs writing to the same subfolder).
    all_csvs = {p.stem.replace('_', ' '): p for p in folder.glob('*.csv')}

    # Canonical order: spectral selection, spectral removal, dual set,
    # leverage scores, random columns — matches all other experiments so
    # colors are consistent across all plots.
    canonical = ['spectral selection', 'spectral removal', 'frobenius removal',
                 'dual set', 'leverage scores', 'random columns']
    ordered = canonical + sorted(n for n in all_csvs if n not in canonical)
    cfg['_all_algo_names'] = ordered

    data = {}
    for display, csv_path in all_csvs.items():
        df = pd.read_csv(csv_path)
        data[display] = df
    return cfg, data


def k_split_from_config(cfg: dict) -> int:
    """Left subplot shows k up to 2*m."""
    if 'rows' in cfg.get('matrix', {}):
        return 2 * cfg['matrix']['rows']
    # File-based matrices: m = first k value (minimum selectable subset size)
    k_vals = cfg.get('k_values', [])
    return 2 * k_vals[0] if k_vals else 200


def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', length=2, width=0.5,
                   color='gray', direction='in')
    ax.tick_params(axis='both', which='minor', length=1, width=0.5,
                   color='gray', direction='in')


def plot_metric_subplot(ax, data, algo_names, metric_col, k_mask_fn,
                        show_ylabel: bool, ylabel: str, show_std: bool,
                        show_bound: bool = True, k_lim: tuple = None,
                        bound_algos: set = None):
    """Plot mean (± std) of *metric_col* on *ax* for k values passing k_mask_fn.
    If bound_algos is given, only draw bounds for algorithms in that set."""
    for idx, name in enumerate(algo_names):
        if name not in data:
            continue
        df    = data[name]
        color = COLORS[idx % len(COLORS)]

        df_filt = df[df['k'].apply(k_mask_fn)].copy()
        if df_filt.empty:
            continue

        df_filt = df_filt[df_filt[metric_col].notna()]
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

        if (show_bound and 'spectral_bound' in df_filt.columns
                and metric_col in ('pinv_spectral_norm_ratio', 'X_S_dag_X_spectral_norm_inv')
                and (bound_algos is None or name in bound_algos)):
            bound = df_filt.groupby('k')['spectral_bound'].mean().reindex(k_vals).values
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
    if k_lim is not None:
        ax.set_xlim(k_lim[0], k_lim[1])


def has_nonzero_bound(data: dict) -> bool:
    for df in data.values():
        if 'spectral_bound' in df.columns and (df['spectral_bound'] != 0).any():
            return True
    return False


def make_legend(fig, algo_names, data, show_std: bool, show_bound: bool):
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

    ncols = 4
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


def make_spectral_figure(exp_name, cfg, data, algo_names, out_stem):
    """Two subplots: left = zoomed k range, right = full k range (spectral norm)."""
    k_split    = k_split_from_config(cfg)
    show_std   = cfg.get('trials_per_k', 1) > 1
    show_bound = has_nonzero_bound(data)
    ylabel   = r'$\Vert X^\dag \Vert_2 \,/\, \Vert X_\mathcal{S}^\dag \Vert_2$'

    all_k = sorted({k for df in data.values() for k in df['k'].unique()})
    k_min, k_max = all_k[0], all_k[-1]

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(TEXT_WIDTH, 0.45 * TEXT_WIDTH), layout='constrained')

    for ax, mask, show_y, lim in [
        (ax_left,  lambda k: k <= k_split, True,  (k_min, k_split)),
        (ax_right, lambda k: True,         False,  (k_min, k_max)),
    ]:
        plot_metric_subplot(ax, data, algo_names,
                            metric_col='pinv_spectral_norm_ratio',
                            k_mask_fn=mask,
                            show_ylabel=show_y, ylabel=ylabel,
                            show_std=show_std, show_bound=show_bound,
                            k_lim=lim)

    make_legend(fig, algo_names, data, show_std=show_std, show_bound=show_bound)
    save_figure(fig, out_stem)


def make_abalone_figure(exp_name, cfg, data, algo_names, out_stem):
    """Two subplots: left = 1/‖X_S† X‖, right = ‖X†‖/‖X_S†‖ (both full k range)."""
    show_std   = cfg.get('trials_per_k', 1) > 1
    show_bound = has_nonzero_bound(data)
    ylabel_left  = r'$1 \,/\, \Vert X_\mathcal{S}^\dag X \Vert_2$'
    ylabel_right = r'$\Vert X^\dag \Vert_2 \,/\, \Vert X_\mathcal{S}^\dag \Vert_2$'

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(TEXT_WIDTH, 0.45 * TEXT_WIDTH), layout='constrained')

    plot_metric_subplot(ax_left, data, algo_names,
                        metric_col='X_S_dag_X_spectral_norm_inv',
                        k_mask_fn=lambda k: True,
                        show_ylabel=True, ylabel=ylabel_left,
                        show_std=show_std, show_bound=show_bound,
                        bound_algos={'spectral selection', 'spectral removal', 'dual set'})

    plot_metric_subplot(ax_right, data, algo_names,
                        metric_col='pinv_spectral_norm_ratio',
                        k_mask_fn=lambda k: True,
                        show_ylabel=True, ylabel=ylabel_right,
                        show_std=show_std, show_bound=show_bound)

    make_legend(fig, algo_names, data, show_std=show_std, show_bound=show_bound)
    save_figure(fig, out_stem)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    index_file = RESULTS_DIR / 'index.json'
    if not index_file.exists():
        raise FileNotFoundError(
            f"Results index not found: {index_file}\n"
            "Run the tester binary with config.json first."
        )

    with open(index_file) as fh:
        index = json.load(fh)

    for exp_name in index['experiments']:
        print(f'Loading: {exp_name}')
        cfg, data = load_experiment(exp_name)

        algo_names = cfg.get('_all_algo_names',
                             [a.get('display_name', a['name']) for a in cfg['algorithms']])
        stem       = OUT_STEMS.get(exp_name, exp_name.replace(' ', '_').lower())

        if exp_name in ABALONE_FIGURE:
            make_abalone_figure(exp_name, cfg, data, algo_names, stem)
        else:
            make_spectral_figure(exp_name, cfg, data, algo_names, stem)


if __name__ == '__main__':
    main()
