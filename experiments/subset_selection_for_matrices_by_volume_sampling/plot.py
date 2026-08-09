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
    # amsmath for \dfrac in the legend's bound label (the default usetex
    # preamble is minimal and does not provide it).
    "text.latex.preamble": r"\usepackage{amsmath}",
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

# The single shared theoretical bound, drawn once on both subplots. It follows
# from the volume bound ‖X_S† X‖_F² ≤ m (n-m+1)/(k-m+1): dividing m by its square
# root gives m/‖X_S† X‖_F ≥ √m · √((k-m+1)/(n-m+1)), and the same √((k-m+1)/
# (n-m+1)) lower-bounds the pinv Frobenius ratio ‖X†‖_F/‖X_S†‖_F. It is a valid
# lower bound for every deterministic algorithm on both axes (the randomized
# baselines, having no such guarantee, may dip below it). The LaTeX form is a
# legend entry (see make_legend), rendered as a proper fraction.
BOUND_LATEX = r'$\sqrt{\dfrac{k - m + 1}{n - m + 1}}$'


def shared_bound(m: int, n: int, k: np.ndarray) -> np.ndarray:
    """The lower bound √((k-m+1)/(n-m+1)) at k, shared by both subplots."""
    return np.sqrt((k - m + 1) / (n - m + 1))


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
                        m: int = None, n: int = None):
    """Plot mean (± std) of *metric_col* on *ax* for every algorithm, plus the
    single shared lower bound √((k-m+1)/(n-m+1)). The left subplot
    (metric_col 'X_S_dag_X_frobenius_norm_inv') plots √m / ‖X_S† X‖_F; the CSV
    stores 1 / ‖X_S† X‖_F, so its means are scaled by √m."""
    all_k = []
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

        # The left subplot plots √m / ‖X_S† X‖_F (the CSV stores
        # 1 / ‖X_S† X‖_F). The √m makes √((k-m+1)/(n-m+1)) an exact lower bound:
        # ‖X_S† X‖_F² ≤ m (n-m+1)/(k-m+1) ⟹ √m/‖X_S† X‖_F ≥ √((k-m+1)/(n-m+1)).
        scale = (np.sqrt(m) if (metric_col == 'X_S_dag_X_frobenius_norm_inv'
                                and m is not None) else 1.0)
        means = means * scale
        all_k.append(k_vals)

        ax.plot(k_vals, means, color=color, label=name, zorder=2)

        if show_std:
            stds = grp.std().reindex(k_vals).fillna(0).values * scale
            ax.fill_between(k_vals, means - stds, means + stds,
                            color=color, alpha=0.25, linewidth=0, zorder=1)

    # The single shared bound, drawn once on top of everything (zorder=3). Its
    # formula lives in the legend (see make_legend), not as an annotation.
    if show_bound and m is not None and n is not None and all_k:
        k_grid = np.unique(np.concatenate(all_k))
        bound  = shared_bound(m, n, k_grid)
        ax.plot(k_grid, bound, color='black', linewidth=1.4,
                linestyle=(0, (4, 3)), zorder=3)

    style_ax(ax)
    ax.set_xlabel(r'$k$')
    if show_ylabel:
        ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.margins(x=0)


def make_legend(fig, algo_names, data, show_std: bool, show_bound: bool = True,
                ncols: int = 4):
    algo_handles = [
        plt.Line2D([0], [0], color=COLORS[i % len(COLORS)], linewidth=1.2,
                   label=name)
        for i, name in enumerate(algo_names)
        if name in data
    ]
    # Extras: the standard-deviation swatch and the shared bound with its
    # formula as the label (a proper fraction), rendered as a black dashed line
    # matching the one drawn on both subplots.
    extra_handles = []
    if show_std:
        extra_handles.append(
            plt.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.25,
                           label='standard deviation')
        )
    if show_bound:
        extra_handles.append(
            plt.Line2D([0], [0], color='black', linewidth=1.4,
                       linestyle=(0, (4, 3)), label=BOUND_LATEX)
        )

    # fig.legend fills column-major: with 9 algorithms and ncols=4 (3 rows) the
    # two extras appended last fill the top two rows of the fourth column —
    # 'standard deviation' on row 1, the bound fraction on row 2. No blank
    # spacer handles: they reserved a full column each and blew the gap before
    # the last column wide open.
    handles = algo_handles + extra_handles

    fig.legend(handles=handles, loc='upper center',
               bbox_to_anchor=(0.5, 0.0), ncols=ncols,
               frameon=False, handlelength=1.4, handletextpad=0.5,
               columnspacing=1.0)


def save_figure(fig, stem: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in [FIGURES_DIR / f'{stem}_{timestamp}.pdf',
                 FIGURES_DIR / f'{stem}.pdf']:
        fig.savefig(path, bbox_inches='tight')
        print(f'Saved: {path}')
    plt.close(fig)


def make_figure(exp_name, cfg, data, algo_names, out_stem):
    """Two subplots: left = √m/‖X_S† X‖_F, right = ‖X†‖_F/‖X_S†‖_F, each with
    the shared lower bound √((k-m+1)/(n-m+1)) drawn (labelled in the legend)."""
    # Randomized algorithms have multiple rows per k; deterministic ones a
    # single row (std = 0, no visible band). Show the std machinery whenever
    # any algorithm has more than one trial at some k.
    show_std   = any((df.groupby('k').size() > 1).any() for df in data.values())
    m, n       = infer_m_n(cfg)
    ylabel_left  = r'$\sqrt{m} \,/\, \Vert X_\mathcal{S}^\dag X \Vert_F$'
    ylabel_right = r'$\Vert X^\dag \Vert_F \,/\, \Vert X_\mathcal{S}^\dag \Vert_F$'

    # Slightly wider than TEXT_WIDTH: the 4-column legend below the axes is
    # the widest element, so the axes may as well use the same width.
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(1.15 * TEXT_WIDTH, 0.5 * TEXT_WIDTH),
        layout='constrained')

    plot_metric_subplot(ax_left, data, algo_names,
                        metric_col='X_S_dag_X_frobenius_norm_inv',
                        show_ylabel=True, ylabel=ylabel_left,
                        show_std=show_std, m=m, n=n)

    plot_metric_subplot(ax_right, data, algo_names,
                        metric_col='pinv_frobenius_norm_ratio',
                        show_ylabel=True, ylabel=ylabel_right,
                        show_std=show_std, m=m, n=n)

    # 9 algorithms + 2 extras at 4 columns → three rows filled column-major:
    # algorithms occupy the first three columns; 'standard deviation' and the
    # bound fraction fill the top two rows of the fourth.
    make_legend(fig, algo_names, data, show_std=show_std, show_bound=True,
                ncols=4)
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
