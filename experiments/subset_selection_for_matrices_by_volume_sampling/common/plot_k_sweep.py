#!/usr/bin/env python3
"""
Plot script for the subset_selection_for_matrices_by_volume_sampling experiment.

Adapted from the spectral-norm experiment's plot_k_sweep.py. The deterministic and
randomized tester runs both write into the same results subfolder
(Superconductivity_dataset), so this script globs every CSV in the folder
rather than trusting a single config's algorithm list.

Layout (matrix has a regression target):
  - left  = 1 / ‖X_S† X‖_F   (Frobenius norm of the projected pseudo-inverse)
  - right = ‖X†‖_F / ‖X_S†‖_F (Frobenius norm ratio)

Shared by every experiment in this folder. It reads and writes inside one
experiment's own directory, named by BASE_DIR — the run_*.sh scripts set it, and
run by hand it defaults to the working directory:

    cd experiments/subset_selection_for_matrices_by_volume_sampling/superconductivity
    python ../common/plot_k_sweep.py

Every experiment found under RESULTS_DIR is plotted. To plot just one, name it
(or any part of its name, case-insensitively) as an argument or in EXPERIMENT:

    python ../common/plot_k_sweep.py superconductivity   # only the dataset figure
    python ../common/plot_k_sweep.py unfolding           # only the unfolding one
    EXPERIMENT=superconductivity python ../common/plot_k_sweep.py

Environment overrides:
    BASE_DIR     – the experiment folder          (default: the working directory)
    RESULTS_DIR  – path to the results directory  (default: <BASE_DIR>/results)
    FIGURES_DIR  – path to save figures           (default: <BASE_DIR>/figures)
    EXPERIMENT   – plot only experiments whose name contains this (default: all)
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
# This script lives in common/, one level above the experiment it is plotting.
# Everything it touches — the results it reads, the figures it writes, and the
# matrix paths its configs name, which the Tester resolved against the folder it
# was run from — is relative to that experiment folder, not to this file.
BASE_DIR    = Path(os.environ.get('BASE_DIR', Path.cwd()))
RESULTS_DIR = Path(os.environ.get('RESULTS_DIR', BASE_DIR / 'results'))
FIGURES_DIR = Path(os.environ.get('FIGURES_DIR', BASE_DIR / 'figures'))
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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

# tab10 first, so every algorithm that already had a colour keeps it, then
# tab20b for the overflow: the canonical roster is longer than 10 now that DEIM
# and QDEIM are in it, and wrapping with % 10 would silently hand two
# algorithms the same colour.
COLORS = plt.cm.tab10.colors + plt.cm.tab20b.colors

# Map experiment name → output file stem
OUT_STEMS = {
    'Superconductivity dataset': 'plot_superconductivity',
    # A TT unfolding of the Allen-Cahn state taken from the middle of the run
    # (see allen_cahn/run.sh) — the same two-panel figure on a matrix the
    # PDE genuinely produces rather than on a static dataset.
    'Allen-Cahn unfolding': 'plot_allen_cahn_unfolding',
    # The same, on the acoustic wave equation (see acoustic/run.sh). That run
    # is tolerance-driven rather than fixed-rank, so its unfolding is taken
    # from the solver's own state mid-run.
    'Acoustic unfolding': 'plot_acoustic_unfolding',
    # POD modes of NOAA OI SST V2 weekly means, where selecting a column is
    # placing a sensor (see sensor_placement/run.sh). The reconstruction figure
    # that experiment is really about is its own plot_error.py; this is the
    # house k-sweep on the same matrix.
    'SST sensor placement': 'plot_sensor_placement_k_sweep',
}

# Experiments drawn as a single wide panel rather than two.
#
# The figure's two metrics — √m/‖X_S† X‖_F and ‖X†‖_F/‖X_S†‖_F — are different
# quantities and separate on the dataset experiments, which is why both are
# shown there. On the TT unfoldings of the PDE runs they track each other so
# closely that the panels are the same picture twice, so those keep only the
# pinv ratio, stretched across the full width.
#
# The SST modes are there for a stronger reason: that matrix has orthonormal
# rows, so the LQ factor L is orthogonal and Q = L^T X. Then ‖Q_S†‖ = ‖X_S†‖
# exactly, and the two metrics are equal, not merely close — the panels would
# be the same numbers, not just the same shape.
SINGLE_PANEL = {'Allen-Cahn unfolding', 'Acoustic unfolding',
                'SST sensor placement'}

# Canonical algorithm order (by display_name) so colours are stable across runs.
CANONICAL = ['FDVS', 'RDVS', 'Frobenius selection', 'Frobenius removal',
             'Dominant', 'Dominant-split', 'VS', 'DEIM', 'QDEIM',
             'Leverage scores', 'Random columns', 'Rect-maxvol']


def canonical_label(name: str) -> str:
    """The display name to draw for an algorithm, as a sentence-cased label.

    Names reach the plot from the result *filenames*, so they carry whatever
    capitalization the config that produced them used — and the older PDE
    unfolding configs wrote 'leverage scores' / 'random columns' where the
    dataset ones wrote them capitalized. Matching CANONICAL case-insensitively
    lets a run of either vintage keep its colour slot and print the same label,
    so old results need not be regenerated to get a consistent legend."""
    for canon in CANONICAL:
        if name.casefold() == canon.casefold():
            return canon
    return name

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
    if target_file:
        target_path = BASE_DIR / target_file
        if target_path.exists():
            with open(target_path) as fh:
                n = sum(1 for line in fh if line.strip())
            return m, n

    # No regression target (e.g. the Allen-Cahn unfolding): fall back to the
    # matrix file itself. It is stored tall and auto-transposed on load, so the
    # column count n of X is its number of data rows.
    file_path = cfg.get('matrix', {}).get('file_path')
    if not file_path:
        return m, None
    matrix_path = BASE_DIR / file_path
    if not matrix_path.exists():
        return m, None
    with open(matrix_path) as fh:
        rows = sum(1 for line in fh
                   if line.strip() and not line.lstrip().startswith('#'))
    return m, max(rows, m)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_experiment(exp_name: str):
    folder = RESULTS_DIR / exp_name.replace(' ', '_')
    with open(folder / 'config.json') as fh:
        cfg = json.load(fh)

    # Collect all CSVs present in the folder (covers results from multiple
    # tester runs with different configs writing to the same subfolder).
    all_csvs = {canonical_label(p.stem.replace('_', ' ')): p
                for p in folder.glob('*.csv')}

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

        # DEIM and Q-DEIM return one interpolation point per basis vector, so
        # they contribute a single k and a line through one point draws
        # nothing. Give a lone point a marker instead.
        if k_vals.size == 1:
            ax.plot(k_vals, means, color=color, label=name, marker='o',
                    markersize=3.5, linestyle='none', zorder=3)
            continue

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
    """The k-sweep figure: √m/‖X_S† X‖_F and ‖X†‖_F/‖X_S†‖_F against k, with the
    shared lower bound √((k-m+1)/(n-m+1)) drawn (labelled in the legend).

    Both metrics are shown side by side for the dataset experiments, where they
    do come apart. On the PDE unfoldings they agree so closely that the two
    panels are visually the same plot drawn twice, so those get the pinv ratio
    alone, across the full width (see SINGLE_PANEL)."""
    # Randomized algorithms have multiple rows per k; deterministic ones a
    # single row (std = 0, no visible band). Show the std machinery whenever
    # any algorithm has more than one trial at some k.
    show_std   = any((df.groupby('k').size() > 1).any() for df in data.values())
    m, n       = infer_m_n(cfg)
    ylabel_left  = r'$\sqrt{m} \,/\, \Vert X_\mathcal{S}^\dag X \Vert_F$'
    ylabel_right = r'$\Vert X^\dag \Vert_F \,/\, \Vert X_\mathcal{S}^\dag \Vert_F$'

    if exp_name in SINGLE_PANEL:
        # Plain text width, the same aspect as the single-panel error figures
        # (allen_cahn/plot_error.py): one axes stretched to the two-panel
        # version's 1.15 * TEXT_WIDTH reads as uncomfortably wide, and the
        # legend still fits at four columns.
        fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 0.55 * TEXT_WIDTH),
                               layout='constrained')
        plot_metric_subplot(ax, data, algo_names,
                            metric_col='pinv_frobenius_norm_ratio',
                            show_ylabel=True, ylabel=ylabel_right,
                            show_std=show_std, m=m, n=n)
        make_legend(fig, algo_names, data, show_std=show_std, show_bound=True,
                    ncols=4)
        save_figure(fig, out_stem)
        return

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

def discover_experiments():
    """Every experiment present in RESULTS_DIR, by name, in a stable order.

    The results folder is its own manifest: the Tester writes one subfolder per
    experiment, each carrying the config.json it ran with, and the folder name
    is the experiment name with spaces turned into underscores. Scanning for
    those is authoritative in a way index.json is not — each Tester run
    *overwrites* index.json with only the experiments of the config it just
    ran, so running one config by hand silently drops every other experiment's
    figure until a runner rebuilds the file.

    index.json is still read when it lists something this scan missed, so a
    hand-curated index (or results laid out some other way) keeps working.
    """
    found = {cfg.parent.name.replace('_', ' ')
             for cfg in RESULTS_DIR.glob('*/config.json')}

    index_file = RESULTS_DIR / 'index.json'
    if index_file.exists():
        with open(index_file) as fh:
            found |= set(json.load(fh).get('experiments', []))

    return sorted(found)


def main():
    experiments = discover_experiments()
    if not experiments:
        raise FileNotFoundError(
            f"No experiment results found under {RESULTS_DIR}\n"
            "Run the tester binary with the config(s) first."
        )

    # Optional filter: a substring of the experiment name, from the command
    # line or EXPERIMENT. Substring rather than exact match so the names in the
    # results folder ("Superconductivity dataset", "Allen-Cahn unfolding") do
    # not have to be typed in full.
    selector = (sys.argv[1] if len(sys.argv) > 1
                else os.environ.get('EXPERIMENT', '')).strip()
    if selector:
        matched = [e for e in experiments if selector.lower() in e.lower()]
        if not matched:
            raise SystemExit(
                f"No experiment matches {selector!r}.\n"
                f"Available: {', '.join(experiments)}"
            )
        experiments = matched

    for exp_name in experiments:
        print(f'Loading: {exp_name}')
        cfg, data = load_experiment(exp_name)

        algo_names = cfg.get('_all_algo_names',
                             [a.get('display_name', a['name']) for a in cfg['algorithms']])
        stem       = OUT_STEMS.get(exp_name, exp_name.replace(' ', '_').lower())

        make_figure(exp_name, cfg, data, algo_names, stem)


if __name__ == '__main__':
    main()
