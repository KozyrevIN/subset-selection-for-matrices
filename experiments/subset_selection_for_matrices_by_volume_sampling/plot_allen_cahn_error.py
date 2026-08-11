#!/usr/bin/env python3
"""
Plot script for the Allen-Cahn accuracy experiment of the
subset_selection_for_matrices_by_volume_sampling experiment.

Companion to plot_k_sweep.py (the superconductivity figure). Where that one compares
the selectors on a fixed data matrix, this one compares them on the matrices a
time-dependent problem produces: the TT unfoldings of the evolving state of

    df/dt = kappa * lap(f) + f - f^3   on the periodic cube [0, 2*pi]^3,

integrated with the fixed-skeleton TT-cross Solver under RK2. Accuracy is the
relative Frobenius error against a dense full-grid reference running the
identical discretization, so what is plotted is purely the low-rank error each
selector's skeleton choice incurs.

The runs are fixed-rank (atol = rtol = 0, every bond pinned to RANK), so all
selectors are compared at exactly equal cost and no rank adaptivity muddies the
comparison.

A single panel: one curve per algorithm from the AllenCahnTester run, plus the
black dashed 'best rank-r' curve — the error of the optimal TT approximation of
the reference at that same fixed rank, the floor no rank-limited integrator can
beat.

Usage (from repo root or from this directory):
    python experiments/subset_selection_for_matrices_by_volume_sampling/plot_allen_cahn_error.py

Environment overrides:
    RESULTS_DIR    – path to the results directory (default: <script_dir>/results)
    FIGURES_DIR    – path to save figures          (default: <script_dir>/figures)
    RESULTS_SUBDIR – subfolder of RESULTS_DIR with the run
                                                   (default: allen_cahn)
    RANK           – fixed TT rank, named in the best-rank-r legend entry
                     (default: read from the run's own max_rank column)
"""

import os
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

RESULTS_SUBDIR = os.environ.get('RESULTS_SUBDIR', 'allen_cahn')

# The fixed rank is only ever *named* in the best-rank-r legend entry, never
# used to compute anything. run_allen_cahn.sh passes it from the config, but a
# hardcoded fallback silently mislabels the figure whenever the config's rank
# changes and the plotter is run standalone — so when RANK is unset it is read
# back from the run's own max_rank column (see resolve_rank).
RANK_ENV = os.environ.get('RANK')

# ── style (shared with plot_k_sweep.py) ───────────────────────────────────────────────
CM         = 1 / 2.54
TEXT_WIDTH = 17 * CM

plt.rcParams.update({
    "text.usetex":     True,
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

# Same canonical order as plot_k_sweep.py, so an algorithm keeps its colour across both
# figures of the experiment. Every name keeps its slot whether or not the run
# included it — the roster is a config choice now (the shipped configs leave out
# leverage scores and random columns, which diverge on this problem), and
# holding the positions keeps each remaining algorithm's colour identical to the
# unfolding figure, which does run them.
CANONICAL = ['FDVS', 'RDVS', 'Frobenius selection', 'Frobenius removal',
             'Dominant', 'Dominant-split', 'VS', 'DEIM', 'QDEIM',
             'leverage scores', 'random columns']


def resolve_rank(df) -> int:
    """The rank named in the best-rank-r legend entry.

    RANK from the environment wins (run_allen_cahn.sh passes the config's
    value), but otherwise it comes from the run itself: every row records the
    max_rank it was integrated at, so the label cannot drift out of sync with
    the data the way a hardcoded default does."""
    if RANK_ENV is not None:
        return int(RANK_ENV)

    ranks = df['max_rank'].unique()
    if len(ranks) != 1:
        raise ValueError(
            f"Run mixes several max_rank values ({sorted(ranks)}); "
            "set RANK explicitly to say which one the figure is about."
        )
    return int(ranks[0])


def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', length=2, width=0.5,
                   color='gray', direction='in')
    ax.tick_params(axis='both', which='minor', length=1, width=0.5,
                   color='gray', direction='in')


def load_errors(folder: Path) -> pd.DataFrame:
    csv = folder / 'allen_cahn_errors.csv'
    if not csv.exists():
        raise FileNotFoundError(
            f"Errors CSV not found: {csv}\n"
            "Run the AllenCahnTester binary first (see run_allen_cahn.sh)."
        )
    return pd.read_csv(csv)


def error_ylim(df_list):
    """y-limits snug around what is actually drawn: the best-rank-r floor at the
    bottom, the top of the widest ± std band at the top, plus a small margin.

    The margin is a fixed factor on the log axis, so it reads as the same
    visual gap at either end. It used to be two decades of headroom, which was
    there to keep a diverging selector from squashing every converged curve —
    with no divergence left, that space is simply empty."""
    margin = 1.3

    tops, floors = [], []
    for df in df_list:
        # The band as drawn, not the raw samples: exp(mu + sigma) in log space,
        # matching plot_error_subplot exactly.
        log_err = np.log(df['error'].clip(lower=np.finfo(float).tiny))
        grp = log_err.groupby([df['algorithm'], df['t']])
        tops.append(np.exp(grp.mean() + grp.std().fillna(0)).max())
        floors.append(df['best_error'].min())

    top = max(tops)
    floor_min = min(floors)
    if not (top > 0) or not (floor_min > 0):
        return None
    return floor_min / margin, top * margin


def plot_error_subplot(ax, df, show_ylabel: bool, title: str):
    """Relative Frobenius error against the dense reference vs. time, one curve per
    algorithm (mean over samples, ± std band for the randomized ones), plus the
    black dashed best-rank-r floor."""
    algo_names = [a for a in CANONICAL if a in set(df['algorithm'])]
    algo_names += sorted(set(df['algorithm']) - set(CANONICAL))

    for idx, name in enumerate(algo_names):
        sub = df[df['algorithm'] == name]
        if sub.empty:
            continue
        # Colour by canonical position, not by plotting order, so a missing
        # algorithm does not shift everyone else's colour.
        color = COLORS[CANONICAL.index(name) % len(COLORS)] \
            if name in CANONICAL else 'gray'

        t   = np.sort(sub['t'].unique())
        grp = sub.groupby('t')['error']

        # Mean and std are taken in log space, then mapped back: the centre is
        # the geometric mean and the band exp(mu ± sigma). On a log axis that
        # is the natural summary — the band is symmetric as *drawn*, and being
        # a multiplicative factor it stays strictly positive however wide the
        # spread gets, where an arithmetic mean - std would cross zero.
        log_err = np.log(sub['error'].clip(lower=np.finfo(float).tiny))
        log_grp = log_err.groupby(sub['t'])
        mu      = log_grp.mean().reindex(t).values
        centre  = np.exp(mu)

        ax.plot(t, centre, color=color, label=name, zorder=2)

        if (sub.groupby('t').size() > 1).any():
            sigma = log_grp.std().reindex(t).values
            ax.fill_between(t, np.exp(mu - sigma), np.exp(mu + sigma),
                            color=color, alpha=0.25, linewidth=0, zorder=1)

    # The best rank-r floor. Each algorithm reaches its own ranks, so its floor
    # differs slightly; the curve drawn is the floor of the best-performing
    # (lowest-error) rank profile at each time — the most honest single line, and
    # a genuine lower envelope of what any of these runs could have achieved.
    floor = df.groupby('t')['best_error'].min()
    ax.plot(floor.index.values, floor.values, color='black', linewidth=1.4,
            linestyle=(0, (4, 3)), zorder=3)

    # An unstable skeleton makes the error diverge by tens of orders of
    # magnitude, which on a shared log axis would squash every converged curve
    # into one line. Clamp the top to a couple of decades above the worst
    # *converged* error (anything above 1 has already lost the solution, so it
    # carries no further information) and let the divergent curves run off the
    # top of the axis, which reads as the blow-up it is.
    style_ax(ax)
    ax.set_yscale('log')
    ax.set_xlabel(r'$t$')
    if show_ylabel:
        ax.set_ylabel(r'$\Vert f - f_{\mathrm{ref}} \Vert_F \,/\, '
                      r'\Vert f_{\mathrm{ref}} \Vert_F$')
    if title:
        ax.set_title(title)
    ax.margins(x=0)


def make_legend(fig, df_list, best_label: str, ncols: int = 3):
    present = set()
    for df in df_list:
        present |= set(df['algorithm'])

    handles = [
        plt.Line2D([0], [0], color=COLORS[i % len(COLORS)], linewidth=1.2,
                   label=name)
        for i, name in enumerate(CANONICAL) if name in present
    ]
    handles.append(
        plt.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.25,
                      label=r'standard deviation')
    )
    handles.append(
        plt.Line2D([0], [0], color='black', linewidth=1.4,
                   linestyle=(0, (4, 3)), label=best_label)
    )

    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.0),
               ncols=ncols, frameon=False, handlelength=1.4,
               handletextpad=0.5, columnspacing=1.0)


def save_figure(fig, stem: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in [FIGURES_DIR / f'{stem}_{timestamp}.pdf',
                 FIGURES_DIR / f'{stem}.pdf']:
        fig.savefig(path, bbox_inches='tight')
        print(f'Saved: {path}')
    plt.close(fig)


def main():
    df = load_errors(RESULTS_DIR / RESULTS_SUBDIR)

    # Full text width: this is a standalone single panel, so it spans the page
    # rather than sitting in the half-width slot the two-panel version used.
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 0.55 * TEXT_WIDTH),
                           layout='constrained')

    plot_error_subplot(ax, df, show_ylabel=True, title=None)

    ylim = error_ylim([df])
    if ylim is not None:
        ax.set_ylim(*ylim)

    make_legend(fig, [df], rf'best rank-${resolve_rank(df)}$', ncols=4)
    save_figure(fig, 'plot_allen_cahn')


if __name__ == '__main__':
    main()
