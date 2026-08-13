#!/usr/bin/env python3
"""
Plot script for the acoustic-wave accuracy experiment of the
subset_selection_for_matrices_by_volume_sampling experiment.

Companion to plot_allen_cahn_error.py, and deliberately its opposite. The
Allen-Cahn run is fixed-rank: every bond is pinned, so all selectors cost the
same and the only free variable is accuracy, measured against the best rank-r
approximation. A wave has no such luxury — the wavefront sweeps outward and the
rank needed to represent it grows with it, so that run is tolerance-driven and
the rank is an output, not an input:

    1/c^2 p_tt = lap(p) + s(x) f(t)   on the cube [0, extent]^3,

integrated with the adaptive TT-cross AdaptiveSolver under leapfrog. Accuracy
is the relative L2 error against a dense full-grid reference running the
identical discretization, so what is plotted is purely the low-rank error each
selector's skeleton choice incurs.

Two panels, because with the rank free neither one tells the story alone:
  left   the relative error against the dense reference vs. time;
  right  the rank each selector's state actually reached vs. time.
A selector that holds a given error at a lower rank is doing better work, and
one that lets the rank run away is failing even when its error looks fine.

There is deliberately no 'best rank-r' curve here. With the rank moving, the
floor would move under every curve and differently for each selector, so it is
not a single line — the rank panel replaces it.

Usage (from repo root or from this directory):
    python experiments/subset_selection_for_matrices_by_volume_sampling/plot_acoustic_error.py

Environment overrides:
    RESULTS_DIR    – path to the results directory (default: <script_dir>/results)
    FIGURES_DIR    – path to save figures          (default: <script_dir>/figures)
    RESULTS_SUBDIR – subfolder of RESULTS_DIR with the run
                                                   (default: acoustic)
    RANK_TOP       – upper limit of the rank panel (default: follow the
                     deterministic selectors, so a runaway randomized curve
                     does not set the scale for everyone)
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

RESULTS_SUBDIR = os.environ.get('RESULTS_SUBDIR', 'acoustic')

# Upper limit of the rank panel. Unset, it follows the deterministic selectors
# (see plot_rank_subplot) so a runaway randomized curve cannot squash them.
RANK_TOP = os.environ.get('RANK_TOP')

# ── style (shared with plot_k_sweep.py / plot_allen_cahn_error.py) ────────────
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

# tab10 first, so every algorithm keeps the colour it has in the other figures
# of this experiment, then tab20b for the overflow.
COLORS = plt.cm.tab10.colors + plt.cm.tab20b.colors

# Same canonical order as the other plot scripts, so an algorithm keeps its
# colour across every figure of the experiment. Names absent from this run keep
# their slot rather than shifting everyone else's colour. DEIM and QDEIM are
# listed for that reason alone — they cannot run here, since they require the
# skeleton width to equal the rank, which is exactly the case in which the rank
# cannot adapt (see main_acoustic_tester.cpp).
CANONICAL = ['FDVS', 'RDVS', 'Frobenius selection', 'Frobenius removal',
             'Dominant', 'Dominant-split', 'VS', 'DEIM', 'QDEIM',
             'Leverage scores', 'Random columns']


def canonical_label(name: str) -> str:
    """The display name to draw for an algorithm, as a sentence-cased label.

    Names come from whatever "display_name" the run's config used, and the
    older configs spelled 'leverage scores' / 'random columns' in lower case.
    Matching CANONICAL case-insensitively keeps the legend consistent without
    having to re-run anything."""
    for canon in CANONICAL:
        if name.casefold() == canon.casefold():
            return canon
    return name


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
    csv = folder / 'acoustic_errors.csv'
    if not csv.exists():
        raise FileNotFoundError(
            f"Errors CSV not found: {csv}\n"
            "Run the AcousticTester binary first (see run_acoustic.sh)."
        )
    df = pd.read_csv(csv)
    df['algorithm'] = df['algorithm'].map(canonical_label)
    return df


def algorithms_in(df) -> list:
    """The run's algorithms in canonical order, unknown ones appended."""
    names = [a for a in CANONICAL if a in set(df['algorithm'])]
    return names + sorted(set(df['algorithm']) - set(CANONICAL))


def colour_of(name: str):
    """Colour by canonical position, not by plotting order, so a missing
    algorithm does not shift everyone else's colour."""
    return (COLORS[CANONICAL.index(name) % len(COLORS)]
            if name in CANONICAL else 'gray')


def warmup_end(df) -> float:
    """The last time at which every selector still agrees exactly.

    The run opens with a warm-up in which the solver takes exact TT steps and
    no skeleton is ever selected, so every algorithm produces bit-identical
    states: what the error measures there is the plain TT truncation at rtol,
    with no subset selection involved at all. Those samples are one curve drawn
    once per algorithm, and they say nothing about the comparison — the panel
    starts where the curves first differ.

    Detected from the data rather than read from the config, so the figure
    stays correct whether the warm-up was given as "warmup_time" or as
    "warmup_steps", and whatever dt the grid implies."""
    spread = df.groupby('t')['error'].agg(lambda e: e.max() - e.min())
    identical = spread[spread == 0]
    if identical.empty:
        return float(df['t'].min())
    return float(identical.index.max())


def plot_error_subplot(ax, df):
    """Relative L2 error against the dense reference vs. time, one curve per
    algorithm (geometric mean over samples, ± std band for the randomized
    ones).

    Only the post-warm-up part is drawn: before that every selector holds the
    identical state, so the samples carry no comparison (see warmup_end)."""
    df = df[df['t'] > warmup_end(df)]

    for name in algorithms_in(df):
        sub = df[df['algorithm'] == name]
        if sub.empty:
            continue

        t = np.sort(sub['t'].unique())

        # Mean and std are taken in log space, then mapped back: the centre is
        # the geometric mean and the band exp(mu ± sigma). On a log axis that
        # is the natural summary — the band is symmetric as *drawn*, and being
        # a multiplicative factor it stays strictly positive however wide the
        # spread gets, where an arithmetic mean - std would cross zero.
        log_err = np.log(sub['error'].clip(lower=np.finfo(float).tiny))
        log_grp = log_err.groupby(sub['t'])
        mu      = log_grp.mean().reindex(t).values

        ax.plot(t, np.exp(mu), color=colour_of(name), label=name, zorder=2)

        if (sub.groupby('t').size() > 1).any():
            sigma = log_grp.std().reindex(t).values
            ax.fill_between(t, np.exp(mu - sigma), np.exp(mu + sigma),
                            color=colour_of(name), alpha=0.25, linewidth=0,
                            zorder=1)

    style_ax(ax)
    ax.set_yscale('log')
    ax.set_xlabel(r'$t$, s')
    ax.set_ylabel(r'$\Vert p - p_{\mathrm{ref}} \Vert_2 \,/\, '
                  r'\Vert p_{\mathrm{ref}} \Vert_2$')
    ax.margins(x=0)


def plot_rank_subplot(ax, df):
    """The rank each selector's state reached vs. time.

    The counterpart of the error panel: with the rank tolerance-driven rather
    than pinned, two selectors reaching the same error at different ranks are
    not doing equally well, and the difference is only visible here. Plotted on
    a linear axis — the rank is a small integer count, and reading it off
    directly is the point."""
    for name in algorithms_in(df):
        sub = df[df['algorithm'] == name]
        if sub.empty:
            continue

        t   = np.sort(sub['t'].unique())
        grp = sub.groupby('t')['max_rank']
        mean = grp.mean().reindex(t).values

        ax.plot(t, mean, color=colour_of(name), label=name, zorder=2)

        # The rank is an integer count, so it is averaged and banded directly
        # rather than in log space as the error is.
        if (sub.groupby('t').size() > 1).any():
            sigma = grp.std().reindex(t).fillna(0).values
            ax.fill_between(t, mean - sigma, mean + sigma,
                            color=colour_of(name), alpha=0.25, linewidth=0,
                            zorder=1)

    style_ax(ax)
    ax.set_xlabel(r'$t$, s')
    ax.set_ylabel(r'TT rank')
    ax.margins(x=0)

    # The top is set by the deterministic selectors rather than by the largest
    # rank in the run. A randomized selector that lets the rank run away would
    # otherwise set the scale for everyone and squash the band the comparison
    # actually lives in; clipping keeps that band readable and lets the
    # runaway curve leave the top of the axis, which reads as the blow-up it
    # is. RANK_TOP overrides it when a particular figure wants a fixed scale.
    if RANK_TOP is not None:
        ax.set_ylim(0, float(RANK_TOP))
    else:
        deterministic = df[df.groupby('algorithm')['sample']
                             .transform('nunique') == 1]
        reference = (deterministic if not deterministic.empty else df)
        ax.set_ylim(0, float(reference['max_rank'].max()) * 1.25)


def make_legend(fig, df, ncols: int = 4):
    present = set(df['algorithm'])
    handles = [
        plt.Line2D([0], [0], color=colour_of(name), linewidth=1.2, label=name)
        for name in CANONICAL if name in present
    ]
    handles.append(
        plt.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.25,
                      label=r'standard deviation')
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

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 0.42 * TEXT_WIDTH),
                             layout='constrained')

    plot_error_subplot(axes[0], df)
    plot_rank_subplot(axes[1], df)

    make_legend(fig, df, ncols=4)
    save_figure(fig, 'plot_acoustic')


if __name__ == '__main__':
    main()
