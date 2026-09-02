#!/usr/bin/env python3
"""
Plot script for the r x f grid of the NOAA OI SST V2 sensor-placement
experiment.

Where plot_error.py follows one rank across every sensor count, this draws the
other cut: a few ranks against a few oversampling factors. One panel per cell,
rows the retained modes r, columns the sensor count k = f * r, and inside each
panel one bar per algorithm of

    ‖X†‖ / ‖X_S†‖,

the pseudo-inverse norm ratio the rest of this folder plots — bounded above by
1, larger is better. Psi_r has orthonormal columns, so the matrix the selectors
see has ‖X†‖ = 1 in the spectral norm and this ratio is exactly 1 / ‖Psi_r[S,
:]†‖₂, the reciprocal of the error constant of plot_error.py's right panel.

The bars share one x order — the canonical algorithm order, so a colour means
the same algorithm in every panel — but each panel is scaled to its own data.
The question a cell answers is which algorithm wins there, and the ratio grows
by an order of magnitude between k = r and k = 2r, so a shared scale would
flatten the k = r column into invisibility. Read across a row by the axis
numbers, not by bar height.

DEIM and Q-DEIM produce one interpolation point per mode, so they appear only
in the k = r column; their bars are simply absent from the others.

Usage (from this directory):
    python plot_grid.py

Environment overrides:
    BASE_DIR     – the experiment folder          (default: this script's own)
    RESULTS_DIR  – path to the results directory  (default: <BASE_DIR>/results)
    FIGURES_DIR  – path to save figures           (default: <BASE_DIR>/figures)
    CONFIG       – the experiment config          (default: <BASE_DIR>/config.json)
    NORM         – 'frobenius' or 'spectral'      (default: spectral)
"""

import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
BASE_DIR    = Path(os.environ.get('BASE_DIR', SCRIPT_DIR))
RESULTS_DIR = Path(os.environ.get('RESULTS_DIR', BASE_DIR / 'results'))
FIGURES_DIR = Path(os.environ.get('FIGURES_DIR', BASE_DIR / 'figures'))
CONFIG      = Path(os.environ.get('CONFIG', BASE_DIR / 'config.json'))
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── norm ──────────────────────────────────────────────────────────────────────
# Both norms are in every Tester CSV. The spectral one is the default here: it
# is the error constant the sensor-placement literature reports, and on an
# orthonormal basis it is exactly 1 / ‖Psi_r[S, :]†‖₂.
NORM = os.environ.get('NORM', 'spectral').strip().lower()
if NORM not in ('frobenius', 'spectral'):
    raise SystemExit(f"NORM must be 'frobenius' or 'spectral', not {NORM!r}")

PINV_COL    = f'pinv_{NORM}_norm_ratio'
STEM        = 'plot_sensor_placement_grid' + ('' if NORM == 'frobenius'
                                              else '_spectral')
NORM_SUFFIX = '_F' if NORM == 'frobenius' else '_2'

# ── style (shared with ../common/plot_k_sweep.py) ─────────────────────────────
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

# tab10 first, then one shade per tab20b hue family — the same palette as
# ../common/plot_k_sweep.py, so an algorithm keeps its colour across figures.
_TAB20B = plt.cm.tab20b.colors
COLORS = (plt.cm.tab10.colors + _TAB20B[::4] +
          tuple(c for i, c in enumerate(_TAB20B) if i % 4))

CANONICAL = ['FDVS', 'RDVS', 'Frobenius selection', 'Frobenius removal',
             'Dominant', 'Dominant-split', 'VS', 'DEIM', 'QDEIM',
             'Leverage scores', 'Random columns', 'Rect-maxvol',
             'GappyPOD+E', 'Spectral selection']


def canonical_label(name: str) -> str:
    """The display name to draw, matched case-insensitively against CANONICAL
    so results written with a different capitalization keep their colour."""
    for canon in CANONICAL:
        if name.casefold() == canon.casefold():
            return canon
    return name


def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', length=2, width=0.5,
                   color='gray', direction='in')


def load_rank(cfg: dict, r: int) -> dict:
    """Every algorithm's rows for one rank, keyed by display name. The
    deterministic and randomized runs write into the same folder, so the folder
    is globbed rather than the config's algorithm list trusted."""
    folder = RESULTS_DIR / f'{cfg["experiment_name"]} r{r}'.replace(' ', '_')
    if not folder.is_dir():
        return {}
    return {canonical_label(p.stem.replace('_', ' ')): pd.read_csv(p)
            for p in folder.glob('*.csv')}


def value_at(df: pd.DataFrame, k: int):
    """The mean ratio at sensor count k, or None if this algorithm has no row
    there (DEIM and Q-DEIM outside the k = r column)."""
    rows = df[(df['k'] == k) & df[PINV_COL].notna()]
    return None if rows.empty else float(rows[PINV_COL].mean())


def error_at(df: pd.DataFrame, k: int):
    """The standard deviation over trials at k, for the randomized selectors;
    0 where a single trial was run."""
    rows = df[(df['k'] == k) & df[PINV_COL].notna()]
    return 0.0 if len(rows) < 2 else float(rows[PINV_COL].std())


def make_figure(cfg: dict, ranks: list, data: dict, k_grid: dict):
    """The r x f grid. Rows are ranks, columns oversampling factors, and each
    panel is one bar per algorithm."""
    factors = cfg.get('oversampling', [1, 1.5, 2])
    names = [n for n in CANONICAL
             if any(n in data[r] for r in ranks)]
    names += sorted({n for r in ranks for n in data[r]} - set(names))

    n_rows, n_cols = len(ranks), len(factors)
    # Not sharey: see the module docstring. Sharing it would also have to wait
    # for the whole row to be drawn before the limits are set, since fixing the
    # bottom at 0 on a shared axis freezes whatever top the first panel
    # autoscaled to and clips the rest.
    fig, axes = plt.subplots(
        n_rows, n_cols, squeeze=False,
        figsize=(1.15 * TEXT_WIDTH, 0.36 * TEXT_WIDTH * n_rows),
        layout='constrained')

    for i, r in enumerate(ranks):
        for j, factor in enumerate(factors):
            ax = axes[i][j]
            k  = k_grid[r][j]

            heights, errors, colors, ticks = [], [], [], []
            for idx, name in enumerate(names):
                df = data[r].get(name)
                value = None if df is None else value_at(df, k)
                # A missing bar is meaningful (DEIM away from k = r), so the
                # slot is kept and left empty rather than closed up: the bars
                # then line up across the row.
                heights.append(np.nan if value is None else value)
                errors.append(0.0 if value is None else error_at(df, k))
                colors.append(COLORS[idx % len(COLORS)])
                ticks.append(idx)

            ax.bar(ticks, heights, yerr=errors, color=colors, width=0.75,
                   error_kw={'elinewidth': 0.8, 'ecolor': '0.3'})
            style_ax(ax)
            ax.set_xticks([])
            ax.set_xlim(-0.8, len(names) - 0.2)
            ax.set_ylim(bottom=0)

            factor_text = (f'{factor:g}' if factor != 1 else '')
            ax.set_title(rf'$k = {factor_text}r = {k}$')
            if j == 0:
                ax.set_ylabel(rf'$r = {r}$')

    fig.supylabel(rf'$\Vert X^\dag \Vert{NORM_SUFFIX} \,/\, '
                  rf'\Vert X_\mathcal{{S}}^\dag \Vert{NORM_SUFFIX}$')

    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[i % len(COLORS)],
                             label=name)
               for i, name in enumerate(names)]
    if any(error_at(df, k_grid[r][j]) > 0
           for r in ranks for df in data[r].values()
           for j in range(len(factors))):
        handles.append(plt.Line2D([0], [0], color='0.3', linewidth=0.8,
                                  label='standard deviation'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.0),
               ncols=4, frameon=False, handlelength=1.4, handletextpad=0.5,
               columnspacing=1.0)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for path in [FIGURES_DIR / f'{STEM}_{timestamp}.pdf',
                 FIGURES_DIR / f'{STEM}.pdf']:
        fig.savefig(path, bbox_inches='tight')
        print(f'Saved: {path}')
    plt.close(fig)


def main():
    cfg = json.loads(CONFIG.read_text())
    ranks = [int(r) for r in cfg.get('modes_grid', [])]
    if not ranks:
        raise SystemExit(f'{CONFIG.name} has no "modes_grid" to plot')

    factors = cfg.get('oversampling', [1, 1.5, 2])
    k_grid  = {r: [int(round(f * r)) for f in factors] for r in ranks}

    data  = {r: load_rank(cfg, r) for r in ranks}
    empty = [r for r in ranks if not data[r]]
    if empty:
        raise SystemExit(
            'No results for rank(s) ' + ', '.join(map(str, empty)) +
            f' under {RESULTS_DIR}\nRun the grid configs first '
            '(see run_grid.sh).')

    for r in ranks:
        print(f'Loaded r = {r}: {len(data[r])} algorithms, '
              f'k = {", ".join(str(k) for k in k_grid[r])}')
    make_figure(cfg, ranks, data, k_grid)


if __name__ == '__main__':
    main()
