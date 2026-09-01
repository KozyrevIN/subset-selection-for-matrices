#!/usr/bin/env python3
"""
Plot script for the NOAA OI SST V2 sensor-placement experiment.

The figure of Manohar, Brunton, Kutz & Brunton (2018), drawn from this repo's
selectors: how well a held-out sea-surface-temperature field is reconstructed
from k point sensors, and how badly the reconstruction amplifies noise.

Two panels, because a sensor set is judged on both at once:
  left   relative reconstruction error ‖Psi_r a_hat - y‖ / ‖y‖ vs. the number
         of sensors, where a_hat = Psi_r[S, :]^† y_S. The Tester writes the
         mean squared error of exactly this reconstruction into its
         `regression_mse` column (see prepare_data.py for why), so the relative
         error is √mse / rms(y).
  right  the error constant ‖Psi_r[S, :]^†‖₂, which multiplies any sensor noise
         on its way into a_hat. Psi_r has orthonormal columns, so Psi_r^T — the
         matrix the selectors see — has all singular values equal to 1 and
         ‖X^†‖₂ = 1; the CSV's `pinv_spectral_norm_ratio` is therefore exactly
         1 / ‖X_S^†‖₂, and this panel is its reciprocal.

A selector that reconstructs well with a small error constant is doing real
work; one that reaches a low error with a huge constant has fitted the test
field through an ill-conditioned inverse and will fall apart on noisy sensors.

DEIM and Q-DEIM produce one interpolation point per mode, so they exist only at
k = r and are drawn as single markers rather than curves.

Usage (from this directory):
    python plot_error.py

Environment overrides:
    RESULTS_DIR    – path to the results directory (default: <script_dir>/results)
    FIGURES_DIR    – path to save figures          (default: <script_dir>/figures)
    RESULTS_SUBDIR – subfolder of RESULTS_DIR with the run
                                                   (default: SST_sensor_placement)
    TARGET_FILE    – the held-out field the error is relative to
                                                   (default: data/sst_target.csv)
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
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_SUBDIR = os.environ.get('RESULTS_SUBDIR', 'SST_sensor_placement')
TARGET_FILE    = Path(os.environ.get('TARGET_FILE',
                                     SCRIPT_DIR / 'data' / 'sst_target.csv'))

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

COLORS = plt.cm.tab10.colors + plt.cm.tab20b.colors

# Same canonical order as the other plot scripts, so an algorithm keeps its
# colour across every figure of this folder.
CANONICAL = ['FDVS', 'RDVS', 'Frobenius selection', 'Frobenius removal',
             'Dominant', 'Dominant-split', 'VS', 'DEIM', 'QDEIM',
             'Leverage scores', 'Random columns', 'Rect-maxvol']

# The randomized selectors are the ones run many times per k, so they are the
# ones that get a spread band. Everything else contributes one row per k.
RANDOMIZED = {'VS', 'Leverage scores', 'Random columns'}


def canonical_label(name: str) -> str:
    """The display name to draw, matched case-insensitively against CANONICAL
    so a config that spelled its algorithm in lower case keeps the same colour
    slot and label as one that capitalized it."""
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
    ax.tick_params(axis='both', which='minor', length=1, width=0.5,
                   color='gray', direction='in')


def load_results(folder: Path) -> dict:
    """Every algorithm's CSV in the run folder, keyed by display name. Both
    Tester runs (deterministic and randomized) write here, and the k = r only
    entry for DEIM/QDEIM writes here too, so the folder is globbed rather than
    any one config trusted for the roster."""
    if not folder.is_dir():
        raise FileNotFoundError(
            f"No results folder: {folder}\n"
            "Run the Tester first (see run.sh)."
        )
    data = {canonical_label(p.stem.replace('_', ' ')): pd.read_csv(p)
            for p in sorted(folder.glob('*.csv'))}
    if not data:
        raise FileNotFoundError(f"No result CSVs in {folder}")
    return data


def target_rms() -> float:
    """rms of the held-out field, the denominator of the relative error."""
    y = np.loadtxt(TARGET_FILE)
    return float(np.sqrt((y ** 2).mean()))


def ordered_names(data: dict) -> list:
    return CANONICAL + sorted(n for n in data if n not in CANONICAL)


def plot_subplot(ax, data, names, values_of, ylabel: str):
    """Draw one panel. `values_of` maps an algorithm's rows to the quantity
    plotted, so both panels share the grouping, the spread band and the
    single-point handling."""
    for idx, name in enumerate(names):
        if name not in data:
            continue
        df = data[name]
        col = values_of(df)
        df = df.assign(_v=col)
        df = df[df['_v'].notna() & np.isfinite(df['_v'])]
        if df.empty:
            continue

        color  = COLORS[idx % len(COLORS)]
        k_vals = np.sort(df['k'].unique())
        grp    = df.groupby('k')['_v']
        means  = grp.mean().reindex(k_vals).values

        # DEIM and Q-DEIM exist at k = r alone; a one-point line renders as
        # nothing at all, so those are drawn as markers.
        if k_vals.size == 1:
            ax.plot(k_vals, means, color=color, label=name, marker='o',
                    markersize=3.5, linestyle='none', zorder=3)
            continue

        ax.plot(k_vals, means, color=color, label=name, zorder=2)
        if name in RANDOMIZED:
            stds = grp.std().reindex(k_vals).fillna(0).values
            ax.fill_between(k_vals, means - stds, means + stds,
                            color=color, alpha=0.25, linewidth=0, zorder=1)

    ax.set_yscale('log')
    style_ax(ax)
    ax.set_xlabel(r'number of sensors $k$')
    ax.set_ylabel(ylabel)
    ax.margins(x=0)


def make_legend(fig, data, names):
    handles = [
        plt.Line2D([0], [0], color=COLORS[i % len(COLORS)], linewidth=1.2,
                   label=name)
        for i, name in enumerate(names) if name in data
    ]
    handles.append(plt.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.25,
                                 label='standard deviation'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.0),
               ncols=4, frameon=False, handlelength=1.4, handletextpad=0.5,
               columnspacing=1.0)


def save_figure(fig, stem: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in [FIGURES_DIR / f'{stem}_{timestamp}.pdf',
                 FIGURES_DIR / f'{stem}.pdf']:
        fig.savefig(path, bbox_inches='tight')
        print(f'Saved: {path}')


def main():
    data  = load_results(RESULTS_DIR / RESULTS_SUBDIR)
    names = ordered_names(data)
    rms   = target_rms()

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 6 * CM))

    plot_subplot(
        axes[0], data, names,
        lambda df: np.sqrt(df['regression_mse']) / rms,
        r'$\|\Psi_r \hat{a} - y\|_2 \,/\, \|y\|_2$')

    # ‖X^†‖₂ = 1 for a matrix with orthonormal rows, so the stored ratio is the
    # reciprocal of the error constant this panel wants.
    plot_subplot(
        axes[1], data, names,
        lambda df: 1.0 / df['pinv_spectral_norm_ratio'],
        r'$\|\Psi_r[\mathcal{S}, :]^{\dag}\|_2$')

    fig.tight_layout()
    make_legend(fig, data, names)
    save_figure(fig, 'plot_sensor_placement')


if __name__ == '__main__':
    main()
