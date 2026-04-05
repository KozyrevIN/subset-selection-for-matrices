#!/usr/bin/env python3
"""
Plots per-swap traces of Frobenius norm ratio and volume ratio
as dominant runs from the volume-add-remove starting point.
Both metrics are shown relative to their value at swap 0
(end of volume-add-remove).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOT_CONFIG = {
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 0.8,
    "axes.linewidth": 0.5,
    "axes.edgecolor": "gray",
}

CM = 1 / 2.54
TEXT_WIDTH = 17 * CM


def main():
    parser = argparse.ArgumentParser(
        description="Plot dominant iteration traces starting from volume-add-remove."
    )
    parser.add_argument("--results-path", "-r", type=Path, default=Path("results"))
    parser.add_argument("--output-path",  "-o", type=Path, default=None)
    parser.add_argument("--output-filename", "-f", type=str,
                        default="iteration_trace.pdf")
    args = parser.parse_args()

    output_path = args.output_path or args.results_path
    df = pd.read_csv(args.results_path / "trace.csv")

    plt.rcParams.update(PLOT_CONFIG)

    fig, (ax_frob, ax_vol) = plt.subplots(
        nrows=1, ncols=2,
        figsize=(TEXT_WIDTH, 0.45 * TEXT_WIDTH),
        layout="constrained"
    )

    trials = df["trial"].unique()
    cmap = plt.cm.Blues(np.linspace(0.35, 0.9, len(trials)))

    for color, trial in zip(cmap, trials):
        t = df[df["trial"] == trial].sort_values("swap_index")
        v0_frob = t["frob_ratio_sq"].iloc[0]
        ax_frob.plot(t["swap_index"], t["frob_ratio_sq"] / v0_frob,
                     color=color, linewidth=0.7, alpha=0.8)
        ax_vol.plot(t["swap_index"], t["vol_sq_relative"],
                    color=color, linewidth=0.7, alpha=0.8)

    # Mean across trials (normalise each trial before averaging)
    def normalised_mean(col):
        normed = df.groupby("trial").apply(
            lambda t: t.set_index("swap_index")[col] / t[col].iloc[0],
            include_groups=False
        )
        return normed.groupby(level=1).mean()

    mean_frob = normalised_mean("frob_ratio_sq")
    mean_vol  = df.groupby("swap_index")["vol_sq_relative"].mean()

    ax_frob.plot(mean_frob.index, mean_frob.values,
                 color="black", linewidth=1.4, zorder=5, label="mean")
    ax_vol.plot(mean_vol.index, mean_vol.values,
                color="black", linewidth=1.4, zorder=5, label="mean")

    # Reference line at 1 (= value at swap 0)
    ax_frob.axhline(1, color="gray", linewidth=0.8, linestyle=":")
    ax_frob.axvline(0, color="gray", linewidth=0.8, linestyle=":")

    ax_frob.set_xlabel("Dominant swap index")
    ax_frob.set_ylabel(
        r"$\Vert X_\mathcal{S}^\dag \Vert_F^{-2}\,/\,"
        r"\Vert X_{\mathcal{S}_0}^\dag \Vert_F^{-2}$"
    )
    ax_frob.set_title(r"Frobenius norm ratio (relative to iter.\ 0)")

    ax_vol.set_xlabel("Dominant swap index")
    ax_vol.set_ylabel(
        r"$\log(\mathrm{vol}^2(X_\mathcal{S})\,/\,\mathrm{vol}^2(X_{\mathcal{S}_0}))$"
    )
    ax_vol.set_title(r"log squared volume ratio (relative to iter.\ 0)")

    for ax in (ax_frob, ax_vol):
        ax.minorticks_on()
        ax.tick_params(axis="both", which="major", length=2, width=0.5,
                       color="gray", direction="in")
        ax.tick_params(axis="both", which="minor", length=1, width=0.5,
                       color="gray", direction="in")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
        ax.legend()

    out_file = output_path / args.output_filename
    plt.savefig(out_file, bbox_inches="tight")
    print(f"Plot saved to: {out_file}")


if __name__ == "__main__":
    main()
