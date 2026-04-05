#!/usr/bin/env python3
"""
Scatter plotter for the volume-vs-frobenius experiment.
Reads results produced by the Tester binary and generates a scatter plot of
vol(X_S)/vol(X)  vs  ||X†||_F / ||X_S†||_F  for each experiment (k value).
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Style                                                                        #
# --------------------------------------------------------------------------- #
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


def load_experiment(results_path: Path, experiment_name: str) -> dict:
    folder = results_path / experiment_name.replace(" ", "_")
    with open(folder / "config.json") as f:
        config = json.load(f)

    data = {}
    for alg in config["algorithms"]:
        name = alg.get("display_name", alg["name"])
        csv_path = folder / (name.replace(" ", "_") + ".csv")
        if csv_path.exists():
            data[name] = pd.read_csv(csv_path)

    return {"config": config, "data": data}


def make_scatter(results_path: Path, output_path: Path,
                 output_filename: str, log_scale: bool,
                 n_bins: int = 30) -> None:

    plt.rcParams.update(PLOT_CONFIG)

    with open(results_path / "index.json") as f:
        index = json.load(f)
    experiment_names = index["experiments"]

    n = len(experiment_names)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig_width = TEXT_WIDTH
    fig_height = TEXT_WIDTH * 0.8 * nrows / ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(fig_width, fig_height),
                             layout="constrained")
    axes = np.array(axes).flatten()

    for ax_idx, exp_name in enumerate(experiment_names):
        exp = load_experiment(results_path, exp_name)
        config = exp["config"]
        ax = axes[ax_idx]

        m = config["matrix"]["rows"]
        k_values = config.get("k_values", [])
        k = k_values[0] if k_values else "?"

        for alg_idx, (alg_name, df) in enumerate(exp["data"].items()):
            color = plt.cm.tab10(alg_idx)

            vol = df["volume_ratio"].values
            frob = df["pinv_frobenius_norm_ratio"].values ** 2

            ax.scatter(vol, frob, s=2, alpha=0.1, color=color,
                       label=alg_name, rasterized=True)

            # Binned mean overlay
            bins = np.quantile(vol, np.linspace(0, 1, n_bins + 1))
            bin_ids = np.digitize(vol, bins[1:-1])
            bin_centers, bin_means = [], []
            for b in range(n_bins):
                mask = bin_ids == b
                if mask.sum() >= 2:
                    bin_centers.append(vol[mask].mean())
                    bin_means.append(frob[mask].mean())
            ax.plot(bin_centers, bin_means, color="black",
                    linewidth=1.2, zorder=3)

        ax.set_xlabel(r"$\mathrm{vol}(X_\mathcal{S})\,/\,\mathrm{vol}(X)$")
        ax.set_ylabel(r"$\Vert X^\dag \Vert_F^2\,/\,\Vert X_\mathcal{S}^\dag \Vert_F^2$")
        ax.set_title(f"$k = {k}$,\\ $m = {m}$")

        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")

        ax.minorticks_on()
        ax.tick_params(axis="both", which="major", length=2, width=0.5,
                       color="gray", direction="in")
        ax.tick_params(axis="both", which="minor", length=1, width=0.5,
                       color="gray", direction="in")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)

        if len(exp["data"]) > 1:
            ax.legend(markerscale=4, framealpha=0.9)

    for ax in axes[n:]:
        ax.set_visible(False)

    out_file = output_path / output_filename
    plt.savefig(out_file, bbox_inches="tight")
    print(f"Scatter plot saved to: {out_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Scatter plot: vol(X_S)/vol(X) vs Frobenius norm ratio."
    )
    parser.add_argument("--results-path", "-r", type=Path,
                        default=Path("results"),
                        help="Path to results directory (default: results)")
    parser.add_argument("--output-path", "-o", type=Path, default=None,
                        help="Output directory (default: same as results-path)")
    parser.add_argument("--output-filename", "-f", type=str,
                        default="volume_vs_frobenius.pdf",
                        help="Output filename (default: volume_vs_frobenius.pdf)")
    parser.add_argument("--log-scale", action="store_true",
                        help="Use log scale on both axes")
    parser.add_argument("--bins", type=int, default=30,
                        help="Number of bins for the mean overlay (default: 30)")
    args = parser.parse_args()

    output_path = args.output_path or args.results_path
    make_scatter(args.results_path, output_path,
                 args.output_filename, args.log_scale, args.bins)


if __name__ == "__main__":
    main()
