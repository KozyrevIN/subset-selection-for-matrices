#!/usr/bin/env python3
"""
Plot generation script for subset selection experiments.
Reads experimental data from the new structure and generates publication-quality plots.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ExperimentPlotter:
    """Generates plots from experiment results following the new data structure."""

    # LaTeX configuration and style setup
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
        "axes.edgecolor": 'gray'
    }

    # Document dimensions
    CM = 1/2.54  # centimeters in inches
    TEXT_WIDTH = 17 * CM

    # Plot parameters
    FIG_SIZE = (TEXT_WIDTH, 0.5 * TEXT_WIDTH * 0.8)  # Golden ratio

    def __init__(self, results_path: Path, output_path: Optional[Path] = None,
                 show_legend: bool = True, show_titles: bool = True,
                 log_scale: bool = False):
        """
        Initialize the plotter.

        Args:
            results_path: Path to the results directory containing index.json
            output_path: Path where plots will be saved (default: same as results_path)
            show_legend: Whether to show legend in plots
            show_titles: Whether to show subplot titles
            log_scale: Whether to use logarithmic y-axis scale
        """
        self.results_path = Path(results_path)
        self.output_path = Path(output_path) if output_path else self.results_path
        self.show_legend = show_legend
        self.show_titles = show_titles
        self.log_scale = log_scale

        # Apply plot configuration
        plt.rcParams.update(self.PLOT_CONFIG)

        # Load experiment index
        index_file = self.results_path / "index.json"
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")

        with open(index_file, 'r') as f:
            self.index = json.load(f)

    def load_experiment_data(self, experiment_name: str) -> Dict:
        """
        Load data for a single experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Dictionary containing config and data for all algorithms
        """
        # Convert experiment name to folder name (with underscores)
        folder_name = experiment_name.replace(' ', '_')
        experiment_path = self.results_path / folder_name

        if not experiment_path.exists():
            raise FileNotFoundError(f"Experiment folder not found: {experiment_path}")

        # Load config
        config_file = experiment_path / "config.json"
        with open(config_file, 'r') as f:
            config = json.load(f)

        # Load algorithm data
        algorithms_data = {}
        for algorithm_config in config['algorithms']:
            algorithm_name = algorithm_config['name']
            csv_filename = algorithm_name.replace(' ', '_') + '.csv'
            csv_path = experiment_path / csv_filename

            if csv_path.exists():
                df = pd.read_csv(csv_path)
                algorithms_data[algorithm_name] = df

        return {
            'config': config,
            'data': algorithms_data
        }

    def create_subplot(self, ax: plt.Axes, experiment_data: Dict,
                      csv_column: str, ylabel: str, position: str = 'left',
                      square_norms: bool = True):
        """
        Create a subplot for a single experiment.

        Args:
            ax: Matplotlib axes to plot on
            experiment_data: Dictionary with experiment config and data
            csv_column: Name of the column in CSV to plot
            ylabel: Label for y-axis
            position: Position of subplot ('left' or 'right'), affects y-axis label
            square_norms: Whether to square norm ratios before plotting (default: True)
        """
        config = experiment_data['config']
        data = experiment_data['data']

        max_y = 0
        algorithm_names = []

        # Determine the bound column name based on the metric
        bound_column = None
        if csv_column == 'pinv_spectral_norm_ratio':
            bound_column = 'spectral_bound'
        elif csv_column == 'pinv_frobenius_norm_ratio':
            bound_column = 'frobenius_bound'

        for idx, (algorithm_name, df) in enumerate(data.items()):
            algorithm_names.append(algorithm_name)

            # Group by k values
            k_values = np.sort(df['k'].unique())

            # Use specified metric column
            values = df[csv_column]

            # Square the norm ratios if requested (default for norm metrics)
            if square_norms and 'norm' in csv_column:
                values = values ** 2

            # Calculate statistics
            mean_values = values.groupby(df['k']).mean()
            std_values = values.groupby(df['k']).std()
            ci = std_values

            color = plt.cm.tab10(idx)

            # Main plot with confidence interval
            ax.plot(k_values, mean_values, color=color, label=algorithm_name)
            ax.fill_between(k_values, mean_values - ci, mean_values + ci,
                           color=color, alpha=0.3, linewidth=0)

            # Plot theoretical bounds if available and applicable
            if bound_column is not None and bound_column in df.columns:
                bound_values = df[bound_column]

                # Square the bounds if requested (to match the plotted values)
                if square_norms:
                    bound_values = bound_values ** 2

                # Get unique bound values per k (they should all be the same for a given k)
                mean_bound_values = bound_values.groupby(df['k']).mean()

                # Plot bounds as dashed lines with the same color
                ax.plot(k_values, mean_bound_values, color=color,
                       linestyle='--', linewidth=0.8, alpha=0.8)

                max_y = max(max_y, np.max(bound_values))

            max_y = max(max_y, np.max(values))

        # Set axis limits
        if len(k_values) > 0:
            ax.set_xlim(k_values[0], k_values[0] + (k_values[-1] - k_values[0]) * (51.0 / 50))
            if self.log_scale:
                ax.set_yscale('log')
            else:
                ax.set_ylim(0, max_y * (51.0 / 50))

        # Manage ticks
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', length=2, width=0.5,
                      color='gray', direction='in')
        ax.tick_params(axis='both', which='minor', length=1, width=0.5,
                      color='gray', direction='in')

        # Axis labels
        ax.set_xlabel(r'$k$')
        if position == 'left':
            ax.set_ylabel(ylabel)

        # Grid and borders
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add experiment name as title if enabled
        if self.show_titles:
            ax.set_title(config['name'])

        return algorithm_names

    def plot_experiments(self, experiment_names: Optional[List[str]] = None,
                        metric: str = 'spectral_norm',
                        output_filename: str = "plot.pdf",
                        square_norms: bool = False):
        """
        Create plots for specified experiments.

        Args:
            experiment_names: List of experiment names to plot.
                            If None, plots all experiments from index.json
            metric: Metric to plot ('spectral_norm', 'frobenius_norm', or 'wall_time')
            output_filename: Name of the output file
            square_norms: Whether to plot squared norm ratios (default: False)
        """
        if experiment_names is None:
            experiment_names = self.index['experiments']

        if not experiment_names:
            raise ValueError("No experiments to plot")

        # Map simplified metric names to CSV column names
        metric_column_map = {
            'spectral_norm': 'pinv_spectral_norm_ratio',
            'frobenius_norm': 'pinv_frobenius_norm_ratio',
            'wall_time': 'wall_time_ms'
        }

        # Metric display names
        metric_labels = {
            'spectral_norm': r'$\Vert X^\dag \Vert_2 / \Vert X_\mathcal{S}^\dag \Vert_2$',
            'frobenius_norm': r'$\Vert X^\dag \Vert_F / \Vert X_\mathcal{S}^\dag \Vert_F$',
            'wall_time': 'Wall time (ms)'
        }

        # Metric display names for squared norms
        metric_labels_squared = {
            'spectral_norm': r'$\Vert X^\dag \Vert_2^2 / \Vert X_\mathcal{S}^\dag \Vert_2^2$',
            'frobenius_norm': r'$\Vert X^\dag \Vert_F^2 / \Vert X_\mathcal{S}^\dag \Vert_F^2$',
            'wall_time': 'Wall time (ms)'
        }

        csv_column = metric_column_map.get(metric, metric)

        # Choose appropriate label based on square_norms setting
        if square_norms:
            ylabel = metric_labels_squared.get(metric, metric)
        else:
            ylabel = metric_labels.get(metric, metric)

        # Determine layout based on number of experiments
        n_experiments = len(experiment_names)

        if n_experiments == 1:
            fig, axes = plt.subplots(figsize=self.FIG_SIZE, nrows=1, ncols=1,
                                    layout='constrained')
            axes = [axes]  # Make it iterable
        else:
            # For multiple experiments, create subplots
            ncols = min(n_experiments, 2)
            nrows = (n_experiments + ncols - 1) // ncols
            fig_width = self.TEXT_WIDTH
            fig_height = 0.5 * self.TEXT_WIDTH * 0.8 * (nrows / 1.0)
            fig, axes = plt.subplots(figsize=(fig_width, fig_height),
                                    nrows=nrows, ncols=ncols,
                                    layout='constrained')
            axes = axes.flatten() if n_experiments > 1 else [axes]

        # Plot each experiment
        all_algorithm_names = []
        for idx, experiment_name in enumerate(experiment_names):
            print(f"Loading experiment: {experiment_name}")
            experiment_data = self.load_experiment_data(experiment_name)

            position = 'left' if (idx % 2 == 0 or n_experiments == 1) else 'right'
            algorithm_names = self.create_subplot(axes[idx], experiment_data,
                                                  csv_column, ylabel, position,
                                                  square_norms)

            if not all_algorithm_names:
                all_algorithm_names = algorithm_names

        # Hide unused subplots
        for idx in range(n_experiments, len(axes)):
            axes[idx].set_visible(False)

        # Create custom legend
        if self.show_legend and all_algorithm_names:
            custom_legend = []
            for idx, algorithm_name in enumerate(all_algorithm_names):
                color = plt.cm.tab10(idx)
                custom_legend.append(plt.Line2D([0], [0], marker='s', color='w',
                                               markerfacecolor=color,
                                               markeredgecolor='white',
                                               label=algorithm_name, markersize=6))

            custom_legend.append(plt.Line2D([0], [0], linestyle='-',
                                          label='mean value', color='black'))
            custom_legend.append(plt.Line2D([0], [0], marker='s', color='w',
                                          markerfacecolor='black',
                                          markeredgecolor='white', alpha=0.3,
                                          label='standard deviation', markersize=6))
            custom_legend.append(plt.Line2D([0], [0], linestyle='--',
                                          label='theoretical bound', color='black',
                                          linewidth=0.8, alpha=0.8))

            leg = fig.legend(handles=custom_legend, framealpha=0.9,
                           loc='upper center', bbox_to_anchor=(0.5, 0.00),
                           fancybox=True, ncols=4)
            leg.get_frame().set_edgecolor("1")

        # Save plot
        output_file = self.output_path / output_filename
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
        plt.close()


def main():
    """Main entry point for the plotting script."""
    parser = argparse.ArgumentParser(
        description='Generate plots from subset selection experiment results.'
    )
    parser.add_argument(
        '--results-path', '-r',
        type=Path,
        default=Path('out/results'),
        help='Path to the results directory containing index.json (default: out/results)'
    )
    parser.add_argument(
        '--output-path', '-o',
        type=Path,
        default=None,
        help='Path where plots will be saved (default: same as results-path)'
    )
    parser.add_argument(
        '--experiments', '-e',
        nargs='+',
        default=None,
        help='List of experiment names to plot (default: all experiments from index.json)'
    )
    parser.add_argument(
        '--output-filename', '-f',
        type=str,
        default='plot.pdf',
        help='Output filename (default: plot.pdf)'
    )
    parser.add_argument(
        '--metric', '-m',
        type=str,
        default='spectral_norm',
        choices=['spectral_norm', 'frobenius_norm', 'wall_time'],
        help='Metric to plot (default: spectral_norm)'
    )
    parser.add_argument(
        '--no-legend',
        action='store_true',
        help='Disable legend in plots'
    )
    parser.add_argument(
        '--no-titles',
        action='store_true',
        help='Disable subplot titles (useful when adding titles in LaTeX)'
    )
    parser.add_argument(
        '--square-norms',
        action='store_true',
        help='Plot squared norm ratios instead of regular ratios'
    )
    parser.add_argument(
        '--log-scale',
        action='store_true',
        help='Use logarithmic scale for y-axis'
    )

    args = parser.parse_args()

    # Create plotter
    plotter = ExperimentPlotter(
        results_path=args.results_path,
        output_path=args.output_path,
        show_legend=not args.no_legend,
        show_titles=not args.no_titles,
        log_scale=args.log_scale
    )

    # Generate plots
    plotter.plot_experiments(
        experiment_names=args.experiments,
        metric=args.metric,
        output_filename=args.output_filename,
        square_norms=args.square_norms
    )


if __name__ == '__main__':
    main()
