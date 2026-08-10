#!/usr/bin/env python3
"""
Rebuild results/index.json from every experiment present in the results folder.

Each Tester run overwrites index.json with only the experiments of the config it
just ran, so after running several configs in sequence the file describes the
last one alone. This rebuilds it from the folders on disk — every subfolder
carrying a config.json is an experiment that really ran — so all of them stay
listed alongside each other.

The plotters no longer depend on this being correct (plot_k_sweep.py scans the
results folder itself, precisely because index.json is so easy to clobber), but
the file is a repo-wide convention that other tooling still reads, so the
runners keep it up to date.

Environment:
    RESULTS_DIR – results directory to index (default: <script_dir>/results)
"""

import json
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = Path(os.environ.get('RESULTS_DIR', SCRIPT_DIR / 'results'))


def main():
    if not RESULTS_DIR.is_dir():
        raise FileNotFoundError(f'No results directory: {RESULTS_DIR}')

    experiments = sorted(cfg.parent.name.replace('_', ' ')
                         for cfg in RESULTS_DIR.glob('*/config.json'))

    (RESULTS_DIR / 'index.json').write_text(
        json.dumps({'experiments': experiments}, indent=4) + '\n')
    print(f'index.json updated: {experiments}')


if __name__ == '__main__':
    main()
