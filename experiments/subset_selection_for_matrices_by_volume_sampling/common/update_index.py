#!/usr/bin/env python3
"""
Rebuild results/index.json from every experiment present in the results folder.

Shared by every experiment in this folder; each one indexes its own results
directory, named by BASE_DIR.

Each Tester run overwrites index.json with only the experiments of the config it
just ran, so after running several configs in sequence the file describes the
last one alone. This rebuilds it from the folders on disk — every subfolder
carrying a config.json is an experiment that really ran — so all of them stay
listed alongside each other, whichever config was run last.

The plotters no longer depend on this being correct (plot_k_sweep.py scans the
results folder itself, precisely because index.json is so easy to clobber), but
the file is a repo-wide convention that other tooling still reads, so the
runners keep it up to date.

Environment:
    BASE_DIR    – the experiment folder (default: the working directory)
    RESULTS_DIR – results directory to index (default: <BASE_DIR>/results)
"""

import json
import os
from pathlib import Path

# This script lives in common/, one level above the experiment it is indexing,
# so paths are resolved against the experiment folder rather than against the
# script's own directory.
BASE_DIR = Path(os.environ.get('BASE_DIR', Path.cwd()))
RESULTS_DIR = Path(os.environ.get('RESULTS_DIR', BASE_DIR / 'results'))


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
