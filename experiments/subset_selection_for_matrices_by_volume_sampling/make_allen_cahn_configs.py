#!/usr/bin/env python3
"""
Generate the Tester configs for the mid-run Allen-Cahn unfolding.

The unfolding dumped by AllenCahnTester is a tall matrix whose shape depends
on the grid and on the ranks the solver happened to reach, so the k range cannot
be hard-coded the way the superconductivity configs do. This script reads the
dumped CSV, works out the (auto-transposed) matrix shape, and writes a
deterministic and a randomized config over k in [m, K_MAX_FACTOR * m] — m is
the TT rank for this matrix, so by default the sweep runs up to 4r rather than
all the way to n.

`MatrixFromFileGenerator` transposes a tall matrix to be wide, so the selectors
see m = (CSV column count) rows and n = (CSV row count) columns.

Environment (all set by run_allen_cahn.sh):
    UNFOLDING    – path to the dumped unfolding CSV
    RESULTS_DIR  – results directory the Tester writes into
    CONFIG_DET   – path to write the deterministic config to
    CONFIG_RAND  – path to write the randomized config to
    SAMPLES      – trials per k for the randomized algorithms (default 16)
    K_MAX_FACTOR – sweep k up to this multiple of m (default 4, i.e. up to 4r)
"""

import json
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

UNFOLDING   = Path(os.environ['UNFOLDING'])
RESULTS_DIR = Path(os.environ.get('RESULTS_DIR', SCRIPT_DIR / 'results'))
CONFIG_DET  = Path(os.environ.get('CONFIG_DET',
                                  SCRIPT_DIR / 'config_allen_cahn_deterministic.json'))
CONFIG_RAND = Path(os.environ.get('CONFIG_RAND',
                                  SCRIPT_DIR / 'config_allen_cahn_randomized.json'))
SAMPLES     = int(os.environ.get('SAMPLES', 16))

# k sweeps over [m, K_MAX_FACTOR * m] rather than all the way to n — see
# write_config. m is the TT rank here, so this is "up to 4r".
K_MAX_FACTOR = int(os.environ.get('K_MAX_FACTOR', 4))

# The experiment name doubles as the results subfolder (spaces → underscores)
# and as the key plot_k_sweep.py looks up in OUT_STEMS.
EXPERIMENT_NAME = 'Allen-Cahn unfolding'

# The nine algorithms of the superconductivity experiment, split the same way:
# deterministic ones run once per k, randomized ones run SAMPLES times. Dominant
# and Dominant-split take c = 1 + 1e-6 — at exactly 1 the swap loop can cycle
# forever on floating-point ties.
DETERMINISTIC = [
    {"display_name": "FDVS",                "name": "derandomized volume"},
    {"display_name": "RDVS",                "name": "spectral removal"},
    {"display_name": "Frobenius selection", "name": "frobenius selection"},
    {"display_name": "Frobenius removal",   "name": "frobenius removal"},
    {"display_name": "Dominant",            "name": "dominant",
     "c": 1.000001, "initialization": "greedy"},
    {"display_name": "Dominant-split",      "name": "volume add-remove",
     "c": 1.000001, "initialization": "greedy"},
]

RANDOMIZED = [
    {"display_name": "VS",              "name": "forward iterative volume sampling"},
    {"display_name": "leverage scores", "name": "leverage scores"},
    {"display_name": "random columns",  "name": "random columns"},
]


def matrix_shape(path: Path) -> tuple:
    """(m, n) of the matrix as the selectors will see it: the CSV is tall and
    MatrixFromFileGenerator transposes it, so m = columns and n = rows."""
    rows = 0
    cols = 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            rows += 1
            if cols == 0:
                cols = len(line.split(','))
    if rows == 0 or cols == 0:
        raise RuntimeError(f'Empty or unreadable unfolding: {path}')
    m, n = (cols, rows) if rows > cols else (rows, cols)
    return m, n


def write_config(path: Path, algorithms: list, m: int, n: int, trials: int,
                 matrix_path: Path):
    # The Tester resolves "file_path" relative to the working directory it is
    # run from (this experiment's folder) and "output_path" relative to the
    # config's own parent, matching how the superconductivity configs are used.
    #
    # Subset selection needs k >= m, and for this matrix m *is* the TT rank r
    # (the bond-1 left unfolding has one row per bond index). The interesting
    # part of the curve is the first few multiples of r: past that every
    # algorithm has long since flattened onto the unselected matrix, so
    # sweeping all the way to k = n only stretches the axis. Stop at
    # K_MAX_FACTOR * r, clamped to n for a matrix too narrow to reach it.
    k_stop = min(K_MAX_FACTOR * m, n)
    config = {
        "scalar": "double",
        "output_path": "results",
        "experiments": [
            {
                "name": EXPERIMENT_NAME,
                "enabled": True,
                "matrix": {
                    "type": "matrix from file",
                    "file_path": str(matrix_path),
                },
                "algorithms": algorithms,
                "k_values_ranges": [
                    {"start": m, "stop": k_stop, "step": 1}
                ],
                "trials_per_k": trials,
            }
        ],
    }
    path.write_text(json.dumps(config, indent=4) + '\n')
    print(f'Wrote {path} (m = {m}, n = {n}, k = {m}..{k_stop}, '
          f'trials = {trials})')


def main():
    m, n = matrix_shape(UNFOLDING)

    # Relative to this experiment's folder — the directory the Tester runs from.
    try:
        matrix_path = UNFOLDING.resolve().relative_to(SCRIPT_DIR.resolve())
    except ValueError:
        matrix_path = UNFOLDING.resolve()

    write_config(CONFIG_DET, DETERMINISTIC, m, n, 1, matrix_path)
    write_config(CONFIG_RAND, RANDOMIZED, m, n, SAMPLES, matrix_path)


if __name__ == '__main__':
    main()
