#!/usr/bin/env bash
# Build, run experiments, and produce plots for the
# subset_selection_for_matrices_in_spectral_norm experiment.
#
# Usage (from repo root or from this directory):
#   bash experiments/subset_selection_for_matrices_in_spectral_norm/run_all.sh
#
# Optional env overrides:
#   BUILD_DIR   – path to CMake build directory (default: <repo_root>/build/experiments)
#   BUILD_TYPE  – cmake --config value          (default: Release)
#   PYTHON      – python interpreter            (default: python3)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build/experiments}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
PYTHON="${PYTHON:-python3}"

RESULTS_DIR="$SCRIPT_DIR/results"
FIGURES_DIR="$SCRIPT_DIR/figures"
PLOTTER="$SCRIPT_DIR/plot.py"

# ── 1. build ──────────────────────────────────────────────────────────────────
echo "==> Building (config: $BUILD_TYPE) …"
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" --target MatSubsetExperiments -j"$(nproc)"

TESTER="$BUILD_DIR/tools/Tester"
if [[ ! -x "$TESTER" ]]; then
    echo "ERROR: Tester binary not found at $TESTER" >&2
    exit 1
fi

# ── 2. prepare data ───────────────────────────────────────────────────────────
echo ""
echo "==> Preparing abalone dataset …"
"$PYTHON" "$SCRIPT_DIR/prepare_abalone.py"

# ── 3. run experiments ────────────────────────────────────────────────────────
# Random experiments (orthonormal rows, graph) — 32 trials each
echo ""
echo "==> Running random-matrix experiments …"
"$TESTER" "$SCRIPT_DIR/config.json"

# Abalone: deterministic algorithms — 1 trial (matrix is fixed)
echo ""
echo "==> Running abalone experiment (deterministic algorithms) …"
"$TESTER" "$SCRIPT_DIR/config_abalone_deterministic.json"

# Abalone: randomized algorithms — 32 trials
echo ""
echo "==> Running abalone experiment (randomized algorithms) …"
"$TESTER" "$SCRIPT_DIR/config_abalone_randomized.json"

# ── 4. merge index.json ───────────────────────────────────────────────────────
# Each tester run overwrites index.json with only its experiments.
# Rebuild it from all result subfolders that contain a config.json.
echo ""
echo "==> Merging index.json …"
"$PYTHON" -c "
import json, pathlib
results = pathlib.Path('$RESULTS_DIR')
experiments = sorted(p.parent.name.replace('_', ' ') for p in results.glob('*/config.json'))
(results / 'index.json').write_text(json.dumps({'experiments': experiments}, indent=4))
print('index.json updated:', experiments)
"

# ── 5. plots ──────────────────────────────────────────────────────────────────
echo ""
echo "==> Plotting …"
RESULTS_DIR="$RESULTS_DIR" FIGURES_DIR="$FIGURES_DIR" "$PYTHON" "$PLOTTER"

echo ""
echo "==> All done. Figures saved to $FIGURES_DIR/"
