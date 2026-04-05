#!/usr/bin/env bash
# Master script: build, run experiments, then produce all three plots.
#
# Usage (from repo root or from this directory):
#   bash experiments/exchange_based_subset_selection/run_all.sh [config.json]
#
# If no config is given, defaults to config_small.json (30x500, fast test run).
# Pass config.json for the full 100x5000 experiment.
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
CONFIG="${1:-$SCRIPT_DIR/config_small.json}"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config file not found: $CONFIG" >&2
    exit 1
fi

# Resolve RESULTS_DIR from the config's output_path so plotters point at the
# right folder (config_small uses results_small, config uses results).
OUTPUT_PATH="$(python3 -c "import json,sys; print(json.load(open('$CONFIG'))['output_path'])")"
RESULTS_DIR="$SCRIPT_DIR/$OUTPUT_PATH"

# ── 1. build ──────────────────────────────────────────────────────────────────
echo "==> Building (config: $BUILD_TYPE) …"
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" --target MatSubsetExperiments -j"$(nproc)"

TESTER="$BUILD_DIR/tools/Tester"
if [[ ! -x "$TESTER" ]]; then
    echo "ERROR: Tester binary not found at $TESTER" >&2
    exit 1
fi

# ── 2. run experiments ────────────────────────────────────────────────────────
echo ""
echo "==> Running experiments (config: $CONFIG) …"
"$TESTER" "$CONFIG"

# ── 3. plots ──────────────────────────────────────────────────────────────────
echo ""
echo "==> Plotting dispersion vs k …"
RESULTS_DIR="$RESULTS_DIR" "$PYTHON" "$SCRIPT_DIR/plot_dispersion_vs_k.py"

echo ""
echo "==> Plotting algorithm comparison (violin) …"
RESULTS_DIR="$RESULTS_DIR" "$PYTHON" "$SCRIPT_DIR/plot_frob_orthonormal_comparison.py"

echo ""
echo "==> Plotting swap counts …"
RESULTS_DIR="$RESULTS_DIR" "$PYTHON" "$SCRIPT_DIR/plot_swap_counts.py"

echo ""
echo "==> All done. Figures saved to $SCRIPT_DIR/figures/"
