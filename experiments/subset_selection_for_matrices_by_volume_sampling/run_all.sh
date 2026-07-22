#!/usr/bin/env bash
# Build, run experiments, and produce plots for the
# subset_selection_for_matrices_by_volume_sampling experiment.
#
# Usage (from repo root or from this directory):
#   bash experiments/subset_selection_for_matrices_by_volume_sampling/run_all.sh
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
echo "==> Preparing superconductivity dataset …"
"$PYTHON" "$SCRIPT_DIR/prepare_superconductivity.py"

# ── 3. run experiments ────────────────────────────────────────────────────────
# The tester resolves the matrix "file_path" relative to the current working
# directory, so run from the experiment folder. The "output_path" (results) is
# resolved relative to the config file's parent directory. Both configs write
# into the same results subfolder (Superconductivity_dataset).
echo ""
echo "==> Running superconductivity experiment (deterministic algorithms) …"
(cd "$SCRIPT_DIR" && "$TESTER" "$SCRIPT_DIR/config_superconductivity_deterministic.json")

echo ""
echo "==> Running superconductivity experiment (randomized algorithms) …"
(cd "$SCRIPT_DIR" && "$TESTER" "$SCRIPT_DIR/config_superconductivity_randomized.json")

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
