#!/usr/bin/env bash
# Build, run experiments, and produce plots for the
# subset_selection_for_matrices_by_volume_sampling experiment.
#
# Usage (from repo root or from this directory):
#   bash experiments/subset_selection_for_matrices_by_volume_sampling/run_superconductivity.sh
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
PLOTTER="$SCRIPT_DIR/plot_k_sweep.py"

# ── 1. build ──────────────────────────────────────────────────────────────────
echo "==> Building (config: $BUILD_TYPE) …"
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" --target MatSubsetExperiments -j"$(nproc)"

# The Tester's output directory has moved between CMake layouts (tools/ in
# older build trees, matrix_tools/ in freshly configured ones), so locate it
# rather than assuming either.
TESTER=""
for candidate in "$BUILD_DIR/matrix_tools/Tester" "$BUILD_DIR/tools/Tester"; do
    if [[ -x "$candidate" ]]; then
        TESTER="$candidate"
        break
    fi
done
if [[ -z "$TESTER" ]]; then
    TESTER="$(find "$BUILD_DIR" -name Tester -type f -perm -u+x -print -quit)"
fi
if [[ -z "$TESTER" ]]; then
    echo "ERROR: Tester binary not found under $BUILD_DIR" >&2
    exit 1
fi
echo "    using Tester at $TESTER"

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

# ── 4. refresh index.json ─────────────────────────────────────────────────────
# Each tester run overwrites index.json with only its experiments; rebuild it
# from all result subfolders that contain a config.json.
echo ""
echo "==> Refreshing index.json …"
RESULTS_DIR="$RESULTS_DIR" "$PYTHON" "$SCRIPT_DIR/update_index.py"

# ── 5. plots ──────────────────────────────────────────────────────────────────
echo ""
echo "==> Plotting …"
RESULTS_DIR="$RESULTS_DIR" FIGURES_DIR="$FIGURES_DIR" "$PYTHON" "$PLOTTER" \
    "Superconductivity dataset"

echo ""
echo "==> All done. Figures saved to $FIGURES_DIR/"
