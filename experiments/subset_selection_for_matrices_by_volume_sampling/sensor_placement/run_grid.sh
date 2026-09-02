#!/usr/bin/env bash
# Build, run the r x f grid of the sensor-placement experiment, and produce its
# figures.
#
# The sweep run.sh drives is one rank against every sensor count. This is the
# other cut: the ranks in config.json's "modes_grid" against the oversampling
# factors in "oversampling" (k = f * r), with every algorithm at each of the
# r x f cells. Each rank is its own matrix, so its own Tester experiment and
# its own results folder ("SST sensor placement r<r>" → results/..._r<r>/); the
# bases are the leading columns of one SVD, so prepare_data.py writes them all
# without factorizing more than once.
#
# The grid entries set "save_indices", so each run also writes the selected
# subsets themselves to results/<folder>/indices/<Algorithm>.csv, one row per
# (k, trial) with the indices in selection order. Those are what a later map of
# sensor positions reads — the metric CSVs do not carry them.
#
# Figures (one per norm, both from plot_grid.py):
#   plot_sensor_placement_grid_spectral.pdf  the 3 x 3 grid of ‖X†‖₂/‖X_S†‖₂,
#     one bar per algorithm per cell. On an orthonormal basis this ratio is
#     exactly 1/‖Psi_r[S, :]†‖₂, the reciprocal of the usual error constant.
#   plot_sensor_placement_grid.pdf           the same in the Frobenius norm.
#
# Usage (from repo root or from this directory):
#   bash experiments/subset_selection_for_matrices_by_volume_sampling/sensor_placement/run_grid.sh
#
# The randomized selectors run "samples" trials at every cell, which is the
# expensive part of the grid; RANDOMIZED=0 skips them.
#
# Optional env overrides:
#   BUILD_DIR   – CMake build dir for the Tester (default: <repo_root>/build/experiments)
#   BUILD_TYPE  – cmake --config value           (default: Release)
#   PYTHON      – python interpreter             (default: python3)
#   CONFIG      – the experiment config          (default: config.json)
#   RANDOMIZED  – run the randomized config too  (default: 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
COMMON_DIR="$SCRIPT_DIR/../common"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build/experiments}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
PYTHON="${PYTHON:-python3}"
RANDOMIZED="${RANDOMIZED:-1}"

CONFIG="${CONFIG:-$SCRIPT_DIR/config.json}"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2
    exit 1
fi

RESULTS_DIR="$SCRIPT_DIR/results"
FIGURES_DIR="$SCRIPT_DIR/figures"

# ── 1. build ──────────────────────────────────────────────────────────────────
echo "==> Building the Tester (config: $BUILD_TYPE) …"
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" --target MatSubsetExperiments \
      -j"$(nproc)"

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

# ── 2. POD ────────────────────────────────────────────────────────────────────
# Downloads the dataset on a first run, takes the SVD of the training weeks,
# writes the sweep's mode matrix and every rank of the grid, and regenerates
# all four Tester configs from config.json. A grid basis that is already on
# disk and matches this SVD is kept rather than rewritten — they are hundreds
# of megabytes each.
echo ""
echo "==> Preparing the SST modes …"
BASE_DIR="$SCRIPT_DIR" CONFIG="$CONFIG" "$PYTHON" "$SCRIPT_DIR/prepare_data.py"

# ── 3. run the grid ───────────────────────────────────────────────────────────
# The Tester resolves the matrix "file_path" relative to the working directory,
# so run from the experiment folder.
echo ""
echo "==> Running the grid (deterministic algorithms) …"
(cd "$SCRIPT_DIR" && "$TESTER" "$SCRIPT_DIR/config_grid_deterministic.json")

if [[ "$RANDOMIZED" != "0" ]]; then
    echo ""
    echo "==> Running the grid (randomized algorithms) …"
    (cd "$SCRIPT_DIR" && "$TESTER" "$SCRIPT_DIR/config_grid_randomized.json")
else
    echo ""
    echo "==> Skipping the randomized algorithms (RANDOMIZED=0)"
fi

# ── 4. refresh index.json ─────────────────────────────────────────────────────
# Each tester run overwrites index.json with only its own experiments; rebuild
# it from every result subfolder of this experiment that carries a config.json,
# so the sweep's folder survives a grid run and vice versa.
echo ""
echo "==> Refreshing index.json …"
BASE_DIR="$SCRIPT_DIR" RESULTS_DIR="$RESULTS_DIR" \
"$PYTHON" "$COMMON_DIR/update_index.py"

# ── 5. plots ──────────────────────────────────────────────────────────────────
echo ""
echo "==> Plotting the grid (spectral norm) …"
BASE_DIR="$SCRIPT_DIR" RESULTS_DIR="$RESULTS_DIR" FIGURES_DIR="$FIGURES_DIR" \
    CONFIG="$CONFIG" NORM=spectral "$PYTHON" "$SCRIPT_DIR/plot_grid.py"

echo ""
echo "==> Plotting the grid (Frobenius norm) …"
BASE_DIR="$SCRIPT_DIR" RESULTS_DIR="$RESULTS_DIR" FIGURES_DIR="$FIGURES_DIR" \
    CONFIG="$CONFIG" NORM=frobenius "$PYTHON" "$SCRIPT_DIR/plot_grid.py"

echo ""
echo "==> All done. Figures saved to $FIGURES_DIR/"
echo "    Selected subsets: $RESULTS_DIR/<experiment>/indices/"
