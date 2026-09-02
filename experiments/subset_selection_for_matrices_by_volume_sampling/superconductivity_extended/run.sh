#!/usr/bin/env bash
# Build, run the extended superconductivity-dataset experiment of the
# subset_selection_for_matrices_by_volume_sampling experiment, and produce its
# figure.
#
# This is superconductivity/ with three more algorithms on the same dataset:
# DEIM, GappyPOD+E and spectral selection. DEIM returns one interpolation point
# per basis vector, so it only runs at k = m = 82 and gets its own experiment
# block in config_deterministic.json (a single marker in the figure); the other
# two sweep the full k range alongside the volume-based selectors.
#
# Figures, both from the standard Tester + ../common/plot_k_sweep.py:
#   plot_superconductivity_extended.pdf          the two-panel k-sweep over the
#     prepared dataset, one curve per selection algorithm, in the Frobenius norm
#   plot_superconductivity_extended_spectral.pdf the same figure in the spectral
#     norm (NORM=spectral; both norms are in every CSV)
#
# Unlike the two PDE experiments this one selects from a static dataset, so
# there is no integration run and no generated config: the k range is fixed and
# lives in config_deterministic.json / config_randomized.json.
#
# Usage (from repo root or from this directory):
#   bash experiments/subset_selection_for_matrices_by_volume_sampling/superconductivity_extended/run.sh
#
# Optional env overrides:
#   BUILD_DIR   – path to CMake build directory (default: <repo_root>/build/experiments)
#   BUILD_TYPE  – cmake --config value          (default: Release)
#   PYTHON      – python interpreter            (default: python3)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# The scripts every experiment in this folder shares. They resolve their paths
# against BASE_DIR — this experiment's own folder — rather than against
# themselves, so each experiment keeps its results and figures to itself.
COMMON_DIR="$SCRIPT_DIR/../common"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build/experiments}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
PYTHON="${PYTHON:-python3}"

RESULTS_DIR="$SCRIPT_DIR/results"
FIGURES_DIR="$SCRIPT_DIR/figures"
PLOTTER="$COMMON_DIR/plot_k_sweep.py"

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
"$PYTHON" "$SCRIPT_DIR/prepare_data.py"

# ── 3. run experiments ────────────────────────────────────────────────────────
# The tester resolves the matrix "file_path" relative to the current working
# directory, so run from the experiment folder. The "output_path" (results) is
# resolved relative to the config file's parent directory. Both configs write
# into the same results subfolder (Superconductivity_extended).
echo ""
echo "==> Running extended superconductivity experiment (deterministic algorithms) …"
(cd "$SCRIPT_DIR" && "$TESTER" "$SCRIPT_DIR/config_deterministic.json")

echo ""
echo "==> Running extended superconductivity experiment (randomized algorithms) …"
(cd "$SCRIPT_DIR" && "$TESTER" "$SCRIPT_DIR/config_randomized.json")

# ── 4. refresh index.json ─────────────────────────────────────────────────────
# Each tester run overwrites index.json with only its own experiments; rebuild
# it from every result subfolder of this experiment that carries a config.json.
echo ""
echo "==> Refreshing index.json …"
BASE_DIR="$SCRIPT_DIR" RESULTS_DIR="$RESULTS_DIR" \
"$PYTHON" "$COMMON_DIR/update_index.py"

# ── 5. plots ──────────────────────────────────────────────────────────────────
echo ""
echo "==> Plotting …"
BASE_DIR="$SCRIPT_DIR" RESULTS_DIR="$RESULTS_DIR" FIGURES_DIR="$FIGURES_DIR" \
    "$PYTHON" "$PLOTTER" "Superconductivity extended"

# The same figure in the spectral norm. Both norms are in every CSV, so this is
# a second pass over the same results, saved as ..._spectral.pdf.
BASE_DIR="$SCRIPT_DIR" RESULTS_DIR="$RESULTS_DIR" FIGURES_DIR="$FIGURES_DIR" \
    NORM=spectral "$PYTHON" "$PLOTTER" "Superconductivity extended"

echo ""
echo "==> All done. Figures saved to $FIGURES_DIR/"
