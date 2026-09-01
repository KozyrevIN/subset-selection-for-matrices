#!/usr/bin/env bash
# Build, run the sensor-placement experiment of the
# subset_selection_for_matrices_by_volume_sampling experiment, and produce its
# figures.
#
# The problem of Manohar, Brunton, Kutz & Brunton, "Data-Driven Sparse Sensor
# Placement for Reconstruction" (IEEE CSM 38(3), 2018), on the dataset that
# paper uses: NOAA OI SST V2 weekly means. POD on the training weeks gives r
# modes; placing k sensors is choosing k columns of Psi_r^T, so every selector
# in this library answers it directly. See prepare_data.py for the reduction.
#
# Figure 1 (plot_sensor_placement.pdf): relative reconstruction error of a
#   held-out weekly field vs. the number of sensors, and the error constant
#   ‖Psi_r[S, :]^†‖₂ beside it. Produced by the Tester + plot_error.py.
#
# Figure 2 (plot_sensor_placement_k_sweep.pdf): this folder's usual two-panel
#   k-sweep on the same matrix. Produced by the Tester + ../common/plot_k_sweep.py.
#
# Usage (from repo root or from this directory):
#   bash experiments/subset_selection_for_matrices_by_volume_sampling/sensor_placement/run.sh
#
# The run is described entirely by one JSON config, config.json, which carries
# the snapshots the SVD sees, the number of retained modes, how far the sensor
# count sweeps and the algorithm roster. Editing it is how the experiment
# changes; prepare_data.py regenerates config_deterministic.json and
# config_randomized.json from it, so the mode count is never written twice.
#
# The first run downloads the dataset (~215 MB) into data/, which is gitignored.
#
# Optional env overrides:
#   BUILD_DIR   – CMake build dir for the Tester (default: <repo_root>/build/experiments)
#   BUILD_TYPE  – cmake --config value           (default: Release)
#   PYTHON      – python interpreter             (default: python3)
#   CONFIG      – the experiment config          (default: config.json)

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

CONFIG="${CONFIG:-$SCRIPT_DIR/config.json}"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2
    exit 1
fi

RESULTS_DIR="$SCRIPT_DIR/results"
FIGURES_DIR="$SCRIPT_DIR/figures"

# The plots read their rows from the folder the run wrote, which the config
# names, so it is read back out rather than repeated here — the config stays
# the single source of truth.
read_json() {  # read_json <config> <key>
    "$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" \
              "$1" "$2"
}

EXPERIMENT_NAME="$(read_json "$CONFIG" experiment_name)"
RESULTS_SUBDIR="${EXPERIMENT_NAME// /_}"

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
# writes the mode matrix and the held-out field, and regenerates the two Tester
# configs from config.json.
echo ""
echo "==> Preparing the SST modes …"
BASE_DIR="$SCRIPT_DIR" CONFIG="$CONFIG" "$PYTHON" "$SCRIPT_DIR/prepare_data.py"

# ── 3. run experiments ────────────────────────────────────────────────────────
# The Tester resolves the matrix "file_path" relative to the working directory,
# so run from the experiment folder. Both configs write into the same results
# subfolder, named after the experiment.
echo ""
echo "==> Running the sensor-placement experiment (deterministic algorithms) …"
(cd "$SCRIPT_DIR" && "$TESTER" "$SCRIPT_DIR/config_deterministic.json")

echo ""
echo "==> Running the sensor-placement experiment (randomized algorithms) …"
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
echo "==> Plotting figure 1 (reconstruction error vs. sensors) …"
RESULTS_DIR="$RESULTS_DIR" FIGURES_DIR="$FIGURES_DIR" \
RESULTS_SUBDIR="$RESULTS_SUBDIR" \
"$PYTHON" "$SCRIPT_DIR/plot_error.py"

echo ""
echo "==> Plotting figure 2 (the k-sweep) …"
BASE_DIR="$SCRIPT_DIR" RESULTS_DIR="$RESULTS_DIR" FIGURES_DIR="$FIGURES_DIR" \
"$PYTHON" "$COMMON_DIR/plot_k_sweep.py" "$EXPERIMENT_NAME"

echo ""
echo "==> All done. Figures saved to $FIGURES_DIR/"
