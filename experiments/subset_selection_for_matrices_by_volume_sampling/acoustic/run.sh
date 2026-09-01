#!/usr/bin/env bash
# Build and run the acoustic-wave accuracy experiment of the
# subset_selection_for_matrices_by_volume_sampling experiment, and produce its
# figure.
#
# The companion of ../allen_cahn/run.sh, and its opposite in the one respect that
# matters: Allen-Cahn is diffusive and is run at a fixed rank, so the selectors
# are compared at equal cost against a best-rank-r floor. A wavefront's rank
# grows as it propagates, so this run is tolerance-driven and the rank is an
# output — hence a two-panel figure (error and rank) and no floor curve.
#
# Figure 1 (plot_acoustic.pdf): relative L2 error against a dense full-grid
#   reference vs. time on the left, the TT rank each selector's state reached on
#   the right, one curve per selection algorithm.
#   Produced by the AcousticTester binary + plot_error.py.
#
# Figure 2 (plot_acoustic_unfolding.pdf): the exact two-panel figure of the
#   superconductivity experiment, but on a matrix this PDE genuinely produces —
#   the absorbed TT unfolding a cross step of the mid-run wavefield hands its
#   selector. Produced by the standard Tester + ../common/plot_k_sweep.py.
#
# Usage (from repo root or from this directory):
#   bash experiments/subset_selection_for_matrices_by_volume_sampling/acoustic/run.sh
#
# The run is described entirely by one JSON config, config.json, which
# carries the grid, the tolerance, the final time and the algorithm roster. Edit
# it to change the experiment; this script only wires it up.
#
# Optional env overrides:
#   BUILD_DIR       – CMake build dir for the Tester   (default: <repo_root>/build/experiments)
#   TT_BUILD_DIR    – CMake build dir for the TT tools (default: <repo_root>/build/dlra_deim_tools)
#   BUILD_TYPE      – cmake --config value             (default: Release)
#   PYTHON          – python interpreter               (default: python3)
#   CONFIG          – the integration config           (default: config.json)
#   SAMPLES         – trials per k for the unfolding figure (default 16)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# The scripts every experiment in this folder shares. They resolve their paths
# against BASE_DIR — this experiment's own folder — rather than against
# themselves, so each experiment keeps its results and figures to itself.
COMMON_DIR="$SCRIPT_DIR/../common"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build/experiments}"
TT_BUILD_DIR="${TT_BUILD_DIR:-$REPO_ROOT/build/dlra_deim_tools}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
PYTHON="${PYTHON:-python3}"

SAMPLES="${SAMPLES:-16}"

CONFIG="${CONFIG:-$SCRIPT_DIR/config.json}"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2
    exit 1
fi

RESULTS_DIR="$SCRIPT_DIR/results"
FIGURES_DIR="$SCRIPT_DIR/figures"

# The plot reads its rows from the folder the run wrote, which the config names,
# so it is read back out rather than repeated here — the config stays the single
# source of truth.
read_json() {  # read_json <config> <key>
    "$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" \
              "$1" "$2"
}

# "output_path" is relative to the config's parent (this folder), and the plot
# wants the leaf name under results/.
RESULTS_SUBDIR="$(basename "$(read_json "$CONFIG" output_path)")"

# ── 1. build ──────────────────────────────────────────────────────────────────
echo "==> Building the TT tools (config: $BUILD_TYPE) …"
cmake -S "$REPO_ROOT/experiments/dlra_deim_tools" -B "$TT_BUILD_DIR" \
      -DCMAKE_BUILD_TYPE="$BUILD_TYPE" > /dev/null
cmake --build "$TT_BUILD_DIR" --config "$BUILD_TYPE" \
      --target AcousticTester -j"$(nproc)"

SOLVER="$TT_BUILD_DIR/AcousticTester"
if [[ ! -x "$SOLVER" ]]; then
    echo "ERROR: AcousticTester binary not found at $SOLVER" >&2
    exit 1
fi

echo ""
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

# ── 2. the integration run ────────────────────────────────────────────────────
# Grid, tolerance, final time and the algorithm roster all live in the config,
# which is the single place to edit the experiment.
echo ""
echo "==> Acoustic integration run …"
"$SOLVER" "$CONFIG"

# ── 3. figure 2: the mid-run unfolding through the standard Tester ────────────
# The unfolding dumped by the run above is a tall matrix; MatrixFromFileGenerator
# transposes it to be wide, giving m = (its column count) and n = (its row
# count). k starts at m and runs to a few multiples of it (m is the TT rank
# here), so the config is generated from the actual file rather than hard-coded.
UNFOLDING="$RESULTS_DIR/$RESULTS_SUBDIR/acoustic_unfolding.csv"
CONFIG_DET="$SCRIPT_DIR/config_deterministic.json"
CONFIG_RAND="$SCRIPT_DIR/config_randomized.json"

if [[ -f "$UNFOLDING" ]]; then
    echo ""
    echo "==> Generating the unfolding configs from $UNFOLDING …"
    BASE_DIR="$SCRIPT_DIR" \
    UNFOLDING="$UNFOLDING" RESULTS_DIR="$RESULTS_DIR" SAMPLES="$SAMPLES" \
    CONFIG_DET="$CONFIG_DET" CONFIG_RAND="$CONFIG_RAND" \
    EXPERIMENT="Acoustic unfolding" \
    "$PYTHON" "$COMMON_DIR/make_unfolding_configs.py"

    echo ""
    echo "==> Running the unfolding experiment (deterministic algorithms) …"
    (cd "$SCRIPT_DIR" && "$TESTER" "$CONFIG_DET")

    echo ""
    echo "==> Running the unfolding experiment (randomized algorithms) …"
    (cd "$SCRIPT_DIR" && "$TESTER" "$CONFIG_RAND")

    # Each tester run overwrites index.json with only its own experiments;
    # rebuild it from every result subfolder of this experiment so both runs
    # stay listed.
    echo ""
    echo "==> Refreshing index.json …"
    BASE_DIR="$SCRIPT_DIR" RESULTS_DIR="$RESULTS_DIR" \
    "$PYTHON" "$COMMON_DIR/update_index.py"
else
    echo ""
    echo "==> No unfolding dumped (\"unfolding_time\" disabled?); skipping figure 2."
fi

# ── 4. plots ──────────────────────────────────────────────────────────────────
echo ""
echo "==> Plotting figure 1 (error and rank vs. time) …"
RESULTS_DIR="$RESULTS_DIR" FIGURES_DIR="$FIGURES_DIR" \
RESULTS_SUBDIR="$RESULTS_SUBDIR" \
"$PYTHON" "$SCRIPT_DIR/plot_error.py"

if [[ -f "$UNFOLDING" ]]; then
    echo ""
    echo "==> Plotting figure 2 (the mid-run unfolding) …"
    BASE_DIR="$SCRIPT_DIR" RESULTS_DIR="$RESULTS_DIR" FIGURES_DIR="$FIGURES_DIR" \
    "$PYTHON" "$COMMON_DIR/plot_k_sweep.py" "Acoustic unfolding"
fi

echo ""
echo "==> All done. Figures saved to $FIGURES_DIR/"
