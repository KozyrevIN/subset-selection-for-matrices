#!/usr/bin/env bash
# Build and run the Allen-Cahn accuracy experiment of the
# subset_selection_for_matrices_by_volume_sampling experiment, and produce both
# of its figures.
#
# Figure 1 (plot_allen_cahn.pdf): relative Frobenius error against a dense full-grid
#   reference vs. time, one curve per selection algorithm, against the best
#   rank-r floor.
#   Produced by the AllenCahnTester binary + plot_allen_cahn_error.py.
#
# Figure 2 (plot_allen_cahn_unfolding.pdf): the exact two-panel figure of the
#   superconductivity experiment, but on a matrix this PDE genuinely produces —
#   a TT unfolding of the reference state taken from the middle of the run.
#   Produced by the standard Tester + plot_k_sweep.py.
#
# Usage (from repo root or from this directory):
#   bash experiments/subset_selection_for_matrices_by_volume_sampling/run_allen_cahn.sh
#
# The integration run of figure 1 is described entirely by one JSON config,
# config_allen_cahn.json, which carries the grid, the fixed rank, the final time
# and the algorithm roster. Edit it to change the experiment; this script only
# wires it up.
#
# Optional env overrides:
#   BUILD_DIR       – CMake build dir for the Tester   (default: <repo_root>/build/experiments)
#   TT_BUILD_DIR    – CMake build dir for the TT tools (default: <repo_root>/build/dlra_deim_tools)
#   BUILD_TYPE      – cmake --config value             (default: Release)
#   PYTHON          – python interpreter               (default: python3)
#   CONFIG          – the integration config           (default: config_allen_cahn.json)
#   SAMPLES         – trials per k for the unfolding figure (default: 16)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build/experiments}"
TT_BUILD_DIR="${TT_BUILD_DIR:-$REPO_ROOT/build/dlra_deim_tools}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
PYTHON="${PYTHON:-python3}"

SAMPLES="${SAMPLES:-16}"

CONFIG="${CONFIG:-$SCRIPT_DIR/config_allen_cahn.json}"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2
    exit 1
fi

RESULTS_DIR="$SCRIPT_DIR/results"
FIGURES_DIR="$SCRIPT_DIR/figures"

# The plot names the fixed rank in its legend and reads its rows from the folder
# the run wrote. Both come from the config, so they are read back out rather
# than repeated here — the config stays the single source of truth.
read_json() {  # read_json <config> <key>
    "$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" \
              "$1" "$2"
}

RANK="$(read_json "$CONFIG" rank)"
# "output_path" is relative to the config's parent (this folder), and the plot
# wants the leaf name under results/.
RESULTS_SUBDIR="$(basename "$(read_json "$CONFIG" output_path)")"

# ── 1. build ──────────────────────────────────────────────────────────────────
echo "==> Building the TT tools (config: $BUILD_TYPE) …"
cmake -S "$REPO_ROOT/experiments/dlra_deim_tools" -B "$TT_BUILD_DIR" \
      -DCMAKE_BUILD_TYPE="$BUILD_TYPE" > /dev/null
cmake --build "$TT_BUILD_DIR" --config "$BUILD_TYPE" \
      --target AllenCahnTester -j"$(nproc)"

SOLVER="$TT_BUILD_DIR/AllenCahnTester"
if [[ ! -x "$SOLVER" ]]; then
    echo "ERROR: AllenCahnTester binary not found at $SOLVER" >&2
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
for candidate in "$BUILD_DIR/tools/Tester" "$BUILD_DIR/matrix_tools/Tester"; do
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

# ── 2. the integration run of figure 1 ────────────────────────────────────────
# Grid, rank, final time and the algorithm roster all live in the config, which
# is the single place to edit the experiment.
echo ""
echo "==> Allen-Cahn integration run …"
"$SOLVER" "$CONFIG"

# ── 3. figure 2: the mid-run unfolding through the standard Tester ────────────
# The unfolding dumped by the run above is a tall matrix; MatrixFromFileGenerator
# transposes it to be wide, giving m = (its column count) and n = (its row
# count). k starts at m and runs to a few multiples of it (m is the TT rank
# here), so the config is generated from the actual file rather than hard-coded.
UNFOLDING="$RESULTS_DIR/$RESULTS_SUBDIR/allen_cahn_unfolding.csv"
CONFIG_DET="$SCRIPT_DIR/config_allen_cahn_deterministic.json"
CONFIG_RAND="$SCRIPT_DIR/config_allen_cahn_randomized.json"

echo ""
echo "==> Generating the unfolding configs from $UNFOLDING …"
UNFOLDING="$UNFOLDING" RESULTS_DIR="$RESULTS_DIR" SAMPLES="$SAMPLES" \
CONFIG_DET="$CONFIG_DET" CONFIG_RAND="$CONFIG_RAND" \
"$PYTHON" "$SCRIPT_DIR/make_allen_cahn_configs.py"

echo ""
echo "==> Running the unfolding experiment (deterministic algorithms) …"
(cd "$SCRIPT_DIR" && "$TESTER" "$CONFIG_DET")

echo ""
echo "==> Running the unfolding experiment (randomized algorithms) …"
(cd "$SCRIPT_DIR" && "$TESTER" "$CONFIG_RAND")

# ── 4. refresh index.json ─────────────────────────────────────────────────────
# Each tester run overwrites index.json with only its own experiments; rebuild
# it from every result subfolder so the superconductivity results stay listed
# alongside this one.
echo ""
echo "==> Refreshing index.json …"
RESULTS_DIR="$RESULTS_DIR" "$PYTHON" "$SCRIPT_DIR/update_index.py"

# ── 5. plots ──────────────────────────────────────────────────────────────────
echo ""
echo "==> Plotting figure 1 (error vs. time) …"
RESULTS_DIR="$RESULTS_DIR" FIGURES_DIR="$FIGURES_DIR" \
RESULTS_SUBDIR="$RESULTS_SUBDIR" RANK="$RANK" \
"$PYTHON" "$SCRIPT_DIR/plot_allen_cahn_error.py"

echo ""
echo "==> Plotting figure 2 (the mid-run unfolding) …"
RESULTS_DIR="$RESULTS_DIR" FIGURES_DIR="$FIGURES_DIR" \
"$PYTHON" "$SCRIPT_DIR/plot_k_sweep.py" "Allen-Cahn unfolding"

echo ""
echo "==> All done. Figures saved to $FIGURES_DIR/"
