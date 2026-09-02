# Subset selection for matrices by volume sampling

One subfolder per experiment. Each is self-contained — its own configs, its own
runner, its own `results/` and `figures/` — and `common/` holds the pieces every
experiment shares.

```
common/                    shared scripts, driven by BASE_DIR (see below)
  plot_k_sweep.py            the two-panel k-sweep figure (NORM=spectral
                             draws it in the spectral norm instead)
  make_unfolding_configs.py  Tester configs for a dumped TT unfolding
  update_index.py            rebuild results/index.json

superconductivity/         a static dataset: one k-sweep figure
superconductivity_extended/
                           the same dataset and k range, with DEIM,
                           GappyPOD+E and spectral selection added
allen_cahn/                a fixed-rank PDE run: error vs. time + its unfolding
acoustic/                  a tolerance-driven PDE run: error and rank vs. time
                           + its unfolding
sensor_placement/          sparse sensors on NOAA OI SST V2: reconstruction
                           error vs. sensor count, after Manohar et al. (2018),
                           plus an r x oversampling grid (run_grid.sh) that
                           also saves the selected sensor sets
```

Run one with its own runner, from anywhere:

```sh
bash experiments/subset_selection_for_matrices_by_volume_sampling/acoustic/run.sh
```

Each runner builds what it needs, runs the Tester (and the PDE solver, where
there is one), and writes its figures into its own `figures/`. The env overrides
each accepts are documented in its header. `sensor_placement/` downloads its
dataset (~215 MB) into its own gitignored `data/` on a first run.

## BASE_DIR

The shared scripts live one level above the experiment they act on, so they
cannot resolve paths against their own directory the way a script sitting in the
experiment folder can. They take the experiment folder from `BASE_DIR` instead,
defaulting to the working directory, and everything else hangs off it:
`results/`, `figures/`, and the matrix paths the configs name (which the Tester
resolves against the folder it was run from). The runners set it; by hand,
`cd` into the experiment first:

```sh
cd experiments/subset_selection_for_matrices_by_volume_sampling/superconductivity
python ../common/plot_k_sweep.py
```

## Adding an experiment

1. `mkdir <name>` and give it a `run.sh`, its configs, and any plot script
   specific to it. Copying the nearest existing experiment is the fastest
   start — `superconductivity/` for a static matrix, `allen_cahn/` for a PDE
   run that also dumps a TT unfolding.
2. In `run.sh`, set `REPO_ROOT` three levels up, point `COMMON_DIR` at
   `../common`, and pass `BASE_DIR="$SCRIPT_DIR"` to every shared script.
3. If the experiment produces a k-sweep figure, add its name to `OUT_STEMS` in
   `common/plot_k_sweep.py` — that map decides the figure's file stem — and to
   `SINGLE_PANEL` there if only the pinv ratio is worth showing.
4. `results/` and `figures/` are gitignored, so there is nothing to create by
   hand and nothing to commit.
