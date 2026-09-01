#!/usr/bin/env python3
"""
Prepare the NOAA OI SST V2 sensor-placement experiment.

This is the sensor-placement problem of Manohar, Brunton, Kutz & Brunton,
"Data-Driven Sparse Sensor Placement for Reconstruction: Recovering Structure
from Few Measurements" (IEEE Control Systems Magazine 38(3), 2018), posed so
that the Tester's existing machinery answers it unchanged.

The reduction is the whole point, so it is worth stating precisely. POD on the
training snapshots gives a basis Psi_r (n_ocean x r) of the r leading modes.
Placing p sensors means choosing p *rows* of Psi_r; reconstructing a field from
those sensors is a_hat = Psi_r[S, :]^dagger y_S, and the reconstruction is
Psi_r a_hat. Written as columns — which is what every selector in this library
picks — the matrix to select from is Psi_r^T, r x n_ocean. So:

    m = r          (retained modes)      k = p  (sensors)      n = n_ocean

and k >= m is exactly "at least as many sensors as modes". This script writes
Psi_r as a *tall* CSV (n_ocean rows x r columns); MatrixFromFileGenerator
transposes tall matrices on load, so the Tester sees Psi_r^T without further
help — the same convention the two PDE experiments use for their unfoldings.

The reconstruction error comes out of the Tester's `regression_mse` column for
free. That column fits beta on the selected columns and evaluates on all of
them: beta = (X_S^T)^dagger y_S with X = Psi_r^T is a_hat = Psi_r[S, :]^dagger
y_S, and the residual ||X^T beta - y||^2 / n is the mean squared error of
Psi_r a_hat against the true field. That is the paper's reconstruction error,
computed by code that already exists.

The one departure from the paper: the Tester carries a single target vector, so
the error is measured on one held-out snapshot ("test_snapshot") rather than
averaged over a test set. Point the config at a different snapshot and rerun to
see another.

Everything is driven by config.json — which snapshots feed the SVD, how many
modes are retained, how far the sensor count sweeps — and this script also
writes the two Tester configs from it, so the mode count lives in exactly one
place.

Usage (from this directory):
    python prepare_data.py

Environment:
    BASE_DIR – the experiment folder (default: the working directory)
    CONFIG   – the experiment config (default: <BASE_DIR>/config.json)
"""

import json
import os
import urllib.request
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file

BASE_DIR = Path(os.environ.get('BASE_DIR', Path.cwd()))
CONFIG   = Path(os.environ.get('CONFIG', BASE_DIR / 'config.json'))


def download_if_missing(path: Path, url: str) -> None:
    """The netCDF files live under data/, which is gitignored, so a fresh
    clone starts with nothing. They are large and never change, so fetch them
    once and leave them alone."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {url}\n         -> {path} …')
    tmp = path.with_suffix(path.suffix + '.part')
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(path)
    print(f'  {path.stat().st_size / 1e6:.1f} MB')


def unpack(var, block) -> np.ndarray:
    """Undo the netCDF packing of an already-sliced block of `var`.

    These variables are int16 with a scale_factor/add_offset pair, and scipy
    hands back the raw integers (maskandscale is off by default), so the
    packing has to be undone by hand. Genuine missing values become NaN — note
    that land is *not* one of them: NOAA fills land cells with a plausible
    -1.8 °C, so this says nothing about whether the mask lines up. That is
    checked in main() instead, on the temporal variance."""
    raw = np.asarray(block, dtype=np.float64)
    missing = var._attributes.get('missing_value')
    if missing is not None:
        raw[np.asarray(block) == missing] = np.nan
    scale  = float(var._attributes.get('scale_factor', 1.0))
    offset = float(var._attributes.get('add_offset', 0.0))
    return raw * scale + offset


def ocean_indices(mask_path: Path, stride: int) -> np.ndarray:
    """Flat indices of the ocean points of the 180 x 360 grid. The mask is 1 on
    water and 0 on land; the paper's 44219 points are exactly its ones."""
    with netcdf_file(str(mask_path), 'r', mmap=False) as fh:
        mask = np.asarray(fh.variables['mask'][:]).squeeze()
    idx = np.flatnonzero(mask.ravel() == 1)
    if stride > 1:
        idx = idx[::stride]
    return idx


def load_snapshots(sst_path: Path, rows: slice, ocean: np.ndarray) -> np.ndarray:
    """Snapshots `rows` of the weekly-mean field, restricted to ocean points.
    Returns (n_ocean, n_snapshots) — space down the columns, as POD wants."""
    with netcdf_file(str(sst_path), 'r', mmap=False) as fh:
        var = fh.variables['sst']
        n_time = var.shape[0]
        if rows.stop is not None and rows.stop > n_time:
            raise SystemExit(
                f'config asks for snapshots up to {rows.stop}, but '
                f'{sst_path.name} holds {n_time}')
        field = unpack(var, var[rows])
    field = field.reshape(field.shape[0], -1)[:, ocean]
    if not np.isfinite(field).all():
        raise SystemExit(f'{sst_path.name}: missing values at ocean points')
    return field.T


def experiment_entry(cfg: dict, algorithms: list, k_values: list,
                     trials: int) -> dict:
    """One entry of a Tester config's "experiments" list. "file_path" is
    relative to the directory the Tester is run from (this experiment's folder)
    and "output_path" to the config's own parent, which is the same folder —
    matching the other experiments here."""
    return {
        "name": cfg['experiment_name'],
        "enabled": True,
        "matrix": {
            "type": "matrix from file",
            "file_path": cfg['modes_file'],
            "target_file": cfg['target_file'],
        },
        "algorithms": algorithms,
        "k_values": k_values,
        "trials_per_k": trials,
    }


def write_tester_config(path: Path, entries: list) -> None:
    path.write_text(json.dumps({
        "scalar": "double",
        "output_path": "results",
        "experiments": entries,
    }, indent=4) + '\n')
    for e in entries:
        print(f'Wrote {path.name}: {len(e["algorithms"])} algorithms, '
              f'k = {e["k_values"][0]}..{e["k_values"][-1]}, '
              f'trials = {e["trials_per_k"]}')


def main():
    cfg = json.loads(CONFIG.read_text())

    sst_path  = BASE_DIR / cfg['sst_file']
    mask_path = BASE_DIR / cfg['mask_file']
    download_if_missing(mask_path, cfg['mask_url'])
    download_if_missing(sst_path,  cfg['sst_url'])

    stride = int(cfg.get('spatial_stride', 1))
    ocean  = ocean_indices(mask_path, stride)
    n_ocean = ocean.size

    train = slice(int(cfg['train_start']), int(cfg['train_stop']),
                  int(cfg.get('train_step', 1)))
    X = load_snapshots(sst_path, train, ocean)
    n_train = X.shape[1]

    # Land is filled with a constant -1.8 °C, so a mask that did not line up
    # with the SST grid would quietly contribute frozen columns rather than an
    # error. Every genuine ocean point moves over 27 years; a constant one is
    # the signature of that mistake.
    frozen = int((X.var(axis=1) == 0).sum())
    if frozen:
        raise SystemExit(
            f'{frozen} of {n_ocean} selected points never change over the '
            f'training window — the land mask and the SST grid disagree')

    # POD is on the fluctuation about the temporal mean: the seasonal cycle is
    # so dominant that without this the first mode is just "the mean ocean" and
    # the sensors chase it instead of the structure being reconstructed.
    if cfg.get('subtract_mean', True):
        mean_field = X.mean(axis=1)
        X = X - mean_field[:, None]
    else:
        mean_field = np.zeros(n_ocean)

    r = int(cfg['modes'])
    if r > min(X.shape):
        raise SystemExit(f'modes = {r} exceeds min(n_ocean, n_train) = '
                         f'{min(X.shape)}')

    U, s, _ = np.linalg.svd(X, full_matrices=False)
    Psi = U[:, :r]

    energy = float((s[:r] ** 2).sum() / (s ** 2).sum())

    # The held-out field the sensors are scored on, in the same fluctuation
    # coordinates as the modes.
    t_test = int(cfg['test_snapshot'])
    if train.start <= t_test < train.stop:
        print(f'  ! warning: test_snapshot {t_test} lies inside the training '
              f'range [{train.start}, {train.stop}) — the error will be '
              f'optimistic')
    y = load_snapshots(sst_path, slice(t_test, t_test + 1), ocean)[:, 0]
    y = y - mean_field

    modes_path  = BASE_DIR / cfg['modes_file']
    target_path = BASE_DIR / cfg['target_file']
    modes_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(modes_path, Psi, delimiter=',', fmt='%.12e')
    np.savetxt(target_path, y, fmt='%.12e')

    print(f'Ocean points:      {n_ocean}'
          + (f'  (stride {stride})' if stride > 1 else ''))
    print(f'Training snapshots: {n_train}  '
          f'[{train.start}:{train.stop}:{train.step or 1}]')
    print(f'Retained modes:     {r}  '
          f'({100 * energy:.2f}% of the fluctuation energy)')
    tail = f'  (next: {s[r]:.4g})' if r < s.size else ''
    print(f'  singular values:  {s[0]:.4g} … {s[r - 1]:.4g}{tail}')
    print(f'Test snapshot:      {t_test}  '
          f'(RMS {np.sqrt((y ** 2).mean()):.4g} °C about the mean)')
    print(f'Saved modes:        {modes_path}  '
          f'({n_ocean} rows x {r} cols, transposed on load)')
    print(f'Saved target:       {target_path}  ({n_ocean} entries)')

    # ── the Tester configs ────────────────────────────────────────────────────
    # Subset selection needs k >= m, and m is r here, so the sweep starts at
    # "one sensor per mode" — the square case the paper's QR pivoting solves —
    # and runs to k_max_factor * r, the oversampled regime.
    k_max = min(int(cfg.get('k_max_factor', 4)) * r, n_ocean)
    k_values = list(range(r, k_max + 1))

    # DEIM and Q-DEIM return one interpolation point per basis vector, so they
    # exist only at k = r. They go in their own experiment entry, listed first
    # so that the sweep entry — sharing the name, and so the results folder —
    # is the one whose k range ends up in the folder's saved config.json.
    write_tester_config(BASE_DIR / 'config_deterministic.json', [
        experiment_entry(cfg, cfg['algorithms_fixed_k'], [r], 1),
        experiment_entry(cfg, cfg['algorithms'], k_values, 1),
    ])
    write_tester_config(BASE_DIR / 'config_randomized.json', [
        experiment_entry(cfg, cfg['algorithms_randomized'], k_values,
                         int(cfg.get('samples', 16))),
    ])


if __name__ == '__main__':
    main()
