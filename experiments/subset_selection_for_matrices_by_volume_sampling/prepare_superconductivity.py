#!/usr/bin/env python3
"""
Prepare the superconductivity dataset for subset selection experiments.

Steps:
  1. Load raw data; separate target (critical_temp, the last column).
  2. Standardize continuous features (zero mean, unit variance).
  3. Append intercept column (all ones).
  4. Save feature matrix and target vector as headerless CSVs.

The tester's MatrixFromFileGenerator auto-transposes tall matrices to wide,
so the saved file (21263 x 82) becomes an 82 x 21263 matrix (m x n).

Prints the largest and smallest singular values of the prepared m x n matrix.

Usage:
    python prepare_superconductivity.py [--input PATH] [--output-matrix PATH]
                                        [--output-target PATH]

Defaults:
    --input          ../../supplementary/superconductivty+data/train.csv
    --output-matrix  data/superconductivity.csv
    --output-target  data/superconductivity_target.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def prepare(input_path: Path, output_matrix: Path, output_target: Path) -> None:
    df = pd.read_csv(input_path)

    # Target is the last column (critical_temp); everything else is a feature.
    y = df.iloc[:, -1].values.astype(float)
    X_cont = df.iloc[:, :-1].values.astype(float)

    col_means = X_cont.mean(axis=0)
    col_stds  = X_cont.std(axis=0)
    col_stds[col_stds == 0] = 1.0
    X_cont_std = (X_cont - col_means) / col_stds

    X = np.hstack([X_cont_std, np.ones((len(df), 1))])

    output_matrix.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_matrix, X, delimiter=',', fmt='%.10f')

    output_target.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_target, y, fmt='%.10f')

    m_file, n_file = X.shape
    print(f"Saved feature matrix: {output_matrix}")
    print(f"  File shape (before auto-transpose): {m_file} rows x {n_file} cols")
    print(f"  Matrix shape in experiment:         {n_file} x {m_file}  (m={n_file}, n={m_file})")
    print(f"Saved target vector:  {output_target}  ({len(y)} entries)")

    sv = np.linalg.svd(X.T, compute_uv=False)
    print(f"  Largest  singular value: {sv[0]:.6g}")
    print(f"  Smallest singular value: {sv[-1]:.6g}")


def main():
    script_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--input', type=Path,
                        default=script_dir / '../../supplementary/superconductivty+data/train.csv')
    parser.add_argument('--output-matrix', type=Path,
                        default=script_dir / 'data/superconductivity.csv')
    parser.add_argument('--output-target', type=Path,
                        default=script_dir / 'data/superconductivity_target.csv')
    args = parser.parse_args()

    prepare(args.input.resolve(),
            args.output_matrix.resolve(),
            args.output_target.resolve())


if __name__ == '__main__':
    main()
