#!/usr/bin/env python
"""Classic protein-wise k-nearest-neighbour imputation adapter for SABER.

Reads a CSV matrix (no header/index) with NaN marking missing entries, imputes
each missing entry (cell c, protein j) by the mean of the k proteins whose
expression profiles are most Pearson-correlated with protein j (evaluated at
cell c, using only observed neighbour values), and writes the full imputed
matrix. This is the standard column-wise / protein-wise KNN-impute used in
proteomics (Troyanskaya/Danner style).

The adapter is task-agnostic: SABER's manifest runner NaN's the relevant mask
(injected missing for imputation, natural missing for LFQ raw-scale) before
calling it, so the same adapter serves both the imputation and LFQ tasks.

CLI contract matches the other SABER external adapters:
  --in PATH --out PATH [--repo PATH] [--config PATH] [--model NAME] [--seed N] [--k N]
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np


def _load_matrix(path: str) -> np.ndarray:
    arr = np.loadtxt(os.path.abspath(path), delimiter=",")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1) if arr.size > 0 else arr.reshape(0, 0)
    return arr


def knn_impute(x: np.ndarray, k: int = 5) -> np.ndarray:
    """Protein-wise (column-wise) KNN imputation.

    Parameters
    ----------
    x : (n_cells, n_proteins) matrix with NaN at missing entries.
    k : number of nearest proteins (by Pearson profile correlation).

    Returns
    -------
    (n_cells, n_proteins) float matrix with all entries finite.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return arr.astype(np.float32)
    n_cells, n_proteins = arr.shape
    observed = np.isfinite(arr)

    # Per-protein column mean over observed entries (0 fallback).
    with np.errstate(invalid="ignore"):
        col_mean = np.nanmean(np.where(observed, arr, np.nan), axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0).astype(np.float64)

    # Column-mean fill purely to compute profile similarities.
    filled = np.where(observed, arr, col_mean[np.newaxis, :])
    centered = filled - filled.mean(axis=0, keepdims=True)
    norms = np.sqrt((centered ** 2).sum(axis=0, keepdims=True))
    norms = np.where(norms > 1e-12, norms, 1.0)
    unit = centered / norms
    sim = unit.T @ unit  # (n_proteins, n_proteins) Pearson correlation
    np.fill_diagonal(sim, -np.inf)  # never pick self as a neighbour

    k_eff = int(max(1, min(int(k), n_proteins - 1))) if n_proteins > 1 else 0
    if k_eff > 0:
        nbr = np.argpartition(-sim, k_eff - 1, axis=1)[:, :k_eff]  # (n_proteins, k_eff)
    else:
        nbr = np.empty((n_proteins, 0), dtype=np.int64)

    out = arr.copy()
    for j in range(n_proteins):
        miss_cells = np.where(~observed[:, j])[0]
        if miss_cells.size == 0:
            continue
        neigh = nbr[j]
        if neigh.size == 0:
            out[miss_cells, j] = col_mean[j]
            continue
        block = arr[np.ix_(miss_cells, neigh)]  # (n_miss, k_eff)
        valid = np.isfinite(block)
        if not valid.any():
            out[miss_cells, j] = col_mean[j]
            continue
        masked = np.where(valid, block, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            with np.errstate(invalid="ignore"):
                means = np.nanmean(masked, axis=1)
        means = np.where(np.isfinite(means), means, col_mean[j])
        out[miss_cells, j] = means

    # Last-resort guard: any remaining non-finite -> column mean.
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = np.broadcast_to(col_mean, out.shape)[bad]
    return out.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", dest="out", required=True)
    parser.add_argument("--repo", dest="repo", required=False, default="")
    parser.add_argument("--config", dest="config", required=False, default=None)
    parser.add_argument("--model", dest="model", required=False, default="")
    parser.add_argument("--seed", dest="seed", type=int, required=False, default=42)
    parser.add_argument("--k", dest="k", type=int, required=False, default=5)
    args = parser.parse_args()

    inp_path = os.path.abspath(args.inp)
    if not os.path.exists(inp_path):
        sys.stderr.write(f"Input file not found: {inp_path}\n")
        return 1

    x = _load_matrix(inp_path)
    k = int(args.k)
    if k <= 0:
        k = 5
    mat = knn_impute(x, k=k)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savetxt(out_path, np.asarray(mat, dtype=np.float32), delimiter=",")
    return 0


if __name__ == "__main__":
    sys.exit(main())
