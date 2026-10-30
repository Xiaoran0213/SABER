from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class MissingInfo:
    n_masked: int
    n_mnar: int
    n_mcar: int
    n_available: int
    fraction_actual: float


def compute_feature_thresholds(
    x: np.ndarray,
    quantile: float,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    thresholds = np.full((x.shape[1],), np.nan, dtype=np.float32)
    global_std = float(np.nanstd(x))
    if not np.isfinite(global_std) or global_std < 1e-8:
        global_std = 1.0
    for j in range(x.shape[1]):
        obs = x[np.isfinite(x[:, j]), j]
        if obs.size < 2:
            continue
        center = float(np.quantile(obs, float(quantile)))
        thresholds[j] = rng.normal(center, float(scale) * global_std)
    return thresholds


def inject_missing(
    x_full: np.ndarray,
    total_fraction: float,
    mnar_fraction: float,
    threshold_quantile: float,
    threshold_scale: float,
    seed: int,
    available_mask: Optional[np.ndarray] = None,
    return_split: bool = False,
) -> tuple:
    """Inject MCAR+MNAR missing values on finite available entries.

    Returns `(x_observed, observed_mask, injected_missing_mask, info)`.
    With ``return_split=True`` a fifth element ``(mnar_mask, mcar_mask)`` is
    appended: the mechanism split of the injected mask (disjoint; their OR
    equals injected_missing_mask). The RNG draw sequence is IDENTICAL with and
    without the flag, so existing masks are reproduced bit-for-bit.
    """
    x = np.asarray(x_full, dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    if available_mask is None:
        available = np.isfinite(x)
    else:
        available = np.asarray(available_mask).astype(bool) & np.isfinite(x)

    n_available = int(available.sum())
    n_to_mask = int(float(total_fraction) * n_available)
    if n_to_mask <= 0:
        observed = available.astype(np.float32)
        empty_mnar = np.zeros_like(available, dtype=np.uint8)
        if return_split:
            return (x.copy(), observed, np.zeros_like(available, dtype=np.uint8),
                    MissingInfo(0, 0, 0, n_available, 0.0), (empty_mnar, empty_mnar.copy()))
        return x.copy(), observed, np.zeros_like(available, dtype=np.uint8), MissingInfo(0, 0, 0, n_available, 0.0)

    thresholds = compute_feature_thresholds(x, threshold_quantile, threshold_scale, rng)
    mnar_candidates = np.zeros_like(available, dtype=bool)
    for j, threshold in enumerate(thresholds):
        if np.isfinite(threshold):
            mnar_candidates[:, j] = (x[:, j] < threshold) & available[:, j]

    n_mnar_target = int(float(mnar_fraction) * n_to_mask)
    mnar_idx = np.argwhere(mnar_candidates)
    n_mnar = min(n_mnar_target, int(mnar_idx.shape[0]))
    mnar_mask = np.zeros_like(available, dtype=bool)
    if n_mnar > 0:
        chosen = rng.choice(mnar_idx.shape[0], size=n_mnar, replace=False)
        coords = mnar_idx[chosen]
        mnar_mask[coords[:, 0], coords[:, 1]] = True

    remaining = available & ~mnar_mask
    n_mcar_target = n_to_mask - n_mnar
    mcar_idx = np.argwhere(remaining)
    n_mcar = min(n_mcar_target, int(mcar_idx.shape[0]))
    mcar_mask = np.zeros_like(available, dtype=bool)
    if n_mcar > 0:
        chosen = rng.choice(mcar_idx.shape[0], size=n_mcar, replace=False)
        coords = mcar_idx[chosen]
        mcar_mask[coords[:, 0], coords[:, 1]] = True

    missing = mnar_mask | mcar_mask
    observed_mask = available & ~missing
    x_obs = x.copy()
    x_obs[~observed_mask] = np.nan
    info = MissingInfo(
        n_masked=int(missing.sum()),
        n_mnar=int(mnar_mask.sum()),
        n_mcar=int(mcar_mask.sum()),
        n_available=n_available,
        fraction_actual=float(missing.sum() / max(1, n_available)),
    )
    if return_split:
        return (x_obs, observed_mask.astype(np.float32), missing.astype(np.uint8), info,
                (mnar_mask.astype(np.uint8), mcar_mask.astype(np.uint8)))
    return x_obs, observed_mask.astype(np.float32), missing.astype(np.uint8), info


def fill_missing_column_mean(x: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32).copy()
    finite = np.isfinite(out)
    counts = finite.sum(axis=0)
    sums = np.where(finite, out, 0.0).sum(axis=0, dtype=np.float64)
    means = np.full((out.shape[1],), float(fallback), dtype=np.float32)
    np.divide(sums, counts, out=means, where=counts > 0)
    rows, cols = np.where(~np.isfinite(out))
    out[rows, cols] = means[cols]
    return out


# Input-fill modes for the cell-encoder input matrix (fill ablation).
# "column_mean" is the default and reproduces
# fill_missing_column_mean exactly; the others fill ALL non-finite entries
# (natural + injected missing) with a low-abundance prior, the standard
# proteomics MNAR treatment (detection-limit / Perseus-style downshift).
FILL_MODES = ("column_mean", "global_min", "protein_q05", "downshift")


def fill_missing(
    x: np.ndarray,
    mode: str = "column_mean",
    seed: int = 42,
    quantile: float = 0.05,
    downshift_scale: float = 1.8,
    downshift_width: float = 0.3,
) -> np.ndarray:
    """Fill non-finite entries of the model INPUT matrix under ``mode``.

    column_mean : per-protein mean of remaining observed (historical default).
    global_min  : the single global minimum over observed values (constant).
    protein_q05 : per-protein quantile (default 5%) of remaining observed.
    downshift   : per-protein draw from N(mu_j - 1.8*sd_j, (0.3*sd_j)^2)
                  (Perseus-style; seeded, so reproducible).
    """
    if mode not in FILL_MODES:
        raise ValueError(f"fill mode must be one of {FILL_MODES}; got {mode!r}")
    if mode == "column_mean":
        return fill_missing_column_mean(x)
    out = np.asarray(x, dtype=np.float32).copy()
    missing = ~np.isfinite(out)
    if not missing.any():
        return out
    obs = np.isfinite(out)
    if mode == "global_min":
        out[missing] = float(np.nanmin(out))
        return out
    counts = obs.sum(axis=0)
    means = np.zeros(out.shape[1], dtype=np.float64)
    np.divide(np.where(obs, out, 0.0).sum(axis=0, dtype=np.float64), counts,
              out=means, where=counts > 0)
    stds = np.nanstd(np.where(obs, out, np.nan), axis=0)
    stds = np.where(np.isfinite(stds) & (stds > 1e-8), stds, 1.0)
    if mode == "protein_q05":
        for j in range(out.shape[1]):
            col_obs = out[obs[:, j], j]
            v = float(np.quantile(col_obs, float(quantile))) if col_obs.size >= 2 else means[j]
            out[missing[:, j], j] = v
        return out.astype(np.float32)
    if mode == "downshift":
        rng = np.random.default_rng(int(seed))
        for j in range(out.shape[1]):
            idx = np.where(missing[:, j])[0]
            if idx.size == 0:
                continue
            out[idx, j] = rng.normal(
                loc=float(means[j] - float(downshift_scale) * stds[j]),
                scale=float(downshift_width) * float(stds[j]),
                size=idx.size,
            )
        return out.astype(np.float32)
    raise ValueError(f"unhandled fill mode {mode!r}")

