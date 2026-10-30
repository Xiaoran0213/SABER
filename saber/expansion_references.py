"""Naive PLM-based reference models for proteome expansion.

These answer the question "is SABER's learned cell encoder + bilinear decoder
better than a naive model that maps raw ESM-2 (PLM) features straight to a
protein's expression profile?" They are *not* learned cell encoders: they use
only the held-out protein's raw PLM feature and the (PLM, expression) pairs of
the training proteins.

Contract with the evaluator
---------------------------
``saber.evaluate.expansion`` auto-discovers any ``pred_target__<method>`` array
stored in ``expansion.npz`` and scores it. So a reference only has to write a
``(n_cells, n_targets)`` float32 array under a ``pred_target__<method>`` key.
No ``evaluate/`` changes are required.

Both references fit / average on the working-scale training expression with
missing entries filled by protein-wise column mean (the project's input-filling
convention). PLM features are the raw ESM-2 cache, never SABER's fine-tuned
features.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


REFERENCE_METHODS: tuple[str, ...] = ("plm_ridge", "plm_knn")


def _column_mean_fill(x: np.ndarray) -> np.ndarray:
    """Fill non-finite entries with per-column (per-protein) mean, 0 fallback."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D matrix, got shape {arr.shape}")
    out = arr.copy()
    bad = ~np.isfinite(out)
    if bad.any():
        for j in np.where(bad.any(axis=0))[0]:
            col = out[:, j]
            good = np.isfinite(col)
            fill = float(col[good].mean()) if good.any() else 0.0
            col[~good] = fill
    return out


def _l2_normalize(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms > eps, norms, 1.0)
    return features / norms


def compute_plm_ridge_predictions(
    plm_train: np.ndarray,
    plm_target: np.ndarray,
    x_train: np.ndarray,
    ridge_lambda: float = 1.0,
) -> np.ndarray:
    """Closed-form ridge from raw PLM features to expression profiles.

    Fits ``B`` mapping standardized PLM features to the (column-mean-filled)
    training expression with an intercept equal to the per-cell mean over
    training proteins (i.e. the ``cell_mean`` baseline). Held-out proteins are
    predicted as ``PLM_target @ B + cell_mean``.

    Returns
    -------
    (n_cells, n_targets) float32 prediction matrix.
    """
    lam = float(ridge_lambda)
    if lam <= 0.0:
        raise ValueError("ridge_lambda must be positive")
    p_train = np.asarray(plm_train, dtype=np.float64)
    p_target = np.asarray(plm_target, dtype=np.float64)
    if p_train.ndim != 2 or p_target.ndim != 2:
        raise ValueError("PLM feature matrices must be 2D")
    if p_train.shape[1] != p_target.shape[1]:
        raise ValueError(
            f"PLM feature dim mismatch: train={p_train.shape[1]} target={p_target.shape[1]}"
        )
    # rows = proteins, cols = cells
    x_filled = _column_mean_fill(x_train).T  # (n_train, n_cells)
    n_train, n_cells = x_filled.shape
    if n_train == 0 or n_cells == 0:
        raise ValueError("empty training matrix for PLM-ridge")

    # Standardize PLM features (fit on training proteins only).
    mu = p_train.mean(axis=0, keepdims=True)
    sigma = p_train.std(axis=0, keepdims=True)
    sigma = np.where(sigma > 1e-8, sigma, 1.0)
    p_train_n = (p_train - mu) / sigma
    p_target_n = (p_target - mu) / sigma

    # Intercept via centering the target by per-cell mean over training proteins.
    y_mean = x_filled.mean(axis=0, keepdims=True)  # (1, n_cells) == cell_mean baseline
    yc = x_filled - y_mean

    # B = (Pn^T Pn + lam I)^{-1} Pn^T Yc   -> (D, n_cells)
    d = p_train_n.shape[1]
    a_mat = p_train_n.T @ p_train_n + lam * np.eye(d)
    b_mat = np.linalg.solve(a_mat, p_train_n.T @ yc)

    pred = p_target_n @ b_mat + y_mean  # (n_targets, n_cells)
    return np.asarray(pred.T, dtype=np.float32)  # (n_cells, n_targets)


def compute_plm_knn_predictions(
    plm_train: np.ndarray,
    plm_target: np.ndarray,
    x_train: np.ndarray,
    ks: Sequence[int] = (5, 10),
) -> dict[int, np.ndarray]:
    """Similarity-weighted k nearest neighbours in raw PLM (cosine) space.

    For each held-out protein, take its ``k`` nearest training proteins by
    cosine similarity and average their (column-mean-filled) expression
    profiles, weighted by clipped similarity (uniform fallback).

    Returns
    -------
    Mapping ``k -> (n_cells, n_targets)`` float32 prediction matrix.
    """
    p_train = np.asarray(plm_train, dtype=np.float64)
    p_target = np.asarray(plm_target, dtype=np.float64)
    if p_train.ndim != 2 or p_target.ndim != 2:
        raise ValueError("PLM feature matrices must be 2D")
    if p_train.shape[1] != p_target.shape[1]:
        raise ValueError(
            f"PLM feature dim mismatch: train={p_train.shape[1]} target={p_target.shape[1]}"
        )
    n_train = p_train.shape[0]
    n_targets = p_target.shape[0]
    if n_train == 0:
        raise ValueError("no training proteins for PLM-kNN")

    x_filled = _column_mean_fill(x_train)  # (n_cells, n_train)
    n_cells = x_filled.shape[0]

    sim = _l2_normalize(p_target) @ _l2_normalize(p_train).T  # (n_targets, n_train)
    # Clip negatives: a negatively-correlated neighbour should not subtract.
    sim_pos = np.clip(sim, 0.0, None)

    out: dict[int, np.ndarray] = {}
    for k in ks:
        kk = max(1, min(int(k), n_train))
        if kk >= n_train:
            idx = np.broadcast_to(np.arange(n_train), (n_targets, n_train)).copy()
        else:
            part = np.argpartition(-sim_pos, kk - 1, axis=1)[:, :kk]
            idx = part  # (n_targets, kk)
        weights = np.take_along_axis(sim_pos, idx, axis=1)  # (n_targets, kk)
        wsum = weights.sum(axis=1, keepdims=True)
        # Uniform fallback when all similarities are ~0.
        weights = np.where(
            wsum > 1e-12,
            weights / np.where(wsum > 1e-12, wsum, 1.0),
            np.full_like(weights, 1.0 / kk),
        )
        neighbor_profiles = x_filled[:, idx]  # (n_cells, n_targets, kk)
        pred = np.einsum("ctk,tk->ct", neighbor_profiles, weights)  # (n_cells, n_targets)
        out[int(k)] = np.asarray(pred, dtype=np.float32)
    return out


def compute_expansion_reference_predictions(
    references: Sequence[str],
    plm_train: np.ndarray,
    plm_target: np.ndarray,
    x_train: np.ndarray,
    ridge_lambda: float = 1.0,
    knn_ks: Sequence[int] = (5, 10),
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Compute all requested reference predictions, isolating failures.

    Each method is computed independently inside its own try/except so a single
    reference failing cannot poison the SABER expansion artifact. The smallest
    ``k`` is also stored under the bare ``plm_knn`` key (primary); every k is
    stored under ``plm_knn_k{k}``.

    Returns
    -------
    predictions: ``{method_key: (n_cells, n_targets) array}``
    status: per-method status dicts (ok/reason) for the failure report.
    """
    refs = [str(r) for r in references if str(r)]
    predictions: dict[str, np.ndarray] = {}
    status: list[dict[str, Any]] = []

    if "plm_ridge" in refs:
        try:
            predictions["plm_ridge"] = compute_plm_ridge_predictions(
                plm_train=plm_train,
                plm_target=plm_target,
                x_train=x_train,
                ridge_lambda=float(ridge_lambda),
            )
            status.append({"method": "plm_ridge", "ok": True})
        except Exception as exc:  # noqa: BLE001 - report, don't crash the run
            status.append({"method": "plm_ridge", "ok": False, "reason": f"{type(exc).__name__}: {exc}"})

    if "plm_knn" in refs and len(knn_ks) > 0:
        primary_k = int(sorted(set(int(k) for k in knn_ks))[0])
        try:
            knn_preds = compute_plm_knn_predictions(
                plm_train=plm_train,
                plm_target=plm_target,
                x_train=x_train,
                ks=sorted(set(int(k) for k in knn_ks)),
            )
            for k_value, pred in knn_preds.items():
                key = "plm_knn" if int(k_value) == primary_k else f"plm_knn_k{int(k_value)}"
                predictions[key] = pred
            status.append({"method": "plm_knn", "ok": True, "ks": sorted(set(int(k) for k in knn_ks))})
        except Exception as exc:  # noqa: BLE001 - report, don't crash the run
            status.append({"method": "plm_knn", "ok": False, "reason": f"{type(exc).__name__}: {exc}"})

    return predictions, status
