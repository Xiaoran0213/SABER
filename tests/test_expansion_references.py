from __future__ import annotations

import numpy as np

from saber.expansion_references import (
    REFERENCE_METHODS,
    compute_expansion_reference_predictions,
    compute_plm_knn_predictions,
    compute_plm_ridge_predictions,
)


def _pearson_per_target(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-target (column-wise) Pearson between (n_cells, n_targets) matrices."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    out = np.empty(p.shape[1], dtype=np.float64)
    for j in range(p.shape[1]):
        a = p[:, j] - p[:, j].mean()
        b = t[:, j] - t[:, j].mean()
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        out[j] = float((a * b).sum() / denom) if denom > 0 else 0.0
    return out


def test_reference_methods_constant() -> None:
    assert REFERENCE_METHODS == ("plm_ridge", "plm_knn")


def test_plm_ridge_shape_and_finite() -> None:
    rng = np.random.default_rng(0)
    n_cells, n_train, n_targets, dim = 30, 80, 8, 16
    plm_train = rng.standard_normal((n_train, dim)).astype(np.float32)
    plm_target = rng.standard_normal((n_targets, dim)).astype(np.float32)
    x_train = rng.standard_normal((n_cells, n_train)).astype(np.float32)

    pred = compute_plm_ridge_predictions(plm_train, plm_target, x_train, ridge_lambda=1.0)

    assert pred.shape == (n_cells, n_targets)
    assert np.isfinite(pred).all()
    assert pred.dtype == np.float32


def test_plm_ridge_recovers_linear_map() -> None:
    """With an exact linear PLM -> expression map, ridge should nearly recover it."""
    rng = np.random.default_rng(1)
    n_cells, n_train, n_targets, dim = 30, 120, 6, 12
    plm_train = rng.standard_normal((n_train, dim))
    plm_target = rng.standard_normal((n_targets, dim))
    w_true = rng.standard_normal((dim, n_cells))
    x_train = (plm_train @ w_true).T.astype(np.float32)  # (n_cells, n_train)
    x_target = (plm_target @ w_true).T  # (n_cells, n_targets)

    pred = compute_plm_ridge_predictions(plm_train, plm_target, x_train, ridge_lambda=1e-4)

    assert pred.shape == x_target.shape
    corrs = _pearson_per_target(pred, x_target)
    assert corrs.min() > 0.95


def test_plm_ridge_handles_nan_inputs() -> None:
    rng = np.random.default_rng(2)
    n_cells, n_train, n_targets, dim = 25, 60, 5, 10
    plm_train = rng.standard_normal((n_train, dim)).astype(np.float32)
    plm_target = rng.standard_normal((n_targets, dim)).astype(np.float32)
    x_train = rng.standard_normal((n_cells, n_train)).astype(np.float32)
    # Punch ~20% NaN holes (structural missing).
    mask = rng.random(x_train.shape) < 0.2
    x_train[mask] = np.nan

    pred = compute_plm_ridge_predictions(plm_train, plm_target, x_train, ridge_lambda=1.0)

    assert pred.shape == (n_cells, n_targets)
    assert np.isfinite(pred).all()


def test_plm_knn_shape_and_keys() -> None:
    rng = np.random.default_rng(3)
    n_cells, n_train, n_targets, dim = 20, 50, 4, 8
    plm_train = rng.standard_normal((n_train, dim)).astype(np.float32)
    plm_target = rng.standard_normal((n_targets, dim)).astype(np.float32)
    x_train = rng.standard_normal((n_cells, n_train)).astype(np.float32)

    out = compute_plm_knn_predictions(plm_train, plm_target, x_train, ks=(5, 10))

    assert set(out.keys()) == {5, 10}
    for k, pred in out.items():
        assert pred.shape == (n_cells, n_targets)
        assert np.isfinite(pred).all()
        assert pred.dtype == np.float32


def test_plm_knn_clamps_k_to_n_train() -> None:
    rng = np.random.default_rng(4)
    n_train = 3
    plm_train = rng.standard_normal((n_train, 6)).astype(np.float32)
    plm_target = rng.standard_normal((2, 6)).astype(np.float32)
    x_train = rng.standard_normal((8, n_train)).astype(np.float32)

    out = compute_plm_knn_predictions(plm_train, plm_target, x_train, ks=(5, 10))

    # k larger than n_train collapses to n_train; still produces finite output.
    assert set(out.keys()) == {5, 10}
    for pred in out.values():
        assert pred.shape == (8, 2)
        assert np.isfinite(pred).all()


def test_compute_predictions_keys_and_status() -> None:
    rng = np.random.default_rng(5)
    n_cells, n_train, n_targets, dim = 20, 50, 4, 8
    plm_train = rng.standard_normal((n_train, dim)).astype(np.float32)
    plm_target = rng.standard_normal((n_targets, dim)).astype(np.float32)
    x_train = rng.standard_normal((n_cells, n_train)).astype(np.float32)

    preds, status = compute_expansion_reference_predictions(
        references=("plm_ridge", "plm_knn"),
        plm_train=plm_train,
        plm_target=plm_target,
        x_train=x_train,
        ridge_lambda=1.0,
        knn_ks=(5, 10),
    )

    assert set(preds.keys()) == {"plm_ridge", "plm_knn", "plm_knn_k10"}
    assert preds["plm_ridge"].shape == (n_cells, n_targets)
    assert preds["plm_knn"].shape == (n_cells, n_targets)
    methods_ok = {entry["method"]: bool(entry["ok"]) for entry in status}
    assert methods_ok == {"plm_ridge": True, "plm_knn": True}


def test_compute_predictions_empty_when_no_references() -> None:
    rng = np.random.default_rng(6)
    preds, status = compute_expansion_reference_predictions(
        references=(),
        plm_train=rng.standard_normal((10, 4)),
        plm_target=rng.standard_normal((3, 4)),
        x_train=rng.standard_normal((5, 10)),
    )
    assert preds == {}
    assert status == []
