from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from saber.artifacts import load_json, write_json
from saber.metrics import compute_matrix_prediction_metrics, pearson, spearman

from .common import metric_rows, write_rows_csv


def _protein_mean_baseline(x_true_train: np.ndarray, n_targets: int) -> np.ndarray:
    """Honest floor for held-out proteins: the global training mean as a constant.

    The previous version took the per-column mean of ``x_true_target`` -- i.e. it
    peeked at the held-out truth to set each target protein's level, which is
    unavailable in a real expansion scenario and made MAE/RMSE artificially low.
    A held-out protein has no observed values, so its own mean is undefined; the
    only real-obtainable floor is the grand mean of the *training* proteins,
    broadcast to every (cell, target). The prediction is constant, so per-target
    Pearson is None (the floor every method should beat) and MAE/RMSE reflect the
    true spread rather than a cheated level.
    """
    x = np.asarray(x_true_train, dtype=np.float64)
    finite = np.isfinite(x)
    grand = float(x[finite].mean()) if finite.any() else 0.0
    return np.full((x.shape[0], int(n_targets)), grand, dtype=np.float32)


def _cell_mean_baseline(x_true_train: np.ndarray, n_targets: int) -> np.ndarray:
    x = np.asarray(x_true_train, dtype=np.float32)
    finite = np.isfinite(x)
    counts = finite.sum(axis=1, keepdims=True)
    sums = np.where(finite, x, 0.0).sum(axis=1, keepdims=True, dtype=np.float64)
    means = np.zeros((x.shape[0], 1), dtype=np.float32)
    np.divide(sums, counts, out=means, where=counts > 0)
    means = np.where(np.isfinite(means), means, 0.0).astype(np.float32)
    return np.broadcast_to(means, (x.shape[0], int(n_targets))).copy().astype(np.float32)


def _target_protein_rows(manifest: dict[str, Any], n_targets: int) -> list[dict[str, Any]]:
    labels_path = manifest.get("artifacts", {}).get("labels")
    if labels_path:
        try:
            with np.load(labels_path, allow_pickle=True) as labels:
                ids = labels["target_protein_ids"].astype(str).tolist() if "target_protein_ids" in labels.files else []
                indices = labels["target_protein_indices"].astype(int).tolist() if "target_protein_indices" in labels.files else []
            if len(ids) == n_targets:
                if len(indices) != n_targets:
                    indices = list(range(n_targets))
                return [
                    {"protein_id": str(pid), "protein_index": int(idx), "target_position": int(pos)}
                    for pos, (pid, idx) in enumerate(zip(ids, indices))
                ]
        except Exception:
            pass
    return [{"protein_id": f"target_{pos}", "protein_index": int(pos), "target_position": int(pos)} for pos in range(n_targets)]


def _per_protein_metrics(
    x_true: np.ndarray,
    x_pred: np.ndarray,
    mask: np.ndarray,
    protein_rows: list[dict[str, Any]],
    method: str,
    method_kind: str,
) -> list[dict[str, Any]]:
    true = np.asarray(x_true, dtype=np.float32)
    pred = np.asarray(x_pred, dtype=np.float32)
    eval_mask = np.asarray(mask).astype(bool)
    rows: list[dict[str, Any]] = []
    for j in range(true.shape[1]):
        col_mask = eval_mask[:, j] & np.isfinite(true[:, j]) & np.isfinite(pred[:, j])
        n_valid = int(col_mask.sum())
        errors = pred[col_mask, j] - true[col_mask, j]
        base = protein_rows[j] if j < len(protein_rows) else {
            "protein_id": f"target_{j}",
            "protein_index": int(j),
            "target_position": int(j),
        }
        row: dict[str, Any] = {
            **base,
            "method": method,
            "method_kind": method_kind,
            "n_valid": n_valid,
            "rmse_target": None,
            "mae_target": None,
            "pearson_target": None,
            "spearman_target": None,
        }
        if n_valid > 0:
            row["rmse_target"] = float(np.sqrt(np.mean(errors**2)))
            row["mae_target"] = float(np.mean(np.abs(errors)))
        if n_valid >= 2:
            row["pearson_target"] = pearson(true[col_mask, j], pred[col_mask, j])
            row["spearman_target"] = spearman(true[col_mask, j], pred[col_mask, j])
        rows.append(row)
    return rows


def compute_expansion_evaluation(manifest_path: str | Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = load_json(manifest_path)
    expansion_path = manifest.get("artifacts", {}).get("expansion")
    if not expansion_path:
        raise ValueError(f"Manifest has no artifacts.expansion: {manifest_path}")

    with np.load(expansion_path) as data:
        x_true = data["x_true_target"]
        mask_key = "eval_target_mask" if "eval_target_mask" in data.files else "available_target_mask"
        mask = data[mask_key].astype(bool)
        predictions: dict[str, np.ndarray] = {}
        method_kinds: dict[str, str] = {}
        # Naive PLM reference models (plm_ridge, plm_knn, plm_knn_k<N>) are written
        # by the runner as pred_target__<method> arrays and recorded in the
        # manifest's references.written_keys. Tag them as "reference" (the
        # reference-ablation tier, alongside no-PLM/no-W) instead of the generic
        # "model" so downstream grouping keeps them distinct from SABER itself.
        reference_methods = set(manifest.get("references", {}).get("written_keys", []) or [])
        for key in data.files:
            if not key.startswith("pred_target__"):
                continue
            method = key.removeprefix("pred_target__")
            predictions[method] = np.asarray(data[key], dtype=np.float32)
            method_kinds[method] = "reference" if method in reference_methods else "model"
        if "x_true_train" in data.files and x_true.shape[1] > 0:
            predictions["cell_mean"] = _cell_mean_baseline(data["x_true_train"], x_true.shape[1])
            method_kinds["cell_mean"] = "baseline"
            predictions["protein_mean"] = _protein_mean_baseline(data["x_true_train"], x_true.shape[1])
            method_kinds["protein_mean"] = "baseline"

    results = {
        method: compute_matrix_prediction_metrics(x_true, pred, mask, prefix="target")
        for method, pred in sorted(predictions.items())
    }
    protein_rows = _target_protein_rows(manifest, int(x_true.shape[1]))
    per_protein: list[dict[str, Any]] = []
    for method, pred in sorted(predictions.items()):
        per_protein.extend(
            _per_protein_metrics(
                x_true,
                pred,
                mask,
                protein_rows,
                method=method,
                method_kind=method_kinds.get(method, "model"),
            )
        )
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path.resolve()),
        "mask_key": mask_key,
        "method_kinds": method_kinds,
        "results": results,
        "per_protein": per_protein,
    }


def evaluate_expansion_manifest(manifest_path: str | Path, outdir: str | Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    payload = compute_expansion_evaluation(manifest_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "expansion_metrics.json"
    csv_path = outdir / "expansion_metrics.csv"
    source_path = outdir / "source_expansion_metrics.csv"
    per_protein_path = outdir / "source_expansion_per_protein.csv"

    write_json(
        json_path,
        {
            "manifest": payload["manifest_path"],
            "mask_key": payload["mask_key"],
            "metric_contract": {
                "scope": "Held-out target proteins absent from training matrix.",
                "mask": "Uses eval_target_mask when present, otherwise available_target_mask.",
                "primary_metric": "pearson_target_mean is the mean per-target-protein Pearson correlation.",
            },
            "method_kinds": payload["method_kinds"],
            "results": payload["results"],
        },
    )
    write_rows_csv(
        csv_path,
        [
            {"method": method, "method_kind": payload["method_kinds"].get(method, "model"), **metrics}
            for method, metrics in sorted(payload["results"].items())
        ],
        preferred=["method", "method_kind"],
    )
    source_rows: list[dict[str, Any]] = []
    for method, metrics in sorted(payload["results"].items()):
        source_rows.extend(
            metric_rows(
                "proteome_expansion",
                manifest_path,
                method,
                metrics,
                manifest=payload["manifest"],
                extra={"method_kind": payload["method_kinds"].get(method, "model"), "mask_key": payload["mask_key"]},
            )
        )
    write_rows_csv(
        source_path,
        source_rows,
        preferred=[
            "task",
            "dataset",
            "method",
            "method_kind",
            "condition",
            "seed",
            "scale_mode",
            "target_fraction",
            "metric",
            "value",
            "artifact_path",
        ],
    )
    write_rows_csv(
        per_protein_path,
        payload["per_protein"],
        preferred=[
            "method",
            "method_kind",
            "protein_id",
            "protein_index",
            "target_position",
            "n_valid",
            "pearson_target",
            "spearman_target",
            "rmse_target",
            "mae_target",
        ],
    )
    return {
        "json": str(json_path.resolve()),
        "csv": str(csv_path.resolve()),
        "source_csv": str(source_path.resolve()),
        "per_protein_csv": str(per_protein_path.resolve()),
        "results": payload["results"],
        "per_protein": payload["per_protein"],
    }

