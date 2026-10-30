from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from saber.artifacts import load_json, write_json
from saber.metrics import compute_imputation_metrics, pearson, spearman

from .common import metric_rows, write_rows_csv


def _protein_mean_baseline(x_input: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
    x = np.asarray(x_input, dtype=np.float32)
    observed = np.asarray(observed_mask).astype(bool) & np.isfinite(x)
    counts = observed.sum(axis=0)
    sums = np.where(observed, x, 0.0).sum(axis=0, dtype=np.float64)
    means = np.zeros((x.shape[1],), dtype=np.float32)
    np.divide(sums, counts, out=means, where=counts > 0)
    return np.broadcast_to(means[None, :], x.shape).copy().astype(np.float32)


def _cell_mean_baseline(x_input: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
    x = np.asarray(x_input, dtype=np.float32)
    observed = np.asarray(observed_mask).astype(bool) & np.isfinite(x)
    counts = observed.sum(axis=1, keepdims=True)
    sums = np.where(observed, x, 0.0).sum(axis=1, keepdims=True, dtype=np.float64)
    means = np.zeros((x.shape[0], 1), dtype=np.float32)
    np.divide(sums, counts, out=means, where=counts > 0)
    return np.broadcast_to(means, x.shape).copy().astype(np.float32)


def _protein_rows(manifest: dict[str, Any], n_proteins: int) -> list[dict[str, Any]]:
    labels_path = manifest.get("artifacts", {}).get("labels")
    if labels_path:
        try:
            with np.load(labels_path, allow_pickle=True) as labels:
                ids = labels["protein_ids"].astype(str).tolist() if "protein_ids" in labels.files else []
            if len(ids) == n_proteins:
                return [{"protein_id": str(pid), "protein_index": int(idx)} for idx, pid in enumerate(ids)]
        except Exception:
            pass
    return [{"protein_id": f"protein_{idx}", "protein_index": int(idx)} for idx in range(n_proteins)]


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
        base = protein_rows[j] if j < len(protein_rows) else {"protein_id": f"protein_{j}", "protein_index": int(j)}
        row: dict[str, Any] = {
            **base,
            "method": method,
            "method_kind": method_kind,
            "n_valid": n_valid,
            "rmse_missing": None,
            "mae_missing": None,
            "pearson_missing": None,
            "spearman_missing": None,
        }
        if n_valid > 0:
            row["rmse_missing"] = float(np.sqrt(np.mean(errors**2)))
            row["mae_missing"] = float(np.mean(np.abs(errors)))
        if n_valid >= 2:
            row["pearson_missing"] = pearson(true[col_mask, j], pred[col_mask, j])
            row["spearman_missing"] = spearman(true[col_mask, j], pred[col_mask, j])
        rows.append(row)
    return rows


def evaluate_imputation_manifest(manifest_path: str | Path, outdir: str | Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = load_json(manifest_path)
    imputation_path = manifest.get("artifacts", {}).get("imputation")
    if not imputation_path:
        raise ValueError(f"Manifest has no artifacts.imputation: {manifest_path}")

    with np.load(imputation_path) as data:
        x_true = data["x_true"]
        missing_mask = data["injected_missing_mask"].astype(bool)
        x_input = data["x_input"] if "x_input" in data.files else x_true
        observed_mask = data["observed_mask"] if "observed_mask" in data.files else np.isfinite(x_input)
        predictions: dict[str, np.ndarray] = {
            key.removeprefix("pred__"): np.asarray(data[key], dtype=np.float32)
            for key in data.files
            if key.startswith("pred__")
        }
        if "protein_mean" not in predictions:
            predictions["protein_mean"] = _protein_mean_baseline(x_input, observed_mask)
        if "cell_mean" not in predictions:
            predictions["cell_mean"] = _cell_mean_baseline(x_input, observed_mask)
        results: dict[str, Any] = {}
        rows: list[dict[str, Any]] = []
        per_protein_rows: list[dict[str, Any]] = []
        protein_rows = _protein_rows(manifest, int(x_true.shape[1]))
        for method, pred in sorted(predictions.items()):
            metrics = compute_imputation_metrics(x_true, pred, missing_mask)
            results[method] = metrics
            rows.extend(metric_rows("imputation", manifest_path, method, metrics, manifest=manifest))
            method_kind = "baseline" if method in {"protein_mean", "cell_mean"} else "model"
            per_protein_rows.extend(
                _per_protein_metrics(
                    x_true,
                    pred,
                    missing_mask,
                    protein_rows,
                    method=method,
                    method_kind=method_kind,
                )
            )

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "imputation_metrics.json"
    csv_path = outdir / "imputation_metrics.csv"
    source_path = outdir / "source_imputation_metrics.csv"
    per_protein_path = outdir / "source_imputation_per_protein.csv"
    write_json(
        json_path,
        {
            "manifest": str(manifest_path.resolve()),
            "metric_contract": {
                "scope": "Metrics are computed on injected_missing_mask only.",
                "scale": "Metrics use the model/data scale stored in x_true. For zscore datasets this matches the zscored training scale; LFQ has a separate raw-scale evaluator.",
                "primary_metric": "pearson_missing_mean is the mean of per-protein Pearson correlations over proteins with at least two evaluated masked cells.",
            },
            "results": results,
        },
    )
    wide_rows = [{"method": method, **metrics} for method, metrics in sorted(results.items())]
    write_rows_csv(csv_path, wide_rows, preferred=["method"])
    write_rows_csv(
        source_path,
        rows,
        preferred=["task", "dataset", "method", "condition", "seed", "scale_mode", "metric", "value", "artifact_path"],
    )
    write_rows_csv(
        per_protein_path,
        per_protein_rows,
        preferred=[
            "method",
            "method_kind",
            "protein_id",
            "protein_index",
            "n_valid",
            "pearson_missing",
            "spearman_missing",
            "rmse_missing",
            "mae_missing",
        ],
    )
    return {
        "json": str(json_path.resolve()),
        "csv": str(csv_path.resolve()),
        "source_csv": str(source_path.resolve()),
        "per_protein_csv": str(per_protein_path.resolve()),
        "results": results,
        "rows": rows,
        "per_protein": per_protein_rows,
    }
