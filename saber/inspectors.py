from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .feature_cache import load_protein_feature_cache


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(v) for v in value]
    return value


def _read_h5ad_protein_ids(data_path: str | Path, protein_id_field: str = "") -> np.ndarray:
    import anndata as ad

    adata = ad.read_h5ad(data_path, backed="r")
    if protein_id_field and protein_id_field in adata.var.columns:
        ids = adata.var[protein_id_field].astype(str).to_numpy(dtype=object)
    else:
        ids = np.asarray([str(v) for v in adata.var_names.tolist()], dtype=object)
    adata.file.close()
    return ids


def _matrix_slice_to_array(x: Any) -> np.ndarray:
    arr = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    return np.asarray(arr, dtype=np.float32)


def inspect_h5ad(
    data_path: str | Path,
    batch_key: str = "batch",
    cell_type_key: str = "cell_type",
    protein_id_field: str = "",
    max_cells: int = 256,
) -> dict[str, Any]:
    import anndata as ad

    data_path = Path(data_path)
    adata = ad.read_h5ad(data_path, backed="r")
    obs_columns = [str(c) for c in adata.obs.columns]
    var_columns = [str(c) for c in adata.var.columns]
    if protein_id_field and protein_id_field in adata.var.columns:
        protein_ids = adata.var[protein_id_field].astype(str).to_numpy(dtype=object)
        protein_id_source = f"var.{protein_id_field}"
    else:
        protein_ids = np.asarray([str(v) for v in adata.var_names.tolist()], dtype=object)
        protein_id_source = "var_names"

    summary: dict[str, Any] = {
        "path": str(data_path.resolve()),
        "n_cells": int(adata.n_obs),
        "n_proteins": int(adata.n_vars),
        "obs_columns": obs_columns,
        "var_columns": var_columns,
        "batch_key": batch_key,
        "batch_key_present": bool(batch_key in obs_columns) if batch_key else False,
        "cell_type_key": cell_type_key,
        "cell_type_key_present": bool(cell_type_key in obs_columns) if cell_type_key else False,
        "protein_id_source": protein_id_source,
        "protein_id_field_present": bool(protein_id_field in var_columns) if protein_id_field else False,
        "n_unique_protein_ids": int(len(set(str(x) for x in protein_ids.tolist()))),
        "n_duplicate_protein_ids": int(protein_ids.shape[0] - len(set(str(x) for x in protein_ids.tolist()))),
        "protein_id_preview": [str(x) for x in protein_ids[:5].tolist()],
    }
    if batch_key and batch_key in obs_columns:
        summary["n_batches"] = int(adata.obs[batch_key].astype(str).nunique())
    if cell_type_key and cell_type_key in obs_columns:
        summary["n_cell_types"] = int(adata.obs[cell_type_key].astype(str).nunique())

    n_sample = min(int(max_cells), int(adata.n_obs))
    if n_sample > 0:
        idx = np.linspace(0, adata.n_obs - 1, num=n_sample, dtype=np.int64)
        x = _matrix_slice_to_array(adata.X[idx, :])
        finite = np.isfinite(x)
        summary["sampled_cells"] = int(n_sample)
        summary["sample_finite_fraction"] = float(finite.mean())
        summary["sample_nan_fraction"] = float(np.isnan(x).mean())
        summary["sample_value_mean"] = float(np.nanmean(x)) if np.isfinite(x).any() else None
        summary["sample_value_std"] = float(np.nanstd(x)) if np.isfinite(x).any() else None
    adata.file.close()
    return _as_jsonable(summary)


def inspect_feature_cache(
    cache_path: str | Path,
    cache_protein_ids_path: str | Path = "",
    data_path: str | Path = "",
    protein_id_field: str = "",
    expected_dim: int | None = None,
) -> dict[str, Any]:
    cache = load_protein_feature_cache(
        cache_path,
        expected_dim=expected_dim,
        cache_protein_ids_path=cache_protein_ids_path,
    )
    cache_ids = np.asarray(cache.protein_ids, dtype=object)
    summary: dict[str, Any] = {
        "cache_path": str(Path(cache_path).resolve()),
        "cache_protein_ids_path": str(Path(cache_protein_ids_path).resolve()) if cache_protein_ids_path else "",
        "n_feature_rows": int(cache.features.shape[0]),
        "feature_dim": int(cache.features.shape[1]),
        "has_cache_protein_ids": bool(cache_ids.size),
        "n_cache_protein_ids": int(cache_ids.shape[0]),
        "n_unique_cache_protein_ids": int(len(set(str(x) for x in cache_ids.tolist()))) if cache_ids.size else 0,
        "metadata": cache.metadata,
    }
    if expected_dim is not None:
        summary["expected_dim"] = int(expected_dim)
        summary["expected_dim_match"] = bool(int(expected_dim) == int(cache.features.shape[1]))
    if data_path:
        requested = _read_h5ad_protein_ids(data_path, protein_id_field=protein_id_field)
        requested_set = {str(x) for x in requested.tolist()}
        if cache_ids.size:
            cache_set = {str(x) for x in cache_ids.tolist()}
            missing = sorted(requested_set - cache_set)
            extra = sorted(cache_set - requested_set)
            summary.update(
                {
                    "data_path": str(Path(data_path).resolve()),
                    "n_requested_proteins": int(requested.shape[0]),
                    "n_requested_unique_proteins": int(len(requested_set)),
                    "n_missing_requested_proteins": int(len(missing)),
                    "missing_requested_preview": missing[:10],
                    "n_extra_cache_proteins": int(len(extra)),
                    "coverage_fraction": float((len(requested_set) - len(missing)) / max(1, len(requested_set))),
                    "row_order_compatible": False,
                    "protein_id_alignment_ready": bool(len(missing) == 0),
                }
            )
        else:
            summary.update(
                {
                    "data_path": str(Path(data_path).resolve()),
                    "n_requested_proteins": int(requested.shape[0]),
                    "row_order_compatible": bool(cache.features.shape[0] == requested.shape[0]),
                    "protein_id_alignment_ready": False,
                }
            )
    return _as_jsonable(summary)
