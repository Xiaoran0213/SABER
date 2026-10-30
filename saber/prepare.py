from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import write_json
from .feature_cache import FeatureCacheError, load_protein_feature_cache, save_protein_feature_cache


@dataclass(frozen=True)
class PreparedInputs:
    h5ad_path: Path
    feature_cache_path: Path
    report_path: Path
    report: dict[str, Any]


def _as_str_array(values: np.ndarray) -> np.ndarray:
    return np.asarray([str(v) for v in np.asarray(values).tolist()], dtype=object)


def _duplicates(values: np.ndarray) -> list[str]:
    seen: set[str] = set()
    dup: set[str] = set()
    for value in values.tolist():
        text = str(value)
        if text in seen:
            dup.add(text)
        seen.add(text)
    return sorted(dup)


def _first_index(ids: np.ndarray, duplicate_policy: str) -> dict[str, int]:
    if duplicate_policy not in {"fail", "first"}:
        raise ValueError("duplicate_cache_policy must be 'fail' or 'first'")
    duplicates = _duplicates(ids)
    if duplicates and duplicate_policy == "fail":
        preview = ", ".join(duplicates[:10])
        raise FeatureCacheError(f"Duplicate cache protein IDs: {preview}")
    index: dict[str, int] = {}
    for i, value in enumerate(ids.tolist()):
        text = str(value)
        if text not in index:
            index[text] = i
    return index


def _adata_protein_ids(adata: Any, protein_id_field: str) -> np.ndarray:
    if protein_id_field and protein_id_field in adata.var.columns:
        return _as_str_array(adata.var[protein_id_field].to_numpy())
    return _as_str_array(np.asarray(adata.var_names.tolist(), dtype=object))


def prepare_aligned_inputs(
    data_path: str | Path,
    cache_path: str | Path,
    outdir: str | Path,
    name: str,
    cache_protein_ids_path: str | Path = "",
    protein_id_field: str = "",
    expected_dim: int | None = None,
    duplicate_cache_policy: str = "fail",
    drop_missing_proteins: bool = False,
    duplicate_dataset_policy: str = "fail",
) -> PreparedInputs:
    try:
        import anndata as ad
    except Exception as exc:
        raise RuntimeError("anndata is required to prepare aligned inputs") from exc

    data_path = Path(data_path)
    cache_path = Path(cache_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(data_path)
    requested_ids = _adata_protein_ids(adata, protein_id_field)
    requested_duplicates = _duplicates(requested_ids)
    if duplicate_dataset_policy not in {"fail", "first"}:
        raise ValueError("duplicate_dataset_policy must be 'fail' or 'first'")
    if requested_duplicates and duplicate_dataset_policy == "fail":
        preview = ", ".join(requested_duplicates[:10])
        raise ValueError(f"Duplicate dataset protein IDs: {preview}")

    cache = load_protein_feature_cache(
        cache_path,
        expected_dim=expected_dim,
        cache_protein_ids_path=cache_protein_ids_path,
    )
    cache_ids = _as_str_array(cache.protein_ids)
    if not cache_ids.size:
        raise FeatureCacheError("A cache protein ID sidecar is required for aligned preparation")
    if cache_ids.shape[0] != cache.features.shape[0]:
        raise FeatureCacheError("cache protein ID count does not match feature rows")

    cache_duplicates = _duplicates(cache_ids)
    index = _first_index(cache_ids, duplicate_cache_policy)
    missing_requested = [str(pid) for pid in requested_ids.tolist() if str(pid) not in index]
    if missing_requested and not drop_missing_proteins:
        preview = ", ".join(missing_requested[:10])
        raise FeatureCacheError(
            f"Feature cache is missing {len(missing_requested)} dataset proteins: {preview}. "
            "Use --drop-missing-proteins to create an explicit filtered dataset."
        )

    seen_dataset: set[str] = set()
    kept_positions: list[int] = []
    dropped_duplicate_dataset: list[str] = []
    for i, pid in enumerate(requested_ids.tolist()):
        text = str(pid)
        if text in seen_dataset:
            dropped_duplicate_dataset.append(text)
            continue
        seen_dataset.add(text)
        if text in index:
            kept_positions.append(i)
    kept_ids = requested_ids[kept_positions]
    feature_rows = [index[str(pid)] for pid in kept_ids.tolist()]
    aligned_features = cache.features[np.asarray(feature_rows, dtype=np.int64)].astype(np.float32, copy=False)
    prepared = adata[:, kept_positions].copy()

    h5ad_path = outdir / f"{name}.h5ad"
    feature_cache_path = outdir / f"{name}_features.npz"
    report_path = outdir / f"{name}_alignment_report.json"
    prepared.write_h5ad(h5ad_path)
    save_protein_feature_cache(
        feature_cache_path,
        aligned_features,
        kept_ids,
        metadata={
            "source_cache": str(cache_path.resolve()),
            "source_cache_protein_ids": str(Path(cache_protein_ids_path).resolve()) if cache_protein_ids_path else "",
            "source_h5ad": str(data_path.resolve()),
            "duplicate_cache_policy": duplicate_cache_policy,
            "duplicate_dataset_policy": duplicate_dataset_policy,
            "drop_missing_proteins": bool(drop_missing_proteins),
        },
    )

    cache_unique = set(str(x) for x in cache_ids.tolist())
    requested_unique = set(str(x) for x in requested_ids.tolist())
    extra_cache = sorted(cache_unique - requested_unique)
    report = {
        "name": name,
        "source_h5ad": str(data_path.resolve()),
        "source_cache": str(cache_path.resolve()),
        "source_cache_protein_ids": str(Path(cache_protein_ids_path).resolve()) if cache_protein_ids_path else "",
        "prepared_h5ad": str(h5ad_path.resolve()),
        "prepared_feature_cache": str(feature_cache_path.resolve()),
        "n_cells": int(adata.n_obs),
        "n_requested_proteins": int(requested_ids.shape[0]),
        "n_kept_proteins": int(kept_ids.shape[0]),
        "n_dropped_missing_requested_proteins": int(len(missing_requested)),
        "dropped_missing_requested_preview": missing_requested[:25],
        "n_dropped_duplicate_dataset_proteins": int(len(dropped_duplicate_dataset)),
        "dropped_duplicate_dataset_preview": dropped_duplicate_dataset[:25],
        "n_cache_rows": int(cache.features.shape[0]),
        "n_cache_unique_proteins": int(len(cache_unique)),
        "n_duplicate_cache_protein_ids": int(len(cache_duplicates)),
        "duplicate_cache_protein_id_preview": cache_duplicates[:25],
        "n_extra_cache_proteins": int(len(extra_cache)),
        "extra_cache_protein_preview": extra_cache[:25],
        "feature_dim": int(aligned_features.shape[1]),
        "protein_id_field": protein_id_field,
        "duplicate_cache_policy": duplicate_cache_policy,
        "duplicate_dataset_policy": duplicate_dataset_policy,
        "drop_missing_proteins": bool(drop_missing_proteins),
    }
    write_json(report_path, report)
    return PreparedInputs(
        h5ad_path=h5ad_path,
        feature_cache_path=feature_cache_path,
        report_path=report_path,
        report=report,
    )
