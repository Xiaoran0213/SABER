from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


class FeatureCacheError(ValueError):
    """Raised when a protein feature cache is malformed."""


@dataclass(frozen=True)
class FeatureCache:
    features: np.ndarray
    protein_ids: np.ndarray
    metadata: dict[str, Any]


def _as_str_array(values: np.ndarray | list[str]) -> np.ndarray:
    return np.asarray([str(v) for v in np.asarray(values).tolist()], dtype=object)


def _require_unique(ids: np.ndarray, label: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in ids.tolist():
        text = str(value)
        if text in seen:
            duplicates.add(text)
        seen.add(text)
    if duplicates:
        preview = ", ".join(sorted(duplicates)[:5])
        raise FeatureCacheError(f"Duplicate {label}: {preview}")


def read_protein_id_file(path: str | Path) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _as_str_array(np.load(path, allow_pickle=True))
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            key = "protein_ids" if "protein_ids" in data.files else data.files[0]
            return _as_str_array(data[key])

    delimiter = "\t" if suffix == ".tsv" else ","
    with path.open("r", newline="") as f:
        first = f.readline()
        f.seek(0)
        if delimiter in first:
            reader = csv.reader(f, delimiter=delimiter)
            rows = [row for row in reader if row]
            if not rows:
                return np.asarray([], dtype=object)
            header = [cell.strip().lower() for cell in rows[0]]
            start = 1 if header[0] in {"protein_id", "protein", "id", "uniprot"} else 0
            return _as_str_array([row[0] for row in rows[start:] if row and row[0].strip()])
        return _as_str_array([line.strip() for line in f if line.strip()])


def _load_csv_cache(path: Path) -> FeatureCache:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader, None)
        if header is None or len(header) < 2:
            raise FeatureCacheError("CSV protein feature cache must have an id column and at least one feature column")
        protein_ids: list[str] = []
        rows: list[list[float]] = []
        for row in reader:
            if not row:
                continue
            protein_ids.append(str(row[0]))
            rows.append([float(x) for x in row[1:]])
    return FeatureCache(
        features=np.asarray(rows, dtype=np.float32),
        protein_ids=_as_str_array(protein_ids),
        metadata={"source_format": path.suffix.lower().lstrip(".")},
    )


def _load_raw_cache(path: Path) -> FeatureCache:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if "features" not in data.files:
                raise FeatureCacheError("NPZ protein feature cache must contain a 'features' array")
            features = np.asarray(data["features"], dtype=np.float32)
            protein_ids = _as_str_array(data["protein_ids"]) if "protein_ids" in data.files else np.asarray([], dtype=object)
            metadata: dict[str, Any] = {"source_format": "npz"}
            if "metadata" in data.files:
                raw = str(np.asarray(data["metadata"]).item())
                metadata.update(json.loads(raw))
            return FeatureCache(features=features, protein_ids=protein_ids, metadata=metadata)
    if suffix == ".npy":
        features = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        return FeatureCache(features=features, protein_ids=np.asarray([], dtype=object), metadata={"source_format": "npy"})
    if suffix in {".csv", ".tsv"}:
        return _load_csv_cache(path)
    raise FeatureCacheError(f"Unsupported protein feature cache format: {path.suffix}")


def load_protein_feature_cache(
    path: str | Path,
    protein_ids: Optional[np.ndarray] = None,
    expected_dim: Optional[int] = None,
    cache_protein_ids_path: str | Path = "",
) -> FeatureCache:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    cache = _load_raw_cache(path)
    features = np.asarray(cache.features, dtype=np.float32)
    if features.ndim != 2:
        raise FeatureCacheError("Protein feature cache must be a 2D matrix")
    if expected_dim is not None and int(expected_dim) != int(features.shape[1]):
        raise FeatureCacheError(
            f"Protein feature dim mismatch: config expects {expected_dim}, cache has {features.shape[1]}"
        )

    cache_ids = _as_str_array(cache.protein_ids) if cache.protein_ids.size else np.asarray([], dtype=object)
    if cache_protein_ids_path:
        sidecar_ids = read_protein_id_file(cache_protein_ids_path)
        if cache_ids.size and not np.array_equal(cache_ids.astype(str), sidecar_ids.astype(str)):
            raise FeatureCacheError("cache protein IDs and sidecar protein IDs disagree")
        cache_ids = sidecar_ids

    requested_ids = None if protein_ids is None else _as_str_array(protein_ids)
    if requested_ids is None:
        if cache_ids.size and cache_ids.shape[0] != features.shape[0]:
            raise FeatureCacheError("protein_ids length does not match feature rows")
        return FeatureCache(features=features, protein_ids=cache_ids, metadata=cache.metadata)

    _require_unique(requested_ids, "requested protein id")
    if not cache_ids.size:
        if features.shape[0] != requested_ids.shape[0]:
            raise FeatureCacheError(
                f"Feature rows={features.shape[0]} but dataset has {requested_ids.shape[0]} proteins"
            )
        return FeatureCache(features=features, protein_ids=requested_ids, metadata={**cache.metadata, "id_alignment": "row_order"})

    if cache_ids.shape[0] != features.shape[0]:
        raise FeatureCacheError("protein_ids length does not match feature rows")
    _require_unique(cache_ids, "cache protein id")
    index = {str(pid): i for i, pid in enumerate(cache_ids.tolist())}
    missing = [str(pid) for pid in requested_ids.tolist() if str(pid) not in index]
    if missing:
        preview = ", ".join(missing[:5])
        raise FeatureCacheError(f"Protein feature cache is missing {len(missing)} requested proteins: {preview}")
    order = np.asarray([index[str(pid)] for pid in requested_ids.tolist()], dtype=np.int64)
    return FeatureCache(
        features=features[order].astype(np.float32, copy=False),
        protein_ids=requested_ids,
        metadata={**cache.metadata, "id_alignment": "protein_id"},
    )


def save_protein_feature_cache(
    path: str | Path,
    features: np.ndarray,
    protein_ids: np.ndarray,
    metadata: Optional[dict[str, Any]] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    features = np.asarray(features, dtype=np.float32)
    protein_ids = _as_str_array(protein_ids)
    if features.ndim != 2:
        raise FeatureCacheError("features must be a 2D matrix")
    if protein_ids.shape[0] != features.shape[0]:
        raise FeatureCacheError("protein_ids length must match feature rows")
    _require_unique(protein_ids, "protein id")
    payload = json.dumps(metadata or {}, sort_keys=True)
    np.savez_compressed(path, features=features, protein_ids=protein_ids.astype(str), metadata=np.asarray(payload))
    return path
