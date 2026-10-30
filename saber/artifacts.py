from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .config import SaberConfig
from .data import DataBundle


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(_jsonable(payload), f, indent=2, sort_keys=True)
        f.write("\n")


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r") as f:
        return json.load(f)


def save_run_artifacts(
    run_dir: str | Path,
    cfg: SaberConfig,
    data: DataBundle,
    prediction: np.ndarray,
    cell_embeddings: np.ndarray,
    protein_embeddings: np.ndarray,
    history: list[dict[str, float]],
    command: str,
    model_path: str | Path,
    device: str,
    resources: dict | None = None,
) -> Path:
    run_dir = Path(run_dir)
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    labels_path = artifact_dir / "labels.npz"
    imputation_path = artifact_dir / "imputation.npz"
    embeddings_path = artifact_dir / "embeddings.npz"
    history_path = artifact_dir / "train_history.json"

    np.savez_compressed(
        labels_path,
        batches=np.asarray(data.batches, dtype=np.int64),
        cell_types=np.asarray(data.cell_types, dtype=np.int64),
        protein_ids=np.asarray(data.protein_ids, dtype=str),
        x_raw=np.asarray(data.x_raw, dtype=np.float32),
        scale_mode=np.asarray(str(data.scale_info.get("mode", "none"))),
        scale_center=np.asarray(data.scale_info.get("center", []), dtype=np.float32),
        scale_scale=np.asarray(data.scale_info.get("scale", []), dtype=np.float32),
    )
    np.savez_compressed(
        imputation_path,
        x_true=np.asarray(data.x_true, dtype=np.float32),
        x_true_raw=np.asarray(data.x_raw, dtype=np.float32),
        x_input=np.asarray(data.x_input, dtype=np.float32),
        observed_mask=np.asarray(data.observed_mask, dtype=np.float32),
        injected_missing_mask=np.asarray(data.injected_missing_mask, dtype=np.uint8),
        injected_mnar_mask=(
            np.asarray(data.injected_mnar_mask, dtype=np.uint8)
            if data.injected_mnar_mask is not None
            else np.zeros_like(np.asarray(data.injected_missing_mask), dtype=np.uint8)
        ),
        available_mask=np.asarray(data.available_mask, dtype=np.uint8),
        pred__saber=np.asarray(prediction, dtype=np.float32),
        scale_mode=np.asarray(str(data.scale_info.get("mode", "none"))),
        scale_center=np.asarray(data.scale_info.get("center", []), dtype=np.float32),
        scale_scale=np.asarray(data.scale_info.get("scale", []), dtype=np.float32),
    )
    np.savez_compressed(
        embeddings_path,
        z__saber=np.asarray(cell_embeddings, dtype=np.float32),
        protein_embeddings__saber=np.asarray(protein_embeddings, dtype=np.float32),
    )
    write_json(history_path, {"history": history})

    manifest_path = run_dir / "artifact_manifest.json"
    manifest = {
        "schema_version": "1.0",
        "method_name": "SABER",
        "package_version": __version__,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "device": device,
        "resources": resources or {},
        "config": cfg.to_dict(),
        "data": {
            "n_cells": int(data.x_true.shape[0]),
            "n_proteins": int(data.x_true.shape[1]),
            "n_batches": int(len(np.unique(data.batches))),
            "n_cell_types": int(len(np.unique(data.cell_types))),
            "missing_info": data.missing_info,
            "feature_cache_info": data.feature_cache_info,
            "scale_info": data.scale_info,
        },
        "artifacts": {
            "labels": str(labels_path.resolve()),
            "imputation": str(imputation_path.resolve()),
            "embeddings": str(embeddings_path.resolve()),
            "history": str(history_path.resolve()),
            "model": str(Path(model_path).resolve()),
        },
        "evaluation_status": {
            "imputation": "pending",
        },
    }
    write_json(manifest_path, manifest)
    return manifest_path
