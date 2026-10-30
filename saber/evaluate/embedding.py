from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from saber.artifacts import load_json, write_json
from saber.metrics import compute_embedding_metrics

from .common import metric_rows, write_rows_csv


def evaluate_embedding_manifest(manifest_path: str | Path, outdir: str | Path, seed: int = 42) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = load_json(manifest_path)
    artifacts = manifest.get("artifacts", {})
    labels_path = artifacts.get("labels")
    embeddings_path = artifacts.get("embeddings")
    if not labels_path:
        raise ValueError(f"Manifest has no artifacts.labels: {manifest_path}")
    if not embeddings_path:
        raise ValueError(f"Manifest has no artifacts.embeddings: {manifest_path}")

    with np.load(labels_path) as labels, np.load(embeddings_path) as embeddings:
        batches = labels["batches"]
        cell_types = labels["cell_types"]
        results: dict[str, Any] = {}
        rows: list[dict[str, Any]] = []
        for key in embeddings.files:
            if not key.startswith("z__"):
                continue
            method = key.removeprefix("z__")
            metrics = compute_embedding_metrics(embeddings[key], cell_types, batches, seed=seed)
            results[method] = metrics
            rows.extend(metric_rows("cell_embedding", manifest_path, method, metrics, manifest=manifest))

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "embedding_metrics.json"
    csv_path = outdir / "embedding_metrics.csv"
    source_path = outdir / "source_embedding_metrics.csv"
    write_json(
        json_path,
        {
            "manifest": str(manifest_path.resolve()),
            "metric_contract": {
                "cell_metrics": "ARI/NMI/purity use KMeans clusters against cell-type labels; ASW uses cell-type labels directly.",
                "batch_metrics": "ARI/NMI/purity use KMeans clusters against batch labels; ASW/kBET/iLISI summarize batch structure.",
                "interpretation": "Cell embedding is supportive representation evidence, not a dedicated batch-correction superiority claim.",
            },
            "results": results,
        },
    )
    write_rows_csv(csv_path, [{"method": method, **metrics} for method, metrics in sorted(results.items())], preferred=["method"])
    write_rows_csv(
        source_path,
        rows,
        preferred=["task", "dataset", "method", "condition", "seed", "scale_mode", "metric", "value", "artifact_path"],
    )
    return {
        "json": str(json_path.resolve()),
        "csv": str(csv_path.resolve()),
        "source_csv": str(source_path.resolve()),
        "results": results,
        "rows": rows,
    }

