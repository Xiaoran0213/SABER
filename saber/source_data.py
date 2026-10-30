from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import csv

from .artifacts import load_json, write_json
from .evaluate.common import expand_manifest_inputs, write_rows_csv
from .evaluate.embedding import evaluate_embedding_manifest
from .evaluate.expansion import evaluate_expansion_manifest
from .evaluate.imputation import evaluate_imputation_manifest
from .evaluate.lfq import evaluate_lfq_manifest
from .evaluate.protein_embedding import evaluate_protein_embedding_manifest


def _read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _manifest_dataset(manifest_path: str | Path) -> str:
    try:
        manifest = load_json(manifest_path)
    except Exception:
        return ""
    config = manifest.get("config", {}) or {}
    data_cfg = config.get("data", {}) or {}
    project = config.get("project", {}) or {}
    dataset = (
        manifest.get("dataset")
        or data_cfg.get("dataset")
        or data_cfg.get("name")
        or project.get("dataset")
        or ""
    )
    if not dataset:
        data_path = str(data_cfg.get("path", ""))
        dataset = Path(data_path).stem if data_path else ""
    return str(dataset)


def _protein_ext_dir_for_manifest(manifest_path: str | Path, protein_ext_dir: str | Path) -> str:
    if protein_ext_dir:
        base = Path(protein_ext_dir)
    else:
        from .paths import benchmark_root
        base = benchmark_root() / "external_annotations"
    dataset = _manifest_dataset(manifest_path)
    if dataset and (base / dataset).is_dir():
        return str((base / dataset).resolve())
    if base.is_dir() and (base / "string_interactions.csv").exists():
        return str(base.resolve())
    return str(base / dataset) if dataset else str(base)


def collect_evaluation_source_data(
    manifest_paths: Iterable[str | Path],
    outdir: str | Path,
    tasks: Iterable[str] = ("imputation", "cell_embedding", "proteome_expansion", "protein_embedding", "lfq"),
    lfq_data_path: str | Path = "",
    protein_labels_csv: str | Path = "",
    protein_ext_dir: str | Path = "",
) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    task_set = {str(t) for t in tasks}
    rows: list[dict[str, Any]] = []
    outputs: dict[str, list[str]] = {}
    for manifest_path in expand_manifest_inputs([str(p) for p in manifest_paths], []):
        stem = manifest_path.parent.name or manifest_path.stem
        if "imputation" in task_set:
            try:
                result = evaluate_imputation_manifest(manifest_path, outdir / stem / "imputation")
                rows.extend(result["rows"])
                outputs.setdefault("imputation", []).append(result["source_csv"])
                if result.get("per_protein_csv"):
                    outputs.setdefault("imputation_per_protein", []).append(result["per_protein_csv"])
            except ValueError:
                pass
        if "cell_embedding" in task_set:
            try:
                result = evaluate_embedding_manifest(manifest_path, outdir / stem / "cell_embedding")
                rows.extend(result["rows"])
                outputs.setdefault("cell_embedding", []).append(result["source_csv"])
            except ValueError:
                pass
        if "proteome_expansion" in task_set:
            try:
                result = evaluate_expansion_manifest(manifest_path, outdir / stem / "proteome_expansion")
                rows.extend(_read_csv_rows(result["source_csv"]))
                outputs.setdefault("proteome_expansion", []).append(result["source_csv"])
                if result.get("per_protein_csv"):
                    outputs.setdefault("proteome_expansion_per_protein", []).append(result["per_protein_csv"])
            except ValueError:
                pass
        if "lfq" in task_set:
            try:
                result = evaluate_lfq_manifest(manifest_path, outdir / stem / "lfq", data_path=lfq_data_path)
                rows.extend(_read_csv_rows(result["source_csv"]))
                outputs.setdefault("lfq", []).append(result["source_csv"])
            except ValueError:
                pass
        if "protein_embedding" in task_set:
            try:
                result = evaluate_protein_embedding_manifest(
                    manifest_path,
                    outdir / stem / "protein_embedding",
                    labels_csv=protein_labels_csv,
                    ext_dir=_protein_ext_dir_for_manifest(manifest_path, protein_ext_dir),
                )
                rows.extend(result["rows"])
                outputs.setdefault("protein_embedding", []).append(result["source_csv"])
            except ValueError:
                pass

    source_csv = outdir / "source_metrics_long.csv"
    write_rows_csv(
        source_csv,
        rows,
        preferred=["task", "dataset", "method", "condition", "seed", "scale_mode", "metric", "value", "artifact_path"],
    )
    manifest = {"source_csv": str(source_csv.resolve()), "task_outputs": outputs, "n_rows": len(rows)}
    write_json(outdir / "source_data_manifest.json", manifest)
    return {**manifest, "rows": rows}
