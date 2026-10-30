from __future__ import annotations

import csv
import glob
from pathlib import Path
from typing import Any, Iterable

from saber.artifacts import load_json


def expand_manifest_inputs(manifests: Iterable[str], manifest_globs: Iterable[str]) -> list[Path]:
    paths: list[Path] = [Path(p) for p in manifests]
    for pattern in manifest_globs:
        paths.extend(Path(p) for p in glob.glob(pattern))
    unique = sorted({str(path.resolve()): path.resolve() for path in paths}.values())
    if not unique:
        raise ValueError("No manifest paths were provided or matched")
    return unique


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def write_rows_csv(path: str | Path, rows: list[dict[str, Any]], preferred: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    if preferred:
        fieldnames = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def manifest_metadata(manifest_path: str | Path, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = load_json(path) if manifest is None else manifest
    config = manifest.get("config", {}) or {}
    project = config.get("project", {}) or {}
    data_cfg = config.get("data", {}) or {}
    missing = config.get("missing", {}) or {}
    expansion = config.get("expansion", {}) or {}
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
    condition = manifest.get("condition", project.get("condition", ""))
    if not condition:
        run_name = str(project.get("name", ""))
        if run_name.endswith("_noplm") or "_noplm_" in run_name:
            condition = "noplm"
        elif run_name.endswith("_now") or "_now_" in run_name:
            condition = "now"
        elif run_name.endswith("_saber") or "_saber_" in run_name:
            condition = "saber"
    return {
        "dataset": dataset,
        "run_name": project.get("name", ""),
        "condition": condition,
        "seed": project.get("seed", manifest.get("seed", "")),
        "scale_mode": data_cfg.get("scale_mode", ""),
        "missing_fraction": missing.get("total_fraction", ""),
        "mnar_fraction": missing.get("mnar_fraction", ""),
        "target_fraction": expansion.get("target_fraction", manifest.get("expansion", {}).get("target_fraction", "")),
        "artifact_path": str(path.resolve()),
    }


def metric_rows(
    task: str,
    manifest_path: str | Path,
    method: str,
    metrics: dict[str, Any],
    manifest: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    meta = manifest_metadata(manifest_path, manifest)
    rows: list[dict[str, Any]] = []
    for metric, value in sorted(metrics.items()):
        rows.append(
            {
                "task": task,
                "dataset": meta["dataset"],
                "method": method,
                "condition": meta["condition"],
                "seed": meta["seed"],
                "scale_mode": meta["scale_mode"],
                "metric": metric,
                "value": value,
                "artifact_path": meta["artifact_path"],
                **{k: v for k, v in meta.items() if k not in {"dataset", "condition", "seed", "scale_mode", "artifact_path"}},
                **(extra or {}),
            }
        )
    return rows
