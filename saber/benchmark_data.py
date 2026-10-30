from __future__ import annotations

import hashlib
import os
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from . import paths as _paths
from .artifacts import write_json


# The dataset registry lives in configs/benchmarks/saber_benchmark.yaml; this
# module loads and materializes it. Path placeholders
# ${SABER_*} resolve through saber.paths so the behaviour matches every other
# config in the repository.
DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "configs" / "benchmarks" / "saber_benchmark.yaml"

_ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _resolve_placeholders(value: str) -> str:
    """Expand ${SABER_*} through saber.paths, with clear errors when unset."""
    def _sub(match: re.Match) -> str:
        var = match.group(1)
        if var == "SABER_DATA_ROOT":
            return str(_paths.data_root())
        if var == "SABER_PLM_ROOT":
            return str(_paths.plm_root())
        if var == "SABER_BENCHMARK_ROOT":
            return str(_paths.benchmark_root())
        if var == "SABER_PROTAE_ROOT":
            root = _paths.protae_root()
            if root is None:
                raise RuntimeError(
                    "SABER_PROTAE_ROOT is not configured but the registry references it "
                    "(external-method annotations). Export SABER_PROTAE_ROOT or remove the "
                    "external_annotations block if you do not need Layer-2/3 protein-embedding "
                    "evaluations."
                )
            return str(root)
        # Non-SABER variables (e.g. ${HOME}) go through the shell environment.
        return os.environ.get(var, match.group(0))

    return _ENV_PATTERN.sub(_sub, value)


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    source_path: Path
    target_name: str
    batch_key: str
    cell_type_key: str
    species_hint: str
    scale_mode: str
    task_families: tuple[str, ...]
    note: str = ""


@dataclass(frozen=True)
class AnnotationSpec:
    dataset: str
    source_dir: Path
    target_dir: Path
    filenames: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkRegistry:
    benchmark_root: Path
    data_root: Path
    protae_root: Path | None
    datasets: tuple[DatasetSpec, ...]
    annotations: tuple[AnnotationSpec, ...]


ANNOTATION_FILES = (
    "string_interactions.csv",
    "reactome_pathways.json",
    "go_annotations.json",
    "annotation_summary.json",
)


def _format_path(value: str, values: dict[str, str]) -> Path:
    return Path(str(value).format(**values))


def default_benchmark_registry(benchmark_root: str | Path = "") -> BenchmarkRegistry:
    """Load the repository's default dataset registry YAML."""
    return load_benchmark_registry(DEFAULT_REGISTRY_PATH, benchmark_root)


def load_benchmark_registry(path: str | Path = "", benchmark_root: str | Path = "") -> BenchmarkRegistry:
    """Load a benchmark registry YAML.

    ``path`` may be empty, in which case the repository default
    (``configs/benchmarks/saber_benchmark.yaml``) is used. ``${SABER_*}``
    placeholders in the ``paths:`` block resolve through ``saber.paths`` (i.e.
    environment variables or ``.saber_env``), so an unset variable raises a
    clear error instead of producing a literal ``${SABER_DATA_ROOT}`` path.
    """
    registry_path = Path(path) if path else DEFAULT_REGISTRY_PATH
    with registry_path.open("r") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Benchmark registry must be a mapping: {registry_path}")

    block = raw.get("paths", {}) or {}
    data_root = Path(_resolve_placeholders(str(block.get("data_root", "${SABER_DATA_ROOT}"))))
    if benchmark_root:
        root = Path(str(benchmark_root))
    else:
        root = Path(_resolve_placeholders(str(block.get("benchmark_root", "${SABER_BENCHMARK_ROOT}"))))
    # SABER_PROTAE_ROOT is optional: absent is fine unless the registry's
    # external_annotations block actually references it (checked below).
    protae_root: Path | None = None
    if str(block.get("protae_root", "")).strip():
        try:
            protae_root = Path(_resolve_placeholders(str(block["protae_root"])))
        except RuntimeError:
            protae_root = None
    fmt = {
        "data_root": str(data_root),
        "benchmark_root": str(root),
        "protae_root": str(protae_root) if protae_root is not None else "",
    }

    datasets: list[DatasetSpec] = []
    for row in raw.get("datasets", []) or []:
        datasets.append(
            DatasetSpec(
                dataset=str(row["dataset"]),
                source_path=_format_path(_resolve_placeholders(str(row["source_path"])), fmt),
                target_name=str(row.get("target_name") or f"{row['dataset']}.h5ad"),
                batch_key=str(row.get("batch_key", "batch")),
                cell_type_key=str(row.get("cell_type_key", "cell_type")),
                species_hint=str(row.get("species_hint", "")),
                scale_mode=str(row.get("scale_mode", "none")),
                task_families=tuple(str(x) for x in row.get("task_families", []) or []),
                note=str(row.get("note", "")),
            )
        )
    if not datasets:
        raise ValueError(f"Benchmark registry declares no datasets: {registry_path}")

    annotations: list[AnnotationSpec] = []
    for row in raw.get("external_annotations", []) or []:
        source_dir_template = str(row["source_dir"])
        if "{protae_root}" in source_dir_template and protae_root is None:
            raise RuntimeError(
                "This registry's external_annotations reference the ProtAE annotation "
                "tree but SABER_PROTAE_ROOT is not configured. Export SABER_PROTAE_ROOT "
                "(external-method adapters + annotations, see docs/DATA.md) if you need "
                "the protein-embedding Layer-2/3 evaluations, or remove the "
                "external_annotations block otherwise."
            )
        annotations.append(
            AnnotationSpec(
                dataset=str(row["dataset"]),
                source_dir=_format_path(_resolve_placeholders(source_dir_template), fmt),
                target_dir=_format_path(_resolve_placeholders(str(row["target_dir"])), fmt),
                filenames=tuple(str(x) for x in row.get("filenames", []) or ANNOTATION_FILES),
            )
        )
    return BenchmarkRegistry(
        benchmark_root=root,
        data_root=data_root,
        protae_root=protae_root,
        datasets=tuple(datasets),
        annotations=tuple(annotations),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_registry_rows(
    benchmark_root: str | Path = "",
    registry: BenchmarkRegistry | None = None,
) -> list[dict[str, Any]]:
    registry = registry or default_benchmark_registry(benchmark_root)
    root = Path(registry.benchmark_root)
    rows: list[dict[str, Any]] = []
    for spec in registry.datasets:
        target = root / "anndata" / spec.target_name
        row = asdict(spec)
        row["source_path"] = str(spec.source_path)
        row["target_path"] = str(target)
        row["task_families"] = list(spec.task_families)
        rows.append(row)
    return rows


def prepare_benchmark_data(
    benchmark_root: str | Path = "",
    registry_path: str | Path = "",
    dry_run: bool = False,
    overwrite: bool = False,
    copy_annotations: bool = True,
    checksum: bool = False,
) -> dict[str, Any]:
    registry = load_benchmark_registry(registry_path, benchmark_root)
    root = Path(registry.benchmark_root)
    anndata_dir = root / "anndata"
    actions: list[dict[str, Any]] = []

    if not dry_run:
        anndata_dir.mkdir(parents=True, exist_ok=True)
        (root / "external_annotations").mkdir(parents=True, exist_ok=True)

    for spec in registry.datasets:
        target = anndata_dir / spec.target_name
        action: dict[str, Any] = {
            "kind": "dataset",
            "dataset": spec.dataset,
            "source": str(spec.source_path),
            "target": str(target),
            "exists_source": spec.source_path.exists(),
            "exists_target": target.exists(),
            "copied": False,
            "skipped": False,
            "scale_mode": spec.scale_mode,
        }
        if not spec.source_path.exists():
            action["skipped"] = True
            action["reason"] = "source_missing"
        elif target.exists() and not overwrite:
            action["skipped"] = True
            action["reason"] = "target_exists"
        elif dry_run:
            action["skipped"] = True
            action["reason"] = "dry_run"
        else:
            shutil.copy2(spec.source_path, target)
            action["copied"] = True
            action["exists_target"] = True
        if checksum and spec.source_path.exists() and (target.exists() or dry_run):
            action["source_sha256"] = _sha256(spec.source_path)
            if target.exists():
                action["target_sha256"] = _sha256(target)
        actions.append(action)

    if copy_annotations:
        for spec in registry.annotations:
            if not dry_run:
                spec.target_dir.mkdir(parents=True, exist_ok=True)
            for filename in spec.filenames:
                source = spec.source_dir / filename
                target = spec.target_dir / filename
                action = {
                    "kind": "external_annotation",
                    "dataset": spec.dataset,
                    "source": str(source),
                    "target": str(target),
                    "exists_source": source.exists(),
                    "exists_target": target.exists(),
                    "copied": False,
                    "skipped": False,
                }
                if not source.exists():
                    action["skipped"] = True
                    action["reason"] = "source_missing"
                elif target.exists() and not overwrite:
                    action["skipped"] = True
                    action["reason"] = "target_exists"
                elif dry_run:
                    action["skipped"] = True
                    action["reason"] = "dry_run"
                else:
                    shutil.copy2(source, target)
                    action["copied"] = True
                    action["exists_target"] = True
                if checksum and source.exists() and (target.exists() or dry_run):
                    action["source_sha256"] = _sha256(source)
                    if target.exists():
                        action["target_sha256"] = _sha256(target)
                actions.append(action)

    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_root": str(root),
        "registry_path": str(Path(registry_path).resolve()) if registry_path else "",
        "copy_strategy": "physical_copy",
        "dry_run": bool(dry_run),
        "overwrite": bool(overwrite),
        "datasets": dataset_registry_rows(root, registry=registry),
        "actions": actions,
    }
    if not dry_run:
        write_json(root / "saber_benchmark_data_manifest.json", manifest)
    return manifest
