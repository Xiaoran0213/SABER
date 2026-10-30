from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import load_json, write_json
from .paths import ext_methods_root as _ext_methods_root
from .paths import external_repos_root as _external_repos_root


def _required_ext_methods_root() -> Path:
    """Adapters ship in the repository (see external_adapters/README.md)."""
    root = _ext_methods_root()
    if root is None:
        raise RuntimeError(
            "External-method adapters not found. They ship with the repository "
            "under external_adapters/ (run from a git checkout, or export "
            "SABER_EXT_METHODS_ROOT); external baselines are optional for "
            "training SABER."
        )
    return root


def _required_repos_root() -> Path:
    """Upstream baseline clones (see external_adapters/README.md)."""
    root = _external_repos_root()
    if root is None:
        raise RuntimeError(
            "Upstream external-method repositories are not configured. Clone the "
            "baseline repositories (external_adapters/README.md) and export "
            "SABER_EXTERNAL_REPOS pointing at the directory containing them."
        )
    return root


@dataclass(frozen=True)
class ExternalMethodSpec:
    name: str
    adapter_name: str
    capabilities: tuple[str, ...]
    model: str = ""
    description: str = ""


METHODS: dict[str, ExternalMethodSpec] = {
    "pimms_dae": ExternalMethodSpec(
        name="pimms_dae",
        adapter_name="pimms",
        capabilities=("imputation", "lfq"),
        model="DAE",
        description="PIMMS denoising autoencoder; imputation only.",
    ),
    "pimms_vae": ExternalMethodSpec(
        name="pimms_vae",
        adapter_name="pimms",
        capabilities=("imputation", "lfq"),
        model="VAE",
        description="PIMMS variational autoencoder; imputation only.",
    ),
    "omicsgmf": ExternalMethodSpec(
        name="omicsgmf",
        adapter_name="omicsgmf",
        capabilities=("imputation", "cell_embedding", "protein_embedding", "lfq"),
        description="omicsGMF adapter; can emit imputation and embedding sidecars when supported by the adapter.",
    ),
    "knn": ExternalMethodSpec(
        name="knn",
        adapter_name="knn",
        capabilities=("imputation", "lfq"),
        description="Classic protein-wise k-nearest-neighbour imputation (k=5, Pearson similarity); imputation + LFQ.",
    ),
    "scprotein": ExternalMethodSpec(
        name="scprotein",
        adapter_name="scprotein",
        capabilities=("cell_embedding",),
        description="scProtein cell embedding adapter; not used for imputation.",
    ),
    "cell_mean": ExternalMethodSpec(
        name="cell_mean",
        adapter_name="",
        capabilities=("imputation", "lfq"),
        description="Internal cell mean imputation baseline.",
    ),
    "protein_mean": ExternalMethodSpec(
        name="protein_mean",
        adapter_name="",
        capabilities=("imputation", "lfq"),
        description="Internal protein mean imputation baseline.",
    ),
}


def method_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "adapter_name": spec.adapter_name,
            "capabilities": list(spec.capabilities),
            "model": spec.model,
            "description": spec.description,
        }
        for spec in METHODS.values()
    ]


def methods_for_task(task: str) -> list[str]:
    return sorted(name for name, spec in METHODS.items() if task in spec.capabilities)


def require_capability(method: str, task: str) -> ExternalMethodSpec:
    key = method.lower()
    if key not in METHODS:
        raise ValueError(f"Unknown external method: {method}")
    spec = METHODS[key]
    if task not in spec.capabilities:
        supported = ", ".join(spec.capabilities)
        raise ValueError(f"External method {method} does not support task {task}; supported: {supported}")
    return spec


def resolve_adapter_dir(adapter_name: str, ext_methods_root: str | Path = "") -> Path | None:
    if not adapter_name:
        return None
    root = Path(ext_methods_root) if ext_methods_root else _required_ext_methods_root()
    for candidate in (adapter_name, adapter_name.lower(), adapter_name.upper(), adapter_name.capitalize()):
        path = root / candidate
        if path.is_dir():
            return path
    return None


def load_adapter_config(adapter_dir: str | Path) -> dict[str, Any]:
    adapter_dir = Path(adapter_dir)
    for name in ("spec.json", "spec.yaml"):
        path = adapter_dir / name
        if not path.exists():
            continue
        if path.suffix == ".json":
            import json

            with path.open("r") as f:
                return json.load(f)
        import yaml

        with path.open("r") as f:
            return yaml.safe_load(f) or {}
    return {}


def check_external_method_available(
    method: str,
    task: str,
    ext_methods_root: str | Path = "",
    dry_run: bool = False,
) -> tuple[bool, str]:
    try:
        spec = require_capability(method, task)
    except ValueError as exc:
        return False, str(exc)
    if not spec.adapter_name:
        return True, "internal"
    adapter_dir = resolve_adapter_dir(spec.adapter_name, ext_methods_root)
    if adapter_dir is None:
        return False, "adapter_dir_not_found"
    adapter_py = adapter_dir / "adapter.py"
    adapter_r = adapter_dir / "adapter.R"
    if not adapter_py.exists() and not adapter_r.exists():
        return False, "adapter_not_found"
    if dry_run:
        # A dry run only reports the command it would execute; interpreter
        # availability is irrelevant.
        return True, "ok (dry run; interpreter checks skipped)"
    config = load_adapter_config(adapter_dir)
    if config.get("conda_env") and shutil.which("conda") is None and shutil.which("mamba") is None:
        return False, "conda_not_found"
    if adapter_r.exists() and not config.get("conda_env") and shutil.which("Rscript") is None:
        return False, "rscript_not_found"
    return True, "ok"


def build_external_command(
    method: str,
    task: str,
    input_csv: str | Path,
    output_csv: str | Path,
    ext_methods_root: str | Path = "",
    repo_root: str | Path = "",
    seed: int | None = None,
) -> tuple[list[str], Path, dict[str, str]]:
    spec = require_capability(method, task)
    if not spec.adapter_name:
        raise ValueError(f"Method {method} is an internal baseline and has no external adapter command")
    adapter_dir = resolve_adapter_dir(spec.adapter_name, ext_methods_root)
    if adapter_dir is None:
        raise FileNotFoundError(f"Adapter directory not found for {method} under {ext_methods_root}")
    adapter_py = adapter_dir / "adapter.py"
    adapter_r = adapter_dir / "adapter.R"
    config_path = ""
    for name in ("spec.json", "spec.yaml"):
        if (adapter_dir / name).exists():
            config_path = str((adapter_dir / name).resolve())
            break

    config = load_adapter_config(adapter_dir)
    cmd: list[str]
    if adapter_r.exists():
        cmd = ["Rscript", str(adapter_r.resolve())]
    elif adapter_py.exists():
        cmd = ["python", str(adapter_py.resolve())]
    else:
        raise FileNotFoundError(f"No adapter.py or adapter.R in {adapter_dir}")
    cmd.extend(["--in", str(Path(input_csv).resolve()), "--out", str(Path(output_csv).resolve())])
    repo = Path(repo_root) if repo_root else _required_repos_root()
    cmd.extend(["--repo", str(repo.resolve())])
    if config_path:
        cmd.extend(["--config", config_path])
    if spec.model:
        cmd.extend(["--model", spec.model])
    if seed is not None:
        cmd.extend(["--seed", str(int(seed))])

    conda_env = str(config.get("conda_env", "") or "")
    if conda_env:
        conda_exe = shutil.which("conda") or shutil.which("mamba")
        if conda_exe:
            cmd = [conda_exe, "run", "-n", conda_env, *cmd]

    env = os.environ.copy()
    env["EXTERNAL_REPO"] = str(repo.resolve())
    return cmd, adapter_dir, env


def run_external_method(
    method: str,
    task: str,
    input_csv: str | Path,
    output_csv: str | Path,
    logs_dir: str | Path,
    ext_methods_root: str | Path = "",
    repo_root: str | Path = "",
    seed: int | None = None,
    timeout_sec: int = 3600,
    dry_run: bool = False,
) -> dict[str, Any]:
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{method}.log"
    ok, reason = check_external_method_available(method, task, ext_methods_root, dry_run=dry_run)
    if not ok:
        log_path.write_text(f"{reason}\n", encoding="utf-8")
        return {"ok": False, "reason": reason, "log": str(log_path.resolve())}
    cmd, cwd, env = build_external_command(
        method,
        task,
        input_csv=input_csv,
        output_csv=output_csv,
        ext_methods_root=ext_methods_root,
        repo_root=repo_root,
        seed=seed,
    )
    payload = {
        "ok": False,
        "dry_run": bool(dry_run),
        "would_run": bool(dry_run),
        "method": method,
        "task": task,
        "command": " ".join(shlex.quote(part) for part in cmd),
        "cwd": str(cwd.resolve()),
        "log": str(log_path.resolve()),
        "output_csv": str(Path(output_csv).resolve()),
    }
    if dry_run:
        payload["ok"] = True
        write_json(logs_dir / f"{method}_dry_run.json", payload)
        return payload
    try:
        with log_path.open("w") as log_f:
            proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=log_f, stderr=log_f, timeout=int(timeout_sec), check=False)
        payload["returncode"] = int(proc.returncode)
        payload["ok"] = int(proc.returncode) == 0
        if not payload["ok"]:
            payload["reason"] = f"returncode_{proc.returncode}"
        return payload
    except subprocess.TimeoutExpired:
        with log_path.open("a") as f:
            f.write("timeout\n")
        payload["reason"] = "timeout"
        return payload


def _load_npz_dict(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _write_npz_dict(path: str | Path, payload: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **payload)


@contextmanager
def _npz_file_lock(path: str | Path, timeout_sec: int = 1800):
    lock_path = Path(f"{path}.lock")
    start = time.time()
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if time.time() - start > float(timeout_sec):
                raise TimeoutError(f"Timed out waiting for artifact lock: {lock_path}")
            time.sleep(1.0)
    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _read_csv_matrix(path: str | Path) -> np.ndarray:
    matrix = np.loadtxt(path, delimiter=",")
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return np.asarray(matrix, dtype=np.float32)


def _write_csv_matrix(path: str | Path, matrix: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(matrix, dtype=np.float32), delimiter=",")


def _fill_nonfinite_column_mean(matrix: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32).copy()
    finite = np.isfinite(arr)
    counts = finite.sum(axis=0)
    sums = np.where(finite, arr, 0.0).sum(axis=0, dtype=np.float64)
    means = np.full((arr.shape[1],), float(fallback), dtype=np.float32)
    np.divide(sums, counts, out=means, where=counts > 0)
    rows, cols = np.where(~finite)
    arr[rows, cols] = means[cols]
    return arr.astype(np.float32)


def _adapter_input_columns(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    arr = np.asarray(matrix, dtype=np.float32)
    keep = np.isfinite(arr).any(axis=0)
    if bool(keep.all()):
        return arr, keep, {"n_input_columns": int(arr.shape[1]), "n_dropped_all_nan_columns": 0, "dropped_all_nan_columns": []}
    dropped = np.where(~keep)[0].astype(int).tolist()
    return (
        arr[:, keep],
        keep,
        {
            "n_input_columns": int(arr.shape[1]),
            "n_adapter_columns": int(keep.sum()),
            "n_dropped_all_nan_columns": int(len(dropped)),
            "dropped_all_nan_columns": dropped,
        },
    )


def _expand_reduced_prediction_csv(
    output_csv: str | Path,
    *,
    keep_columns: np.ndarray,
    fallback_matrix: np.ndarray,
    expected_shape: tuple[int, int],
    expanded_csv: str | Path,
) -> tuple[Path, dict[str, Any]]:
    keep = np.asarray(keep_columns).astype(bool)
    if keep.shape[0] != expected_shape[1]:
        raise ValueError(f"keep_columns has length {keep.shape[0]} but expected {expected_shape[1]} proteins")
    if bool(keep.all()):
        return Path(output_csv), {"expanded_output": False}
    fallback = _fill_nonfinite_column_mean(fallback_matrix)
    if fallback.shape != expected_shape:
        raise ValueError(f"fallback_matrix shape mismatch: got {fallback.shape}, expected {expected_shape}")
    full = fallback.astype(np.float32).copy()
    if int(keep.sum()) > 0:
        reduced = _shape_align(_read_csv_matrix(output_csv), (expected_shape[0], int(keep.sum())))
        full[:, keep] = reduced.astype(np.float32)
    expanded_path = Path(expanded_csv)
    _write_csv_matrix(expanded_path, full)
    return expanded_path, {
        "expanded_output": True,
        "expanded_output_csv": str(expanded_path.resolve()),
        "n_dropped_all_nan_columns": int((~keep).sum()),
    }


def _expand_reduced_protein_embedding_csv(
    output_csv: str | Path,
    *,
    keep_columns: np.ndarray,
    n_proteins: int,
    expanded_csv: str | Path,
) -> tuple[Path, dict[str, Any]]:
    keep = np.asarray(keep_columns).astype(bool)
    if keep.shape[0] != int(n_proteins):
        raise ValueError(f"keep_columns has length {keep.shape[0]} but expected {n_proteins} proteins")
    if bool(keep.all()):
        return Path(output_csv), {"expanded_output": False}
    reduced = _read_csv_matrix(output_csv)
    n_keep = int(keep.sum())
    if reduced.shape[0] == n_keep:
        reduced_embedding = reduced.astype(np.float32)
    elif reduced.shape[1] == n_keep:
        reduced_embedding = reduced.T.astype(np.float32)
    else:
        raise ValueError(f"External protein embedding has shape {reduced.shape}, expected {n_keep} kept proteins")
    full = np.zeros((int(n_proteins), int(reduced_embedding.shape[1])), dtype=np.float32)
    full[keep] = reduced_embedding
    expanded_path = Path(expanded_csv)
    _write_csv_matrix(expanded_path, full)
    return expanded_path, {
        "expanded_output": True,
        "expanded_output_csv": str(expanded_path.resolve()),
        "n_dropped_all_nan_columns": int((~keep).sum()),
    }


def _shape_align(matrix: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.shape == shape:
        return arr
    if arr.T.shape == shape:
        return arr.T.astype(np.float32)
    raise ValueError(f"External output shape mismatch: got {arr.shape}, expected {shape}")


def add_external_imputation_to_manifest(
    manifest_path: str | Path,
    method: str,
    output_csv: str | Path,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    imputation_path = manifest.get("artifacts", {}).get("imputation")
    if not imputation_path:
        raise ValueError(f"Manifest has no artifacts.imputation: {manifest_path}")
    with _npz_file_lock(imputation_path):
        payload = _load_npz_dict(imputation_path)
        expected_shape = tuple(payload["x_true"].shape)
        pred = _shape_align(_read_csv_matrix(output_csv), expected_shape)
        payload[f"pred__{method}"] = pred.astype(np.float32)
        _write_npz_dict(imputation_path, payload)
    return {"manifest": str(Path(manifest_path).resolve()), "imputation": imputation_path, "method": method, "key": f"pred__{method}"}


def add_external_lfq_to_manifest(
    manifest_path: str | Path,
    method: str,
    output_csv: str | Path,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    imputation_path = manifest.get("artifacts", {}).get("imputation")
    if not imputation_path:
        raise ValueError(f"Manifest has no artifacts.imputation: {manifest_path}")
    with _npz_file_lock(imputation_path):
        payload = _load_npz_dict(imputation_path)
        shape_key = "x_true_raw" if "x_true_raw" in payload else "x_true"
        expected_shape = tuple(payload[shape_key].shape)
        pred = _shape_align(_read_csv_matrix(output_csv), expected_shape)
        payload[f"pred_raw__{method}"] = pred.astype(np.float32)
        _write_npz_dict(imputation_path, payload)
    return {"manifest": str(Path(manifest_path).resolve()), "imputation": imputation_path, "method": method, "key": f"pred_raw__{method}"}


def add_external_cell_embedding_to_manifest(
    manifest_path: str | Path,
    method: str,
    output_csv: str | Path,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    embeddings_path = manifest.get("artifacts", {}).get("embeddings")
    labels_path = manifest.get("artifacts", {}).get("labels")
    if not embeddings_path:
        raise ValueError(f"Manifest has no artifacts.embeddings: {manifest_path}")
    n_cells = None
    if labels_path:
        with np.load(labels_path, allow_pickle=True) as labels:
            if "batches" in labels.files:
                n_cells = int(labels["batches"].shape[0])
    with _npz_file_lock(embeddings_path):
        payload = _load_npz_dict(embeddings_path)
        embedding = _read_csv_matrix(output_csv)
        if n_cells is not None and embedding.shape[0] != n_cells and embedding.shape[1] == n_cells:
            embedding = embedding.T
        if n_cells is not None and embedding.shape[0] != n_cells:
            raise ValueError(f"External embedding has {embedding.shape[0]} rows but manifest has {n_cells} cells")
        payload[f"z__{method}"] = embedding.astype(np.float32)
        _write_npz_dict(embeddings_path, payload)
    return {"manifest": str(Path(manifest_path).resolve()), "embeddings": embeddings_path, "method": method, "key": f"z__{method}"}


def add_external_protein_embedding_to_manifest(
    manifest_path: str | Path,
    method: str,
    output_csv: str | Path,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    embeddings_path = manifest.get("artifacts", {}).get("embeddings")
    labels_path = manifest.get("artifacts", {}).get("labels")
    if not embeddings_path:
        raise ValueError(f"Manifest has no artifacts.embeddings: {manifest_path}")
    n_proteins = None
    if labels_path:
        with np.load(labels_path, allow_pickle=True) as labels:
            if "protein_ids" in labels.files:
                n_proteins = int(labels["protein_ids"].shape[0])
    with _npz_file_lock(embeddings_path):
        payload = _load_npz_dict(embeddings_path)
        embedding = _read_csv_matrix(output_csv)
        if n_proteins is not None and embedding.shape[0] != n_proteins and embedding.shape[1] == n_proteins:
            embedding = embedding.T
        if n_proteins is not None and embedding.shape[0] != n_proteins:
            raise ValueError(f"External protein embedding has {embedding.shape[0]} rows but manifest has {n_proteins} proteins")
        payload[f"protein_embeddings__{method}"] = embedding.astype(np.float32)
        _write_npz_dict(embeddings_path, payload)
    return {
        "manifest": str(Path(manifest_path).resolve()),
        "embeddings": embeddings_path,
        "method": method,
        "key": f"protein_embeddings__{method}",
    }


def run_external_for_manifest(
    manifest_path: str | Path,
    method: str,
    task: str,
    workdir: str | Path,
    ext_methods_root: str | Path = "",
    repo_root: str | Path = "",
    seed: int | None = None,
    timeout_sec: int = 3600,
    dry_run: bool = False,
    import_csv: str | Path = "",
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if task == "imputation":
        imputation_path = manifest.get("artifacts", {}).get("imputation")
        if not imputation_path:
            raise ValueError(f"Manifest has no artifacts.imputation: {manifest_path}")
        with np.load(imputation_path) as data:
            x_input = np.asarray(data["x_input"], dtype=np.float32)
            if "observed_mask" in data.files:
                adapter_input = x_input.copy()
                adapter_input[~data["observed_mask"].astype(bool)] = np.nan
            else:
                adapter_input = x_input
        fallback_matrix = _fill_nonfinite_column_mean(adapter_input)
        adapter_input, keep_columns, column_filter = _adapter_input_columns(adapter_input)
        input_csv = workdir / "X_external_input.csv"
        output_csv = Path(import_csv) if import_csv else workdir / f"{method}_imputed.csv"
        _write_csv_matrix(input_csv, adapter_input)
        if import_csv:
            imported = add_external_imputation_to_manifest(manifest_path, method, output_csv)
            return {"mode": "import", "task": task, "method": method, **imported}
        run = run_external_method(
            method,
            task,
            input_csv=input_csv,
            output_csv=output_csv,
            logs_dir=workdir / "logs",
            ext_methods_root=ext_methods_root,
            repo_root=repo_root,
            seed=seed,
            timeout_sec=timeout_sec,
            dry_run=dry_run,
        )
        if dry_run or not run.get("ok"):
            return {"mode": "run", "column_filter": column_filter, **run}
        expanded_csv, expansion = _expand_reduced_prediction_csv(
            output_csv,
            keep_columns=keep_columns,
            fallback_matrix=fallback_matrix,
            expected_shape=tuple(fallback_matrix.shape),
            expanded_csv=workdir / f"{method}_imputed.expanded.csv",
        )
        imported = add_external_imputation_to_manifest(manifest_path, method, expanded_csv)
        return {"mode": "run", "column_filter": column_filter, "output_expansion": expansion, **run, "imported": imported}

    if task == "cell_embedding":
        imputation_path = manifest.get("artifacts", {}).get("imputation")
        if not imputation_path:
            raise ValueError(f"Manifest has no artifacts.imputation for embedding input: {manifest_path}")
        with np.load(imputation_path) as data:
            x_input = np.asarray(data["x_input"], dtype=np.float32)
        adapter_input, keep_columns, column_filter = _adapter_input_columns(x_input)
        input_csv = workdir / "X_embedding_input.csv"
        output_csv = Path(import_csv) if import_csv else workdir / f"{method}_embedding.csv"
        _write_csv_matrix(input_csv, adapter_input)
        if import_csv:
            imported = add_external_cell_embedding_to_manifest(manifest_path, method, output_csv)
            return {"mode": "import", "task": task, "method": method, **imported}
        run = run_external_method(
            method,
            task,
            input_csv=input_csv,
            output_csv=output_csv,
            logs_dir=workdir / "logs",
            ext_methods_root=ext_methods_root,
            repo_root=repo_root,
            seed=seed,
            timeout_sec=timeout_sec,
            dry_run=dry_run,
        )
        if dry_run or not run.get("ok"):
            return {"mode": "run", "column_filter": column_filter, **run}
        imported = add_external_cell_embedding_to_manifest(manifest_path, method, output_csv)
        return {"mode": "run", "column_filter": column_filter, **run, "imported": imported}

    if task == "lfq":
        imputation_path = manifest.get("artifacts", {}).get("imputation")
        if not imputation_path:
            raise ValueError(f"Manifest has no artifacts.imputation for LFQ input: {manifest_path}")
        with np.load(imputation_path) as data:
            if "x_true_raw" not in data.files:
                raise ValueError(f"LFQ external methods require raw-scale x_true_raw in artifact: {manifest_path}")
            adapter_input = np.asarray(data["x_true_raw"], dtype=np.float32).copy()
        fallback_matrix = _fill_nonfinite_column_mean(adapter_input)
        adapter_input, keep_columns, column_filter = _adapter_input_columns(adapter_input)
        input_csv = workdir / "X_lfq_raw_input.csv"
        output_csv = Path(import_csv) if import_csv else workdir / f"{method}_lfq_imputed_raw.csv"
        _write_csv_matrix(input_csv, adapter_input)
        if import_csv:
            imported = add_external_lfq_to_manifest(manifest_path, method, output_csv)
            return {"mode": "import", "task": task, "method": method, **imported}
        run = run_external_method(
            method,
            task,
            input_csv=input_csv,
            output_csv=output_csv,
            logs_dir=workdir / "logs",
            ext_methods_root=ext_methods_root,
            repo_root=repo_root,
            seed=seed,
            timeout_sec=timeout_sec,
            dry_run=dry_run,
        )
        if dry_run or not run.get("ok"):
            return {"mode": "run", "column_filter": column_filter, **run}
        expanded_csv, expansion = _expand_reduced_prediction_csv(
            output_csv,
            keep_columns=keep_columns,
            fallback_matrix=fallback_matrix,
            expected_shape=tuple(fallback_matrix.shape),
            expanded_csv=workdir / f"{method}_lfq_imputed_raw.expanded.csv",
        )
        imported = add_external_lfq_to_manifest(manifest_path, method, expanded_csv)
        return {"mode": "run", "column_filter": column_filter, "output_expansion": expansion, **run, "imported": imported}

    if task == "protein_embedding":
        require_capability(method, task)
        imputation_path = manifest.get("artifacts", {}).get("imputation")
        if not imputation_path:
            raise ValueError(f"Manifest has no artifacts.imputation for protein embedding input: {manifest_path}")
        with np.load(imputation_path) as data:
            x_input = np.asarray(data["x_input"], dtype=np.float32)
            if "observed_mask" in data.files:
                adapter_input = x_input.copy()
                adapter_input[~data["observed_mask"].astype(bool)] = np.nan
            else:
                adapter_input = x_input
        adapter_input, keep_columns, column_filter = _adapter_input_columns(adapter_input)
        input_csv = workdir / "X_protein_embedding_input.csv"
        output_csv = Path(import_csv) if import_csv else workdir / f"{method}_protein_embedding.csv"
        _write_csv_matrix(input_csv, adapter_input)
        if import_csv:
            imported = add_external_protein_embedding_to_manifest(manifest_path, method, output_csv)
            return {"mode": "import", "task": task, "method": method, **imported}
        run = run_external_method(
            method,
            "imputation",
            input_csv=input_csv,
            output_csv=output_csv,
            logs_dir=workdir / "logs",
            ext_methods_root=ext_methods_root,
            repo_root=repo_root,
            seed=seed,
            timeout_sec=timeout_sec,
            dry_run=dry_run,
        )
        run["adapter_task"] = run.get("task", "imputation")
        run["task"] = task
        if dry_run or not run.get("ok"):
            return {"mode": "run", "column_filter": column_filter, **run}
        sidecar = Path(str(output_csv) + ".protein.csv")
        if not sidecar.exists():
            return {"mode": "run", **run, "ok": False, "reason": "protein_sidecar_missing", "sidecar": str(sidecar.resolve())}
        expanded_sidecar, expansion = _expand_reduced_protein_embedding_csv(
            sidecar,
            keep_columns=keep_columns,
            n_proteins=int(keep_columns.shape[0]),
            expanded_csv=workdir / f"{method}_protein_embedding.expanded.protein.csv",
        )
        imported = add_external_protein_embedding_to_manifest(manifest_path, method, expanded_sidecar)
        return {
            "mode": "run",
            "column_filter": column_filter,
            "output_expansion": expansion,
            **run,
            "protein_sidecar": str(expanded_sidecar.resolve()),
            "imported": imported,
        }

    raise ValueError("Manifest-aware external runner currently supports imputation, cell_embedding, protein_embedding, and lfq")
