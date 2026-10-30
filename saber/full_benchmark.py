from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml

from .benchmark_data import DatasetSpec, load_benchmark_registry
from .evaluate.common import write_rows_csv
from .evaluate.embedding import evaluate_embedding_manifest
from .evaluate.expansion import evaluate_expansion_manifest
from .evaluate.imputation import evaluate_imputation_manifest
from .evaluate.lfq import evaluate_lfq_manifest
from .evaluate.protein_embedding import evaluate_protein_embedding_manifest
from .prepare import prepare_aligned_inputs
from .source_data import _protein_ext_dir_for_manifest
from .artifacts import write_json
from .paths import benchmark_root as _benchmark_root_fn
from .paths import esm2_cache_dir as _esm2_cache_dir_fn
from .paths import outputs_root as _outputs_root_fn


# Path defaults resolve lazily (at call time) so that importing this module
# never requires the data roots to be configured.
def _default_plm_features() -> Path:
    return _esm2_cache_dir_fn() / "protae_dataset_embeddings.npy"


def _default_plm_ids() -> Path:
    return _esm2_cache_dir_fn() / "protae_dataset_protein_ids.npy"


def _default_benchmark_root() -> Path:
    return _benchmark_root_fn()
NON_LFQ_DATASETS = (
    "Specht_2019",
    "Derks_2022",
    "Huffman_2023_melanoma_HEK",
    "Woo_2021",
)
PROTEIN_EMBEDDING_DATASETS = ("Specht_2019", "Derks_2022")
MISSING_LEVELS = (0.20, 0.50, 0.80)
EXPANSION_TARGET_FRACTIONS = (0.025, 0.05, 0.075, 0.10)

# Optional robustness arms (off by default; enabled via materialize flags).
# §1 missingness grid: full missing-fraction x MNAR-share sweep (supp robustness figure).
MISSING_GRID_FRACTIONS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
# every 10% so the by-fraction imputation panel can plot the full 10-90% axis
MNAR_LEVELS = (0.00, 0.25, 0.50, 0.75, 1.00)
# §2 sequence-cluster holdout: MMseqs2 identity thresholds x holdout fractions on
# the HUMAN datasets only (mouse Dou/Woo lack a human-proteome cluster TSV).
SEQUENCE_HOLDOUT_DATASETS = ("Derks_2022", "Specht_2019", "Huffman_2023_melanoma_HEK")
SEQUENCE_ID_THRESHOLDS = (0.30, 0.50, 0.70)
SEQUENCE_HOLDOUT_FRACTIONS = (0.05, 0.10, 0.20, 0.40)
def _cluster_dir() -> Path:
    return _outputs_root_fn() / "expansion_cluster_holdout" / "clusters"
SEED = 42
DATASET_PLM_CACHE_NAMES = {
    "Specht_2019": "Specht_2019",
    "Derks_2022": "Derks_2022_3ct",
    "Huffman_2023_melanoma_HEK": "Huffman_mel_HEK_filt",
    "Woo_2021": "Woo_2021_filt",
    "EasySCP_liver": "EasySCP_liver",
    "Brunner2022": "Brunner2022",
    "PXD071205_LFQ": "PXD071205_Full_All",
}


@dataclass(frozen=True)
class MaterializedBenchmark:
    root: Path
    plan_path: Path
    train_commands_path: Path
    external_commands_path: Path
    train_slurm_path: Path
    external_slurm_path: Path
    collect_slurm_path: Path
    submit_script_path: Path
    sequential_train_slurm_path: Path
    sequential_external_slurm_path: Path
    sequential_submit_script_path: Path
    n_train_jobs: int
    n_external_jobs: int


def _today_run_root() -> Path:
    return Path("outputs") / f"paper_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _q(value: str | Path) -> str:
    return shlex.quote(str(value))


def _fmt_fraction(value: float) -> str:
    pct = value * 100.0
    return f"{pct:g}".replace(".", "p")


def _dataset_path(spec: DatasetSpec, benchmark_root: Path) -> Path:
    return benchmark_root / "anndata" / spec.target_name


def _condition_label(condition: str) -> str:
    return {"saber": "saber", "no_plm": "noplm", "no_w": "now"}.get(condition, condition)


def _base_config(
    *,
    run_name: str,
    run_root: Path,
    spec: DatasetSpec,
    data_path: Path,
    plm_features: Path,
    plm_ids: str | Path,
    condition: str,
    missing_enabled: bool,
    missing_fraction: float,
    mnar_fraction: float,
    missing_seed: int,
    epochs: int,
    batch_size: int,
    device: str,
    target_fraction: float = 0.10,
    holdout_mode: str = "random",
    cluster_map_path: str | Path = "",
) -> dict[str, Any]:
    no_plm = condition == "no_plm"
    no_w = condition == "no_w"
    return {
        "project": {
            "name": run_name,
            "outdir": str((run_root / "runs").resolve()),
            "seed": int(missing_seed),
        },
        "data": {
            "synthetic": False,
            "path": str(data_path.resolve()),
            "protein_features_path": "" if no_plm else str(plm_features.resolve()),
            "protein_features_ids_path": "" if no_plm or not str(plm_ids) else str(Path(plm_ids).resolve()),
            "protein_ids_path": "",
            "batch_key": spec.batch_key,
            "cell_type_key": spec.cell_type_key,
            "protein_id_field": "",
            "scale_mode": spec.scale_mode,
            "allow_random_protein_features": bool(no_plm),
        },
        "missing": {
            "enabled": bool(missing_enabled),
            "total_fraction": float(missing_fraction),
            "mnar_fraction": float(mnar_fraction),
            "threshold_quantile": 0.10,
            "threshold_scale": 0.30,
            "seed": int(missing_seed),
        },
        "model": {
            "latent_dim": 16,
            "ae_h1": 128,
            "ae_h2": 64,
            "protein_feature_dim": 1280,
            "protein_proj_dim": 256,
            "protein_proj_hidden_dim": 256,
            "protein_proj_dropout": 0.10,
            "bilinear_with_w": not no_w,
            "trainable_protein_features": False,
        },
        "training": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": 0.0005,
            "weight_decay": 0.01,
            "val_frac": 0.05,
            "device": device,
            "lr_mult_protein_features": 1.0,
            "lr_mult_protein_encoder": 1.0,
        },
        "evaluation": {"imputation": bool(missing_enabled)},
        "expansion": {
            "target_fraction": float(target_fraction),
            "seed": int(missing_seed),
            "min_target_observed": 0,
            "references": ["plm_ridge", "plm_knn"],
            "ridge_lambda": 1.0,
            "knn_ks": [5, 10],
            "holdout_mode": str(holdout_mode),
            "cluster_map_path": str(cluster_map_path) if cluster_map_path else "",
        },
    }


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _run_command(conda_env: str, subcommand: str, config_path: Path) -> str:
    return f"conda run -n {_q(conda_env)} saber {subcommand} --config {_q(config_path.resolve())}"


def _external_command(
    *,
    conda_env: str,
    manifest_path: Path,
    method: str,
    task: str,
    workdir: Path,
    out_json: Path,
    timeout_sec: int,
    seed: int,
) -> str:
    return (
        f"conda run -n {_q(conda_env)} saber run-external-manifest "
        f"--manifest {_q(manifest_path.resolve())} "
        f"--method {_q(method)} "
        f"--task {_q(task)} "
        f"--workdir {_q(workdir.resolve())} "
        f"--seed {int(seed)} "
        f"--timeout-sec {int(timeout_sec)} "
        f"--out {_q(out_json.resolve())}"
    )


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _slurm_array_script(
    *,
    path: Path,
    job_name: str,
    commands_path: Path,
    logs_dir: Path,
    partition: str,
    time_limit: str,
    mem: str,
    cpus: int,
    n_jobs: int,
    max_parallel: int,
    gres: str,
    repo_root: Path,
) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    array_spec = f"1-{max(1, int(n_jobs))}%{max(1, int(max_parallel))}"
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --cpus-per-task={int(cpus)}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --array={array_spec}",
        f"#SBATCH --output={logs_dir.resolve()}/{job_name}_%A_%a.out",
        f"#SBATCH --error={logs_dir.resolve()}/{job_name}_%A_%a.err",
    ]
    if gres:
        lines.append(f"#SBATCH --gres={gres}")
    lines.extend(
        [
            "set -euo pipefail",
            f"CMD_FILE={_q(commands_path.resolve())}",
            f"cd {_q(repo_root.resolve())}",
            'CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CMD_FILE")',
            'if [[ -z "${CMD}" ]]; then',
            '  echo "No command for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2',
            "  exit 2",
            "fi",
            'echo "[saber-full-benchmark] ${CMD}"',
            'eval "${CMD}"',
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def _collect_script(path: Path, *, conda_env: str, plan_path: Path, root: Path, repo_root: Path) -> None:
    source_dir = root / "source_data"
    figures_dir = root / "paper_figures"
    lines = [
        "#!/usr/bin/env bash",
        "#SBATCH --job-name=saber_collect",
        "#SBATCH --partition=stat",
        "#SBATCH --cpus-per-task=4",
        "#SBATCH --mem=32G",
        "#SBATCH --time=12:00:00",
        f"#SBATCH --output={(root / 'logs' / 'slurm').resolve()}/saber_collect_%j.out",
        f"#SBATCH --error={(root / 'logs' / 'slurm').resolve()}/saber_collect_%j.err",
        "set -euo pipefail",
        f"cd {_q(repo_root.resolve())}",
        (
            f"conda run -n {_q(conda_env)} saber collect-full-benchmark-source-data "
            f"--plan {_q(plan_path.resolve())} "
            f"--outdir {_q(source_dir.resolve())}"
        ),
        (
            f"conda run -n {_q(conda_env)} saber make-paper-figures "
            f"--source-data {_q((source_dir / 'source_metrics_long.csv').resolve())} "
            f"--outdir {_q(figures_dir.resolve())} "
            "--formats png,pdf"
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def _submit_script(path: Path, *, train_slurm: Path, external_slurm: Path, collect_slurm: Path, root: Path) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"TRAIN_JOB=$(sbatch --parsable {_q(train_slurm.resolve())})",
        f"echo \"${{TRAIN_JOB}}\" > {_q((root / 'train_jobid.txt').resolve())}",
        'echo "Submitted train array: ${TRAIN_JOB}"',
        f"EXTERNAL_JOB=$(sbatch --parsable --dependency=afterok:${{TRAIN_JOB}} {_q(external_slurm.resolve())})",
        f"echo \"${{EXTERNAL_JOB}}\" > {_q((root / 'external_jobid.txt').resolve())}",
        'echo "Submitted external array: ${EXTERNAL_JOB}"',
        f"COLLECT_JOB=$(sbatch --parsable --dependency=afterany:${{EXTERNAL_JOB}} {_q(collect_slurm.resolve())})",
        f"echo \"${{COLLECT_JOB}}\" > {_q((root / 'collect_jobid.txt').resolve())}",
        'echo "Submitted collect/plot job: ${COLLECT_JOB}"',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def _slurm_sequential_script(
    *,
    path: Path,
    job_name: str,
    commands_path: Path,
    logs_dir: Path,
    partition: str,
    time_limit: str,
    mem: str,
    cpus: int,
    gres: str,
    repo_root: Path,
) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --cpus-per-task={int(cpus)}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --output={logs_dir.resolve()}/{job_name}_%j.out",
        f"#SBATCH --error={logs_dir.resolve()}/{job_name}_%j.err",
    ]
    if gres:
        lines.append(f"#SBATCH --gres={gres}")
    lines.extend(
        [
            "set -euo pipefail",
            f"CMD_FILE={_q(commands_path.resolve())}",
            f"cd {_q(repo_root.resolve())}",
            'i=0',
            'while IFS= read -r CMD; do',
            '  i=$((i + 1))',
            '  [[ -z "${CMD}" ]] && continue',
            '  echo "[saber-full-benchmark] job ${i}: ${CMD}"',
            '  eval "${CMD}"',
            'done < "${CMD_FILE}"',
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def _submit_sequential_script(path: Path, *, train_slurm: Path, external_slurm: Path, collect_slurm: Path, root: Path) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"TRAIN_JOB=$(sbatch --parsable {_q(train_slurm.resolve())})",
        f"echo \"${{TRAIN_JOB}}\" > {_q((root / 'train_sequential_jobid.txt').resolve())}",
        'echo "Submitted sequential train job: ${TRAIN_JOB}"',
        f"EXTERNAL_JOB=$(sbatch --parsable --dependency=afterok:${{TRAIN_JOB}} {_q(external_slurm.resolve())})",
        f"echo \"${{EXTERNAL_JOB}}\" > {_q((root / 'external_sequential_jobid.txt').resolve())}",
        'echo "Submitted sequential external job: ${EXTERNAL_JOB}"',
        f"COLLECT_JOB=$(sbatch --parsable --dependency=afterany:${{EXTERNAL_JOB}} {_q(collect_slurm.resolve())})",
        f"echo \"${{COLLECT_JOB}}\" > {_q((root / 'collect_sequential_jobid.txt').resolve())}",
        'echo "Submitted collect/plot job: ${COLLECT_JOB}"',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def materialize_full_benchmark(
    root: str | Path = "",
    registry_path: str | Path = "",
    benchmark_root: str | Path = "",
    plm_features: str | Path = "",
    plm_ids: str | Path = "",
    plm_cache_dir: str | Path = "",
    conda_env: str = "saber",
    epochs: int = 300,
    batch_size: int = 32,
    device: str = "auto",
    seed: int = SEED,
    train_partition: str = "stat",
    external_partition: str = "stat",
    train_max_parallel: int = 4,
    external_max_parallel: int = 6,
    dry_run: bool = False,
    missingness_grid: bool = False,
    sequence_holdout: bool = False,
) -> MaterializedBenchmark:
    repo_root = Path(__file__).resolve().parents[1]
    root_path = (Path(root) if root else _today_run_root()).resolve()
    benchmark_root_path = Path(benchmark_root or _default_benchmark_root()).resolve()
    registry = load_benchmark_registry(registry_path, benchmark_root_path)
    specs = {spec.dataset: spec for spec in registry.datasets}
    plm_features_path = Path(plm_features or _default_plm_features()).resolve()
    plm_ids_path = Path(plm_ids or _default_plm_ids()).resolve()
    plm_cache_dir_path = Path(plm_cache_dir or _esm2_cache_dir_fn()).resolve()

    configs_dir = root_path / "configs"
    commands_dir = root_path / "commands"
    slurm_dir = root_path / "slurm"
    logs_dir = root_path / "logs" / "slurm"
    external_root = root_path / "external_work"
    prepared_root = root_path / "prepared_inputs"
    train_jobs: list[dict[str, Any]] = []
    external_jobs: list[dict[str, Any]] = []
    prepared_inputs: dict[str, dict[str, Any]] = {}
    train_commands: list[str] = []
    external_commands: list[str] = []
    task_manifests: dict[str, list[str]] = {
        "imputation": [],
        "cell_embedding": [],
        "protein_embedding": [],
        "proteome_expansion": [],
        "lfq": [],
    }

    root_path.mkdir(parents=True, exist_ok=True)

    def prepared_for_dataset(dataset: str) -> tuple[Path, Path]:
        if dataset in prepared_inputs:
            row = prepared_inputs[dataset]
            return Path(row["h5ad"]), Path(row["feature_cache"])
        spec = specs[dataset]
        data_path = _dataset_path(spec, benchmark_root_path)
        cache_name = DATASET_PLM_CACHE_NAMES.get(dataset, dataset)
        cache_path = plm_cache_dir_path / f"{cache_name}_embeddings.npy"
        cache_ids_path = plm_cache_dir_path / f"{cache_name}_protein_ids.npy"
        if not cache_path.exists() or not cache_ids_path.exists():
            cache_path = plm_features_path
            cache_ids_path = plm_ids_path
        prep = prepare_aligned_inputs(
            data_path=data_path,
            cache_path=cache_path,
            outdir=prepared_root / dataset,
            name=dataset,
            cache_protein_ids_path=cache_ids_path,
            protein_id_field="",
            expected_dim=1280,
            duplicate_cache_policy="first",
            duplicate_dataset_policy="first",
            drop_missing_proteins=True,
        )
        prepared_inputs[dataset] = {
            "source_h5ad": str(data_path.resolve()),
            "source_feature_cache": str(cache_path.resolve()),
            "source_feature_cache_ids": str(cache_ids_path.resolve()),
            "plm_cache_name": cache_name,
            "h5ad": str(prep.h5ad_path.resolve()),
            "feature_cache": str(prep.feature_cache_path.resolve()),
            "report": str(prep.report_path.resolve()),
            "n_cells": prep.report["n_cells"],
            "n_requested_proteins": prep.report["n_requested_proteins"],
            "n_kept_proteins": prep.report["n_kept_proteins"],
            "n_dropped_missing_requested_proteins": prep.report["n_dropped_missing_requested_proteins"],
            "n_dropped_duplicate_dataset_proteins": prep.report["n_dropped_duplicate_dataset_proteins"],
        }
        return prep.h5ad_path, prep.feature_cache_path

    def add_train_job(kind: str, dataset: str, condition: str, cfg: dict[str, Any], subcommand: str, cfg_name: str) -> Path:
        cfg_path = configs_dir / f"{cfg_name}.yaml"
        manifest_path = root_path / "runs" / cfg["project"]["name"] / "artifact_manifest.json"
        if not dry_run:
            _write_yaml(cfg_path, cfg)
        command = _run_command(conda_env, subcommand, cfg_path)
        train_commands.append(command)
        train_jobs.append(
            {
                "kind": kind,
                "dataset": dataset,
                "condition": condition,
                "config": str(cfg_path.resolve()),
                "manifest": str(manifest_path.resolve()),
                "command": command,
            }
        )
        return manifest_path

    miss50_main: dict[str, Path] = {}
    miss50_noplm: dict[str, Path] = {}
    for dataset in NON_LFQ_DATASETS:
        spec = specs[dataset]
        data_path, feature_cache_path = prepared_for_dataset(dataset)
        grid_fractions = MISSING_GRID_FRACTIONS if missingness_grid else MISSING_LEVELS
        grid_mnars = MNAR_LEVELS if missingness_grid else (0.50,)
        for missing_fraction in grid_fractions:
            for mnar_fraction in grid_mnars:
                missing_tag = f"miss{_fmt_fraction(missing_fraction)}_mnar{_fmt_fraction(mnar_fraction)}"
                # Grid (robustness heatmap) needs only saber + no_plm; the main sweep
                # keeps no_w too (figE ablation panel).
                grid_conditions = ("saber", "no_plm") if missingness_grid else ("saber", "no_plm", "no_w")
                for condition in grid_conditions:
                    run_name = f"{dataset}_{missing_tag}_{_condition_label(condition)}"
                    cfg = _base_config(
                        run_name=run_name,
                        run_root=root_path,
                        spec=spec,
                        data_path=data_path,
                        plm_features=feature_cache_path,
                        plm_ids="",
                        condition=condition,
                        missing_enabled=True,
                        missing_fraction=missing_fraction,
                        mnar_fraction=mnar_fraction,
                        missing_seed=seed,
                        epochs=epochs,
                        batch_size=batch_size,
                        device=device,
                    )
                    manifest_path = add_train_job("imputation", dataset, condition, cfg, "train", f"imputation/{run_name}")
                    task_manifests["imputation"].append(str(manifest_path.resolve()))
                    is_canonical = abs(missing_fraction - 0.50) < 1e-9 and abs(mnar_fraction - 0.50) < 1e-9
                    if is_canonical and condition == "saber":
                        miss50_main[dataset] = manifest_path
                        task_manifests["cell_embedding"].append(str(manifest_path.resolve()))
                        if dataset in PROTEIN_EMBEDDING_DATASETS:
                            task_manifests["protein_embedding"].append(str(manifest_path.resolve()))
                    if is_canonical and condition == "no_plm" and dataset in PROTEIN_EMBEDDING_DATASETS:
                        miss50_noplm[dataset] = manifest_path
                        task_manifests["protein_embedding"].append(str(manifest_path.resolve()))
                    # External baselines span the missing-fraction axis at mnar50 (the
                    # "SABER advantage grows with missing" comparison): in the lean main
                    # sweep that's miss{20,50,80}; in the full grid it's all miss fractions
                    # at mnar50. They are NOT run across the MNAR axis (secondary).
                    fire_externals = condition == "saber" and abs(mnar_fraction - 0.50) < 1e-9
                    if fire_externals:
                        for method in ("pimms_dae", "pimms_vae", "omicsgmf", "knn"):
                            workdir = external_root / dataset / missing_tag / method
                            out_json = workdir / "run_external.json"
                            command = _external_command(
                                conda_env=conda_env,
                                manifest_path=manifest_path,
                                method=method,
                                task="imputation",
                                workdir=workdir,
                                out_json=out_json,
                                timeout_sec=7200,
                                seed=seed,
                            )
                            external_commands.append(command)
                            external_jobs.append(
                                {
                                    "task": "imputation",
                                    "dataset": dataset,
                                    "missing_fraction": missing_fraction,
                                    "method": method,
                                    "manifest": str(manifest_path.resolve()),
                                    "command": command,
                                }
                            )

    for dataset in NON_LFQ_DATASETS:
        spec = specs[dataset]
        data_path, feature_cache_path = prepared_for_dataset(dataset)
        for target_fraction in EXPANSION_TARGET_FRACTIONS:
            target_tag = f"holdout{_fmt_fraction(target_fraction)}"
            for condition in ("saber", "no_plm"):
                run_name = f"{dataset}_{target_tag}_{_condition_label(condition)}"
                cfg = _base_config(
                    run_name=run_name,
                    run_root=root_path,
                    spec=spec,
                    data_path=data_path,
                    plm_features=feature_cache_path,
                    plm_ids="",
                    condition=condition,
                    missing_enabled=True,
                    missing_fraction=0.50,
                    mnar_fraction=0.50,
                    missing_seed=seed,
                    epochs=epochs,
                    batch_size=batch_size,
                    device=device,
                    target_fraction=target_fraction,
                )
                manifest_path = add_train_job("proteome_expansion", dataset, condition, cfg, "train-expansion", f"expansion/{run_name}")
                task_manifests["proteome_expansion"].append(str(manifest_path.resolve()))

    # §2 sequence-cluster holdout (optional): leakage-controlled expansion on human
    # datasets. Holds out whole MMseqs2 sequence-identity clusters so close homologs
    # of the target proteins also leave training (defends the zero-shot claim).
    if sequence_holdout:
        for dataset in SEQUENCE_HOLDOUT_DATASETS:
            if dataset not in specs:
                continue
            spec = specs[dataset]
            data_path, feature_cache_path = prepared_for_dataset(dataset)
            for id_thr in SEQUENCE_ID_THRESHOLDS:
                id_tag = f"id{int(round(id_thr * 100)):02d}"
                tsv = _cluster_dir() / f"{dataset}_{id_tag}_cluster.tsv"
                if not tsv.exists():
                    print(f"[full_benchmark] WARNING: missing cluster TSV {tsv}; skipping sequence holdout {dataset} {id_tag}", flush=True)
                    continue
                for target_fraction in SEQUENCE_HOLDOUT_FRACTIONS:
                    for condition in ("saber", "no_plm"):
                        run_name = f"{dataset}_seq_{id_tag}_hold{_fmt_fraction(target_fraction)}_{_condition_label(condition)}"
                        cfg = _base_config(
                            run_name=run_name,
                            run_root=root_path,
                            spec=spec,
                            data_path=data_path,
                            plm_features=feature_cache_path,
                            plm_ids="",
                            condition=condition,
                            missing_enabled=True,
                            missing_fraction=0.50,
                            mnar_fraction=0.50,
                            missing_seed=seed,
                            epochs=epochs,
                            batch_size=batch_size,
                            device=device,
                            target_fraction=target_fraction,
                            holdout_mode="sequence",
                            cluster_map_path=tsv,
                        )
                        manifest_path = add_train_job("proteome_expansion", dataset, condition, cfg, "train-expansion", f"expansion/{run_name}")
                        task_manifests["proteome_expansion"].append(str(manifest_path.resolve()))

    lfq_spec = specs["PXD071205_LFQ"]
    lfq_data_path, lfq_feature_cache_path = prepared_for_dataset("PXD071205_LFQ")
    for condition in ("saber", "no_plm"):
        run_name = f"PXD071205_LFQ_full_{_condition_label(condition)}"
        cfg = _base_config(
            run_name=run_name,
            run_root=root_path,
            spec=lfq_spec,
            data_path=lfq_data_path,
            plm_features=lfq_feature_cache_path,
            plm_ids="",
            condition=condition,
            missing_enabled=False,
            missing_fraction=0.0,
            mnar_fraction=0.50,
            missing_seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
        )
        manifest_path = add_train_job("lfq", "PXD071205_LFQ", condition, cfg, "train", f"lfq/{run_name}")
        task_manifests["lfq"].append(str(manifest_path.resolve()))
        if condition == "saber":
            for method in ("pimms_dae", "pimms_vae", "omicsgmf", "knn"):
                workdir = external_root / "PXD071205_LFQ" / method
                out_json = workdir / "run_external.json"
                command = _external_command(
                    conda_env=conda_env,
                    manifest_path=manifest_path,
                    method=method,
                    task="lfq",
                    workdir=workdir,
                    out_json=out_json,
                    timeout_sec=7200,
                    seed=seed,
                )
                external_commands.append(command)
                external_jobs.append(
                    {
                        "task": "lfq",
                        "dataset": "PXD071205_LFQ",
                        "method": method,
                        "manifest": str(manifest_path.resolve()),
                        "command": command,
                    }
                )

    for dataset, manifest_path in sorted(miss50_main.items()):
        for method in ("scprotein", "omicsgmf"):
            workdir = external_root / dataset / "cell_embedding" / method
            out_json = workdir / "run_external.json"
            command = _external_command(
                conda_env=conda_env,
                manifest_path=manifest_path,
                method=method,
                task="cell_embedding",
                workdir=workdir,
                out_json=out_json,
                timeout_sec=7200,
                seed=seed,
            )
            external_commands.append(command)
            external_jobs.append(
                {
                    "task": "cell_embedding",
                    "dataset": dataset,
                    "method": method,
                    "manifest": str(manifest_path.resolve()),
                    "command": command,
                }
            )

    for dataset in PROTEIN_EMBEDDING_DATASETS:
        manifest_path = miss50_main[dataset]
        workdir = external_root / dataset / "protein_embedding" / "omicsgmf"
        out_json = workdir / "run_external.json"
        command = _external_command(
            conda_env=conda_env,
            manifest_path=manifest_path,
            method="omicsgmf",
            task="protein_embedding",
            workdir=workdir,
            out_json=out_json,
            timeout_sec=7200,
            seed=seed,
        )
        external_commands.append(command)
        external_jobs.append(
            {
                "task": "protein_embedding",
                "dataset": dataset,
                "method": "omicsgmf",
                "manifest": str(manifest_path.resolve()),
                "command": command,
            }
        )

    train_commands_path = commands_dir / "train_commands.txt"
    external_commands_path = commands_dir / "external_commands.txt"
    if not dry_run:
        _write_lines(train_commands_path, train_commands)
        _write_lines(external_commands_path, external_commands)

    train_slurm_path = slurm_dir / "train_array.sbatch"
    external_slurm_path = slurm_dir / "external_array.sbatch"
    collect_slurm_path = slurm_dir / "collect_and_plot.sbatch"
    submit_script_path = root_path / "submit_full_benchmark.sh"
    sequential_train_slurm = slurm_dir / "train_sequential.sbatch"
    sequential_external_slurm = slurm_dir / "external_sequential.sbatch"
    sequential_submit_script = root_path / "submit_full_benchmark_sequential.sh"
    if not dry_run:
        _slurm_array_script(
            path=train_slurm_path,
            job_name="saber_train",
            commands_path=train_commands_path,
            logs_dir=logs_dir,
            partition=train_partition,
            time_limit="3-00:00:00",
            mem="48G",
            cpus=4,
            n_jobs=len(train_commands),
            max_parallel=train_max_parallel,
            gres="gpu:1",
            repo_root=repo_root,
        )
        _slurm_array_script(
            path=external_slurm_path,
            job_name="saber_external",
            commands_path=external_commands_path,
            logs_dir=logs_dir,
            partition=external_partition,
            time_limit="1-00:00:00",
            mem="48G",
            cpus=4,
            n_jobs=len(external_commands),
            max_parallel=external_max_parallel,
            gres="",
            repo_root=repo_root,
        )
        _collect_script(collect_slurm_path, conda_env=conda_env, plan_path=root_path / "full_benchmark_plan.json", root=root_path, repo_root=repo_root)
        _submit_script(submit_script_path, train_slurm=train_slurm_path, external_slurm=external_slurm_path, collect_slurm=collect_slurm_path, root=root_path)
        _slurm_sequential_script(
            path=sequential_train_slurm,
            job_name="saber_train_seq",
            commands_path=train_commands_path,
            logs_dir=logs_dir,
            partition=train_partition,
            time_limit="7-00:00:00",
            mem="48G",
            cpus=4,
            gres="gpu:1",
            repo_root=repo_root,
        )
        _slurm_sequential_script(
            path=sequential_external_slurm,
            job_name="saber_ext_seq",
            commands_path=external_commands_path,
            logs_dir=logs_dir,
            partition=external_partition,
            time_limit="7-00:00:00",
            mem="48G",
            cpus=4,
            gres="",
            repo_root=repo_root,
        )
        _submit_sequential_script(
            sequential_submit_script,
            train_slurm=sequential_train_slurm,
            external_slurm=sequential_external_slurm,
            collect_slurm=collect_slurm_path,
            root=root_path,
        )

    plan = {
        "schema_version": "1.0",
        "root": str(root_path),
        "benchmark_root": str(benchmark_root_path),
        "registry_path": str(Path(registry_path).resolve()) if registry_path else "",
        "plm_features": str(plm_features_path),
        "plm_ids": str(plm_ids_path),
        "plm_cache_dir": str(plm_cache_dir_path),
        "prepared_inputs": prepared_inputs,
        "conda_env": conda_env,
        "seed": int(seed),
        "missing_levels": list(MISSING_LEVELS),
        "mnar_fraction": 0.50,
        "expansion_target_fractions": list(EXPANSION_TARGET_FRACTIONS),
        "train_jobs": train_jobs,
        "external_jobs": external_jobs,
        "task_manifests": task_manifests,
        "paths": {
            "train_commands": str(train_commands_path.resolve()),
            "external_commands": str(external_commands_path.resolve()),
            "train_slurm": str(train_slurm_path.resolve()),
            "external_slurm": str(external_slurm_path.resolve()),
            "collect_slurm": str(collect_slurm_path.resolve()),
            "submit_script": str(submit_script_path.resolve()),
            "sequential_train_slurm": str(sequential_train_slurm.resolve()),
            "sequential_external_slurm": str(sequential_external_slurm.resolve()),
            "sequential_submit_script": str(sequential_submit_script.resolve()),
            "lfq_data_path": str(lfq_data_path.resolve()),
            "protein_ext_root": str((benchmark_root_path / "external_annotations").resolve()),
        },
        "counts": {
            "train_jobs": len(train_jobs),
            "external_jobs": len(external_jobs),
            "task_manifests": {key: len(value) for key, value in task_manifests.items()},
        },
    }
    plan_path = root_path / "full_benchmark_plan.json"
    if not dry_run:
        write_json(plan_path, plan)
    else:
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")

    return MaterializedBenchmark(
        root=root_path,
        plan_path=plan_path,
        train_commands_path=train_commands_path,
        external_commands_path=external_commands_path,
        train_slurm_path=train_slurm_path,
        external_slurm_path=external_slurm_path,
        collect_slurm_path=collect_slurm_path,
        submit_script_path=submit_script_path,
        sequential_train_slurm_path=sequential_train_slurm,
        sequential_external_slurm_path=sequential_external_slurm,
        sequential_submit_script_path=sequential_submit_script,
        n_train_jobs=len(train_jobs),
        n_external_jobs=len(external_jobs),
    )


def _read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    import csv

    with Path(path).open("r", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def collect_full_benchmark_source_data(plan_path: str | Path, outdir: str | Path) -> dict[str, Any]:
    plan_path = Path(plan_path)
    with plan_path.open("r") as f:
        plan = json.load(f)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    task_manifests = plan.get("task_manifests", {}) or {}
    rows: list[dict[str, Any]] = []
    outputs: dict[str, list[str]] = {}
    lfq_data_path = plan.get("paths", {}).get("lfq_data_path", "")
    protein_ext_root = plan.get("paths", {}).get("protein_ext_root", "")

    def stem(path: str | Path) -> str:
        return Path(path).parent.name or Path(path).stem

    for manifest_path in task_manifests.get("imputation", []):
        result = evaluate_imputation_manifest(manifest_path, outdir / stem(manifest_path) / "imputation")
        rows.extend(result["rows"])
        outputs.setdefault("imputation", []).append(result["source_csv"])
        if result.get("per_protein_csv"):
            outputs.setdefault("imputation_per_protein", []).append(result["per_protein_csv"])

    for manifest_path in task_manifests.get("cell_embedding", []):
        result = evaluate_embedding_manifest(manifest_path, outdir / stem(manifest_path) / "cell_embedding")
        rows.extend(result["rows"])
        outputs.setdefault("cell_embedding", []).append(result["source_csv"])

    for manifest_path in task_manifests.get("proteome_expansion", []):
        result = evaluate_expansion_manifest(manifest_path, outdir / stem(manifest_path) / "proteome_expansion")
        eval_rows = _read_csv_rows(result["source_csv"])
        # Tag rows with the holdout mode (random vs sequence-cluster) so figures can
        # separate leakage-controlled expansion from the standard random holdout.
        try:
            with Path(manifest_path).open() as _f:
                _m = json.load(_f)
            _hm = _m.get("config", {}).get("expansion", {}).get("holdout_mode", "random")
        except Exception:
            _hm = "random"
        for _r in eval_rows:
            _r.setdefault("holdout_mode", _hm)
        rows.extend(eval_rows)
        outputs.setdefault("proteome_expansion", []).append(result["source_csv"])
        if result.get("per_protein_csv"):
            outputs.setdefault("proteome_expansion_per_protein", []).append(result["per_protein_csv"])

    for manifest_path in task_manifests.get("protein_embedding", []):
        ext_dir = _protein_ext_dir_for_manifest(manifest_path, protein_ext_root)
        result = evaluate_protein_embedding_manifest(manifest_path, outdir / stem(manifest_path) / "protein_embedding", ext_dir=ext_dir)
        rows.extend(result["rows"])
        outputs.setdefault("protein_embedding", []).append(result["source_csv"])

    for manifest_path in task_manifests.get("lfq", []):
        result = evaluate_lfq_manifest(manifest_path, outdir / stem(manifest_path) / "lfq", data_path=lfq_data_path)
        rows.extend(_read_csv_rows(result["source_csv"]))
        outputs.setdefault("lfq", []).append(result["source_csv"])

    source_csv = outdir / "source_metrics_long.csv"
    write_rows_csv(
        source_csv,
        rows,
        preferred=["task", "dataset", "method", "condition", "seed", "scale_mode", "metric", "value", "artifact_path"],
    )
    manifest = {"plan": str(plan_path.resolve()), "source_csv": str(source_csv.resolve()), "task_outputs": outputs, "n_rows": len(rows)}
    write_json(outdir / "source_data_manifest.json", manifest)
    return {**manifest, "rows": rows}


def collect_full_benchmark_multiseed_source_data(
    plan_paths: "Iterable[str | Path]",
    outdir: str | Path,
) -> dict[str, Any]:
    """Evaluate every per-seed plan and merge into one long-form source CSV.

    Each plan is collected into its own ``seedN`` sub-directory (so per-seed
    artifacts stay inspectable), then all rows are concatenated into a single
    ``source_metrics_long.csv`` at ``outdir`` with the ``seed`` column
    distinguishing runs. Use this as the input to ``make-paper-figures``.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    per_seed: list[dict[str, Any]] = []
    resolved_plans: list[str] = []
    for plan_path in plan_paths:
        plan_path = Path(plan_path)
        resolved_plans.append(str(plan_path.resolve()))
        with plan_path.open("r") as f:
            plan = json.load(f)
        seed = plan.get("seed", "")
        seed_tag = f"seed{seed}" if seed != "" else plan_path.parent.name
        result = collect_full_benchmark_source_data(plan_path, outdir / seed_tag)
        all_rows.extend(result["rows"])
        per_seed.append(
            {
                "plan": str(plan_path.resolve()),
                "seed": seed,
                "n_rows": len(result["rows"]),
                "source_csv": result["source_csv"],
            }
        )

    source_csv = outdir / "source_metrics_long.csv"
    write_rows_csv(
        source_csv,
        all_rows,
        preferred=["task", "dataset", "method", "condition", "seed", "scale_mode", "metric", "value", "artifact_path"],
    )
    manifest = {
        "plans": resolved_plans,
        "source_csv": str(source_csv.resolve()),
        "n_rows": len(all_rows),
        "per_seed": per_seed,
    }
    write_json(outdir / "source_data_manifest.json", manifest)
    return {**manifest, "rows": all_rows}
