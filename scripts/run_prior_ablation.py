#!/usr/bin/env python
"""Materialize + collect + compare the SABER PLM input-prior ablation.

Which PLM is the best protein-encoder input prior for SABER? Compares the
registered PLMs on imputation (miss50) + proteome expansion (holdout5, where
the prior contributes most), 5 seeds, on the benchmark datasets:

    esm2_650m  (1280d, esm2)  -- REFERENCE: read from the finished 5-seed campaign
    esm2_150m  ( 640d, esm2)  -- new runs (size ablation partner to 650M)
    prot_t5_xl (1024d, t5)    -- new runs
    prot_bert  (1024d, bert)  -- new runs
    ankh_large (1536d, ankh)  -- new runs

Self-contained and additive: it imports ``_base_config`` / ``prepare_aligned_inputs``
/ the eval functions READ-ONLY and overrides only the PLM input path + feature dim.
It edits NO core code (model.py / config.py / full_benchmark.py / evaluate/ / cli.py).

Usage
-----
  # 1) materialize configs + commands + slurm (no submission):
  conda run -n saber python scripts/run_prior_ablation.py materialize [--smoke]
  # 2) collect metrics after runs finish (reads my runs + 650M campaign CSV):
  conda run -n saber python scripts/run_prior_ablation.py collect --root <root>
  # 3) build the comparison table + figure:
  conda run -n saber python scripts/run_prior_ablation.py compare --root <root>

Modes
-----
  materialize  prepare inputs, write configs/commands/slurm/plan + HANDOFF.
  smoke        materialize a tiny subset (1 dataset, 1 seed, 5 epochs, cpu) AND
               run it end-to-end, to prove every PLM dim trains + evaluates.
  collect      evaluate every run manifest, merge with 650M reference rows.
  compare      build mean+-SD + paired-Wilcoxon table + figure from collected CSV.
"""

from __future__ import annotations

import argparse
import datetime
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

# Read-only reuse of the main campaign's config builder + data prep + registry.
# Same module the 5-seed campaign uses -> hyperparams are byte-identical to the
# main campaign's saber condition; we override ONLY protein_feature_dim + path.
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # resolve `saber` from this checkout, not the env
from saber.full_benchmark import (
    DATASET_PLM_CACHE_NAMES,
    NON_LFQ_DATASETS,
    _base_config,
    _default_benchmark_root,
)
from saber.benchmark_data import load_benchmark_registry
from saber.paths import outputs_root as _outputs_root
from saber.paths import plm_root as _plm_root
from saber.prepare import prepare_aligned_inputs
from saber.plms import PLM_REGISTRY, REFERENCE_PLM

REPO_ROOT = Path(__file__).resolve().parents[1]
PLM_ROOT = _plm_root()
DEFAULT_SEEDS = (42, 123, 2024, 2025, 7)
DEFAULT_CONDA_ENV = "saber"
DATE = datetime.datetime.now().strftime("%Y%m%d")

# (label, cache_subdir, dim, family, role).  role: "reference" = read 650M from
# the finished campaign, do NOT train; "new" = generate train jobs here.
PLMS: list[tuple[str, str, int, str, str]] = [
    (label, label, spec["dim"], spec["family"],
     "reference" if label == REFERENCE_PLM else "new")
    for label, spec in PLM_REGISTRY.items()
]

# 650M reference rows are read from the finished main 5-seed campaign.
CAMPAIGN_650M_CSV = (
    _outputs_root() / "paper_benchmark_20260709_5seed" / "source_data" / "source_metrics_long.csv"
)

PRIMARY_METRIC = {"imputation": "pearson_missing_mean", "proteome_expansion": "pearson_target_mean"}
SECONDARY_METRICS = {
    "imputation": ["mae_missing", "rmse_missing", "spearman_missing_mean"],
    "proteome_expansion": ["mae_target", "rmse_target", "spearman_target_mean"],
}


def _q(value: str | Path) -> str:
    return shlex.quote(str(value))


def _plm_map() -> dict[str, tuple[int, str, str]]:
    return {label: (dim, family, role) for label, _, dim, family, role in PLMS}


# --------------------------------------------------------------------------- #
# materialize
# --------------------------------------------------------------------------- #
def _default_root(tag: str = "5seed") -> Path:
    return REPO_ROOT / "outputs" / f"prior_ablation_{DATE}_{tag}"


def _prepare_one(
    *, plm_label: str, cache_subdir: str, dim: int, dataset: str, spec: Any, prepared_root: Path
) -> tuple[Path, Path, dict[str, Any]]:
    """Align one (PLM, dataset) once; reused across all seeds (seed-independent)."""
    out_dir = prepared_root / plm_label / dataset
    cache_emb = PLM_ROOT / cache_subdir / f"{DATASET_PLM_CACHE_NAMES[dataset]}_embeddings.npy"
    cache_ids = PLM_ROOT / cache_subdir / f"{DATASET_PLM_CACHE_NAMES[dataset]}_protein_ids.npy"
    if not cache_emb.exists() or not cache_ids.exists():
        raise FileNotFoundError(f"PLM cache missing for {plm_label}/{dataset}: {cache_emb}")
    data_path = _default_benchmark_root() / "anndata" / spec.target_name
    prep = prepare_aligned_inputs(
        data_path=data_path,
        cache_path=cache_emb,
        outdir=out_dir,
        name=dataset,
        cache_protein_ids_path=cache_ids,
        protein_id_field="",
        expected_dim=dim,
        duplicate_cache_policy="first",
        duplicate_dataset_policy="first",
        drop_missing_proteins=True,
    )
    return prep.h5ad_path, prep.feature_cache_path, prep.report


def _build_config(
    *,
    run_name: str,
    root: Path,
    spec: Any,
    data_path: Path,
    feature_cache: Path,
    dim: int,
    seed: int,
    epochs: int,
    batch_size: int,
    device: str,
    target_fraction: float,
    missing_fraction: float,
) -> dict[str, Any]:
    """Main-campaign saber config with ONLY the PLM input swapped (dim + path)."""
    cfg = _base_config(
        run_name=run_name,
        run_root=root,
        spec=spec,
        data_path=data_path,
        plm_features=feature_cache,
        plm_ids="",
        condition="saber",
        missing_enabled=True,
        missing_fraction=missing_fraction,
        mnar_fraction=0.50,
        missing_seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        target_fraction=target_fraction,
    )
    cfg["model"]["protein_feature_dim"] = int(dim)  # override the hardcoded 1280
    # Default protocol (Table S5 sequence-prior block): frozen features and a
    # single learning rate, shared by every row of the table. --trainable-features
    # switches to the fine-tuned protocol used by the gene-prior block.
    cfg["model"]["trainable_protein_features"] = bool(trainable_features)
    cfg["training"]["lr_mult_protein_features"] = 1.0
    cfg["training"]["lr_mult_protein_encoder"] = 1.0
    return cfg


def materialize(
    *,
    root: Path,
    seeds: list[int],
    epochs: int,
    batch_size: int,
    device: str,
    plm_filter: list[str] | None,
    datasets: list[str] | None,
    train_partition: str,
    shards: int,
    conda_env: str = DEFAULT_CONDA_ENV,
    dry_run: bool = False,
    trainable_features: bool = False,
) -> dict[str, Any]:
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs" / "slurm").mkdir(parents=True, exist_ok=True)

    registry = load_benchmark_registry("", _default_benchmark_root().resolve())
    specs = {s.dataset: s for s in registry.datasets}

    plm_map = _plm_map()
    # Train EVERY registered PLM fresh (including the reference), so the whole
    # ablation table shares one training recipe. `--plms` subsets this list.
    new_plms = [(lbl, sub, dim, fam) for (lbl, sub, dim, fam, _role) in PLMS]
    if plm_filter:
        new_plms = [p for p in new_plms if p[0] in set(plm_filter)]
    target_datasets = list(datasets) if datasets else list(NON_LFQ_DATASETS)

    prepared_root = root / "prepared_inputs"
    configs_dir = root / "configs"
    commands_dir = root / "commands"
    prepared_cache: dict[tuple[str, str], tuple[Path, Path, dict[str, Any]]] = {}

    train_commands: list[str] = []
    train_jobs: list[dict[str, Any]] = []  # manifest paths for the collector

    alignment_summary: list[dict[str, Any]] = []

    for plm_label, cache_subdir, dim, family in new_plms:
        for dataset in target_datasets:
            if dataset not in specs:
                print(f"[skip] {dataset} not in registry", file=sys.stderr)
                continue
            spec = specs[dataset]
            h5ad, feat, report = _prepare_one(
                plm_label=plm_label, cache_subdir=cache_subdir, dim=dim,
                dataset=dataset, spec=spec, prepared_root=prepared_root,
            )
            prepared_cache[(plm_label, dataset)] = (h5ad, feat, report)
            alignment_summary.append({
                "plm": plm_label, "dataset": dataset,
                "n_cells": report["n_cells"], "n_kept_proteins": report["n_kept_proteins"],
                "n_requested_proteins": report["n_requested_proteins"],
                "n_dropped_missing": report["n_dropped_missing_requested_proteins"],
                "feature_dim": report["feature_dim"],
            })
            for seed in seeds:
                # --- imputation miss50 (subcommand: train) ---
                imp_name = f"{dataset}_miss50_mnar50_saber__{plm_label}_seed{seed}"
                imp_cfg = _build_config(
                    run_name=imp_name, root=root, spec=spec, data_path=h5ad,
                    feature_cache=feat, dim=dim, seed=seed, epochs=epochs,
                    batch_size=batch_size, device=device,
                    target_fraction=0.10, missing_fraction=0.50,
                )
                imp_cfg_path = configs_dir / plm_label / "imputation" / f"{imp_name}.yaml"
                imp_manifest = root / "runs" / imp_name / "artifact_manifest.json"
                if not dry_run:
                    _write_yaml(imp_cfg_path, imp_cfg)
                train_commands.append(_run_cmd("train", imp_cfg_path, conda_env=conda_env))
                train_jobs.append({
                    "task": "imputation", "plm": plm_label, "family": family,
                    "dataset": dataset, "seed": seed, "missing_fraction": 0.50,
                    "config": str(imp_cfg_path.resolve()),
                    "manifest": str(imp_manifest.resolve()),
                })

                # --- proteome expansion holdout5 (subcommand: train-expansion) ---
                exp_name = f"{dataset}_holdout5_saber__{plm_label}_seed{seed}"
                exp_cfg = _build_config(
                    run_name=exp_name, root=root, spec=spec, data_path=h5ad,
                    feature_cache=feat, dim=dim, seed=seed, epochs=epochs,
                    batch_size=batch_size, device=device,
                    target_fraction=0.05, missing_fraction=0.50,
                )
                exp_cfg_path = configs_dir / plm_label / "expansion" / f"{exp_name}.yaml"
                exp_manifest = root / "runs" / exp_name / "artifact_manifest.json"
                if not dry_run:
                    _write_yaml(exp_cfg_path, exp_cfg)
                train_commands.append(_run_cmd("train-expansion", exp_cfg_path, conda_env=conda_env))
                train_jobs.append({
                    "task": "proteome_expansion", "plm": plm_label, "family": family,
                    "dataset": dataset, "seed": seed, "target_fraction": 0.05,
                    "config": str(exp_cfg_path.resolve()),
                    "manifest": str(exp_manifest.resolve()),
                })

    # --- alignment cross-check: per dataset, kept-protein count must match across PLMs ---
    alignment_check = _check_alignment_consistency(alignment_summary)

    if not dry_run:
        commands_dir.mkdir(parents=True, exist_ok=True)
        (commands_dir / "train_commands.txt").write_text("\n".join(train_commands) + "\n")
        # stat QoS caps submitted-JOB COUNT (MaxSubmitJob=100), so a 240-element
        # SLURM array is rejected (every array element counts as a submitted job).
        # Use sharded-sequential jobs instead: split commands into `shards` jobs,
        # each looping over its slice on 1 GPU. Submit count = `shards` (<<100).
        part_sbatches = _write_sharded_sequential(
            root=root, commands=train_commands, partition=train_partition,
            shards=shards, repo_root=REPO_ROOT,
        )
        _write_submit_script(
            path=root / "submit_prior_ablation.sh",
            part_sbatches=part_sbatches, root=root, repo_root=REPO_ROOT,
            conda_env=conda_env,
        )

    plan = {
        "root": str(root),
        "date": DATE,
        "seeds": seeds,
        "epochs": epochs,
        "plms": [{"label": lbl, "dim": dim, "family": fam, "role": role}
                 for (lbl, _sub, dim, fam, role) in PLMS],
        "new_plms_trained": [p[0] for p in new_plms],
        "datasets": target_datasets,
        "n_train_jobs": len(train_commands),
        "shards": shards,
        "conda_env": conda_env,
        "train_jobs": train_jobs,
        "alignment_summary": alignment_summary,
        "alignment_check": alignment_check,
        "reference_650m_csv": str(CAMPAIGN_650M_CSV),
        "submit_script": str(root / "submit_prior_ablation.sh"),
    }
    if not dry_run:
        (root / "prior_ablation_plan.json").write_text(json.dumps(plan, indent=2))
        _write_handoff(root=root, plan=plan)
    return plan


def _check_alignment_consistency(summary: list[dict[str, Any]]) -> dict[str, Any]:
    """For each dataset, kept-protein count must be identical across PLMs (fairness)."""
    by_ds: dict[str, set[int]] = {}
    for row in summary:
        by_ds.setdefault(row["dataset"], set()).add(row["n_kept_proteins"])
    problems = {ds: sorted(s) for ds, s in by_ds.items() if len(s) > 1}
    return {"consistent": not problems, "problems": problems}


def _run_cmd(subcommand: str, cfg_path: Path, *, conda_env: str = DEFAULT_CONDA_ENV) -> str:
    return f"conda run -n {_q(conda_env)} saber {subcommand} --config {_q(cfg_path.resolve())}"


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _write_sharded_sequential(
    *, root: Path, commands: list[str], partition: str, shards: int,
    repo_root: Path, time_limit: str = "3-00:00:00",
) -> list[Path]:
    """Split commands into `shards` sequential SLURM jobs (one GPU each).

    stat QoS caps the submitted-JOB COUNT (MaxSubmitJob=100); a 240-element
    ``--array`` is rejected because every array element counts as a submitted
    job. Sharded-sequential jobs keep the submit count = ``shards`` (<< 100)
    while still parallelising across ``shards`` GPUs. Each shard is one job
    that loops over its slice of commands (mirrors the main campaign's
    sequential-train pattern, generalised to N shards).
    """
    import math

    n = len(commands)
    shards = max(1, min(int(shards), n))
    commands_dir = root / "commands"
    slurm_dir = root / "slurm"
    logs_dir = root / "logs" / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    commands_dir.mkdir(parents=True, exist_ok=True)
    commands_dir.joinpath("train_commands.txt").write_text("\n".join(commands) + "\n")

    chunk = math.ceil(n / shards)
    parts = [commands[i:i + chunk] for i in range(0, n, chunk)]
    sbatch_paths: list[Path] = []
    for k, part in enumerate(parts):
        part_file = commands_dir / f"train_commands.part{k}.txt"
        part_file.write_text("\n".join(part) + "\n")
        sb = slurm_dir / f"train_seq_part{k}.sbatch"
        lines = [
            "#!/usr/bin/env bash",
            f"#SBATCH --job-name=saber_priorabl_p{k}",
            f"#SBATCH --partition={partition}",
            "#SBATCH --cpus-per-task=4",
            "#SBATCH --mem=48G",
            f"#SBATCH --time={time_limit}",
            f"#SBATCH --output={logs_dir.resolve()}/priorabl_p{k}_%j.out",
            f"#SBATCH --error={logs_dir.resolve()}/priorabl_p{k}_%j.err",
            "#SBATCH --gres=gpu:1",
            "set -euo pipefail",
            f"CMD_FILE={_q(part_file.resolve())}",
            f"cd {_q(repo_root.resolve())}",
            "i=0",
            'while IFS= read -r CMD; do',
            "  i=$((i + 1))",
            '  [[ -z "${CMD}" ]] && continue',
            f'  echo "[prior-ablation p{k} job ${{i}}] ${{CMD}}"',
            '  eval "${CMD}"',
            'done < "${CMD_FILE}"',
        ]
        sb.write_text("\n".join(lines) + "\n")
        sb.chmod(0o755)
        sbatch_paths.append(sb)
    return sbatch_paths


def _write_submit_script(
    *, path: Path, part_sbatches: list[Path], root: Path, repo_root: Path,
    conda_env: str = DEFAULT_CONDA_ENV,
) -> None:
    collect_sbatch = root / "slurm" / "collect_prior_ablation.sbatch"
    # collect sbatch calls this driver's `collect` then `compare`.
    collect_sbatch.write_text("\n".join([
        "#!/usr/bin/env bash",
        "#SBATCH --job-name=saber_priorabl_collect",
        "#SBATCH --partition=stat",
        "#SBATCH --cpus-per-task=4",
        "#SBATCH --mem=32G",
        "#SBATCH --time=6:00:00",
        f"#SBATCH --output={root}/logs/slurm/priorabl_collect_%j.out",
        f"#SBATCH --error={root}/logs/slurm/priorabl_collect_%j.err",
        "set -euo pipefail",
        f"cd {_q(repo_root.resolve())}",
        f"conda run -n {_q(conda_env)} python scripts/run_prior_ablation.py collect --root {_q(root)}",
        f"conda run -n {_q(conda_env)} python scripts/run_prior_ablation.py compare --root {_q(root)}",
    ]) + "\n")
    collect_sbatch.chmod(0o755)

    lines = [
        "#!/usr/bin/env bash",
        "# Prior-ablation: sharded-sequential train (stat QoS: submit count = #shards)",
        "# -> collect+compare (afterany on all shards).",
        "set -euo pipefail",
        'TRAIN_IDS=""',
    ]
    for k, sb in enumerate(part_sbatches):
        lines.append(f'P{k}=$(sbatch --parsable {_q(sb.resolve())})')
        lines.append(f'echo "train part {k}: $P{k}"')
        lines.append(f'TRAIN_IDS="${{TRAIN_IDS}}:$P{k}"')
    # collect runs after ALL shards finish (afterany = regardless of pass/fail, so a
    # single failed run does not block collection of the rest).
    lines.append(f'echo "$TRAIN_IDS" | tee {root}/train_jobids.txt')
    lines.append('COLLECT=$(sbatch --parsable --dependency=afterany:${TRAIN_IDS#:}'
                 f' {_q(collect_sbatch.resolve())})')
    lines.append(f'echo "collect+compare: $COLLECT" | tee {root}/collect_jobid.txt')
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o755)


def _write_handoff(*, root: Path, plan: dict[str, Any]) -> None:
    new_plms = plan["new_plms_trained"]
    conda_env = plan.get("conda_env", DEFAULT_CONDA_ENV)
    n = plan["n_train_jobs"]
    shards = plan.get("shards", 8)
    # ~11 min/job average (main-campaign observed). Sequential = 1 GPU; sharded
    # = `shards` GPUs, one shard each.
    hours_seq = n * 11 / 60.0
    hours_par = hours_seq / max(1, shards)
    text = f"""# SABER PLM input-prior ablation plan

Root: `{plan["root"]}`
Date: {plan["date"]}  |  seeds: {plan["seeds"]}  |  epochs: {plan["epochs"]}

## What this is
PLM input-prior choice for SABER. 5 PLMs compared on imputation
(miss50) + proteome expansion (holdout5):
- **esm2_650m** (1280d) = REFERENCE, read from the finished 5-seed campaign
  `{plan["reference_650m_csv"]}` (saber condition, miss50 + holdout5, all 6 datasets x 5 seeds).
- **{", ".join(new_plms)}** = NEW runs materialized here.

New train jobs: **{n}** = {len(new_plms)} PLMs x {len(plan["datasets"])} datasets x {len(plan["seeds"])} seeds x 2 tasks.
Conda environment for generated commands: `{conda_env}`.

## Alignment / fairness
{("- Cross-PLM kept-protein counts are CONSISTENT per dataset (fair comparison)." if plan['alignment_check']['consistent'] else "- WARNING: cross-PLM kept-protein counts DIFFER per dataset: " + json.dumps(plan['alignment_check']['problems']) + " -- investigate before trusting the comparison.")}
- Configs built via `saber.full_benchmark._base_config` (same as main campaign); ONLY
  `model.protein_feature_dim` + `data.protein_features_path` differ per PLM. All other
  hyperparams byte-identical to the main campaign's saber condition. seed->mask mapping
  identical -> paired comparison across PLMs.
- See `prior_ablation_plan.json` (full job list + manifest paths + alignment report).

## Submit (does NOT auto-submit; user decides)
  bash {plan["root"]}/submit_prior_ablation.sh
Sharded-sequential train ({shards} shards = {shards} GPUs; stat QoS caps
submitted-job COUNT at 100, so an array is rejected -- shards keep the submit
count low) -> collect+compare job. Estimated wall-clock:
~{hours_par:.0f}h on {shards} GPUs / ~{hours_seq:.0f}h on 1 GPU.
Edit `slurm/train_seq_part*.sbatch` or re-run with `--shards N` to change parallelism.
Monitor: `squeue`; logs: `logs/slurm/priorabl_p*_*.out`.

## After it finishes
  conda run -n {conda_env} python scripts/run_prior_ablation.py collect  --root {plan["root"]}
  conda run -n {conda_env} python scripts/run_prior_ablation.py compare --root {plan["root"]}
Outputs: `source_data/prior_ablation_long.csv`, `prior_ablation_table.csv`,
`prior_ablation_figure.png/.pdf`, `CONCLUSION.md`.

## Caveats
- 650M rows come from the main campaign (current code, includes plm_ridge/plm_knn) ->
  comparable to the new runs.
- prot_bert & prot_t5_xl both 1024d -> isolates family-at-equal-dim; ankh 1536d is the
  largest dim (does bigger help?).
- Expansion absolute rho is low (Specht holdout5 ~0.030, Derks_2022 ~0.177); PLM gaps may
  be small -- present honestly (mean+-SD + paired Wilcoxon), do not oversell.
"""
    (Path(root) / "HANDOFF_AGENT6.md").write_text(text)


# --------------------------------------------------------------------------- #
# collect
# --------------------------------------------------------------------------- #
def collect(*, root: Path) -> Path:
    import csv
    from saber.evaluate.imputation import evaluate_imputation_manifest
    from saber.evaluate.expansion import evaluate_expansion_manifest

    root = Path(root).resolve()
    plan = json.loads((root / "prior_ablation_plan.json").read_text())
    plm_family = {p["label"]: p["family"] for p in plan["plms"]}

    out_rows: list[dict[str, Any]] = []
    n_evaluated = 0
    n_missing = 0
    for job in plan["train_jobs"]:
        manifest = Path(job["manifest"])
        if not manifest.exists():
            n_missing += 1
            continue
        task = job["task"]
        eval_out = root / "source_data" / "eval" / manifest.parent.name
        if task == "imputation":
            res = evaluate_imputation_manifest(manifest, eval_out)
            rows = res["rows"]
        else:  # proteome_expansion
            res = evaluate_expansion_manifest(manifest, eval_out)
            rows = list(csv.DictReader(open(res["source_csv"])))
        for r in rows:
            r["plm"] = job["plm"]
            r["plm_family"] = plm_family.get(job["plm"], "")
            out_rows.append(r)
        n_evaluated += 1

    # --- reference rows from the finished main campaign, ONLY when the reference
    #     PLM was NOT trained fresh here. When it was (the default now), it is
    #     already collected above and reading the CSV would duplicate rows / mix
    #     training recipes. ---
    trained_plms = {job["plm"] for job in plan["train_jobs"]}
    if REFERENCE_PLM not in trained_plms:
        ref_rows = _read_650m_reference(
            plan["reference_650m_csv"],
            set(plm_family),
            datasets=set(plan.get("datasets", [])),
            seeds=set(int(s) for s in plan.get("seeds", [])),
        )
        out_rows.extend(ref_rows)
    else:
        ref_rows = []

    source_dir = root / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    out_csv = source_dir / "prior_ablation_long.csv"
    preferred = [
        "task", "dataset", "plm", "plm_family", "method", "method_kind", "condition",
        "seed", "scale_mode", "metric", "value", "missing_fraction", "target_fraction",
        "artifact_path", "run_name",
    ]
    fieldnames = sorted({k for r in out_rows for k in r})
    fieldnames = [k for k in preferred if k in fieldnames] + [k for k in fieldnames if k not in preferred]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fieldnames})

    print(f"[collect] evaluated {n_evaluated} runs, {n_missing} manifests missing, "
          f"{len(ref_rows)} 650M-reference rows -> {out_csv}")
    return out_csv


def _read_650m_reference(
    csv_path: str | Path,
    known_plms: set[str],
    datasets: set[str] | None = None,
    seeds: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Read esm2_650m saber rows for miss50 imputation + holdout5 expansion."""
    import csv as _csv
    rows: list[dict[str, Any]] = []
    if not Path(csv_path).exists():
        print(f"[collect] WARNING: 650M campaign CSV not found: {csv_path}", file=sys.stderr)
        return rows
    with open(csv_path) as f:
        for r in _csv.DictReader(f):
            if r.get("method") != "saber" or r.get("condition") != "saber":
                continue
            if datasets and r.get("dataset") not in datasets:
                continue
            if seeds:
                try:
                    if int(r.get("seed", "")) not in seeds:
                        continue
                except ValueError:
                    continue
            task = r.get("task")
            if task == "imputation":
                try:
                    if abs(float(r.get("missing_fraction", "nan")) - 0.5) > 1e-6:
                        continue
                except ValueError:
                    continue
            elif task == "proteome_expansion":
                try:
                    if abs(float(r.get("target_fraction", "nan")) - 0.05) > 1e-6:
                        continue
                except ValueError:
                    continue
            else:
                continue
            r = dict(r)
            r["plm"] = "esm2_650m"
            r["plm_family"] = "esm2"
            rows.append(r)
    return rows


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #
def compare(*, root: Path) -> dict[str, Any]:
    import csv as _csv
    from statistics import mean, stdev

    try:
        import pandas as pd  # noqa: F401
    except Exception:
        pass
    root = Path(root).resolve()
    long_csv = root / "source_data" / "prior_ablation_long.csv"
    if not long_csv.exists():
        raise SystemExit(f"run `collect` first; missing {long_csv}")

    rows = list(_csv.DictReader(open(long_csv)))
    # focus on SABER itself under each PLM (method=saber) for the headline comparison.
    saber = [r for r in rows if r.get("method") == "saber"]

    plm_order = [lbl for lbl, _s, _d, _f, role in PLMS]
    fam_map = {lbl: fam for lbl, _s, _d, fam, _r in PLMS}
    dataset_order = list(NON_LFQ_DATASETS)

    def fnum(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    table_rows: list[dict[str, Any]] = []
    for task in ("imputation", "proteome_expansion"):
        prim = PRIMARY_METRIC[task]
        for ds in dataset_order:
            for plm in plm_order:
                vals = [fnum(r["value"]) for r in saber
                        if r.get("task") == task and r.get("dataset") == ds
                        and r.get("plm") == plm and r.get("metric") == prim]
                vals = [v for v in vals if v is not None]
                if not vals:
                    continue
                table_rows.append({
                    "task": task, "dataset": ds, "plm": plm, "family": fam_map.get(plm, ""),
                    "metric": prim, "n_seeds": len(vals),
                    "mean": mean(vals), "sd": stdev(vals) if len(vals) > 1 else 0.0,
                    "min": min(vals), "max": max(vals),
                })

    table_csv = root / "prior_ablation_table.csv"
    _write_table(table_csv, table_rows)

    # paired Wilcoxon (and mean diff) of each PLM vs esm2_650m, per (task, dataset)
    wilcoxon_rows = _paired_tests(saber, plm_order, dataset_order)
    _write_table(root / "prior_ablation_wilcoxon.csv", wilcoxon_rows)

    figure_path = _make_figure(root, table_rows, plm_order, fam_map)
    conclusion = _conclusion(table_rows, wilcoxon_rows, plm_order)
    (root / "CONCLUSION.md").write_text(conclusion)
    print(f"[compare] table: {table_csv}\n[compare] wilcoxon: {root/'prior_ablation_wilcoxon.csv'}")
    print(f"[compare] figure: {figure_path}\n[compare] conclusion: {root/'CONCLUSION.md'}")
    return {"table": str(table_csv), "figure": str(figure_path), "conclusion": str(root / "CONCLUSION.md")}


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv as _csv
    if not rows:
        path.write_text("(no rows)\n")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _paired_tests(saber_rows, plm_order, dataset_order) -> list[dict[str, Any]]:
    from statistics import mean
    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None

    def series(task, ds, plm, metric):
        out = {}
        for r in saber_rows:
            if (r.get("task") == task and r.get("dataset") == ds and r.get("plm") == plm
                    and r.get("metric") == metric):
                try:
                    out[int(r.get("seed"))] = float(r["value"])
                except (TypeError, ValueError):
                    pass
        return out

    rows: list[dict[str, Any]] = []
    for task in ("imputation", "proteome_expansion"):
        prim = PRIMARY_METRIC[task]
        for ds in dataset_order:
            ref = series(task, ds, "esm2_650m", prim)
            if len(ref) < 2:
                continue
            for plm in plm_order:
                if plm == "esm2_650m":
                    continue
                alt = series(task, ds, plm, prim)
                seeds = sorted(set(ref) & set(alt))
                if len(seeds) < 2:
                    continue
                d = [alt[s] - ref[s] for s in seeds]
                md = mean(d)
                rec = {"task": task, "dataset": ds, "plm": plm, "n_paired": len(seeds),
                       "mean_delta_vs_650m": md}
                if wilcoxon is not None and len(seeds) >= 3:
                    try:
                        stat, p = wilcoxon([alt[s] for s in seeds], [ref[s] for s in seeds])
                        rec["wilcoxon_stat"] = float(stat)
                        rec["wilcoxon_p"] = float(p)
                    except Exception:
                        pass
                rows.append(rec)
    return rows


def _make_figure(root, table_rows, plm_order, fam_map) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not table_rows:
        return root / "prior_ablation_figure.png"
    tasks = ["imputation", "proteome_expansion"]
    task_titles = {"imputation": "Imputation (miss50, pearson_missing_mean)",
                   "proteome_expansion": "Expansion (holdout5, pearson_target_mean)"}
    datasets = sorted({r["dataset"] for r in table_rows},
                      key=lambda d: NON_LFQ_DATASETS.index(d) if d in NON_LFQ_DATASETS else 99)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)
    for ax, task in zip(axes, tasks):
        sub = [r for r in table_rows if r["task"] == task]
        if not sub:
            ax.set_title(f"{task_titles[task]} (no data)")
            continue
        x = np.arange(len(datasets))
        w = 0.8 / max(1, len(plm_order))
        for i, plm in enumerate(plm_order):
            ys = []
            errs = []
            for ds in datasets:
                hit = [r for r in sub if r["dataset"] == ds and r["plm"] == plm]
                ys.append(hit[0]["mean"] if hit else np.nan)
                errs.append(hit[0]["sd"] if hit else 0.0)
            ax.bar(x + i * w - 0.4 + w / 2, ys, w, yerr=errs, label=f"{plm} ({fam_map.get(plm,'')})",
                   capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels([d.replace("_2023_melanoma_HEK", "").replace("_2019", "") for d in datasets],
                           rotation=30, ha="right")
        ax.set_title(task_titles[task])
        ax.set_ylabel("Pearson r (mean)")
        ax.axhline(0, color="k", linewidth=0.5)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("SABER PLM input-prior ablation (5 seeds, mean +/- SD)", fontsize=12)
    fig.tight_layout()
    png = root / "prior_ablation_figure.png"
    fig.savefig(png, dpi=150)
    fig.savefig(root / "prior_ablation_figure.pdf")
    return png


def _conclusion(table_rows, wilcoxon_rows, plm_order) -> str:
    from statistics import mean
    # aggregate mean across datasets per (task, plm), then rank.
    lines = ["# PLM input-prior ablation — conclusion", ""]
    for task in ("imputation", "proteome_expansion"):
        prim = PRIMARY_METRIC[task]
        lines.append(f"## {task} ({prim})")
        agg = {}
        for plm in plm_order:
            ds_means = [r["mean"] for r in table_rows
                        if r["task"] == task and r["plm"] == plm]
            if ds_means:
                agg[plm] = mean(ds_means)
        if not agg:
            lines.append("(no data)\n")
            continue
        ranked = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
        for rank, (plm, val) in enumerate(ranked, 1):
            lines.append(f"{rank}. **{plm}** ({val:.4f} mean across datasets)")
        lines.append("")
    best_exp = None
    exp_agg = {}
    for plm in plm_order:
        m = [r["mean"] for r in table_rows if r["task"] == "proteome_expansion" and r["plm"] == plm]
        if m:
            exp_agg[plm] = mean(m)
    if exp_agg:
        best_exp = max(exp_agg, key=exp_agg.get)
        lines.append("## One-line conclusion")
        line = (
            f"On expansion (PLM contributes most), best prior = **{best_exp}** "
            f"({exp_agg[best_exp]:.4f}); "
            f"ESM-2 650M = {exp_agg.get('esm2_650m', float('nan')):.4f}."
        )
        if "esm2_650m" in exp_agg and "esm2_150m" in exp_agg:
            size_eff = exp_agg["esm2_650m"] - exp_agg["esm2_150m"]
            line += f" ESM-2 size effect (650M - 150M) on expansion = {size_eff:+.4f}."
        lines.append(line)
    lines.append("")
    lines.append("_Auto-generated by `scripts/run_prior_ablation.py compare`. "
                 "Absolute expansion rho is low; treat rankings with the paired Wilcoxon table._")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
def smoke(*, root: Path, conda_env: str = DEFAULT_CONDA_ENV) -> None:
    """Materialize a 1-dataset/1-seed/5-epoch/cpu subset for all NEW PLMs and RUN it."""
    root = Path(root).resolve()
    plan = materialize(
        root=root, seeds=[42], epochs=5, batch_size=32, device="cpu",
        plm_filter=None, datasets=["Derks_2022"], train_partition="stat",
        shards=4, conda_env=conda_env, dry_run=False,
    )
    print(f"\n[smoke] materialized {plan['n_train_jobs']} jobs; running them on CPU now...\n")
    cmd_file = root / "commands" / "train_commands.txt"
    cmds = [c for c in cmd_file.read_text().splitlines() if c.strip()]
    failures = 0
    for i, cmd in enumerate(cmds, 1):
        print(f"[smoke] ({i}/{len(cmds)}) {cmd}", flush=True)
        rc = subprocess.run(cmd, shell=True, executable="/bin/bash").returncode
        if rc != 0:
            failures += 1
            print(f"[smoke] FAILED (rc={rc}): {cmd}", file=sys.stderr)
    print(f"\n[smoke] done. {len(cmds)-failures}/{len(cmds)} succeeded. "
          f"Verifying artifacts + collecting...")
    if failures:
        raise SystemExit(f"[smoke] {failures} job(s) failed -- inspect runs/ above.")
    # collect on the smoke subset to prove the eval pipeline + 650M merge work.
    collect(root=root)
    print("[smoke] OK: train + eval + 650M-merge pipeline verified for all new PLMs.")
    print(f"[smoke] see {root}/source_data/prior_ablation_long.csv")


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_mat = sub.add_parser("materialize", help="prepare inputs + write configs/commands/slurm.")
    p_mat.add_argument("--root", default=str(_default_root()))
    p_mat.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    p_mat.add_argument("--epochs", type=int, default=300)
    p_mat.add_argument("--batch-size", type=int, default=32)
    p_mat.add_argument("--device", default="auto")
    p_mat.add_argument("--conda-env", default=DEFAULT_CONDA_ENV)
    p_mat.add_argument("--plms", default="", help="comma list subset of new PLMs (default: all)")
    p_mat.add_argument("--datasets", default="", help="comma list subset (default: all 6 NON_LFQ)")
    p_mat.add_argument("--train-partition", default="stat")
    p_mat.add_argument("--shards", type=int, default=8,
                       help="#parallel sequential SLURM jobs (= #GPUs). stat QoS caps "
                            "submitted-job count at 100, so keep shards well under that.")
    p_mat.add_argument("--trainable-features", action="store_true",
                       help="fine-tune the protein features (gene-prior block protocol); "
                            "default is the frozen protocol of the sequence-prior block")
    p_mat.add_argument("--dry-run", action="store_true")

    p_smoke = sub.add_parser("smoke", help="tiny end-to-end run to validate every PLM dim.")
    p_smoke.add_argument("--root", default=str(_default_root("smoke")))
    p_smoke.add_argument("--conda-env", default=DEFAULT_CONDA_ENV)

    p_col = sub.add_parser("collect", help="evaluate all runs + merge 650M reference.")
    p_col.add_argument("--root", required=True)

    p_cmp = sub.add_parser("compare", help="build table + figure + conclusion.")
    p_cmp.add_argument("--root", required=True)

    args = parser.parse_args()

    if args.mode == "materialize":
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        plm_filter = [p for p in args.plms.split(",") if p.strip()] or None
        datasets = [d for d in args.datasets.split(",") if d.strip()] or None
        plan = materialize(
            root=args.root, seeds=seeds, epochs=args.epochs, batch_size=args.batch_size,
            device=args.device, plm_filter=plm_filter, datasets=datasets,
            train_partition=args.train_partition, shards=args.shards,
            conda_env=args.conda_env, dry_run=args.dry_run,
            trainable_features=args.trainable_features,
        )
        print(f"\n[materialize] {plan['n_train_jobs']} train jobs -> {plan['root']}")
        print(f"[materialize] alignment consistent: {plan['alignment_check']['consistent']}")
        print(f"[materialize] submit: bash {plan['root']}/submit_prior_ablation.sh")
    elif args.mode == "smoke":
        smoke(root=args.root, conda_env=args.conda_env)
    elif args.mode == "collect":
        collect(root=args.root)
    elif args.mode == "compare":
        compare(root=args.root)


if __name__ == "__main__":
    main()
