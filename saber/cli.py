from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

from . import __version__
from .artifacts import save_run_artifacts
from .config import load_config
from .data import load_data_bundle
from .expansion import train_expansion
from .inspectors import inspect_feature_cache, inspect_h5ad
from .model import SABER
from .prepare import prepare_aligned_inputs
from .train import predict_all, set_seed, train_model


def _build_model(cfg, data) -> SABER:
    return SABER(
        n_proteins=data.x_input.shape[1],
        protein_features=torch.as_tensor(data.protein_features, dtype=torch.float32),
        latent_dim=cfg.model.latent_dim,
        ae_h1=cfg.model.ae_h1,
        ae_h2=cfg.model.ae_h2,
        protein_proj_dim=cfg.model.protein_proj_dim,
        protein_proj_hidden_dim=cfg.model.protein_proj_hidden_dim,
        protein_proj_dropout=cfg.model.protein_proj_dropout,
        bilinear_with_w=cfg.model.bilinear_with_w,
        trainable_protein_features=cfg.model.trainable_protein_features,
    )


def _cmd_version(_args: argparse.Namespace) -> int:
    print(__version__)
    return 0


def _cmd_validate_config(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    print(f"Config OK: {args.config}")
    print(f"Run name: {cfg.project.name}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    data = load_data_bundle(cfg.data, cfg.missing, cfg.model, seed=cfg.project.seed)
    set_seed(cfg.project.seed)
    model = _build_model(cfg, data)
    result = train_model(model, data, cfg.training, seed=cfg.project.seed)
    prediction, z, protein_embeddings = predict_all(model, data.x_input, result.device)

    from .paths import resolve_outdir
    run_dir = resolve_outdir(cfg.project.outdir) / cfg.project.name
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg.to_dict(),
            "protein_ids": [str(x) for x in data.protein_ids.tolist()],
        },
        model_path,
    )

    command = getattr(args, "command", None) or " ".join(sys.argv)
    manifest_path = save_run_artifacts(
        run_dir=run_dir,
        cfg=cfg,
        data=data,
        prediction=prediction,
        cell_embeddings=z,
        protein_embeddings=protein_embeddings,
        history=result.history,
        command=command,
        model_path=model_path,
        device=result.device,
        resources=result.resources,
    )
    print(str(manifest_path.resolve()))
    return 0


def _cmd_train_expansion(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    command = getattr(args, "command", None) or " ".join(sys.argv)
    manifest_path = train_expansion(cfg, command=command)
    print(str(manifest_path.resolve()))
    return 0


def _cmd_inspect_data(args: argparse.Namespace) -> int:
    from .artifacts import write_json
    import json

    result = inspect_h5ad(
        args.data_path,
        batch_key=args.batch_key,
        cell_type_key=args.cell_type_key,
        protein_id_field=args.protein_id_field,
        max_cells=args.max_cells,
    )
    if args.out:
        write_json(args.out, result)
        print(str(Path(args.out).resolve()))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_inspect_feature_cache(args: argparse.Namespace) -> int:
    from .artifacts import write_json
    import json

    result = inspect_feature_cache(
        args.cache,
        cache_protein_ids_path=args.cache_protein_ids,
        data_path=args.data_path,
        protein_id_field=args.protein_id_field,
        expected_dim=args.expected_dim,
    )
    if args.out:
        write_json(args.out, result)
        print(str(Path(args.out).resolve()))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_prepare_aligned_inputs(args: argparse.Namespace) -> int:
    result = prepare_aligned_inputs(
        data_path=args.data_path,
        cache_path=args.cache,
        outdir=args.outdir,
        name=args.name,
        cache_protein_ids_path=args.cache_protein_ids,
        protein_id_field=args.protein_id_field,
        expected_dim=args.expected_dim,
        duplicate_cache_policy=args.duplicate_cache_policy,
        drop_missing_proteins=args.drop_missing_proteins,
        duplicate_dataset_policy=args.duplicate_dataset_policy,
    )
    print(str(result.h5ad_path.resolve()))
    print(str(result.feature_cache_path.resolve()))
    print(str(result.report_path.resolve()))
    return 0


def _cmd_prepare_benchmark_data(args: argparse.Namespace) -> int:
    from .benchmark_data import prepare_benchmark_data

    result = prepare_benchmark_data(
        benchmark_root=args.root,
        registry_path=args.registry,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        copy_annotations=not args.no_annotations,
        checksum=args.checksum,
    )
    if args.out:
        from .artifacts import write_json

        write_json(args.out, result)
        print(str(Path(args.out).resolve()))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_list_external_methods(args: argparse.Namespace) -> int:
    from .artifacts import write_json
    from .external_methods import check_external_method_available, method_specs

    rows = method_specs()
    if args.check:
        for row in rows:
            row["availability"] = {
                task: dict(zip(("available", "reason"), check_external_method_available(row["name"], task, args.ext_methods_root)))
                for task in row["capabilities"]
            }
    payload = {"methods": rows}
    if args.out:
        write_json(args.out, payload)
        print(str(Path(args.out).resolve()))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_run_external(args: argparse.Namespace) -> int:
    from .artifacts import write_json
    from .external_methods import run_external_method

    result = run_external_method(
        method=args.method,
        task=args.task,
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        logs_dir=args.logs_dir,
        ext_methods_root=args.ext_methods_root,
        repo_root=args.repo_root,
        seed=args.seed,
        timeout_sec=args.timeout_sec,
        dry_run=args.dry_run,
    )
    if args.out:
        write_json(args.out, result)
        print(str(Path(args.out).resolve()))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") or args.dry_run else 1


def _cmd_run_external_manifest(args: argparse.Namespace) -> int:
    import time as _time
    from .artifacts import write_json
    from .external_methods import run_external_for_manifest

    _t0 = _time.perf_counter()
    result = run_external_for_manifest(
        manifest_path=args.manifest,
        method=args.method,
        task=args.task,
        workdir=args.workdir,
        ext_methods_root=args.ext_methods_root,
        repo_root=args.repo_root,
        seed=args.seed,
        timeout_sec=args.timeout_sec,
        dry_run=args.dry_run,
        import_csv=args.import_csv,
    )
    result["wall_time_sec"] = float(_time.perf_counter() - _t0)
    if args.out:
        write_json(args.out, result)
        print(str(Path(args.out).resolve()))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") or args.dry_run or result.get("mode") == "import" else 1


def _cmd_evaluate_imputation(args: argparse.Namespace) -> int:
    from .evaluate.imputation import evaluate_imputation_manifest

    result = evaluate_imputation_manifest(args.manifest, args.outdir)
    print(result["json"])
    return 0


def _cmd_evaluate_cell_embedding(args: argparse.Namespace) -> int:
    from .evaluate.embedding import evaluate_embedding_manifest

    result = evaluate_embedding_manifest(args.manifest, args.outdir, seed=args.seed)
    print(result["json"])
    return 0


def _cmd_evaluate_expansion(args: argparse.Namespace) -> int:
    from .evaluate.expansion import evaluate_expansion_manifest

    result = evaluate_expansion_manifest(args.manifest, args.outdir)
    print(result["json"])
    return 0


def _cmd_evaluate_lfq(args: argparse.Namespace) -> int:
    from .evaluate.lfq import evaluate_lfq_manifest

    result = evaluate_lfq_manifest(
        args.manifest,
        args.outdir,
        data_path=args.data_path,
        condition_key=args.condition_key,
        species_key=args.species_key,
        protein_id_field=args.protein_id_field,
    )
    print(result["json"])
    return 0


def _cmd_evaluate_protein_embedding(args: argparse.Namespace) -> int:
    from .evaluate.protein_embedding import evaluate_protein_embedding_manifest

    ks = tuple(int(x) for x in args.ks.split(",") if x.strip())
    result = evaluate_protein_embedding_manifest(
        args.manifest,
        args.outdir,
        labels_csv=args.labels_csv,
        ext_dir=args.ext_dir,
        ks=ks,
        seed=args.seed,
        min_proteins=args.min_proteins,
        max_proteins=args.max_proteins,
    )
    print(result["json"])
    return 0


def _cmd_collect_source_data(args: argparse.Namespace) -> int:
    from .evaluate.common import expand_manifest_inputs
    from .source_data import collect_evaluation_source_data

    manifests = expand_manifest_inputs(args.manifest, args.manifest_glob)
    tasks = tuple(x.strip() for x in args.tasks.split(",") if x.strip())
    result = collect_evaluation_source_data(
        manifests,
        args.outdir,
        tasks=tasks,
        lfq_data_path=args.lfq_data_path,
        protein_labels_csv=args.protein_labels_csv,
        protein_ext_dir=args.protein_ext_dir,
    )
    print(result["source_csv"])
    return 0


def _cmd_make_paper_figures(args: argparse.Namespace) -> int:
    from .paper_figures import make_paper_figures

    formats = tuple(x.strip() for x in args.formats.split(",") if x.strip())
    tasks = tuple(x.strip() for x in args.tasks.split(",") if x.strip()) if args.tasks else None
    result = make_paper_figures(args.source_data, args.outdir, formats=formats, tasks=tasks)
    print(result["manifest"])
    if result.get("warnings"):
        print(json.dumps({"warnings": result["warnings"]}, indent=2, sort_keys=True))
    return 0


def _cmd_make_expression_maps(args: argparse.Namespace) -> int:
    from .expression_maps import make_expression_maps

    formats = tuple(x.strip() for x in args.formats.split(",") if x.strip())
    result = make_expression_maps(
        args.manifest,
        args.outdir,
        task=args.task,
        method=args.method,
        top_k=args.top_k,
        embedding=args.embedding,
        reducer=args.reducer,
        seed=args.seed,
        formats=formats,
        point_size=args.point_size,
        alpha=args.alpha,
    )
    print(str(Path(args.outdir, "expression_maps_manifest.json").resolve()))
    if not result.get("files"):
        print(json.dumps({"warnings": ["No expression map files were generated"]}, indent=2, sort_keys=True))
    return 0


def _cmd_make_refined_figure(args: argparse.Namespace) -> int:
    from .refined_figures import make_refined_figure

    formats = tuple(x.strip() for x in args.formats.split(",") if x.strip())
    result = make_refined_figure(args.source_data, args.spec, args.outdir, formats=formats)
    print(result["manifest"])
    return 0


def _cmd_materialize_full_benchmark(args: argparse.Namespace) -> int:
    from .full_benchmark import materialize_full_benchmark

    result = materialize_full_benchmark(
        root=args.root,
        registry_path=args.registry,
        benchmark_root=args.benchmark_root,
        plm_features=args.plm_features,
        plm_ids=args.plm_ids,
        plm_cache_dir=args.plm_cache_dir,
        conda_env=args.conda_env,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        train_partition=args.train_partition,
        external_partition=args.external_partition,
        train_max_parallel=args.train_max_parallel,
        external_max_parallel=args.external_max_parallel,
        dry_run=args.dry_run,
    )
    payload = {
        "root": str(result.root.resolve()),
        "plan": str(result.plan_path.resolve()),
        "train_commands": str(result.train_commands_path.resolve()),
        "external_commands": str(result.external_commands_path.resolve()),
        "train_slurm": str(result.train_slurm_path.resolve()),
        "external_slurm": str(result.external_slurm_path.resolve()),
        "collect_slurm": str(result.collect_slurm_path.resolve()),
        "submit_script": str(result.submit_script_path.resolve()),
        "sequential_train_slurm": str(result.sequential_train_slurm_path.resolve()),
        "sequential_external_slurm": str(result.sequential_external_slurm_path.resolve()),
        "sequential_submit_script": str(result.sequential_submit_script_path.resolve()),
        "n_train_jobs": result.n_train_jobs,
        "n_external_jobs": result.n_external_jobs,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_collect_full_benchmark_source_data(args: argparse.Namespace) -> int:
    from .full_benchmark import collect_full_benchmark_source_data

    result = collect_full_benchmark_source_data(args.plan, args.outdir)
    print(result["source_csv"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="saber",
        description="SABER core training and artifact CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    version = subparsers.add_parser("version", help="Print SABER package version")
    version.set_defaults(func=_cmd_version)

    validate = subparsers.add_parser("validate-config", help="Validate a YAML config")
    validate.add_argument("--config", required=True)
    validate.set_defaults(func=_cmd_validate_config)

    train = subparsers.add_parser("train", help="Train SABER and write manifest artifacts")
    train.add_argument("--config", required=True)
    train.set_defaults(func=_cmd_train)

    train_expansion_parser = subparsers.add_parser(
        "train-expansion",
        help="Train SABER with held-out target proteins and write manifest artifacts",
    )
    train_expansion_parser.add_argument("--config", required=True)
    train_expansion_parser.set_defaults(func=_cmd_train_expansion)

    inspect_data = subparsers.add_parser("inspect-data", help="Inspect a real .h5ad input before training")
    inspect_data.add_argument("--data-path", required=True)
    inspect_data.add_argument("--batch-key", default="batch")
    inspect_data.add_argument("--cell-type-key", default="cell_type")
    inspect_data.add_argument("--protein-id-field", default="")
    inspect_data.add_argument("--max-cells", type=int, default=256)
    inspect_data.add_argument("--out", default="")
    inspect_data.set_defaults(func=_cmd_inspect_data)

    inspect_cache = subparsers.add_parser("inspect-feature-cache", help="Inspect a protein feature cache before training")
    inspect_cache.add_argument("--cache", required=True)
    inspect_cache.add_argument("--cache-protein-ids", default="")
    inspect_cache.add_argument("--data-path", default="")
    inspect_cache.add_argument("--protein-id-field", default="")
    inspect_cache.add_argument("--expected-dim", type=int, default=None)
    inspect_cache.add_argument("--out", default="")
    inspect_cache.set_defaults(func=_cmd_inspect_feature_cache)

    prepare = subparsers.add_parser("prepare-aligned-inputs", help="Create explicit aligned h5ad/cache inputs for real runs")
    prepare.add_argument("--data-path", required=True)
    prepare.add_argument("--cache", required=True)
    prepare.add_argument("--cache-protein-ids", required=True)
    prepare.add_argument("--outdir", required=True)
    prepare.add_argument("--name", required=True)
    prepare.add_argument("--protein-id-field", default="")
    prepare.add_argument("--expected-dim", type=int, default=None)
    prepare.add_argument("--duplicate-cache-policy", choices=["fail", "first"], default="fail")
    prepare.add_argument("--duplicate-dataset-policy", choices=["fail", "first"], default="fail")
    prepare.add_argument("--drop-missing-proteins", action="store_true")
    prepare.set_defaults(func=_cmd_prepare_aligned_inputs)

    prepare_benchmark = subparsers.add_parser(
        "prepare-benchmark-data",
        help="Copy canonical SABER benchmark h5ad files and protein-embedding annotations into a stable data area",
    )
    prepare_benchmark.add_argument("--root", default="")
    prepare_benchmark.add_argument("--registry", default="")
    prepare_benchmark.add_argument("--dry-run", action="store_true")
    prepare_benchmark.add_argument("--overwrite", action="store_true")
    prepare_benchmark.add_argument("--no-annotations", action="store_true")
    prepare_benchmark.add_argument("--checksum", action="store_true")
    prepare_benchmark.add_argument("--out", default="")
    prepare_benchmark.set_defaults(func=_cmd_prepare_benchmark_data)

    list_external = subparsers.add_parser("list-external-methods", help="List external method task capabilities")
    list_external.add_argument("--check", action="store_true")
    list_external.add_argument("--ext-methods-root", default=os.environ.get("SABER_EXT_METHODS_ROOT", ""))
    list_external.add_argument("--out", default="")
    list_external.set_defaults(func=_cmd_list_external_methods)

    run_external = subparsers.add_parser("run-external", help="Run one external method adapter on a CSV matrix")
    run_external.add_argument("--method", required=True)
    run_external.add_argument("--task", required=True, choices=["imputation", "cell_embedding", "protein_embedding", "lfq"])
    run_external.add_argument("--input-csv", required=True)
    run_external.add_argument("--output-csv", required=True)
    run_external.add_argument("--logs-dir", required=True)
    run_external.add_argument("--ext-methods-root", default=os.environ.get("SABER_EXT_METHODS_ROOT", ""))
    run_external.add_argument("--repo-root", default=os.environ.get("SABER_EXT_REPO_ROOT", ""))
    run_external.add_argument("--seed", type=int, default=None)
    run_external.add_argument("--timeout-sec", type=int, default=3600)
    run_external.add_argument("--dry-run", action="store_true")
    run_external.add_argument("--out", default="")
    run_external.set_defaults(func=_cmd_run_external)

    run_external_manifest = subparsers.add_parser(
        "run-external-manifest",
        help="Run or import one external method for an existing SABER artifact manifest",
    )
    run_external_manifest.add_argument("--manifest", required=True)
    run_external_manifest.add_argument("--method", required=True)
    run_external_manifest.add_argument("--task", required=True, choices=["imputation", "cell_embedding", "protein_embedding", "lfq"])
    run_external_manifest.add_argument("--workdir", required=True)
    run_external_manifest.add_argument("--import-csv", default="", help="Import an already computed output CSV instead of running the adapter")
    run_external_manifest.add_argument("--ext-methods-root", default=os.environ.get("SABER_EXT_METHODS_ROOT", ""))
    run_external_manifest.add_argument("--repo-root", default=os.environ.get("SABER_EXT_REPO_ROOT", ""))
    run_external_manifest.add_argument("--seed", type=int, default=None)
    run_external_manifest.add_argument("--timeout-sec", type=int, default=3600)
    run_external_manifest.add_argument("--dry-run", action="store_true")
    run_external_manifest.add_argument("--out", default="")
    run_external_manifest.set_defaults(func=_cmd_run_external_manifest)

    eval_imp = subparsers.add_parser("evaluate-imputation", help="Evaluate injected-missing imputation metrics for a manifest")
    eval_imp.add_argument("--manifest", required=True)
    eval_imp.add_argument("--outdir", required=True)
    eval_imp.set_defaults(func=_cmd_evaluate_imputation)

    eval_cell = subparsers.add_parser("evaluate-cell-embedding", help="Evaluate cell embedding structure for a manifest")
    eval_cell.add_argument("--manifest", required=True)
    eval_cell.add_argument("--outdir", required=True)
    eval_cell.add_argument("--seed", type=int, default=42)
    eval_cell.set_defaults(func=_cmd_evaluate_cell_embedding)

    eval_expansion = subparsers.add_parser("evaluate-expansion", help="Evaluate held-out protein expansion for a manifest")
    eval_expansion.add_argument("--manifest", required=True)
    eval_expansion.add_argument("--outdir", required=True)
    eval_expansion.set_defaults(func=_cmd_evaluate_expansion)

    eval_lfq = subparsers.add_parser("evaluate-lfq", help="Evaluate PXD071205 full LFQ fold-change metrics for a manifest")
    eval_lfq.add_argument("--manifest", required=True)
    eval_lfq.add_argument("--outdir", required=True)
    eval_lfq.add_argument("--data-path", default="")
    eval_lfq.add_argument("--condition-key", default="condition")
    eval_lfq.add_argument("--species-key", default="species")
    eval_lfq.add_argument("--protein-id-field", default="")
    eval_lfq.set_defaults(func=_cmd_evaluate_lfq)

    eval_protein = subparsers.add_parser("evaluate-protein-embedding", help="Evaluate protein embedding Layer 1/2/3 metrics")
    eval_protein.add_argument("--manifest", required=True)
    eval_protein.add_argument("--outdir", required=True)
    eval_protein.add_argument("--labels-csv", default="")
    eval_protein.add_argument("--ext-dir", default="")
    eval_protein.add_argument("--ks", default="5,10,20")
    eval_protein.add_argument("--seed", type=int, default=42)
    eval_protein.add_argument("--min-proteins", type=int, default=10)
    eval_protein.add_argument("--max-proteins", type=int, default=100)
    eval_protein.set_defaults(func=_cmd_evaluate_protein_embedding)

    collect = subparsers.add_parser("collect-source-data", help="Evaluate manifests and collect long-form source data")
    collect.add_argument("--manifest", action="append", default=[])
    collect.add_argument("--manifest-glob", action="append", default=[])
    collect.add_argument("--outdir", required=True)
    collect.add_argument("--tasks", default="imputation,cell_embedding,proteome_expansion,protein_embedding,lfq")
    collect.add_argument("--lfq-data-path", default="")
    collect.add_argument("--protein-labels-csv", default="")
    collect.add_argument("--protein-ext-dir", default="")
    collect.set_defaults(func=_cmd_collect_source_data)

    paper_figures = subparsers.add_parser("make-paper-figures", help="Create paper-style figures from SABER source-data rows")
    paper_figures.add_argument("--source-data", required=True)
    paper_figures.add_argument("--outdir", required=True)
    paper_figures.add_argument("--formats", default="png,pdf")
    paper_figures.add_argument("--tasks", default="", help="Optional comma-separated figure groups to render")
    paper_figures.set_defaults(func=_cmd_make_paper_figures)

    expr_maps = subparsers.add_parser("make-expression-maps", help="Create per-protein expression maps from an existing artifact manifest")
    expr_maps.add_argument("--manifest", required=True)
    expr_maps.add_argument("--outdir", required=True)
    expr_maps.add_argument("--task", required=True, choices=["imputation", "proteome_expansion", "expansion"])
    expr_maps.add_argument("--method", default="saber")
    expr_maps.add_argument("--top-k", type=int, default=5)
    expr_maps.add_argument("--embedding", default="observed", choices=["observed"])
    expr_maps.add_argument("--reducer", default="auto", choices=["auto", "umap", "tsne", "raw2d"])
    expr_maps.add_argument("--seed", type=int, default=42)
    expr_maps.add_argument("--point-size", type=float, default=9.0)
    expr_maps.add_argument("--alpha", type=float, default=0.9)
    expr_maps.add_argument("--formats", default="png,pdf")
    expr_maps.set_defaults(func=_cmd_make_expression_maps)

    refined = subparsers.add_parser("make-refined-figure", help="Create a configured refined figure from source-data rows")
    refined.add_argument("--source-data", required=True)
    refined.add_argument("--spec", required=True)
    refined.add_argument("--outdir", required=True)
    refined.add_argument("--formats", default="png,pdf")
    refined.set_defaults(func=_cmd_make_refined_figure)

    materialize_full = subparsers.add_parser("materialize-full-benchmark", help="Generate configs, commands, and Slurm scripts for the SABER full benchmark")
    materialize_full.add_argument("--root", default="")
    materialize_full.add_argument("--registry", default="configs/benchmarks/saber_benchmark.yaml")
    materialize_full.add_argument("--benchmark-root", default="")
    materialize_full.add_argument("--plm-features", default="")
    materialize_full.add_argument("--plm-ids", default="")
    materialize_full.add_argument("--plm-cache-dir", default="")
    materialize_full.add_argument("--conda-env", default="saber")
    materialize_full.add_argument("--epochs", type=int, default=300)
    materialize_full.add_argument("--batch-size", type=int, default=32)
    materialize_full.add_argument("--device", default="auto")
    materialize_full.add_argument("--train-partition", default="stat")
    materialize_full.add_argument("--external-partition", default="stat")
    materialize_full.add_argument("--train-max-parallel", type=int, default=4)
    materialize_full.add_argument("--external-max-parallel", type=int, default=6)
    materialize_full.add_argument("--dry-run", action="store_true")
    materialize_full.set_defaults(func=_cmd_materialize_full_benchmark)

    collect_full = subparsers.add_parser("collect-full-benchmark-source-data", help="Evaluate task-specific manifests from a full benchmark plan")
    collect_full.add_argument("--plan", required=True)
    collect_full.add_argument("--outdir", required=True)
    collect_full.set_defaults(func=_cmd_collect_full_benchmark_source_data)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
