from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from saber.benchmark_data import prepare_benchmark_data
from saber.evaluate.embedding import evaluate_embedding_manifest
from saber.evaluate.expansion import evaluate_expansion_manifest
from saber.evaluate.imputation import evaluate_imputation_manifest
from saber.evaluate.lfq import evaluate_lfq_manifest
from saber.evaluate.protein_embedding import evaluate_protein_embedding_manifest
from saber.expression_maps import make_expression_maps
from saber.external_methods import add_external_protein_embedding_to_manifest, check_external_method_available, methods_for_task
from saber.external_methods import run_external_for_manifest
from saber.refined_figures import make_refined_figure
from saber.prepare import prepare_aligned_inputs


def _write_manifest(run_dir: Path, artifacts: dict[str, Path], dataset: str = "Toy") -> Path:
    manifest = {
        "schema_version": "1.0",
        "dataset": dataset,
        "config": {
            "project": {"name": run_dir.name, "seed": 7},
            "data": {"path": f"{dataset}.h5ad", "scale_mode": "none"},
            "missing": {"total_fraction": 0.2, "mnar_fraction": 0.5},
            "expansion": {"target_fraction": 0.25},
        },
        "data": {"n_cells": 4, "n_proteins": 3},
        "artifacts": {key: str(path) for key, path in artifacts.items()},
    }
    path = run_dir / "artifact_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_prepare_benchmark_data_dry_run_has_canonical_derks_name(tmp_path: Path, saber_env) -> None:
    payload = prepare_benchmark_data(benchmark_root=tmp_path / "saber_benchmark", dry_run=True)
    datasets = {row["dataset"]: row for row in payload["datasets"]}
    assert "Derks_2022" in datasets
    assert datasets["Derks_2022"]["target_name"] == "Derks_2022.h5ad"
    assert "Dou_2019" not in datasets
    assert "Derks_2023" not in datasets
    assert datasets["Woo_2021"]["scale_mode"] == "zscore"
    assert "EasySCP_liver" in datasets
    assert "Brunner_2022" in datasets


def test_prepare_benchmark_data_registry_root_is_configurable(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    registry_root = tmp_path / "canonical"
    registry_path.write_text(
        "\n".join(
            [
                "paths:",
                f"  data_root: {tmp_path}",
                f"  benchmark_root: {registry_root}",
                "datasets:",
                "  - dataset: ToySet",
                '    source_path: "{data_root}/anndata/ToySet.h5ad"',
                "    target_name: ToySet.h5ad",
                "    batch_key: batch",
                "    cell_type_key: cell_type",
                "    species_hint: human",
                "    scale_mode: none",
                "    task_families: [imputation]",
            ]
        ),
        encoding="utf-8",
    )
    payload = prepare_benchmark_data(registry_path=registry_path, dry_run=True, copy_annotations=False)
    assert payload["benchmark_root"] == str(registry_root)
    assert payload["datasets"][0]["target_path"] == str(registry_root / "anndata" / "ToySet.h5ad")


def test_prepare_aligned_inputs_can_keep_first_duplicate_dataset_id(tmp_path: Path) -> None:
    import anndata as ad

    x = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    adata = ad.AnnData(x)
    adata.var_names = ["P1", "P1", "P2"]
    data_path = tmp_path / "dup.h5ad"
    adata.write_h5ad(data_path)
    features = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    cache_path = tmp_path / "features.npz"
    np.savez_compressed(cache_path, features=features, protein_ids=np.asarray(["P1", "P2"]))

    result = prepare_aligned_inputs(
        data_path,
        cache_path,
        tmp_path / "prepared",
        "toy",
        duplicate_dataset_policy="first",
        drop_missing_proteins=True,
    )

    assert result.report["n_kept_proteins"] == 2
    assert result.report["n_dropped_duplicate_dataset_proteins"] == 1


def test_external_method_capabilities() -> None:
    assert "pimms_dae" in methods_for_task("imputation")
    assert "pimms_vae" in methods_for_task("imputation")
    assert "scprotein" in methods_for_task("cell_embedding")
    ok, reason = check_external_method_available("scprotein", "imputation")
    assert not ok
    assert "does not support" in reason


def test_imputation_and_embedding_evaluators(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True)
    labels_path = artifact_dir / "labels.npz"
    imputation_path = artifact_dir / "imputation.npz"
    embeddings_path = artifact_dir / "embeddings.npz"
    np.savez_compressed(
        labels_path,
        batches=np.asarray([0, 0, 1, 1]),
        cell_types=np.asarray([0, 1, 0, 1]),
        protein_ids=np.asarray(["P1", "P2", "P3"]),
    )
    x_true = np.asarray([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]], dtype=np.float32)
    mask = np.zeros_like(x_true, dtype=np.uint8)
    mask[0, 0] = 1
    mask[1, 0] = 1
    pred = x_true.copy()
    pred[0, 0] = 1.5
    pred[1, 0] = 2.5
    np.savez_compressed(
        imputation_path,
        x_true=x_true,
        x_input=x_true,
        injected_missing_mask=mask,
        observed_mask=np.ones_like(x_true),
        available_mask=np.ones_like(x_true),
        pred__saber=pred,
    )
    np.savez_compressed(embeddings_path, z__saber=np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32))
    manifest = _write_manifest(run_dir, {"labels": labels_path, "imputation": imputation_path, "embeddings": embeddings_path})

    imp = evaluate_imputation_manifest(manifest, tmp_path / "eval_imp")
    emb = evaluate_embedding_manifest(manifest, tmp_path / "eval_emb")
    assert imp["results"]["saber"]["n_missing"] == 2
    assert (tmp_path / "eval_imp" / "source_imputation_metrics.csv").exists()
    assert (tmp_path / "eval_imp" / "source_imputation_per_protein.csv").exists()
    assert len(imp["per_protein"]) == 9
    assert emb["results"]["saber"]["n_cells_eval"] == 4


def test_expansion_and_protein_embedding_evaluators(tmp_path: Path) -> None:
    run_dir = tmp_path / "exp"
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True)
    labels_path = artifact_dir / "labels.npz"
    expansion_path = artifact_dir / "expansion.npz"
    embeddings_path = artifact_dir / "embeddings.npz"
    np.savez_compressed(
        labels_path,
        batches=np.asarray([0, 0, 1, 1]),
        cell_types=np.asarray([0, 1, 0, 1]),
        protein_ids=np.asarray(["P1", "P2", "P3"]),
        train_protein_ids=np.asarray(["P1", "P2"]),
        target_protein_ids=np.asarray(["P3"]),
        train_protein_indices=np.asarray([0, 1]),
        target_protein_indices=np.asarray([2]),
    )
    x_target = np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    np.savez_compressed(
        expansion_path,
        x_true_target=x_target,
        available_target_mask=np.ones_like(x_target, dtype=np.uint8),
        eval_target_mask=np.ones_like(x_target, dtype=np.uint8),
        pred_target__saber=x_target + 0.1,
        x_true_train=np.asarray([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.float32),
        pred_train__saber=np.asarray([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.float32),
    )
    np.savez_compressed(
        embeddings_path,
        protein_embeddings_train__saber=np.asarray([[1, 0], [0, 1]], dtype=np.float32),
        protein_embeddings_target__saber=np.asarray([[1, 1]], dtype=np.float32),
    )
    raw_features = tmp_path / "raw_features.npz"
    np.savez_compressed(
        raw_features,
        features=np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
        protein_ids=np.asarray(["P1", "P2", "P3"]),
    )
    manifest = _write_manifest(run_dir, {"labels": labels_path, "expansion": expansion_path, "embeddings": embeddings_path})
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["config"]["data"]["protein_features_path"] = str(raw_features)
    payload["config"]["model"] = {"protein_feature_dim": 3}
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    labels_csv = tmp_path / "protein_labels.csv"
    labels_csv.write_text("protein_id,label_id\nP1,A\nP2,B\nP3,A\n", encoding="utf-8")

    exp = evaluate_expansion_manifest(manifest, tmp_path / "eval_exp")
    prot = evaluate_protein_embedding_manifest(manifest, tmp_path / "eval_prot", labels_csv=labels_csv)
    assert exp["results"]["saber"]["n_target"] == 4
    assert (tmp_path / "eval_exp" / "source_expansion_per_protein.csv").exists()
    assert "saber_train" in prot["results"]
    assert "raw_plm" in prot["results"]


def test_expansion_protein_mean_floor_is_honest_global_mean(tmp_path: Path) -> None:
    """Regression: the protein_mean floor must not peek at held-out truth.

    The old implementation used x_true_target's per-column mean (cheating: that
    truth is unavailable in a real expansion scenario and made MAE/RMSE
    artificially low). The honest floor is the global *training* mean broadcast
    as a constant, so it is invariant to the held-out values.
    """
    from saber.evaluate.expansion import _protein_mean_baseline

    # Training matrix with a structural NaN hole.
    x_true_train = np.asarray(
        [[1.0, 5.0, np.nan], [3.0, 5.0, 9.0], [2.0, 5.0, 9.0]], dtype=np.float32
    )
    n_targets = 4
    expected = float(np.nanmean(x_true_train))  # finite-only grand mean = 4.875

    pred = _protein_mean_baseline(x_true_train, n_targets)

    assert pred.shape == (x_true_train.shape[0], n_targets)
    assert pred.dtype == np.float32
    assert np.isfinite(pred).all()
    assert np.unique(pred).size == 1            # constant floor, no per-target variation
    assert np.isclose(pred[0, 0], expected)     # == global finite training mean

    # End-to-end: the wired baseline is the honest floor.
    run_dir = tmp_path / "floor"
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True)
    expansion_path = artifact_dir / "expansion.npz"
    x_target = np.asarray([[1.0], [2.0], [8.0]], dtype=np.float32)
    np.savez_compressed(
        expansion_path,
        x_true_target=x_target,
        available_target_mask=np.ones_like(x_target, dtype=np.uint8),
        eval_target_mask=np.ones_like(x_target, dtype=np.uint8),
        pred_target__saber=x_target + 0.5,
        x_true_train=x_true_train,
        pred_train__saber=x_true_train,
    )
    labels_path = artifact_dir / "labels.npz"
    np.savez_compressed(
        labels_path,
        batches=np.asarray([0, 0, 1]),
        cell_types=np.asarray([0, 1, 0]),
        protein_ids=np.asarray(["T1", "P1", "P2", "P3"]),
        target_protein_ids=np.asarray(["T1"]),
        target_protein_indices=np.asarray([0]),
    )
    manifest = _write_manifest(run_dir, {"labels": labels_path, "expansion": expansion_path})
    exp = evaluate_expansion_manifest(manifest, tmp_path / "eval_floor")
    # Constant prediction -> per-target Pearson undefined: the absolute floor.
    assert exp["results"]["protein_mean"]["pearson_target_mean"] is None
    # MAE matches the honest global-mean floor, NOT the cheated per-target column mean.
    honest_mae = float(np.mean(np.abs(x_target - expected)))
    assert np.isclose(exp["results"]["protein_mean"]["mae_target"], honest_mae)


def test_expansion_reference_methods_tagged_as_reference(tmp_path: Path) -> None:
    """PLM reference models carry method_kind='reference', not 'model'.

    plm_ridge / plm_knn / plm_knn_k<N> belong to the Reference-ablation tier
    (alongside no-PLM/no-W), distinct from SABER itself. The evaluator reads the
    runner's references.written_keys to tag them so downstream grouping can keep
    them separate from the learned model.
    """
    from saber.evaluate.expansion import compute_expansion_evaluation

    run_dir = tmp_path / "refkind"
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True)
    expansion_path = artifact_dir / "expansion.npz"
    x_target = np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    np.savez_compressed(
        expansion_path,
        x_true_target=x_target,
        available_target_mask=np.ones_like(x_target, dtype=np.uint8),
        eval_target_mask=np.ones_like(x_target, dtype=np.uint8),
        pred_target__saber=x_target + 0.1,
        pred_target__plm_ridge=x_target + 1.0,
        pred_target__plm_knn=x_target + 0.5,
        pred_target__plm_knn_k10=x_target + 0.2,
        x_true_train=np.asarray([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.float32),
    )
    manifest = _write_manifest(run_dir, {"expansion": expansion_path})
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    # Mirror what the runner records for a references-enabled expansion run.
    payload["references"] = {"written_keys": ["plm_knn", "plm_knn_k10", "plm_ridge"]}
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    mk = compute_expansion_evaluation(manifest)["method_kinds"]
    assert mk["saber"] == "model"
    assert mk["plm_ridge"] == "reference"
    assert mk["plm_knn"] == "reference"
    assert mk["plm_knn_k10"] == "reference"
    assert mk["cell_mean"] == "baseline"
    assert mk["protein_mean"] == "baseline"


def test_protein_embedding_layer23_evaluator(tmp_path: Path) -> None:
    run_dir = tmp_path / "prot23"
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True)
    labels_path = artifact_dir / "labels.npz"
    embeddings_path = artifact_dir / "embeddings.npz"
    protein_ids = np.asarray([f"P{i}" for i in range(12)], dtype=object)
    emb = np.asarray(
        [[1.0, 0.08 * i, 0.0] for i in range(6)]
        + [[0.0, 1.0, 0.08 * i] for i in range(6)],
        dtype=np.float32,
    )
    np.savez_compressed(labels_path, protein_ids=protein_ids)
    np.savez_compressed(embeddings_path, protein_embeddings__saber=emb)
    manifest = _write_manifest(run_dir, {"labels": labels_path, "embeddings": embeddings_path})

    ext_dir = tmp_path / "external"
    ext_dir.mkdir()
    edges = ["protein1,protein2,score"]
    edges.extend(f"P{i},P{j},0.9" for i in range(6) for j in range(i + 1, 6))
    ext_dir.joinpath("string_interactions.csv").write_text("\n".join(edges), encoding="utf-8")
    reactome = [
        {"pathway_id": "R1", "pathway_name": "Cluster 1", "proteins": [f"P{i}" for i in range(6)]},
        {"pathway_id": "R2", "pathway_name": "Cluster 2", "proteins": [f"P{i}" for i in range(6, 12)]},
    ]
    go = [
        {"go_id": "GO:1", "go_name": "Cluster 1", "proteins": [f"P{i}" for i in range(6)]},
        {"go_id": "GO:2", "go_name": "Cluster 2", "proteins": [f"P{i}" for i in range(6, 12)]},
    ]
    ext_dir.joinpath("reactome_pathways.json").write_text(json.dumps(reactome), encoding="utf-8")
    ext_dir.joinpath("go_annotations.json").write_text(json.dumps(go), encoding="utf-8")
    ext_dir.joinpath("annotation_summary.json").write_text("{}", encoding="utf-8")

    result = evaluate_protein_embedding_manifest(
        manifest,
        tmp_path / "eval_prot23",
        ext_dir=ext_dir,
        ks=(3, 5),
        min_proteins=3,
        max_proteins=10,
    )

    assert "saber" in result["layer23_results"]
    assert "string.edge_auroc" in result["layer23_results"]["saber"]
    assert "reactome.pathway_hit_at_3" in result["layer23_results"]["saber"]
    assert (tmp_path / "eval_prot23" / "protein_embedding_layer23_metrics.json").exists()
    assert any(row["protein_embedding_layer"] == "layer2_3" for row in result["rows"])


def test_lfq_evaluator_with_manifest_labels(tmp_path: Path) -> None:
    run_dir = tmp_path / "lfq"
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True)
    labels_path = artifact_dir / "labels.npz"
    imputation_path = artifact_dir / "imputation.npz"
    conditions = np.asarray(["A", "A", "B", "B", "C", "C"], dtype=object)
    species = np.asarray(["HUMAN", "YEAST", "ECOLI"], dtype=object)
    protein_ids = np.asarray(["H1", "Y1", "E1"], dtype=object)
    x = np.asarray(
        [
            [10.0, 3.0, 0.5],
            [10.2, 3.0, 0.5],
            [10.1, 2.0, 2.0],
            [10.0, 2.0, 2.0],
            [10.0, 0.3, 3.2],
            [10.1, 0.3, 3.2],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(labels_path, conditions=conditions, species=species, protein_ids=protein_ids, x_raw=x)
    np.savez_compressed(
        imputation_path,
        x_true=x,
        x_input=x,
        injected_missing_mask=np.zeros_like(x, dtype=np.uint8),
        observed_mask=np.ones_like(x),
        available_mask=np.ones_like(x),
        pred__saber=x,
    )
    manifest = _write_manifest(run_dir, {"labels": labels_path, "imputation": imputation_path}, dataset="PXD071205_LFQ")
    result = evaluate_lfq_manifest(manifest, tmp_path / "eval_lfq")
    assert result["results"]["saber"]["n_protein_comparisons"] == 9
    assert result["results"]["protein_mean"]["n_protein_comparisons"] == 9
    assert (tmp_path / "eval_lfq" / "source_lfq_metrics.csv").exists()


def test_expression_maps_from_imputation_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "expr_imp"
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True)
    labels_path = artifact_dir / "labels.npz"
    imputation_path = artifact_dir / "imputation.npz"
    x = np.asarray(
        [
            [1.0, 0.0, 1.0],
            [2.0, 0.5, 1.5],
            [3.0, 1.0, 2.0],
            [4.0, 1.5, 2.5],
            [5.0, 2.0, 3.0],
        ],
        dtype=np.float32,
    )
    mask = np.zeros_like(x, dtype=np.uint8)
    mask[:, 0] = 1
    np.savez_compressed(labels_path, protein_ids=np.asarray(["P1", "P2", "P3"]))
    np.savez_compressed(
        imputation_path,
        x_true=x,
        x_input=x,
        injected_missing_mask=mask,
        observed_mask=np.ones_like(x),
        available_mask=np.ones_like(x),
        pred__saber=x + 0.01,
    )
    manifest = _write_manifest(run_dir, {"labels": labels_path, "imputation": imputation_path})

    result = make_expression_maps(
        manifest,
        tmp_path / "expr_out",
        task="imputation",
        method="saber",
        top_k=1,
        reducer="raw2d",
        formats=("png",),
    )

    assert len(result["files"]) == 1
    assert Path(result["files"][0]).exists()


def test_refined_figure_spec_bar(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text(
        "\n".join(
            [
                "task,dataset,method,condition,metric,value,missing_fraction",
                "imputation,D1,saber,saber,pearson_missing_mean,0.6,0.5",
                "imputation,D1,pimms_dae,saber,pearson_missing_mean,0.4,0.5",
                "imputation,D2,saber,saber,pearson_missing_mean,0.7,0.5",
                "imputation,D2,pimms_dae,saber,pearson_missing_mean,0.3,0.5",
            ]
        ),
        encoding="utf-8",
    )
    spec = tmp_path / "spec.yaml"
    spec.write_text(
        "\n".join(
            [
                "name: midterm_imp",
                "task: imputation",
                "metric: pearson_missing_mean",
                "datasets: [D1, D2]",
                "methods: [SABER, PIMMS-DAE]",
                "filters:",
                "  missing_fraction: 0.50",
                "ylim: [0.3, 0.8]",
                "ylabel: Pearson r",
            ]
        ),
        encoding="utf-8",
    )

    result = make_refined_figure(source, spec, tmp_path / "figs", formats=("png",))

    assert Path(result["files"][0]).exists()
    assert Path(result["figure_source"]).exists()
    assert Path(result["plot_source"]).exists()


def test_lfq_evaluator_inverse_transforms_zscore_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "lfq_zscore"
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True)
    labels_path = artifact_dir / "labels.npz"
    imputation_path = artifact_dir / "imputation.npz"
    conditions = np.asarray(["A", "A", "B", "B", "C", "C"], dtype=object)
    species = np.asarray(["HUMAN", "YEAST", "ECOLI"], dtype=object)
    protein_ids = np.asarray(["H1", "Y1", "E1"], dtype=object)
    x_raw = np.asarray(
        [
            [10.0, 3.0, 0.5],
            [10.2, 3.0, 0.5],
            [10.1, 2.0, 2.0],
            [10.0, 2.0, 2.0],
            [10.0, 0.3, 3.2],
            [10.1, 0.3, 3.2],
        ],
        dtype=np.float32,
    )
    center = np.nanmean(x_raw, axis=0).astype(np.float32)
    scale = np.nanstd(x_raw, axis=0).astype(np.float32)
    x_scaled = ((x_raw - center.reshape(1, -1)) / scale.reshape(1, -1)).astype(np.float32)
    np.savez_compressed(labels_path, conditions=conditions, species=species, protein_ids=protein_ids, x_raw=x_raw)
    np.savez_compressed(
        imputation_path,
        x_true=x_scaled,
        x_true_raw=x_raw,
        x_input=x_scaled,
        injected_missing_mask=np.zeros_like(x_scaled, dtype=np.uint8),
        observed_mask=np.ones_like(x_scaled),
        available_mask=np.ones_like(x_scaled),
        pred__saber=x_scaled,
        pred_raw__pimms_dae=x_raw,
        scale_mode=np.asarray("zscore"),
        scale_center=center,
        scale_scale=scale,
    )
    manifest = _write_manifest(run_dir, {"labels": labels_path, "imputation": imputation_path}, dataset="PXD071205_LFQ")
    result = evaluate_lfq_manifest(manifest, tmp_path / "eval_lfq_zscore")

    assert result["results"]["saber"]["n_protein_comparisons"] == 9
    assert math.isclose(result["results"]["saber"]["mean_abs_error"], result["results"]["raw_true"]["mean_abs_error"], rel_tol=1e-6)
    assert math.isclose(result["results"]["pimms_dae"]["mean_abs_error"], result["results"]["raw_true"]["mean_abs_error"], rel_tol=1e-6)


def test_external_manifest_import_adds_prediction_and_embedding(tmp_path: Path) -> None:
    run_dir = tmp_path / "external"
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True)
    labels_path = artifact_dir / "labels.npz"
    imputation_path = artifact_dir / "imputation.npz"
    embeddings_path = artifact_dir / "embeddings.npz"
    x = np.asarray([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]], dtype=np.float32)
    np.savez_compressed(
        labels_path,
        batches=np.asarray([0, 0, 1, 1]),
        cell_types=np.asarray([0, 1, 0, 1]),
        protein_ids=np.asarray(["P1", "P2", "P3"]),
    )
    np.savez_compressed(
        imputation_path,
        x_true=x,
        x_input=x,
        injected_missing_mask=np.zeros_like(x, dtype=np.uint8),
        observed_mask=np.ones_like(x),
        available_mask=np.ones_like(x),
        pred__saber=x,
    )
    np.savez_compressed(embeddings_path, z__saber=np.ones((4, 2), dtype=np.float32))
    manifest = _write_manifest(run_dir, {"labels": labels_path, "imputation": imputation_path, "embeddings": embeddings_path})
    pred_csv = tmp_path / "pimms.csv"
    emb_csv = tmp_path / "scprotein.csv"
    np.savetxt(pred_csv, x + 1.0, delimiter=",")
    np.savetxt(emb_csv, np.arange(12, dtype=np.float32).reshape(4, 3), delimiter=",")

    run_external_for_manifest(manifest, "pimms_dae", "imputation", tmp_path / "work_imp", import_csv=pred_csv)
    run_external_for_manifest(manifest, "scprotein", "cell_embedding", tmp_path / "work_emb", import_csv=emb_csv)

    with np.load(imputation_path) as data:
        assert "pred__pimms_dae" in data.files
        assert data["pred__pimms_dae"].shape == x.shape
    with np.load(embeddings_path) as data:
        assert "z__scprotein" in data.files
        assert data["z__scprotein"].shape == (4, 3)


def test_external_manifest_dry_run_drops_all_nan_adapter_columns(tmp_path: Path, saber_env) -> None:
    run_dir = tmp_path / "external_all_nan"
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True)
    labels_path = artifact_dir / "labels.npz"
    imputation_path = artifact_dir / "imputation.npz"
    x = np.asarray([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]], dtype=np.float32)
    observed = np.ones_like(x, dtype=np.uint8)
    observed[:, 1] = 0
    np.savez_compressed(labels_path, protein_ids=np.asarray(["P1", "P2", "P3"]))
    np.savez_compressed(
        imputation_path,
        x_true=x,
        x_input=x,
        injected_missing_mask=(~observed.astype(bool)).astype(np.uint8),
        observed_mask=observed,
        available_mask=np.ones_like(x),
        pred__saber=x,
    )
    manifest = _write_manifest(run_dir, {"labels": labels_path, "imputation": imputation_path})

    result = run_external_for_manifest(
        manifest,
        "pimms_dae",
        "imputation",
        tmp_path / "work_all_nan",
        dry_run=True,
    )

    assert result["ok"]
    assert result["column_filter"]["n_dropped_all_nan_columns"] == 1
    assert result["column_filter"]["dropped_all_nan_columns"] == [1]
    written = np.loadtxt(tmp_path / "work_all_nan" / "X_external_input.csv", delimiter=",")
    assert written.shape == (4, 2)


def test_external_protein_embedding_import(tmp_path: Path) -> None:
    run_dir = tmp_path / "external_protein"
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True)
    labels_path = artifact_dir / "labels.npz"
    embeddings_path = artifact_dir / "embeddings.npz"
    np.savez_compressed(labels_path, protein_ids=np.asarray(["P1", "P2", "P3"]))
    np.savez_compressed(embeddings_path, protein_embeddings__saber=np.ones((3, 2), dtype=np.float32))
    manifest = _write_manifest(run_dir, {"labels": labels_path, "embeddings": embeddings_path})
    protein_csv = tmp_path / "omicsgmf.protein.csv"
    np.savetxt(protein_csv, np.arange(12, dtype=np.float32).reshape(4, 3), delimiter=",")

    add_external_protein_embedding_to_manifest(manifest, "omicsgmf", protein_csv)

    with np.load(embeddings_path) as data:
        assert "protein_embeddings__omicsgmf" in data.files
        assert data["protein_embeddings__omicsgmf"].shape == (3, 4)
