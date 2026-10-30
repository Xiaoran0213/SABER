from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import __version__
from .artifacts import write_json
from .config import SaberConfig
from .data import DataBundle, load_data_bundle
from .expansion_references import compute_expansion_reference_predictions
from .model import SABER
from .train import predict_all, set_seed, train_model


def choose_expansion_targets(data: DataBundle, target_fraction: float, seed: int, min_target_observed: int = 2) -> np.ndarray:
    available_counts = np.asarray(data.available_mask, dtype=bool).sum(axis=0)
    candidates = np.where(available_counts >= int(min_target_observed))[0]
    if candidates.size < 2:
        raise ValueError("Not enough proteins with sufficient observed values for expansion holdout")
    n_targets = max(1, int(math.ceil(float(target_fraction) * data.x_true.shape[1])))
    n_targets = min(n_targets, candidates.size - 1)
    rng = np.random.RandomState(int(seed))
    return np.sort(rng.choice(candidates, size=n_targets, replace=False).astype(np.int64))


def choose_expansion_targets_clustered(
    data: DataBundle,
    target_fraction: float,
    seed: int,
    min_target_observed: int = 2,
    cosine_threshold: float = 0.10,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Hold out whole ESM-2 cosine-similarity clusters of proteins.

    Unlike :func:`choose_expansion_targets` (uniform random proteins), this holds
    out entire sequence-similarity clusters so that close sequence neighbours of
    the target proteins are also removed from training -- a leakage-controlled
    zero-shot expansion test.

    Clusters are built over ALL proteins from L2-normalized ESM-2 features via
    average-linkage hierarchical clustering at ``cosine_threshold`` cosine
    distance. Clusters are then sampled (seeded) and accumulated until the held-out
    protein count reaches the nominal ``ceil(target_fraction * n_proteins)``.
    Only candidate proteins (>= ``min_target_observed`` observed values) within
    selected clusters are scored as targets; the whole cluster still leaves
    training.

    Returns ``(target_idx, diagnostics)`` where diagnostics carries the realised
    holdout fraction and per-target nearest-training-protein cosine similarity
    (min / median / per-target array), used to plot performance vs leakage.
    """
    from scipy.cluster.hierarchy import average, fcluster
    from scipy.spatial.distance import pdist, squareform

    feats = np.asarray(data.protein_features, dtype=np.float64)
    n_proteins = int(data.x_true.shape[1])
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    feats_norm = feats / norms

    available_counts = np.asarray(data.available_mask, dtype=bool).sum(axis=0)
    candidates = np.where(available_counts >= int(min_target_observed))[0]
    if candidates.size < 2:
        raise ValueError("Not enough proteins with sufficient observed values for expansion holdout")

    condensed = pdist(feats_norm, metric="cosine")
    linkage = average(condensed)
    labels = fcluster(linkage, t=float(cosine_threshold), criterion="distance")

    cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)
    n_targets_nominal = max(1, int(math.ceil(float(target_fraction) * n_proteins)))

    rng = np.random.RandomState(int(seed))
    order = rng.permutation(cluster_ids.size)
    selected_clusters: list[int] = []
    cumulative = 0
    for ci in order:
        selected_clusters.append(int(cluster_ids[ci]))
        cumulative += int(cluster_sizes[ci])
        if cumulative >= n_targets_nominal:
            break
    selected_mask = np.isin(labels, np.asarray(selected_clusters, dtype=labels.dtype))
    # Score only candidates within selected clusters; whole cluster still leaves training.
    target_idx = np.sort(np.intersect1d(candidates, np.where(selected_mask)[0]).astype(np.int64))
    if target_idx.size < 1:
        # Degenerate: no candidate in the selected clusters -- fall back to candidates in them.
        target_idx = np.sort(np.where(selected_mask)[0].astype(np.int64))
    if target_idx.size >= n_proteins:
        # Never hold out everything; keep at least one training protein.
        target_idx = target_idx[: n_proteins - 1]

    train_idx = np.setdiff1d(np.arange(n_proteins, dtype=np.int64), target_idx, assume_unique=True)
    sim = feats_norm[target_idx] @ feats_norm[train_idx].T
    nearest_train_cos = sim.max(axis=1).astype(np.float32)
    # Tightness check: mean cosine between all target pairs (cluster holdout should be tighter than random).
    if target_idx.size >= 2:
        tgt_pair = pdist(feats_norm[target_idx], metric="cosine")
        mean_target_pair_cos = float(1.0 - np.mean(tgt_pair))
    else:
        mean_target_pair_cos = float("nan")

    diagnostics: dict[str, Any] = {
        "holdout_mode": "cluster",
        "cosine_threshold": float(cosine_threshold),
        "nominal_target_fraction": float(target_fraction),
        "realised_target_fraction": float(target_idx.size / n_proteins),
        "n_clusters_total": int(cluster_ids.size),
        "n_clusters_selected": int(len(selected_clusters)),
        "n_targets": int(target_idx.size),
        "nearest_train_cosine_min": float(np.min(nearest_train_cos)),
        "nearest_train_cosine_median": float(np.median(nearest_train_cos)),
        "nearest_train_cosine_mean": float(np.mean(nearest_train_cos)),
        "mean_target_pair_cosine": mean_target_pair_cos,
        "_nearest_train_cosine": nearest_train_cos.astype(np.float32),  # per-target array (popped into npz; not JSON)
    }
    return target_idx, diagnostics



def choose_expansion_targets_sequence_clustered(
    data: DataBundle,
    cluster_map_path: str,
    target_fraction: float,
    seed: int,
    min_target_observed: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Hold out whole MMseqs2/CD-HIT sequence-identity clusters of proteins.

    Leakage-controlled zero-shot holdout: clusters are defined by sequence identity
    (the field standard, UniRef-style), so holding out a whole cluster removes all
    close sequence homologs of the targets from training. ``cluster_map_path`` is a
    TSV with ``rep<TAB>member`` per line (mmseqs ``easy-cluster *_cluster.tsv``).

    Returns ``(target_idx, diagnostics)`` with the realised holdout fraction and the
    per-target nearest-training-protein ESM-2 cosine (continuous leakage axis).
    """
    cluster_of: dict[str, str] = {}
    with Path(cluster_map_path).open() as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                cluster_of[parts[1]] = parts[0]  # member -> rep

    feats = np.asarray(data.protein_features, dtype=np.float64)
    n_proteins = int(data.x_true.shape[1])
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    feats_norm = feats / norms

    protein_ids = [str(x) for x in data.protein_ids.tolist()]
    available_counts = np.asarray(data.available_mask, dtype=bool).sum(axis=0)
    candidates = np.where(available_counts >= int(min_target_observed))[0]
    if candidates.size < 2:
        raise ValueError("Not enough proteins with sufficient observed values for expansion holdout")

    groups: dict[str, list[int]] = {}
    unmapped = 0
    for ci in candidates:
        pid = protein_ids[int(ci)]
        rep = cluster_of.get(pid)
        if rep is None:
            unmapped += 1
            rep = f"__unmapped_{int(ci)}"  # treat unmapped as its own singleton cluster
        groups.setdefault(rep, []).append(int(ci))

    n_targets_nominal = max(1, int(math.ceil(float(target_fraction) * n_proteins)))
    rng = np.random.RandomState(int(seed))
    cluster_keys = list(groups.keys())
    order = rng.permutation(len(cluster_keys))
    cumulative = 0
    selected: list[int] = []
    for oi in order:
        members = groups[cluster_keys[int(oi)]]
        selected.extend(members)
        cumulative += len(members)
        if cumulative >= n_targets_nominal:
            break
    target_idx = np.sort(np.asarray(selected, dtype=np.int64))
    if target_idx.size >= n_proteins:
        target_idx = target_idx[: n_proteins - 1]

    train_idx = np.setdiff1d(np.arange(n_proteins, dtype=np.int64), target_idx, assume_unique=True)
    sim = feats_norm[target_idx] @ feats_norm[train_idx].T
    nearest_train_cos = sim.max(axis=1).astype(np.float32)

    diagnostics: dict[str, Any] = {
        "holdout_mode": "sequence",
        "cluster_map_path": str(cluster_map_path),
        "n_clusters_total": len(groups),
        "n_unmapped_candidates": int(unmapped),
        "nominal_target_fraction": float(target_fraction),
        "realised_target_fraction": float(target_idx.size / n_proteins),
        "n_targets": int(target_idx.size),
        "nearest_train_cosine_min": float(np.min(nearest_train_cos)),
        "nearest_train_cosine_median": float(np.median(nearest_train_cos)),
        "nearest_train_cosine_mean": float(np.mean(nearest_train_cos)),
        "_nearest_train_cosine": nearest_train_cos,
    }
    return target_idx, diagnostics


def choose_expansion_targets_explicit(
    data: DataBundle,
    target_protein_ids: list[str],
    min_target_observed: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Hold out an explicit, user-specified set of proteins by ID.

    Case-study holdout: hold out a biologically defined marker set (e.g. liver
    zonation markers) and ask whether SABER recovers their cell-wise abundance /
    spatial pattern from sequence alone. Proteins are matched to the data axis by
    ``data.protein_ids`` (UniProt AC). Requested IDs absent from the data, or with
    fewer than ``min_target_observed`` observed cells, are reported but skipped.

    Returns ``(target_idx, diagnostics)``.
    """
    protein_ids = [str(x) for x in data.protein_ids.tolist()]
    id_to_idx = {pid: i for i, pid in enumerate(protein_ids)}
    available_counts = np.asarray(data.available_mask, dtype=bool).sum(axis=0)
    n_proteins = int(data.x_true.shape[1])

    wanted = [str(x) for x in target_protein_ids]
    target_idx: list[int] = []
    skipped_not_in_data: list[str] = []
    skipped_low_obs: list[str] = []
    seen: set[int] = set()
    for pid in dict.fromkeys(wanted):  # preserve order, dedup
        if pid not in id_to_idx:
            skipped_not_in_data.append(pid)
            continue
        j = int(id_to_idx[pid])
        if j in seen:
            continue
        seen.add(j)
        if int(available_counts[j]) < int(min_target_observed):
            skipped_low_obs.append(pid)
            continue
        target_idx.append(j)
    target_idx_arr = np.sort(np.asarray(target_idx, dtype=np.int64))

    diagnostics: dict[str, Any] = {
        "holdout_mode": "explicit",
        "n_requested": len(set(wanted)),
        "n_in_data": len(set(wanted) & set(id_to_idx)),
        "n_targets": int(target_idx_arr.size),
        "realised_target_fraction": float(target_idx_arr.size / n_proteins) if n_proteins else 0.0,
        "n_skipped_not_in_data": int(len(skipped_not_in_data)),
        "n_skipped_low_obs": int(len(skipped_low_obs)),
        "skipped_not_in_data_preview": skipped_not_in_data[:25],
        "skipped_low_obs_preview": skipped_low_obs[:25],
    }
    return target_idx_arr, diagnostics


def subset_data_bundle(data: DataBundle, protein_indices: np.ndarray) -> DataBundle:
    idx = np.asarray(protein_indices, dtype=np.int64)
    scale_info = dict(data.scale_info)
    if np.asarray(scale_info.get("center", [])).size:
        scale_info["center"] = np.asarray(scale_info["center"], dtype=np.float32)[idx]
    if np.asarray(scale_info.get("scale", [])).size:
        scale_info["scale"] = np.asarray(scale_info["scale"], dtype=np.float32)[idx]
    return DataBundle(
        x_raw=data.x_raw[:, idx],
        x_true=data.x_true[:, idx],
        x_input=data.x_input[:, idx],
        observed_mask=data.observed_mask[:, idx],
        injected_missing_mask=data.injected_missing_mask[:, idx],
        available_mask=data.available_mask[:, idx],
        batches=data.batches,
        cell_types=data.cell_types,
        protein_ids=data.protein_ids[idx],
        protein_features=data.protein_features[idx],
        missing_info=dict(data.missing_info),
        feature_cache_info=dict(data.feature_cache_info),
        scale_info=scale_info,
    )


def _build_model(cfg: SaberConfig, data: DataBundle) -> SABER:
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


def _predict_heldout(
    model: SABER,
    train_input: np.ndarray,
    heldout_features: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device_obj = torch.device(device)
    model.eval()
    model.to(device_obj)
    x = torch.as_tensor(train_input, dtype=torch.float32, device=device_obj)
    features = torch.as_tensor(heldout_features, dtype=torch.float32, device=device_obj)
    with torch.no_grad():
        z = model.encode_cells(x)
        e = model.encode_proteins(features)
        z_use = z @ model.W if model.W is not None else z
        pred = z_use @ e.t()
    return (
        pred.detach().cpu().numpy().astype(np.float32),
        z.detach().cpu().numpy().astype(np.float32),
        e.detach().cpu().numpy().astype(np.float32),
    )


def train_expansion(cfg: SaberConfig, command: str = "") -> Path:
    full_data = load_data_bundle(cfg.data, cfg.missing, cfg.model, seed=cfg.project.seed)
    holdout_diag: dict[str, Any] | None = None
    if cfg.expansion.holdout_mode == "cluster":
        target_idx, holdout_diag = choose_expansion_targets_clustered(
            full_data,
            target_fraction=cfg.expansion.target_fraction,
            seed=cfg.expansion.seed,
            min_target_observed=cfg.expansion.min_target_observed,
            cosine_threshold=cfg.expansion.cluster_cosine_threshold,
        )
    elif cfg.expansion.holdout_mode == "sequence":
        target_idx, holdout_diag = choose_expansion_targets_sequence_clustered(
            full_data,
            cluster_map_path=cfg.expansion.cluster_map_path,
            target_fraction=cfg.expansion.target_fraction,
            seed=cfg.expansion.seed,
            min_target_observed=cfg.expansion.min_target_observed,
        )
    elif cfg.expansion.holdout_mode == "explicit":
        wanted_ids = list(cfg.expansion.target_protein_ids)
        if cfg.expansion.target_protein_ids_path:
            with Path(cfg.expansion.target_protein_ids_path).open() as f:
                wanted_ids += [ln.strip() for ln in f if ln.strip()]
        target_idx, holdout_diag = choose_expansion_targets_explicit(
            full_data,
            target_protein_ids=wanted_ids,
            min_target_observed=cfg.expansion.min_target_observed,
        )
    else:
        target_idx = choose_expansion_targets(
            full_data,
            target_fraction=cfg.expansion.target_fraction,
            seed=cfg.expansion.seed,
            min_target_observed=cfg.expansion.min_target_observed,
        )
    all_idx = np.arange(full_data.x_true.shape[1], dtype=np.int64)
    train_idx = np.setdiff1d(all_idx, target_idx, assume_unique=True)
    train_data = subset_data_bundle(full_data, train_idx)
    set_seed(cfg.project.seed)
    model = _build_model(cfg, train_data)
    result = train_model(model, train_data, cfg.training, seed=cfg.project.seed)
    train_pred, train_z, train_protein_embeddings = predict_all(model, train_data.x_input, result.device)
    heldout_pred, _, heldout_embeddings = _predict_heldout(
        model,
        train_data.x_input,
        full_data.protein_features[target_idx],
        result.device,
    )

    from .paths import resolve_outdir
    run_dir = resolve_outdir(cfg.project.outdir) / cfg.project.name
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg.to_dict(),
            "train_protein_ids": [str(x) for x in train_data.protein_ids.tolist()],
            "target_protein_ids": [str(x) for x in full_data.protein_ids[target_idx].tolist()],
        },
        model_path,
    )

    labels_path = artifact_dir / "labels.npz"
    expansion_path = artifact_dir / "expansion.npz"
    embeddings_path = artifact_dir / "embeddings.npz"
    history_path = artifact_dir / "train_history.json"
    references_status_path = artifact_dir / "references_status.json"

    np.savez_compressed(
        labels_path,
        batches=np.asarray(full_data.batches, dtype=np.int64),
        cell_types=np.asarray(full_data.cell_types, dtype=np.int64),
        protein_ids=np.asarray(full_data.protein_ids, dtype=str),
        train_protein_ids=np.asarray(full_data.protein_ids[train_idx], dtype=str),
        target_protein_ids=np.asarray(full_data.protein_ids[target_idx], dtype=str),
        train_protein_indices=train_idx,
        target_protein_indices=target_idx,
        x_raw=np.asarray(full_data.x_raw, dtype=np.float32),
        scale_mode=np.asarray(str(full_data.scale_info.get("mode", "none"))),
        scale_center=np.asarray(full_data.scale_info.get("center", []), dtype=np.float32),
        scale_scale=np.asarray(full_data.scale_info.get("scale", []), dtype=np.float32),
    )
    reference_predictions, reference_status = compute_expansion_reference_predictions(
        references=cfg.expansion.references,
        plm_train=full_data.protein_features[train_idx],
        plm_target=full_data.protein_features[target_idx],
        x_train=full_data.x_true[:, train_idx],
        ridge_lambda=cfg.expansion.ridge_lambda,
        knn_ks=cfg.expansion.knn_ks,
    )

    expansion_payload: dict[str, Any] = {
        "x_true_target": np.asarray(full_data.x_true[:, target_idx], dtype=np.float32),
        "available_target_mask": np.asarray(full_data.available_mask[:, target_idx], dtype=np.uint8),
        "eval_target_mask": np.asarray(full_data.observed_mask[:, target_idx], dtype=np.uint8),
        "pred_target__saber": np.asarray(heldout_pred, dtype=np.float32),
        "x_true_train": np.asarray(full_data.x_true[:, train_idx], dtype=np.float32),
        "pred_train__saber": np.asarray(train_pred, dtype=np.float32),
        "scale_mode": np.asarray(str(full_data.scale_info.get("mode", "none"))),
        "scale_center": np.asarray(full_data.scale_info.get("center", []), dtype=np.float32),
        "scale_scale": np.asarray(full_data.scale_info.get("scale", []), dtype=np.float32),
    }
    expansion_payload.update({f"pred_target__{method}": pred for method, pred in reference_predictions.items()})
    if holdout_diag is not None:
        # Per-target nearest-training-protein cosine (for Pearson-vs-leakage scatter);
        # only the cluster/sequence holdout modes compute this.
        nc = holdout_diag.pop("_nearest_train_cosine", None)
        if nc is not None:
            expansion_payload["nearest_train_cosine"] = nc
    np.savez_compressed(expansion_path, **expansion_payload)
    np.savez_compressed(
        embeddings_path,
        z__saber=np.asarray(train_z, dtype=np.float32),
        protein_embeddings_train__saber=np.asarray(train_protein_embeddings, dtype=np.float32),
        protein_embeddings_target__saber=np.asarray(heldout_embeddings, dtype=np.float32),
    )
    write_json(history_path, {"history": result.history})

    manifest_path = run_dir / "artifact_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "task": "proteome_expansion",
        "method_name": "SABER",
        "package_version": __version__,
        "command": command,
        "device": result.device,
        "resources": getattr(result, "resources", {}) or {},
        "config": cfg.to_dict(),
        "data": {
            "n_cells": int(full_data.x_true.shape[0]),
            "n_proteins": int(full_data.x_true.shape[1]),
            "n_train_proteins": int(train_idx.shape[0]),
            "n_target_proteins": int(target_idx.shape[0]),
            "missing_info": full_data.missing_info,
            "feature_cache_info": full_data.feature_cache_info,
            "scale_info": full_data.scale_info,
        },
        "expansion": {
            "target_fraction": float(cfg.expansion.target_fraction),
            "seed": int(cfg.expansion.seed),
            "min_target_observed": int(cfg.expansion.min_target_observed),
            "target_count_rule": "ceil_fraction_of_protein_universe",
            "rng_method": "numpy.random.RandomState",
            "holdout_mode": str(cfg.expansion.holdout_mode),
            **(holdout_diag or {}),
        },
        "artifacts": {
            "labels": str(labels_path.resolve()),
            "expansion": str(expansion_path.resolve()),
            "embeddings": str(embeddings_path.resolve()),
            "history": str(history_path.resolve()),
            "model": str(model_path.resolve()),
            "references_status": str(references_status_path.resolve()),
        },
        "references": {
            "requested": list(cfg.expansion.references),
            "ridge_lambda": float(cfg.expansion.ridge_lambda),
            "knn_ks": list(cfg.expansion.knn_ks),
            "written_keys": sorted(reference_predictions.keys()),
            "status": reference_status,
        },
        "evaluation_status": {"expansion": "pending"},
    }
    write_json(
        references_status_path,
        {
            "requested": list(cfg.expansion.references),
            "ridge_lambda": float(cfg.expansion.ridge_lambda),
            "knn_ks": list(cfg.expansion.knn_ks),
            "written_keys": sorted(reference_predictions.keys()),
            "status": reference_status,
        },
    )
    write_json(manifest_path, manifest)
    return manifest_path
