from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .artifacts import load_json, write_json
from .metrics import pearson
from .plots import apply_paper_style, require_matplotlib, save_figure


def _safe_filename(value: object) -> str:
    text = str(value)
    out = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)
    return out[:120] if out else "protein"


def _finite_fill_column_mean(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(arr)
    counts = finite.sum(axis=0, keepdims=True)
    sums = np.where(finite, arr, 0.0).sum(axis=0, keepdims=True, dtype=np.float64)
    means = np.zeros((1, arr.shape[1]), dtype=np.float32)
    np.divide(sums, counts, out=means, where=counts > 0)
    return np.where(finite, arr, means).astype(np.float32)


def _reduce_to_2d(x: np.ndarray, mode: str = "auto", seed: int = 42) -> tuple[np.ndarray, str]:
    arr = _finite_fill_column_mean(np.asarray(x, dtype=np.float32))
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D expression matrix, got shape={arr.shape}")
    if arr.shape[0] < 2:
        return np.zeros((arr.shape[0], 2), dtype=np.float32), "trivial"
    keep = np.nanstd(arr, axis=0) > 1e-8
    if int(keep.sum()) >= 2:
        arr = arr[:, keep]
    if arr.shape[1] < 2:
        arr = np.concatenate([arr, np.zeros((arr.shape[0], 1), dtype=np.float32)], axis=1)

    try:
        from sklearn.decomposition import PCA

        pca_dim = int(min(50, arr.shape[1], max(2, arr.shape[0] - 1)))
        if pca_dim >= 2:
            arr = PCA(n_components=pca_dim, random_state=int(seed)).fit_transform(arr).astype(np.float32)
    except Exception:
        pass

    requested = str(mode or "auto").lower()
    if requested in {"auto", "umap"}:
        try:
            import umap

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=int(max(5, min(30, arr.shape[0] - 1))),
                min_dist=0.3,
                metric="euclidean",
                random_state=int(seed),
            )
            return reducer.fit_transform(arr).astype(np.float32), "umap"
        except Exception:
            if requested == "umap":
                requested = "tsne"

    if requested in {"auto", "tsne"}:
        try:
            from sklearn.manifold import TSNE

            if arr.shape[0] <= 3:
                emb = np.zeros((arr.shape[0], 2), dtype=np.float32)
                emb[:, 0] = np.arange(arr.shape[0], dtype=np.float32)
                return emb, "tsne"
            perplexity = float(max(2, min(30, (arr.shape[0] - 1) // 3)))
            return TSNE(
                n_components=2,
                random_state=int(seed),
                init="pca",
                learning_rate="auto",
                perplexity=perplexity,
            ).fit_transform(arr).astype(np.float32), "tsne"
        except Exception:
            pass

    emb = np.zeros((arr.shape[0], 2), dtype=np.float32)
    emb[:, 0] = arr[:, 0]
    emb[:, 1] = arr[:, 1] if arr.shape[1] > 1 else 0.0
    return emb, "raw2d"


def _score_proteins(x_true: np.ndarray, x_pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    scores = np.full((x_true.shape[1],), np.nan, dtype=np.float32)
    eval_mask = np.asarray(mask).astype(bool)
    for j in range(x_true.shape[1]):
        keep = eval_mask[:, j] & np.isfinite(x_true[:, j]) & np.isfinite(x_pred[:, j])
        if int(keep.sum()) >= 2:
            value = pearson(x_true[keep, j], x_pred[keep, j])
            if value is not None:
                scores[j] = float(value)
    return scores


def _select_top_indices(scores: np.ndarray, top_k: int, candidates: Sequence[int] | None = None) -> list[int]:
    if int(top_k) <= 0:
        return []
    cand = np.asarray(candidates if candidates is not None else np.arange(scores.shape[0]), dtype=np.int64)
    cand = cand[(cand >= 0) & (cand < scores.shape[0])]
    values = scores[cand]
    finite = np.isfinite(values)
    cand = cand[finite]
    values = values[finite]
    if cand.size == 0:
        return []
    order = np.argsort(-values)
    return [int(i) for i in cand[order[: min(int(top_k), cand.size)]].tolist()]


def _draw_maps(
    outdir: Path,
    emb2: np.ndarray,
    protein_indices: Sequence[int],
    protein_ids: Sequence[str],
    x_true: np.ndarray,
    x_observed: np.ndarray,
    x_pred: np.ndarray,
    method: str,
    task: str,
    formats: Sequence[str],
    point_size: float,
    alpha: float,
) -> list[str]:
    plt = require_matplotlib()
    outdir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#D3D3D3")
    for idx in protein_indices:
        protein = protein_ids[idx] if idx < len(protein_ids) else f"protein_{idx}"
        true = np.asarray(x_true[:, idx], dtype=np.float32)
        obs = np.asarray(x_observed[:, idx], dtype=np.float32)
        pred = np.asarray(x_pred[:, idx], dtype=np.float32)
        finite_vals = np.concatenate([v[np.isfinite(v)] for v in [true, obs, pred]])
        if finite_vals.size == 0:
            continue
        vmin = float(np.nanpercentile(finite_vals, 1))
        vmax = float(np.nanpercentile(finite_vals, 99))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = float(np.nanmin(finite_vals))
            vmax = float(np.nanmax(finite_vals))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = 0.0, 1.0

        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharex=True, sharey=True)
        panels = [("True", true), ("Observed", obs), (f"Predicted ({method})", pred)]
        mappable = None
        for ax, (title, values) in zip(axes, panels):
            mappable = ax.scatter(
                emb2[:, 0],
                emb2[:, 1],
                c=np.ma.masked_invalid(values),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                s=float(point_size),
                alpha=float(alpha),
                linewidths=0,
            )
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        if mappable is not None:
            fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.75, pad=0.015)
        fig.suptitle(f"{task}: {protein}", y=1.05)
        saved.extend(save_figure(fig, outdir / f"{task}_{method}_{idx:04d}_{_safe_filename(protein)}.png", formats=formats))
    return saved


def make_expression_maps(
    manifest_path: str | Path,
    outdir: str | Path,
    task: str,
    method: str = "saber",
    top_k: int = 5,
    embedding: str = "observed",
    reducer: str = "auto",
    seed: int = 42,
    formats: Sequence[str] = ("png", "pdf"),
    point_size: float = 9.0,
    alpha: float = 0.9,
) -> dict[str, Any]:
    apply_paper_style()
    manifest_path = Path(manifest_path)
    manifest = load_json(manifest_path)
    labels_path = manifest.get("artifacts", {}).get("labels")
    if not labels_path:
        raise ValueError(f"Manifest has no artifacts.labels: {manifest_path}")
    with np.load(labels_path, allow_pickle=True) as labels:
        all_protein_ids = labels["protein_ids"].astype(str).tolist() if "protein_ids" in labels.files else []

    task_name = str(task)
    method = str(method)
    if task_name == "imputation":
        artifact_path = manifest.get("artifacts", {}).get("imputation")
        if not artifact_path:
            raise ValueError(f"Manifest has no artifacts.imputation: {manifest_path}")
        with np.load(artifact_path) as data:
            x_true = np.asarray(data["x_true"], dtype=np.float32)
            x_input = np.asarray(data["x_input"] if "x_input" in data.files else data["x_true"], dtype=np.float32)
            if "observed_mask" in data.files:
                observed_mask = np.asarray(data["observed_mask"], dtype=bool)
                x_observed = x_true.copy()
                x_observed[~observed_mask] = np.nan
            else:
                x_observed = x_input
            mask = np.asarray(data["injected_missing_mask"], dtype=bool)
            key = f"pred__{method}"
            if key not in data.files:
                raise ValueError(f"Prediction {key} is missing from {artifact_path}")
            x_pred = np.asarray(data[key], dtype=np.float32)
        coords_source = x_input
        protein_ids = all_protein_ids or [f"protein_{i}" for i in range(x_true.shape[1])]
        scores = _score_proteins(x_true, x_pred, mask)
        selected = _select_top_indices(scores, top_k)
    elif task_name in {"proteome_expansion", "expansion"}:
        task_name = "proteome_expansion"
        artifact_path = manifest.get("artifacts", {}).get("expansion")
        if not artifact_path:
            raise ValueError(f"Manifest has no artifacts.expansion: {manifest_path}")
        with np.load(artifact_path) as data:
            x_true = np.asarray(data["x_true_target"], dtype=np.float32)
            mask_key = "eval_target_mask" if "eval_target_mask" in data.files else "available_target_mask"
            mask = np.asarray(data[mask_key], dtype=bool)
            key = f"pred_target__{method}"
            if key not in data.files:
                raise ValueError(f"Prediction {key} is missing from {artifact_path}")
            x_pred = np.asarray(data[key], dtype=np.float32)
            x_train = np.asarray(data["x_true_train"], dtype=np.float32) if "x_true_train" in data.files else x_true
        with np.load(labels_path, allow_pickle=True) as labels:
            target_ids = labels["target_protein_ids"].astype(str).tolist() if "target_protein_ids" in labels.files else []
        x_observed = np.full_like(x_true, np.nan, dtype=np.float32)
        coords_source = x_train
        protein_ids = target_ids or [f"target_{i}" for i in range(x_true.shape[1])]
        scores = _score_proteins(x_true, x_pred, mask)
        selected = _select_top_indices(scores, top_k)
    else:
        raise ValueError("task must be 'imputation' or 'proteome_expansion'")

    if str(embedding) != "observed":
        raise ValueError("Only embedding='observed' is currently supported for expression maps")
    emb2, reducer_used = _reduce_to_2d(coords_source, mode=reducer, seed=seed)
    outdir = Path(outdir)
    files = _draw_maps(
        outdir=outdir,
        emb2=emb2,
        protein_indices=selected,
        protein_ids=protein_ids,
        x_true=x_true,
        x_observed=x_observed,
        x_pred=x_pred,
        method=method,
        task=task_name,
        formats=formats,
        point_size=point_size,
        alpha=alpha,
    )
    manifest_out = {
        "manifest": str(manifest_path.resolve()),
        "task": task_name,
        "method": method,
        "top_k": int(top_k),
        "embedding": "observed",
        "reducer": reducer_used,
        "selected_indices": selected,
        "files": files,
    }
    write_json(outdir / "expression_maps_manifest.json", manifest_out)
    return manifest_out
