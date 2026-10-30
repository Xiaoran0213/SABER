from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from saber.artifacts import load_json, write_json
from saber.feature_cache import FeatureCacheError, load_protein_feature_cache
from saber.metrics import compute_protein_label_metrics, optional_float

from .common import metric_rows, write_rows_csv


def _read_label_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, str]] = []
        for row in reader:
            protein_id = str(row.get("protein_id", row.get("protein", row.get("id", "")))).strip()
            label_id = str(row.get("label_id", row.get("label", row.get("term_id", "")))).strip()
            label_type = str(row.get("label_type", row.get("type", ""))).strip()
            if protein_id and label_id:
                rows.append({"protein_id": protein_id, "label_id": label_id, "label_type": label_type})
        return rows


def _effective_rank(embedding: np.ndarray) -> float | None:
    x = np.asarray(embedding, dtype=np.float32)
    if x.ndim != 2 or min(x.shape) < 2:
        return None
    x = x - np.nanmean(x, axis=0, keepdims=True)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        s = np.linalg.svd(x, compute_uv=False)
    except Exception:
        return None
    total = float(np.sum(s))
    if total <= 1e-12:
        return None
    p = s / total
    entropy = -float(np.sum(p * np.log(np.clip(p, 1e-12, None))))
    return optional_float(np.exp(entropy))


def _protein_embeddings_from_artifact(
    labels: np.lib.npyio.NpzFile,
    embeddings: np.lib.npyio.NpzFile,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if "protein_ids" in labels.files:
        protein_ids = labels["protein_ids"].astype(str)
    elif "train_protein_ids" in labels.files:
        protein_ids = labels["train_protein_ids"].astype(str)
    else:
        protein_ids = np.asarray([], dtype=str)

    for key in embeddings.files:
        if key.startswith("protein_embeddings__"):
            method = key.removeprefix("protein_embeddings__")
            out[method] = (protein_ids, np.asarray(embeddings[key], dtype=np.float32))
        elif key.startswith("protein__"):
            method = key.removeprefix("protein__")
            out[method] = (protein_ids, np.asarray(embeddings[key], dtype=np.float32))
        elif key.startswith("protein_embeddings_train__"):
            method = key.removeprefix("protein_embeddings_train__")
            ids = labels["train_protein_ids"].astype(str) if "train_protein_ids" in labels.files else protein_ids
            out[f"{method}_train"] = (ids, np.asarray(embeddings[key], dtype=np.float32))
        elif key.startswith("protein_embeddings_target__"):
            method = key.removeprefix("protein_embeddings_target__")
            ids = labels["target_protein_ids"].astype(str) if "target_protein_ids" in labels.files else protein_ids
            out[f"{method}_target"] = (ids, np.asarray(embeddings[key], dtype=np.float32))
    return out


def _raw_plm_embedding_from_manifest(
    manifest: dict[str, Any],
    labels: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, np.ndarray] | None:
    config = manifest.get("config", {}) or {}
    data_cfg = config.get("data", {}) or {}
    model_cfg = config.get("model", {}) or {}
    feature_path = str(data_cfg.get("protein_features_path", "") or "")
    if not feature_path:
        return None
    protein_ids: np.ndarray
    if "protein_ids" in labels.files:
        protein_ids = labels["protein_ids"].astype(str)
    elif "train_protein_ids" in labels.files:
        protein_ids = labels["train_protein_ids"].astype(str)
    else:
        return None
    if protein_ids.size == 0:
        return None
    ids_path = str(data_cfg.get("protein_features_ids_path", "") or "")
    expected_dim = model_cfg.get("protein_feature_dim", None)
    try:
        cache = load_protein_feature_cache(
            feature_path,
            protein_ids=protein_ids,
            expected_dim=int(expected_dim) if expected_dim not in (None, "") else None,
            cache_protein_ids_path=ids_path,
        )
    except (FileNotFoundError, FeatureCacheError, ValueError):
        return None
    return cache.protein_ids.astype(str), np.asarray(cache.features, dtype=np.float32)


def _annotation_status(ext_dir: str | Path) -> dict[str, Any]:
    if not ext_dir:
        return {"enabled": False}
    path = Path(ext_dir)
    files = {
        "string_interactions": path / "string_interactions.csv",
        "reactome_pathways": path / "reactome_pathways.json",
        "go_annotations": path / "go_annotations.json",
        "annotation_summary": path / "annotation_summary.json",
    }
    return {
        "enabled": True,
        "ext_dir": str(path),
        "files": {name: str(file) for name, file in files.items()},
        "available": {name: file.exists() for name, file in files.items()},
        "layer2_ready": bool(files["string_interactions"].exists() and files["reactome_pathways"].exists() and files["go_annotations"].exists()),
        "layer3_ready": bool(files["reactome_pathways"].exists()),
    }


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12, None)


def _knn_indices(emb: np.ndarray, k: int) -> np.ndarray:
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception:
        return np.zeros((emb.shape[0], 0), dtype=np.int64)
    x = _normalize_rows(emb)
    kk = int(max(1, min(int(k), x.shape[0] - 1)))
    if kk <= 0:
        return np.zeros((x.shape[0], 0), dtype=np.int64)
    nn = NearestNeighbors(n_neighbors=kk + 1, metric="cosine")
    nn.fit(x)
    return nn.kneighbors(x, return_distance=False)[:, 1:]


def _load_string_interactions(ext_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(ext_dir) / "string_interactions.csv"
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            p1 = str(row.get("protein1", "")).strip()
            p2 = str(row.get("protein2", "")).strip()
            if not p1 or not p2:
                continue
            rows.append({"protein1": p1, "protein2": p2, "score": float(row.get("score", 0.0) or 0.0)})
        return rows


def _load_json_annotations(ext_dir: str | Path, filename: str) -> list[dict[str, Any]]:
    path = Path(ext_dir) / filename
    if not path.exists():
        return []
    with path.open("r") as f:
        payload = json.load(f)
    return payload if isinstance(payload, list) else []


def _evaluate_string(
    emb: np.ndarray,
    protein_ids: np.ndarray,
    interactions: list[dict[str, Any]],
    ks: Sequence[int],
    seed: int,
) -> dict[str, float]:
    if not interactions or emb.shape[0] < 10:
        return {}
    ids = [str(x) for x in protein_ids.tolist()]
    pid_to_idx = {pid: i for i, pid in enumerate(ids)}
    edges = [
        (pid_to_idx[row["protein1"]], pid_to_idx[row["protein2"]])
        for row in interactions
        if row.get("protein1") in pid_to_idx and row.get("protein2") in pid_to_idx
    ]
    if len(edges) < 10:
        return {}
    x = _normalize_rows(emb)
    pos_sims = np.asarray([float(x[i] @ x[j]) for i, j in edges], dtype=np.float32)
    edge_set = {(min(i, j), max(i, j)) for i, j in edges}
    rng = np.random.RandomState(int(seed))
    n_neg = min(len(edges) * 2, 50000)
    neg_sims: list[float] = []
    attempts = 0
    while len(neg_sims) < n_neg and attempts < n_neg * 10:
        attempts += 1
        i, j = rng.randint(0, emb.shape[0], size=2)
        if i == j or (min(i, j), max(i, j)) in edge_set:
            continue
        neg_sims.append(float(x[i] @ x[j]))
    neg = np.asarray(neg_sims, dtype=np.float32)
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        y_true = np.concatenate([np.ones(pos_sims.size), np.zeros(neg.size)])
        y_score = np.concatenate([pos_sims, neg])
        auroc = float(roc_auc_score(y_true, y_score))
        auprc = float(average_precision_score(y_true, y_score))
    except Exception:
        auroc = float("nan")
        auprc = float("nan")

    protein_edges: dict[int, set[int]] = defaultdict(set)
    for i, j in edges:
        protein_edges[i].add(j)
        protein_edges[j].add(i)
    nbr = _knn_indices(x, max(ks))
    results = {
        "string.edge_auroc": auroc,
        "string.edge_auprc": auprc,
        "string.intra_edge_sim_mean": float(np.mean(pos_sims)),
        "string.random_sim_mean": float(np.mean(neg)) if neg.size else float("nan"),
        "string.n_edges": float(len(edges)),
        "string.n_proteins_with_edges": float(len(protein_edges)),
    }
    for k in ks:
        kk = min(int(k), nbr.shape[1])
        vals = []
        for idx, partners in protein_edges.items():
            total = min(len(partners), kk)
            if total > 0 and kk > 0:
                vals.append(float(len(partners & set(nbr[idx, :kk].tolist())) / total))
        results[f"string.recall_at_{k}"] = float(np.mean(vals)) if vals else float("nan")
    return results


def _term_key(annotation: dict[str, Any]) -> tuple[str, str]:
    term_id = str(annotation.get("pathway_id", annotation.get("go_id", "")))
    term_name = str(annotation.get("pathway_name", annotation.get("go_name", term_id)))
    return term_id, term_name


def _compute_global_clustering(x_valid: np.ndarray, valid_sets: list[set[str]], seed: int) -> dict[str, float]:
    term_sizes = Counter(term for labels in valid_sets for term in labels)
    labels = [sorted(s, key=lambda term: (term_sizes.get(term, 10**9), term))[0] if s else "" for s in valid_sets]
    valid = [i for i, label in enumerate(labels) if label]
    if len(valid) < 10:
        return {}
    y = np.asarray([labels[i] for i in valid])
    x = x_valid[valid]
    if np.unique(y).size < 2:
        return {}
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

        n_clusters = int(max(2, min(np.unique(y).size, 50, x.shape[0] - 1)))
        pred = KMeans(n_clusters=n_clusters, random_state=int(seed), n_init=10).fit_predict(x)
        out = {
            "global_ari": float(adjusted_rand_score(y, pred)),
            "global_nmi": float(normalized_mutual_info_score(y, pred)),
        }
        try:
            out["global_silhouette"] = float(silhouette_score(x, y))
        except Exception:
            out["global_silhouette"] = float("nan")
        return out
    except Exception:
        return {}


def _compute_intra_inter_distance(
    x: np.ndarray,
    term_to_members: dict[str, tuple[str, set[int]]],
    max_terms: int,
    seed: int,
) -> dict[str, float]:
    terms = list(term_to_members.items())
    if len(terms) > int(max_terms):
        rng = np.random.RandomState(int(seed))
        terms = [terms[i] for i in rng.choice(len(terms), size=int(max_terms), replace=False)]
    intra: list[float] = []
    member_sets: list[list[int]] = []
    for _term, (_name, members_set) in terms:
        members = list(members_set)
        if len(members) < 2:
            continue
        member_sets.append(members)
        sim = x[members] @ x[members].T
        intra.extend(1.0 - float(sim[i, j]) for i in range(len(members)) for j in range(i + 1, len(members)))
    if len(member_sets) < 2 or not intra:
        return {}
    rng = np.random.RandomState(int(seed))
    inter: list[float] = []
    for _ in range(min(len(intra) * 2, 20000)):
        t1, t2 = rng.choice(len(member_sets), size=2, replace=False)
        i = int(rng.choice(member_sets[t1]))
        j = int(rng.choice(member_sets[t2]))
        inter.append(1.0 - float(x[i] @ x[j]))
    if not inter:
        return {}
    return {
        "intra_distance_mean": float(np.mean(intra)),
        "inter_distance_mean": float(np.mean(inter)),
        "intra_inter_ratio": float(np.mean(intra) / max(1e-8, np.mean(inter))),
    }


def _evaluate_pathway_coherence(
    emb: np.ndarray,
    protein_ids: np.ndarray,
    annotations: list[dict[str, Any]],
    source_name: str,
    ks: Sequence[int],
    seed: int,
    min_proteins_per_term: int = 3,
    max_terms_for_distance: int = 200,
) -> dict[str, float]:
    if not annotations or emb.shape[0] < 10:
        return {}
    ids = [str(x) for x in protein_ids.tolist()]
    pid_to_idx = {pid: i for i, pid in enumerate(ids)}
    term_to_members: dict[str, tuple[str, set[int]]] = {}
    for ann in annotations:
        term_id, term_name = _term_key(ann)
        members = {pid_to_idx[p] for p in ann.get("proteins", []) if p in pid_to_idx}
        if term_id and len(members) >= int(min_proteins_per_term):
            term_to_members[term_id] = (term_name, members)
    if not term_to_members:
        return {}

    label_sets: list[set[str]] = [set() for _ in ids]
    for term_id, (_name, members) in term_to_members.items():
        for idx in members:
            label_sets[idx].add(term_id)
    valid_idx = [i for i, labels in enumerate(label_sets) if labels]
    if len(valid_idx) < 10:
        return {}

    x = _normalize_rows(emb)
    x_valid = x[valid_idx]
    valid_sets = [label_sets[i] for i in valid_idx]
    nbr = _knn_indices(x_valid, max(ks))
    results: dict[str, float] = {
        f"{source_name}.n_terms": float(len(term_to_members)),
        f"{source_name}.n_proteins_with_labels": float(len(valid_idx)),
    }
    for k in ks:
        kk = min(int(k), nbr.shape[1])
        hit_vals: list[float] = []
        jac_vals: list[float] = []
        for i in range(x_valid.shape[0]):
            q = valid_sets[i]
            if not q or kk <= 0:
                continue
            hits = 0.0
            jaccard = 0.0
            for j in nbr[i, :kk]:
                s = valid_sets[int(j)]
                inter = len(q & s)
                union = len(q | s)
                hits += float(inter > 0)
                jaccard += float(inter / max(1, union))
            hit_vals.append(hits / kk)
            jac_vals.append(jaccard / kk)
        results[f"{source_name}.pathway_hit_at_{k}"] = float(np.mean(hit_vals)) if hit_vals else float("nan")
        results[f"{source_name}.jaccard_at_{k}"] = float(np.mean(jac_vals)) if jac_vals else float("nan")
    results.update({f"{source_name}.{k}": v for k, v in _compute_global_clustering(x_valid, valid_sets, seed).items()})
    results.update({f"{source_name}.{k}": v for k, v in _compute_intra_inter_distance(x, term_to_members, max_terms_for_distance, seed).items()})
    return results


def _build_pathway_index(
    annotations: list[dict[str, Any]],
    protein_ids: np.ndarray,
    min_proteins: int,
    max_proteins: int,
) -> dict[str, dict[str, Any]]:
    ids = [str(x) for x in protein_ids.tolist()]
    pid_to_idx = {pid: i for i, pid in enumerate(ids)}
    result: dict[str, dict[str, Any]] = {}
    for pathway in annotations:
        pathway_id, pathway_name = _term_key(pathway)
        members = [pid_to_idx[p] for p in pathway.get("proteins", []) if p in pid_to_idx]
        if pathway_id and int(min_proteins) <= len(members) <= int(max_proteins):
            result[pathway_id] = {"name": pathway_name, "member_indices": np.asarray(members, dtype=np.int64), "n_members": len(members)}
    return result


def _compute_discrimination(
    emb: np.ndarray,
    pathway_index: dict[str, dict[str, Any]],
    seed: int,
    n_random_pairs: int = 10000,
) -> dict[str, float]:
    if not pathway_index or emb.shape[0] < 10:
        return {}
    x = _normalize_rows(emb)
    rng = np.random.RandomState(int(seed))
    rand = []
    for _ in range(int(n_random_pairs)):
        i, j = rng.randint(0, x.shape[0], size=2)
        if i != j:
            rand.append(float(x[i] @ x[j]))
    baseline = float(np.mean(rand)) if rand else 0.0
    gaps = []
    for info in pathway_index.values():
        members = np.asarray(info["member_indices"], dtype=np.int64)
        if members.size < 2:
            continue
        sim = x[members] @ x[members].T
        intra = [float(sim[i, j]) for i in range(members.size) for j in range(i + 1, members.size)]
        if intra:
            gaps.append(float(np.mean(intra)) - baseline)
    return {
        "reactome.layer3_n_pathways": float(len(pathway_index)),
        "reactome.layer3_random_sim": baseline,
        "reactome.layer3_mean_gap": float(np.mean(gaps)) if gaps else float("nan"),
        "reactome.layer3_median_gap": float(np.median(gaps)) if gaps else float("nan"),
    }


def _evaluate_layer2_layer3(
    protein_embeddings: dict[str, tuple[np.ndarray, np.ndarray]],
    ext_dir: str | Path,
    ks: tuple[int, ...],
    seed: int,
    min_proteins: int,
    max_proteins: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    string_data = _load_string_interactions(ext_dir)
    reactome_data = _load_json_annotations(ext_dir, "reactome_pathways.json")
    go_data = _load_json_annotations(ext_dir, "go_annotations.json")
    coverage = {
        "ext_dir": str(Path(ext_dir).resolve()),
        "string_interactions": int(len(string_data)),
        "reactome_pathways": int(len(reactome_data)),
        "go_terms": int(len(go_data)),
        "min_proteins": int(min_proteins),
        "max_proteins": int(max_proteins),
    }
    results: dict[str, dict[str, Any]] = {}
    for method, (protein_ids, emb) in sorted(protein_embeddings.items()):
        if emb.ndim != 2 or emb.shape[0] != protein_ids.shape[0]:
            continue
        metrics: dict[str, Any] = {}
        metrics.update(_evaluate_string(emb, protein_ids, string_data, ks, seed=seed))
        metrics.update(_evaluate_pathway_coherence(emb, protein_ids, reactome_data, "reactome", ks, seed=seed))
        metrics.update(_evaluate_pathway_coherence(emb, protein_ids, go_data, "go", ks, seed=seed))
        pathway_index = _build_pathway_index(reactome_data, protein_ids, min_proteins=min_proteins, max_proteins=max_proteins)
        metrics.update(_compute_discrimination(emb, pathway_index, seed=seed))
        results[method] = metrics
    return results, coverage


def _dimension_matched_controls(
    raw_plm: tuple[np.ndarray, np.ndarray],
    x_raw: np.ndarray | None,
    dim: int = 16,
    seed: int = 42,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Dimension-matched control protein embeddings (dimensionality control).

    The protein-embedding comparison opposes SABER's low-dimensional embedding
    to the raw 1,280-dimensional ESM-2 prior, which confounds dimensionality
    with expression conditioning. These controls hold the dimension fixed at
    the SABER latent size so the comparison isolates what expression
    conditioning adds beyond compression:

      esm2_pca16 - PCA of the raw ESM-2 features to `dim` components
      esm2_rp16  - seeded Gaussian random projection of the raw ESM-2 features
      expr_pca16 - PCA of the protein-wise observed expression profiles
                   (x_raw transposed, proteins as samples; missing entries
                   filled with the protein-wise observed mean, the same
                   convention as the model input)

    Deterministic: PCA uses random_state=seed, the projection uses
    RandomState(seed). Only meaningful when raw PLM features exist (the
    sequence-informed runs), matching how raw_plm itself is added.
    """
    from sklearn.decomposition import PCA

    protein_ids, features = raw_plm
    features = np.asarray(features, dtype=np.float32)
    n_components = max(2, min(int(dim), features.shape[1], features.shape[0] - 1))
    controls: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    pca = PCA(n_components=n_components, random_state=int(seed))
    controls["esm2_pca16"] = (protein_ids, pca.fit_transform(features).astype(np.float32))

    rng = np.random.RandomState(int(seed))
    projector = rng.normal(size=(features.shape[1], n_components)) / np.sqrt(n_components)
    controls["esm2_rp16"] = (protein_ids, (features @ projector).astype(np.float32))

    if x_raw is not None:
        x = np.asarray(x_raw, dtype=np.float32)
        if x.ndim == 2 and x.shape[1] == features.shape[0] and x.shape[0] > n_components:
            observed = np.isfinite(x)
            column_means = np.where(observed, x, np.nan).mean(axis=0, dtype=np.float64)
            column_means = np.where(np.isfinite(column_means), column_means, 0.0)
            filled = np.where(observed, x, column_means[None, :].astype(np.float32))
            expression_pca = PCA(n_components=n_components, random_state=int(seed))
            controls["expr_pca16"] = (
                protein_ids,
                expression_pca.fit_transform(filled.T).astype(np.float32),
            )
    return controls


def evaluate_protein_embedding_manifest(
    manifest_path: str | Path,
    outdir: str | Path,
    labels_csv: str | Path = "",
    ext_dir: str | Path = "",
    ks: tuple[int, ...] = (5, 10, 20),
    seed: int = 42,
    min_proteins: int = 10,
    max_proteins: int = 100,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = load_json(manifest_path)
    artifacts = manifest.get("artifacts", {})
    labels_path = artifacts.get("labels")
    embeddings_path = artifacts.get("embeddings")
    if not labels_path:
        raise ValueError(f"Manifest has no artifacts.labels: {manifest_path}")
    if not embeddings_path:
        raise ValueError(f"Manifest has no artifacts.embeddings: {manifest_path}")

    label_rows = _read_label_csv(labels_csv) if labels_csv else []
    with np.load(labels_path, allow_pickle=True) as labels, np.load(embeddings_path) as embeddings:
        protein_embeddings = _protein_embeddings_from_artifact(labels, embeddings)
        raw_plm = _raw_plm_embedding_from_manifest(manifest, labels)
        if raw_plm is not None:
            protein_embeddings.setdefault("raw_plm", raw_plm)
            # Dimension-matched controls (16-D) so SABER-vs-raw-ESM is not a
            # dimensionality-confounded comparison. Added only for
            # sequence-informed runs, like raw_plm itself.
            dim_matched = _dimension_matched_controls(
                raw_plm,
                x_raw=labels["x_raw"] if "x_raw" in labels.files else None,
                dim=int(manifest.get("config", {}).get("model", {}).get("latent_dim", 16) or 16),
                seed=seed,
            )
            for control_name, control_entry in dim_matched.items():
                protein_embeddings.setdefault(control_name, control_entry)

    results: dict[str, Any] = {}
    source_rows: list[dict[str, Any]] = []
    for method, (protein_ids, emb) in sorted(protein_embeddings.items()):
        metrics: dict[str, Any] = {
            "n_proteins": int(emb.shape[0]) if emb.ndim == 2 else 0,
            "embedding_dim": int(emb.shape[1]) if emb.ndim == 2 else 0,
            "effective_rank": _effective_rank(emb),
        }
        if label_rows:
            metrics.update(compute_protein_label_metrics(emb, protein_ids, label_rows, ks=ks))
        results[method] = metrics
        source_rows.extend(
            metric_rows(
                "protein_embedding",
                manifest_path,
                method,
                metrics,
                manifest=manifest,
                extra={"protein_embedding_layer": "layer1"},
            )
        )

    annotation = _annotation_status(ext_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "protein_embedding_metrics.json"
    csv_path = outdir / "protein_embedding_metrics.csv"
    source_path = outdir / "source_protein_embedding_metrics.csv"
    write_json(
        json_path,
        {
            "manifest": str(manifest_path.resolve()),
            "labels_csv": str(Path(labels_csv).resolve()) if labels_csv else "",
            "external_annotations": annotation,
            "metric_contract": {
                "layer1": "Dataset-driven protein label hit@k when labels_csv is supplied; always reports embedding dimension and effective rank.",
                "layer2_3": "External annotation files are declared here; full STRING/Reactome/GO metrics are evaluated by the layer2/3 implementation.",
            },
            "results": results,
        },
    )
    write_rows_csv(csv_path, [{"method": method, **metrics} for method, metrics in sorted(results.items())], preferred=["method"])
    write_rows_csv(
        source_path,
        source_rows,
        preferred=[
            "task",
            "dataset",
            "method",
            "protein_embedding_layer",
            "condition",
            "seed",
            "scale_mode",
            "metric",
            "value",
            "artifact_path",
        ],
    )
    outputs: dict[str, Any] = {
        "json": str(json_path.resolve()),
        "csv": str(csv_path.resolve()),
        "source_csv": str(source_path.resolve()),
        "annotation": annotation,
        "results": results,
        "rows": source_rows,
    }
    if ext_dir:
        layer23_results, layer23_coverage = _evaluate_layer2_layer3(
            protein_embeddings,
            ext_dir,
            ks=ks,
            seed=seed,
            min_proteins=min_proteins,
            max_proteins=max_proteins,
        )
        layer23_json = outdir / "protein_embedding_layer23_metrics.json"
        layer23_csv = outdir / "protein_embedding_layer23_metrics.csv"
        layer23_source = outdir / "source_protein_embedding_layer23_metrics.csv"
        layer23_rows: list[dict[str, Any]] = []
        for method, metrics in sorted(layer23_results.items()):
            layer23_rows.extend(
                metric_rows(
                    "protein_embedding",
                    manifest_path,
                    method,
                    metrics,
                    manifest=manifest,
                    extra={"protein_embedding_layer": "layer2_3"},
                )
            )
        write_json(
            layer23_json,
            {
                "manifest": str(manifest_path.resolve()),
                "external_annotations": annotation,
                "coverage": layer23_coverage,
                "metric_contract": {
                    "string": "STRING edge recovery and kNN partner recall.",
                    "reactome_go": "Reactome/GO kNN label hit, Jaccard, global clustering, and intra/inter pathway distance.",
                    "layer3": "Reactome pathway discrimination gap against random protein pairs.",
                },
                "results": layer23_results,
            },
        )
        write_rows_csv(layer23_csv, [{"method": method, **metrics} for method, metrics in sorted(layer23_results.items())], preferred=["method"])
        write_rows_csv(
            layer23_source,
            layer23_rows,
            preferred=[
                "task",
                "dataset",
                "method",
                "protein_embedding_layer",
                "condition",
                "seed",
                "scale_mode",
                "metric",
                "value",
                "artifact_path",
            ],
        )
        outputs.update(
            {
                "layer23_json": str(layer23_json.resolve()),
                "layer23_csv": str(layer23_csv.resolve()),
                "layer23_source_csv": str(layer23_source.resolve()),
                "layer23_results": layer23_results,
                "layer23_rows": layer23_rows,
            }
        )
        source_rows.extend(layer23_rows)
        write_rows_csv(
            source_path,
            source_rows,
            preferred=[
                "task",
                "dataset",
                "method",
                "protein_embedding_layer",
                "condition",
                "seed",
                "scale_mode",
                "metric",
                "value",
                "artifact_path",
            ],
        )
    return {
        **outputs,
        "rows": source_rows,
    }
