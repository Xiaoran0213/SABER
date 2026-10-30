from __future__ import annotations

from typing import Any

import numpy as np


def optional_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size < 2:
        return None
    x0 = x - x.mean()
    y0 = y - y.mean()
    denom = float(np.sqrt(np.sum(x0**2) * np.sum(y0**2)))
    if denom <= 1e-12:
        return None
    return float(np.sum(x0 * y0) / denom)


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size < 2:
        return None
    return pearson(_rankdata(x), _rankdata(y))


def mean_defined(values: list[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return None
    return float(np.mean(finite))


def compute_matrix_prediction_metrics(
    x_true: np.ndarray,
    x_pred: np.ndarray,
    mask: np.ndarray,
    prefix: str,
) -> dict[str, int | float | None]:
    true = np.asarray(x_true, dtype=np.float32)
    pred = np.asarray(x_pred, dtype=np.float32)
    eval_mask = np.asarray(mask).astype(bool)
    if true.shape != pred.shape or true.shape != eval_mask.shape:
        raise ValueError("x_true, x_pred, and mask must have the same shape")

    finite = eval_mask & np.isfinite(true) & np.isfinite(pred)
    n_eval = int(finite.sum())
    result: dict[str, int | float | None] = {
        f"n_{prefix}": n_eval,
        f"n_{prefix}_proteins_eval": 0,
        f"mse_{prefix}": None,
        f"rmse_{prefix}": None,
        f"mae_{prefix}": None,
        f"pearson_{prefix}_all": None,
        f"pearson_{prefix}_mean": None,
        f"spearman_{prefix}_mean": None,
    }
    if n_eval == 0:
        return result

    errors = pred[finite] - true[finite]
    mse = float(np.mean(errors**2))
    result[f"mse_{prefix}"] = optional_float(mse)
    result[f"rmse_{prefix}"] = optional_float(float(np.sqrt(mse)))
    result[f"mae_{prefix}"] = optional_float(float(np.mean(np.abs(errors))))
    result[f"pearson_{prefix}_all"] = pearson(true[finite], pred[finite])

    pearsons: list[float | None] = []
    spearmans: list[float | None] = []
    n_proteins = 0
    for j in range(true.shape[1]):
        col_mask = finite[:, j]
        if int(col_mask.sum()) < 2:
            continue
        n_proteins += 1
        pearsons.append(pearson(true[col_mask, j], pred[col_mask, j]))
        spearmans.append(spearman(true[col_mask, j], pred[col_mask, j]))
    result[f"n_{prefix}_proteins_eval"] = int(n_proteins)
    result[f"pearson_{prefix}_mean"] = mean_defined(pearsons)
    result[f"spearman_{prefix}_mean"] = mean_defined(spearmans)
    return result


def compute_imputation_metrics(
    x_true: np.ndarray,
    x_pred: np.ndarray,
    injected_missing_mask: np.ndarray,
) -> dict[str, int | float | None]:
    return compute_matrix_prediction_metrics(x_true, x_pred, injected_missing_mask, prefix="missing")


def _compute_purity(true_labels: np.ndarray, pred_labels: np.ndarray) -> float | None:
    labels = np.asarray(true_labels).reshape(-1)
    pred = np.asarray(pred_labels).reshape(-1)
    if labels.shape[0] != pred.shape[0] or labels.size == 0:
        return None
    total = 0
    for cluster in np.unique(pred):
        members = labels[pred == cluster]
        if members.size:
            _values, counts = np.unique(members, return_counts=True)
            total += int(counts.max())
    return optional_float(total / max(1, labels.size))


def _safe_silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float | None:
    try:
        from sklearn.metrics import silhouette_score

        return optional_float(silhouette_score(embeddings, labels))
    except Exception:
        return None


def compute_lisi(embeddings: np.ndarray, labels: np.ndarray, k: int = 30) -> float | None:
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception:
        return None
    z = np.asarray(embeddings, dtype=np.float32)
    y = np.asarray(labels).reshape(-1)
    if z.shape[0] != y.shape[0] or z.shape[0] < 5 or np.unique(y).size < 2:
        return None
    kk = int(max(5, min(int(k), z.shape[0] - 1)))
    nn = NearestNeighbors(n_neighbors=kk + 1, metric="euclidean")
    nn.fit(z)
    idx = nn.kneighbors(z, return_distance=False)[:, 1:]
    uniq = np.unique(y)
    mapper = {label: i for i, label in enumerate(uniq)}
    vals: list[float] = []
    for i in range(z.shape[0]):
        p = np.zeros((uniq.size,), dtype=np.float64)
        for label in y[idx[i]]:
            p[mapper[label]] += 1.0
        p /= max(1.0, p.sum())
        simpson = float(np.sum(p**2))
        if simpson > 0:
            vals.append(1.0 / simpson)
    return optional_float(np.mean(vals)) if vals else None


def compute_kbet_rejection(embeddings: np.ndarray, labels: np.ndarray, k: int = 30) -> float | None:
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception:
        return None
    z = np.asarray(embeddings, dtype=np.float32)
    y = np.asarray(labels).reshape(-1)
    if z.shape[0] != y.shape[0] or z.shape[0] < 5 or np.unique(y).size < 2:
        return None
    kk = int(max(5, min(int(k), z.shape[0] - 1)))
    nn = NearestNeighbors(n_neighbors=kk + 1, metric="euclidean")
    nn.fit(z)
    idx = nn.kneighbors(z, return_distance=False)[:, 1:]
    uniq, counts = np.unique(y, return_counts=True)
    global_p = counts / counts.sum()
    mapper = {label: i for i, label in enumerate(uniq)}
    rejected = 0
    for i in range(z.shape[0]):
        obs = np.zeros_like(global_p, dtype=np.float64)
        for label in y[idx[i]]:
            obs[mapper[label]] += 1.0
        exp = global_p * kk
        with np.errstate(divide="ignore", invalid="ignore"):
            chi2 = np.sum((obs - exp) ** 2 / np.clip(exp, 1e-8, None))
        if np.isfinite(chi2) and chi2 > 9.49:
            rejected += 1
    return optional_float(rejected / z.shape[0])


def compute_graph_connectivity(embeddings: np.ndarray, labels: np.ndarray, k: int = 15) -> float | None:
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception:
        return None
    z = np.asarray(embeddings, dtype=np.float32)
    y = np.asarray(labels).reshape(-1)
    if z.shape[0] != y.shape[0] or z.shape[0] < 5 or np.unique(y).size < 2:
        return None
    kk = int(max(3, min(int(k), z.shape[0] - 1)))
    nn = NearestNeighbors(n_neighbors=kk + 1, metric="euclidean")
    nn.fit(z)
    neigh = nn.kneighbors(z, return_distance=False)[:, 1:]
    adj = [set() for _ in range(z.shape[0])]
    for i in range(z.shape[0]):
        for j in neigh[i]:
            adj[i].add(int(j))
            adj[int(j)].add(i)
    scores: list[float] = []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        allowed = set(int(i) for i in idx.tolist())
        seen: set[int] = set()
        max_cc = 0
        for start in allowed:
            if start in seen:
                continue
            stack = [start]
            seen.add(start)
            size = 0
            while stack:
                node = stack.pop()
                size += 1
                for nb in adj[node]:
                    if nb in allowed and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            max_cc = max(max_cc, size)
        scores.append(float(max_cc / max(1, idx.size)))
    return optional_float(np.mean(scores)) if scores else None


def compute_embedding_metrics(
    embeddings: np.ndarray,
    cell_types: np.ndarray,
    batches: np.ndarray,
    seed: int = 42,
) -> dict[str, int | float | None]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    z = np.asarray(embeddings, dtype=np.float32)
    cell_labels = np.asarray(cell_types).reshape(-1)
    batch_labels = np.asarray(batches).reshape(-1)
    if z.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if z.shape[0] != cell_labels.shape[0] or z.shape[0] != batch_labels.shape[0]:
        raise ValueError("embeddings, cell_types, and batches must have the same number of cells")

    finite_rows = np.isfinite(z).all(axis=1)
    z = z[finite_rows]
    cell_labels = cell_labels[finite_rows]
    batch_labels = batch_labels[finite_rows]
    n_cells = int(z.shape[0])
    n_cell_types = int(np.unique(cell_labels).size) if n_cells else 0
    n_batches = int(np.unique(batch_labels).size) if n_cells else 0
    result: dict[str, int | float | None] = {
        "n_cells_eval": n_cells,
        "n_cell_types": n_cell_types,
        "n_batches": n_batches,
        "ARI_cell": None,
        "NMI_cell": None,
        "ASW_cell": None,
        "PS_cell": None,
        "ARI_batch": None,
        "NMI_batch": None,
        "ASW_batch": None,
        "PS_batch": None,
        "kBET_reject_rate": None,
        "iLISI": None,
        "cLISI": None,
        "graph_connectivity": None,
        "batch_mixing_score": None,
    }
    if n_cells < 2:
        return result

    if n_cell_types >= 2 and n_cells > n_cell_types:
        pred_cell = KMeans(n_clusters=n_cell_types, random_state=int(seed), n_init=10).fit_predict(z)
        result["ARI_cell"] = optional_float(adjusted_rand_score(cell_labels, pred_cell))
        result["NMI_cell"] = optional_float(normalized_mutual_info_score(cell_labels, pred_cell))
        result["ASW_cell"] = _safe_silhouette(z, cell_labels)
        result["PS_cell"] = _compute_purity(cell_labels, pred_cell)
        result["cLISI"] = compute_lisi(z, cell_labels)
        result["graph_connectivity"] = compute_graph_connectivity(z, cell_labels)

    if n_batches >= 2 and n_cells > n_batches:
        pred_batch = KMeans(n_clusters=n_batches, random_state=int(seed), n_init=10).fit_predict(z)
        result["ARI_batch"] = optional_float(adjusted_rand_score(batch_labels, pred_batch))
        result["NMI_batch"] = optional_float(normalized_mutual_info_score(batch_labels, pred_batch))
        result["ASW_batch"] = _safe_silhouette(z, batch_labels)
        result["PS_batch"] = _compute_purity(batch_labels, pred_batch)
        result["kBET_reject_rate"] = compute_kbet_rejection(z, batch_labels)
        result["iLISI"] = compute_lisi(z, batch_labels)
        if result["ASW_batch"] is not None:
            result["batch_mixing_score"] = optional_float(max(0.0, 1.0 - abs(float(result["ASW_batch"]))))
    return result


def compute_protein_label_metrics(
    protein_embeddings: np.ndarray,
    protein_ids: np.ndarray,
    label_rows: list[dict[str, str]],
    ks: tuple[int, ...] = (5, 10, 20),
) -> dict[str, int | float | None]:
    emb = np.asarray(protein_embeddings, dtype=np.float32)
    ids = np.asarray(protein_ids).astype(str)
    if emb.ndim != 2:
        raise ValueError("protein_embeddings must be a 2D array")
    if emb.shape[0] != ids.shape[0]:
        raise ValueError("protein_embeddings and protein_ids must have the same length")

    labels_by_protein: dict[str, set[str]] = {}
    for row in label_rows:
        pid = str(row.get("protein_id", "")).strip()
        label_id = str(row.get("label_id", "")).strip()
        if not pid or not label_id:
            continue
        labels_by_protein.setdefault(pid, set()).add(label_id)

    eval_idx = np.asarray([i for i, pid in enumerate(ids.tolist()) if pid in labels_by_protein], dtype=np.int64)
    unique_labels = sorted({label for labels in labels_by_protein.values() for label in labels})
    result: dict[str, int | float | None] = {
        "n_proteins": int(ids.shape[0]),
        "n_labeled_proteins": int(eval_idx.shape[0]),
        "n_label_ids": int(len(unique_labels)),
        "label_coverage": optional_float(eval_idx.shape[0] / max(1, ids.shape[0])),
    }
    for k in ks:
        result[f"hit_at_{int(k)}"] = None
    if eval_idx.shape[0] < 3 or not unique_labels:
        return result

    z = emb[eval_idx]
    z_norm = z / np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1e-12, None)
    sim = z_norm @ z_norm.T
    np.fill_diagonal(sim, -np.inf)
    eval_ids = ids[eval_idx]
    for k in ks:
        k_eff = min(int(k), max(0, eval_idx.shape[0] - 1))
        if k_eff <= 0:
            continue
        hits: list[float] = []
        for i, pid in enumerate(eval_ids.tolist()):
            query_labels = labels_by_protein.get(pid, set())
            if not query_labels:
                continue
            nn = np.argpartition(-sim[i], kth=k_eff - 1)[:k_eff]
            hit = any(query_labels & labels_by_protein.get(eval_ids[j], set()) for j in nn)
            hits.append(float(hit))
        result[f"hit_at_{int(k)}"] = optional_float(np.mean(hits)) if hits else None
    return result
