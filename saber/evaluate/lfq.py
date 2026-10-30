from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from saber.artifacts import load_json, write_json
from saber.metrics import optional_float, pearson, spearman

from .common import metric_rows, write_rows_csv


EXPECTED_LOG2FC: dict[str, dict[str, float]] = {
    "HUMAN": {"A_vs_B": 0.0, "A_vs_C": 0.0, "B_vs_C": 0.0},
    "YEAST": {
        "A_vs_B": float(np.log2(0.30 / 0.15)),
        "A_vs_C": float(np.log2(0.30 / 0.03)),
        "B_vs_C": float(np.log2(0.15 / 0.03)),
    },
    "ECOLI": {
        "A_vs_B": float(np.log2(0.05 / 0.20)),
        "A_vs_C": float(np.log2(0.05 / 0.32)),
        "B_vs_C": float(np.log2(0.20 / 0.32)),
    },
}
COMPARISONS = ("A_vs_B", "A_vs_C", "B_vs_C")


def normalize_species(values: np.ndarray) -> np.ndarray:
    out = []
    for value in np.asarray(values).astype(str).tolist():
        text = value.strip().upper().replace(".", "").replace(" ", "")
        if text in {"HUMAN", "HOMO", "HOMOSAPIENS"}:
            out.append("HUMAN")
        elif text in {"YEAST", "SACCHAROMYCESCEREVISIAE", "SCEREVISIAE"}:
            out.append("YEAST")
        elif text in {"ECOLI", "ECOLI_K12", "ESCHERICHIACOLI"}:
            out.append("ECOLI")
        else:
            out.append(text)
    return np.asarray(out, dtype=object)


def normalize_conditions(values: np.ndarray) -> np.ndarray:
    out = []
    for value in np.asarray(values).astype(str).tolist():
        text = value.strip().upper()
        if text.startswith("A") or text in {"COND_A", "CONDITION_A"}:
            out.append("A")
        elif text.startswith("B") or text in {"COND_B", "CONDITION_B"}:
            out.append("B")
        elif text.startswith("C") or text in {"COND_C", "CONDITION_C"}:
            out.append("C")
        else:
            out.append(text)
    return np.asarray(out, dtype=object)


def load_lfq_labels(
    manifest: dict[str, Any],
    data_path: str | Path = "",
    condition_key: str = "condition",
    species_key: str = "species",
    protein_id_field: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    labels_path = manifest.get("artifacts", {}).get("labels")
    if labels_path:
        with np.load(labels_path, allow_pickle=True) as labels:
            if "conditions" in labels.files and "species" in labels.files:
                conditions = normalize_conditions(labels["conditions"])
                species = normalize_species(labels["species"])
                protein_ids = labels["protein_ids"].astype(str) if "protein_ids" in labels.files else np.asarray([], dtype=str)
                x_raw = labels["x_raw"] if "x_raw" in labels.files else None
                return conditions, species, protein_ids, x_raw

    path = Path(data_path or manifest.get("config", {}).get("data", {}).get("path", ""))
    if not path:
        raise ValueError("LFQ labels require --data-path or config.data.path")
    try:
        import anndata as ad
    except Exception as exc:
        raise RuntimeError("anndata is required to load LFQ labels from .h5ad") from exc
    adata = ad.read_h5ad(path)
    if condition_key not in adata.obs.columns:
        raise ValueError(f"LFQ obs is missing condition column: {condition_key}")
    if species_key not in adata.var.columns:
        raise ValueError(f"LFQ var is missing species column: {species_key}")
    protein_ids = (
        adata.var[protein_id_field].astype(str).to_numpy()
        if protein_id_field and protein_id_field in adata.var.columns
        else np.asarray([str(v) for v in adata.var_names.tolist()], dtype=str)
    )
    x_raw = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    return (
        normalize_conditions(adata.obs[condition_key].to_numpy()),
        normalize_species(adata.var[species_key].to_numpy()),
        protein_ids,
        np.asarray(x_raw, dtype=np.float32),
    )


def compute_per_protein_log2fc(
    x: np.ndarray,
    conditions: np.ndarray,
    species: np.ndarray,
    protein_ids: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    matrix = np.asarray(x, dtype=np.float32)
    conditions = normalize_conditions(conditions)
    species = normalize_species(species)
    if matrix.shape[1] != species.shape[0]:
        raise ValueError(f"LFQ matrix has {matrix.shape[1]} proteins but species has {species.shape[0]} entries")
    ids = np.asarray(protein_ids, dtype=object) if protein_ids is not None and len(protein_ids) else np.asarray([None] * matrix.shape[1], dtype=object)
    if ids.shape[0] != matrix.shape[1]:
        raise ValueError(f"LFQ matrix has {matrix.shape[1]} proteins but protein_ids has {ids.shape[0]} entries")

    records: list[dict[str, Any]] = []
    for species_name, expected_by_comparison in EXPECTED_LOG2FC.items():
        species_mask = species == species_name
        if int(species_mask.sum()) == 0:
            continue
        x_sp = matrix[:, species_mask]
        ids_sp = ids[species_mask]
        for comparison, expected in expected_by_comparison.items():
            c1, c2 = comparison.split("_vs_")
            with np.errstate(all="ignore"):
                med1 = np.nanmedian(x_sp[conditions == c1], axis=0)
                med2 = np.nanmedian(x_sp[conditions == c2], axis=0)
                observed = med1 - med2
            for protein_id, value in zip(ids_sp.tolist(), observed.tolist()):
                if not np.isfinite(value):
                    continue
                error = float(value - expected)
                records.append(
                    {
                        "protein_id": protein_id,
                        "species": species_name,
                        "comparison": comparison,
                        "expected": float(expected),
                        "observed": float(value),
                        "error": error,
                        "abs_error": abs(error),
                    }
                )
    return records


def summarize_lfq_records(records: list[dict[str, Any]]) -> dict[str, int | float | None]:
    if not records:
        return {
            "n_protein_comparisons": 0,
            "mean_abs_error": None,
            "median_abs_error": None,
            "rmse_log2fc": None,
            "pearson_log2fc": None,
            "spearman_log2fc": None,
        }
    expected = np.asarray([row["expected"] for row in records], dtype=np.float64)
    observed = np.asarray([row["observed"] for row in records], dtype=np.float64)
    errors = observed - expected
    out: dict[str, int | float | None] = {
        "n_protein_comparisons": int(len(records)),
        "mean_abs_error": optional_float(np.mean(np.abs(errors))),
        "median_abs_error": optional_float(np.median(np.abs(errors))),
        "rmse_log2fc": optional_float(np.sqrt(np.mean(errors**2))),
        "mean_error": optional_float(np.mean(errors)),
        "pearson_log2fc": pearson(expected, observed),
        "spearman_log2fc": spearman(expected, observed),
    }
    for species_name in ("HUMAN", "YEAST", "ECOLI", "YEAST+ECOLI"):
        if species_name == "YEAST+ECOLI":
            group = [row for row in records if row["species"] in {"YEAST", "ECOLI"}]
        else:
            group = [row for row in records if row["species"] == species_name]
        if group:
            vals = np.asarray([row["abs_error"] for row in group], dtype=np.float64)
            out[f"mean_abs_error_{species_name.lower().replace('+', '_')}"] = optional_float(np.mean(vals))
            out[f"n_{species_name.lower().replace('+', '_')}"] = int(len(group))
    return out


def _split_masks(x_raw: np.ndarray | None) -> dict[str, np.ndarray] | None:
    if x_raw is None:
        return None
    finite = np.isfinite(np.asarray(x_raw, dtype=np.float32))
    return {
        "all": np.ones(finite.shape[1], dtype=bool),
        "has_missing": ~finite.all(axis=0),
        "fully_observed": finite.all(axis=0),
    }


def _npz_string(value: np.ndarray) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 0:
        return ""
    return str(arr.reshape(-1)[0])


def _scale_metadata(data: Any) -> tuple[str, np.ndarray | None, np.ndarray | None]:
    mode = _npz_string(data["scale_mode"]) if "scale_mode" in data.files else "none"
    center = np.asarray(data["scale_center"], dtype=np.float32).reshape(-1) if "scale_center" in data.files else None
    scale = np.asarray(data["scale_scale"], dtype=np.float32).reshape(-1) if "scale_scale" in data.files else None
    if center is not None and center.size == 0:
        center = None
    if scale is not None and scale.size == 0:
        scale = None
    return mode, center, scale


def _inverse_scale_matrix(matrix: np.ndarray, mode: str, center: np.ndarray | None, scale: np.ndarray | None) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if mode != "zscore":
        return arr.astype(np.float32)
    if center is None or scale is None:
        raise ValueError("LFQ zscore artifacts require scale_center and scale_scale for raw-scale evaluation")
    if center.shape[0] != arr.shape[1] or scale.shape[0] != arr.shape[1]:
        raise ValueError(f"Scale metadata has length {center.shape[0] if center is not None else 0}/{scale.shape[0] if scale is not None else 0}, but matrix has {arr.shape[1]} proteins")
    return (arr * scale.reshape(1, -1) + center.reshape(1, -1)).astype(np.float32)


def _protein_mean_baseline(x_input: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
    x = np.asarray(x_input, dtype=np.float32)
    observed = np.asarray(observed_mask).astype(bool) & np.isfinite(x)
    counts = observed.sum(axis=0)
    sums = np.where(observed, x, 0.0).sum(axis=0, dtype=np.float64)
    means = np.zeros((x.shape[1],), dtype=np.float32)
    np.divide(sums, counts, out=means, where=counts > 0)
    return np.broadcast_to(means[None, :], x.shape).copy().astype(np.float32)


def _cell_mean_baseline(x_input: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
    x = np.asarray(x_input, dtype=np.float32)
    observed = np.asarray(observed_mask).astype(bool) & np.isfinite(x)
    counts = observed.sum(axis=1, keepdims=True)
    sums = np.where(observed, x, 0.0).sum(axis=1, keepdims=True, dtype=np.float64)
    means = np.zeros((x.shape[0], 1), dtype=np.float32)
    np.divide(sums, counts, out=means, where=counts > 0)
    return np.broadcast_to(means, x.shape).copy().astype(np.float32)


def _species_block_mean_baseline(
    x_input: np.ndarray,
    observed_mask: np.ndarray,
    conditions: np.ndarray,
    species: np.ndarray,
) -> np.ndarray:
    """Species-only control imputer (species-confound control).

    Fills each missing entry with the mean of its (condition x species) block of
    observed values; observed entries are kept unchanged. The imputer uses species
    identity and condition only -- no sequence, no per-protein learning -- so it
    isolates how much of the fold-change recovery comes from per-protein signal
    versus from species identity alone (the PXD071205 expected fold change is
    species-determined). If this matches SABER on the signal species, the PLM
    benefit is taxonomy; if it is far worse, SABER contributes per-protein
    quantitative signal beyond species identity. Mirrors the cell_mean /
    protein_mean baselines: computed in model scale from x_input over observed
    entries, then inverse-transformed by the caller like the other baselines.
    """
    x = np.asarray(x_input, dtype=np.float32)
    obs = np.asarray(observed_mask).astype(bool) & np.isfinite(x)
    cond = normalize_conditions(conditions)
    spec = normalize_species(species)
    conds = sorted({str(c) for c in cond.tolist()})
    specs = sorted({str(s) for s in spec.tolist()})
    c2i = {c: i for i, c in enumerate(conds)}
    s2i = {s: i for i, s in enumerate(specs)}
    ci = np.array([c2i[str(c)] for c in cond])          # (cells,)
    si = np.array([s2i[str(s)] for s in spec])          # (proteins,)
    block = np.full((len(conds), len(specs)), np.nan, dtype=np.float64)
    masked = np.where(obs, x, np.nan)
    for c in range(len(conds)):
        for s in range(len(specs)):
            sub = masked[ci == c, :][:, si == s]
            if sub.size:
                value = np.nanmean(sub)
                if np.isfinite(value):
                    block[c, s] = value
    grand = np.nanmean(masked)
    if np.isfinite(grand):
        block = np.where(np.isfinite(block), block, grand)
    mu_broadcast = block[np.ix_(ci, si)]                # (cells, proteins)
    return np.where(obs, x, mu_broadcast).astype(np.float32)


def evaluate_lfq_manifest(
    manifest_path: str | Path,
    outdir: str | Path,
    data_path: str | Path = "",
    condition_key: str = "condition",
    species_key: str = "species",
    protein_id_field: str = "",
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = load_json(manifest_path)
    imputation_path = manifest.get("artifacts", {}).get("imputation")
    if not imputation_path:
        raise ValueError(f"Manifest has no artifacts.imputation: {manifest_path}")
    conditions, species, protein_ids, x_raw = load_lfq_labels(
        manifest,
        data_path=data_path,
        condition_key=condition_key,
        species_key=species_key,
        protein_id_field=protein_id_field,
    )

    with np.load(imputation_path) as data:
        scale_mode, scale_center, scale_scale = _scale_metadata(data)
        model_scale_predictions = {
            key.removeprefix("pred__"): np.asarray(data[key], dtype=np.float32)
            for key in data.files
            if key.startswith("pred__")
        }
        raw_scale_predictions = {
            key.removeprefix("pred_raw__"): np.asarray(data[key], dtype=np.float32)
            for key in data.files
            if key.startswith("pred_raw__")
        }
        if "x_input" in data.files:
            observed_mask = data["observed_mask"] if "observed_mask" in data.files else np.isfinite(data["x_input"])
            model_scale_predictions.setdefault("protein_mean", _protein_mean_baseline(data["x_input"], observed_mask))
            model_scale_predictions.setdefault("cell_mean", _cell_mean_baseline(data["x_input"], observed_mask))
        predictions = dict(raw_scale_predictions)
        for method, matrix in model_scale_predictions.items():
            if method not in predictions:
                predictions[method] = _inverse_scale_matrix(matrix, scale_mode, scale_center, scale_scale)
        if "x_true_raw" in data.files:
            predictions.setdefault("raw_true", np.asarray(data["x_true_raw"], dtype=np.float32))
        elif "x_true" in data.files:
            predictions.setdefault("raw_true", _inverse_scale_matrix(data["x_true"], scale_mode, scale_center, scale_scale))
        # Species-only control (species-confound control): fill each
        # missing entry with its (condition x species) block mean. Computed in RAW
        # scale from x_raw and inserted directly into predictions so it BYPASSES
        # inverse-scaling -- a model-scale version would leak per-protein mean/sd
        # through the per-protein inverse transform and would not be a valid
        # species-only (no-per-protein-information) control.
        _species_src = x_raw if x_raw is not None else (
            np.asarray(data["x_true_raw"], dtype=np.float32) if "x_true_raw" in data.files else None
        )
        if _species_src is not None:
            _species_obs = data["observed_mask"] if "observed_mask" in data.files else np.isfinite(_species_src)
            predictions.setdefault(
                "species_mean",
                _species_block_mean_baseline(_species_src, _species_obs, conditions, species),
            )

    results: dict[str, Any] = {}
    details: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    split_masks = _split_masks(x_raw)
    for method, matrix in sorted(predictions.items()):
        records = compute_per_protein_log2fc(matrix, conditions, species, protein_ids=protein_ids)
        results[method] = summarize_lfq_records(records)
        for row in records:
            details.append({"method": method, "subset": "all", **row})
        if split_masks is not None:
            for subset, mask in split_masks.items():
                if int(mask.sum()) == 0:
                    continue
                subset_records = compute_per_protein_log2fc(matrix[:, mask], conditions, species[mask], protein_ids=protein_ids[mask])
                metrics = summarize_lfq_records(subset_records)
                split_rows.append({"method": method, "subset": subset, **metrics})

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "lfq_metrics.json"
    csv_path = outdir / "lfq_summary_metrics.csv"
    details_path = outdir / "lfq_per_protein_details.csv"
    split_path = outdir / "lfq_split_summary.csv"
    source_path = outdir / "source_lfq_metrics.csv"
    write_json(
        json_path,
        {
            "manifest": str(manifest_path.resolve()),
            "metric_contract": {
                "dataset": "PXD071205 full LFQ benchmark.",
                "scale": "If the artifact was trained with data.scale_mode=zscore, predictions are inverse-transformed to the original LFQ log-intensity scale before log2 fold-change metrics are computed.",
                "primary_metric": "Per-protein log2 fold-change absolute error against known mixture ratios.",
                "main_subset": "Use all and has_missing; Yeast+ECOLI is the most informative species group for changing ratios.",
            },
            "results": results,
        },
    )
    write_rows_csv(csv_path, [{"method": method, **metrics} for method, metrics in sorted(results.items())], preferred=["method"])
    write_rows_csv(
        details_path,
        details,
        preferred=["method", "subset", "protein_id", "species", "comparison", "expected", "observed", "error", "abs_error"],
    )
    write_rows_csv(split_path, split_rows, preferred=["method", "subset"])
    source_rows: list[dict[str, Any]] = []
    for method, metrics in sorted(results.items()):
        source_rows.extend(metric_rows("lfq", manifest_path, method, metrics, manifest=manifest, extra={"lfq_subset": "all"}))
    for row in split_rows:
        method = str(row.get("method", ""))
        subset = str(row.get("subset", ""))
        metrics = {k: v for k, v in row.items() if k not in {"method", "subset"}}
        source_rows.extend(metric_rows("lfq", manifest_path, method, metrics, manifest=manifest, extra={"lfq_subset": subset}))
    write_rows_csv(
        source_path,
        source_rows,
        preferred=["task", "dataset", "method", "lfq_subset", "condition", "seed", "scale_mode", "metric", "value", "artifact_path"],
    )
    return {
        "json": str(json_path.resolve()),
        "csv": str(csv_path.resolve()),
        "details_csv": str(details_path.resolve()),
        "split_csv": str(split_path.resolve()),
        "source_csv": str(source_path.resolve()),
        "results": results,
        "details": details,
        "split_rows": split_rows,
    }
