from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DataConfig, MissingConfig, ModelConfig
from .feature_cache import FeatureCacheError, load_protein_feature_cache, read_protein_id_file
from .missing import fill_missing, inject_missing


@dataclass
class DataBundle:
    x_raw: np.ndarray
    x_true: np.ndarray
    x_input: np.ndarray
    observed_mask: np.ndarray
    injected_missing_mask: np.ndarray
    available_mask: np.ndarray
    batches: np.ndarray
    cell_types: np.ndarray
    protein_ids: np.ndarray
    protein_features: np.ndarray
    missing_info: dict
    feature_cache_info: dict
    scale_info: dict
    # Mechanism split of injected_missing_mask (MNAR part), when injection ran
    # with return_split=True. MCAR part = injected & ~mnar.
    injected_mnar_mask: np.ndarray | None = None


def _factorize(values: np.ndarray) -> np.ndarray:
    labels = [str(v) for v in values.tolist()]
    mapping = {v: i for i, v in enumerate(sorted(set(labels)))}
    return np.asarray([mapping[v] for v in labels], dtype=np.int64)


def _scale_matrix(x: np.ndarray, mode: str) -> tuple[np.ndarray, dict]:
    raw = np.asarray(x, dtype=np.float32)
    if mode == "none":
        return raw.astype(np.float32), {"mode": "none"}
    if mode == "zscore":
        mean = np.nanmean(raw, axis=0, keepdims=True)
        std = np.nanstd(raw, axis=0, keepdims=True)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
        return ((raw - mean) / std).astype(np.float32), {
            "mode": "zscore",
            "center": mean.reshape(-1).astype(np.float32),
            "scale": std.reshape(-1).astype(np.float32),
        }
    raise ValueError(f"Unsupported scale_mode: {mode}")


def _synthetic_bundle(data_cfg: DataConfig, missing_cfg: MissingConfig, model_cfg: ModelConfig, seed: int) -> DataBundle:
    rng = np.random.default_rng(int(seed))
    n_cells = int(data_cfg.n_cells)
    n_proteins = int(data_cfg.n_proteins)
    latent = int(model_cfg.latent_dim)
    feature_dim = int(model_cfg.protein_feature_dim)

    batches = np.arange(n_cells, dtype=np.int64) % max(1, int(data_cfg.n_batches))
    cell_types = np.arange(n_cells, dtype=np.int64) % max(1, int(data_cfg.n_cell_types))
    z = rng.normal(0.0, 1.0, size=(n_cells, latent)).astype(np.float32)
    e = rng.normal(0.0, 1.0, size=(n_proteins, latent)).astype(np.float32)
    x = z @ e.T
    x += 0.25 * batches[:, None].astype(np.float32)
    x += 0.35 * cell_types[:, None].astype(np.float32)
    x += rng.normal(0.0, 0.15, size=x.shape).astype(np.float32)

    protein_features = rng.normal(0.0, 1.0, size=(n_proteins, feature_dim)).astype(np.float32)
    protein_features[:, : min(latent, feature_dim)] += e[:, : min(latent, feature_dim)]
    protein_ids = np.asarray([f"P{i:05d}" for i in range(n_proteins)], dtype=object)
    return _finalize_bundle(
        x_raw=x.astype(np.float32),
        x_true=x.astype(np.float32),
        batches=batches,
        cell_types=cell_types,
        protein_ids=protein_ids,
        protein_features=protein_features,
        missing_cfg=missing_cfg,
        feature_cache_info={"feature_source": "synthetic", "feature_dim": feature_dim},
        scale_info={"mode": "none"},
    )


def _load_h5ad_bundle(data_cfg: DataConfig, missing_cfg: MissingConfig, model_cfg: ModelConfig) -> DataBundle:
    try:
        import anndata as ad
    except Exception as exc:
        raise RuntimeError("anndata is required to load .h5ad files") from exc

    adata = ad.read_h5ad(data_cfg.path)
    x_raw = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    x_raw = np.asarray(x_raw, dtype=np.float32)
    x, scale_info = _scale_matrix(x_raw, data_cfg.scale_mode)

    obs_cols = set(str(c) for c in adata.obs.columns)
    if data_cfg.batch_key and data_cfg.batch_key in obs_cols:
        batches = _factorize(adata.obs[data_cfg.batch_key].to_numpy())
    else:
        batches = np.zeros((adata.n_obs,), dtype=np.int64)
    if data_cfg.cell_type_key and data_cfg.cell_type_key in obs_cols:
        cell_types = _factorize(adata.obs[data_cfg.cell_type_key].to_numpy())
    else:
        cell_types = np.zeros((adata.n_obs,), dtype=np.int64)

    if data_cfg.protein_id_field and data_cfg.protein_id_field in adata.var.columns:
        protein_ids = adata.var[data_cfg.protein_id_field].astype(str).to_numpy(dtype=object)
    else:
        protein_ids = np.asarray([str(v) for v in adata.var_names.tolist()], dtype=object)
    if data_cfg.protein_ids_path:
        override_ids = read_protein_id_file(data_cfg.protein_ids_path)
        if override_ids.shape[0] != protein_ids.shape[0]:
            raise ValueError(
                f"protein_ids_path has {override_ids.shape[0]} ids but data has {protein_ids.shape[0]} proteins"
            )
        protein_ids = override_ids

    protein_features, feature_cache_info = _load_or_make_protein_features(data_cfg, model_cfg, protein_ids)
    return _finalize_bundle(
        x_raw=x_raw,
        x_true=x,
        batches=batches,
        cell_types=cell_types,
        protein_ids=protein_ids,
        protein_features=protein_features,
        missing_cfg=missing_cfg,
        feature_cache_info=feature_cache_info,
        scale_info=scale_info,
        fill_mode=data_cfg.fill_mode,
        fill_seed=data_cfg.fill_seed,
    )


def _load_or_make_protein_features(
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    protein_ids: np.ndarray,
) -> tuple[np.ndarray, dict]:
    n = int(protein_ids.shape[0])
    feature_dim = int(model_cfg.protein_feature_dim)
    if data_cfg.protein_features_path:
        cache = load_protein_feature_cache(
            data_cfg.protein_features_path,
            protein_ids=protein_ids,
            expected_dim=feature_dim,
            cache_protein_ids_path=data_cfg.protein_features_ids_path,
        )
        return cache.features.astype(np.float32), {
            "feature_source": "cache",
            "feature_cache_path": str(Path(data_cfg.protein_features_path).resolve()),
            "feature_cache_ids_path": str(Path(data_cfg.protein_features_ids_path).resolve()) if data_cfg.protein_features_ids_path else "",
            "feature_dim": int(cache.features.shape[1]),
            **cache.metadata,
        }
    if not data_cfg.synthetic and not data_cfg.allow_random_protein_features:
        raise FeatureCacheError(
            "data.protein_features_path is required for real datasets. Set "
            "data.allow_random_protein_features=true only for engineering diagnostics."
        )
    rng = np.random.default_rng(42)
    features = rng.normal(0.0, 1.0, size=(n, feature_dim)).astype(np.float32)
    return features, {"feature_source": "deterministic_random", "feature_dim": feature_dim}


def _finalize_bundle(
    x_raw: np.ndarray,
    x_true: np.ndarray,
    batches: np.ndarray,
    cell_types: np.ndarray,
    protein_ids: np.ndarray,
    protein_features: np.ndarray,
    missing_cfg: MissingConfig,
    feature_cache_info: dict,
    scale_info: dict,
    fill_mode: str = "column_mean",
    fill_seed: int = 42,
) -> DataBundle:
    available = np.isfinite(x_true)
    mnar_sub = None
    if missing_cfg.enabled:
        x_observed, observed_mask, injected_missing, info, (mnar_sub, _mcar_sub) = inject_missing(
            x_full=x_true,
            total_fraction=missing_cfg.total_fraction,
            mnar_fraction=missing_cfg.mnar_fraction,
            threshold_quantile=missing_cfg.threshold_quantile,
            threshold_scale=missing_cfg.threshold_scale,
            seed=missing_cfg.seed,
            available_mask=available,
            return_split=True,
        )
        mnar_sub = mnar_sub.astype(np.uint8)
    else:
        x_observed = x_true.copy()
        observed_mask = available.astype(np.float32)
        injected_missing = np.zeros_like(available, dtype=np.uint8)
        info = None

    x_input = fill_missing(x_observed, mode=fill_mode, seed=fill_seed)
    missing_info = {} if info is None else info.__dict__.copy()
    return DataBundle(
        x_raw=x_raw.astype(np.float32),
        x_true=x_true.astype(np.float32),
        x_input=x_input.astype(np.float32),
        observed_mask=observed_mask.astype(np.float32),
        injected_missing_mask=injected_missing.astype(np.uint8),
        available_mask=available.astype(np.uint8),
        batches=batches.astype(np.int64),
        cell_types=cell_types.astype(np.int64),
        protein_ids=protein_ids.astype(object),
        protein_features=protein_features.astype(np.float32),
        missing_info=missing_info,
        feature_cache_info=dict(feature_cache_info),
        scale_info=dict(scale_info),
        injected_mnar_mask=mnar_sub,
    )


def load_data_bundle(
    data_cfg: DataConfig,
    missing_cfg: MissingConfig,
    model_cfg: ModelConfig,
    seed: int,
) -> DataBundle:
    if data_cfg.synthetic:
        return _synthetic_bundle(data_cfg, missing_cfg, model_cfg, seed)
    if not Path(data_cfg.path).exists():
        raise FileNotFoundError(data_cfg.path)
    return _load_h5ad_bundle(data_cfg, missing_cfg, model_cfg)
