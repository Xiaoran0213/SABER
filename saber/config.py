from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a SABER config is invalid."""


def _check_keys(section: str, values: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(values.keys()) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise ConfigError(f"Unknown key(s) in '{section}': {joined}")


@dataclass(frozen=True)
class ProjectConfig:
    name: str = "saber_run"
    outdir: str = "outputs"
    seed: int = 42


@dataclass(frozen=True)
class DataConfig:
    synthetic: bool = False
    path: str = ""
    protein_features_path: str = ""
    protein_features_ids_path: str = ""
    protein_ids_path: str = ""
    batch_key: str = "batch"
    cell_type_key: str = "cell_type"
    protein_id_field: str = ""
    scale_mode: str = "none"
    fill_mode: str = "column_mean"
    fill_seed: int = 42
    allow_random_protein_features: bool = False
    n_cells: int = 64
    n_proteins: int = 48
    n_batches: int = 2
    n_cell_types: int = 2


@dataclass(frozen=True)
class MissingConfig:
    enabled: bool = True
    total_fraction: float = 0.1
    mnar_fraction: float = 0.5
    threshold_quantile: float = 0.1
    threshold_scale: float = 0.3
    seed: int = 42


@dataclass(frozen=True)
class ModelConfig:
    latent_dim: int = 16
    ae_h1: int = 128
    ae_h2: int = 64
    protein_feature_dim: int = 1280
    protein_proj_dim: int = 256
    protein_proj_hidden_dim: int = 256
    protein_proj_dropout: float = 0.1
    bilinear_with_w: bool = True
    trainable_protein_features: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 300
    batch_size: int = 32
    lr: float = 5e-4
    weight_decay: float = 0.01
    val_frac: float = 0.05
    device: str = "auto"
    lr_mult_protein_features: float = 1.0
    lr_mult_protein_encoder: float = 1.0


@dataclass(frozen=True)
class EvaluationConfig:
    imputation: bool = True


@dataclass(frozen=True)
class ExpansionConfig:
    target_fraction: float = 0.1
    seed: int = 42
    min_target_observed: int = 2
    # Naive PLM reference models written as extra pred_target__<method> keys.
    references: tuple[str, ...] = ("plm_ridge", "plm_knn")
    ridge_lambda: float = 1.0
    knn_ks: tuple[int, ...] = (5, 10)
    # How target proteins are chosen for holdout.
    #   "random" - uniform random holdout (original behavior; default).
    #   "cluster" - hold out whole ESM-2 cosine-similarity clusters so close
    #               sequence neighbours leave training too (leakage-controlled
    #               zero-shot expansion test).
    holdout_mode: str = "random"
    # cosine-distance merge threshold for hierarchical clustering of proteins by
    # ESM-2 features; only used when holdout_mode == "cluster".
    cluster_cosine_threshold: float = 0.10
    # path to a MMseqs2/CD-HIT cluster TSV ("rep\tmember" per line) defining
    # sequence-identity clusters; only used when holdout_mode == "sequence".
    cluster_map_path: str = ""
    # explicit holdout: protein IDs (UniProt AC) to hold out, used when
    # holdout_mode == "explicit". Case-study marker holdout (e.g. zonation
    # markers); tests sequence-informed prediction of biologically defined sets.
    target_protein_ids: tuple[str, ...] = ()
    # alternatively, a text file (one protein ID per line) loaded into the above.
    target_protein_ids_path: str = ""


@dataclass(frozen=True)
class SaberConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    missing: MissingConfig = field(default_factory=MissingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    expansion: ExpansionConfig = field(default_factory=ExpansionConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_TOP_KEYS = {"project", "data", "missing", "model", "training", "evaluation", "expansion"}
_SECTION_KEYS: dict[str, set[str]] = {
    "project": set(ProjectConfig.__dataclass_fields__.keys()),
    "data": set(DataConfig.__dataclass_fields__.keys()),
    "missing": set(MissingConfig.__dataclass_fields__.keys()),
    "model": set(ModelConfig.__dataclass_fields__.keys()),
    "training": set(TrainingConfig.__dataclass_fields__.keys()),
    "evaluation": set(EvaluationConfig.__dataclass_fields__.keys()),
    "expansion": set(ExpansionConfig.__dataclass_fields__.keys()),
}


def _known_roots() -> dict[str, str]:
    from .paths import benchmark_root, data_root, plm_root

    roots: dict[str, str] = {}
    for key, getter in (
        ("SABER_DATA_ROOT", data_root),
        ("SABER_PLM_ROOT", plm_root),
        ("SABER_BENCHMARK_ROOT", benchmark_root),
    ):
        env = os.environ.get(key, "").strip()
        if env:
            roots[key] = env
            continue
        try:
            roots[key] = str(getter())
        except RuntimeError:
            pass
    return roots


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        expanded = value
        for key, root in _known_roots().items():
            expanded = expanded.replace("${" + key + "}", root)
            expanded = expanded.replace("$" + key, root)
        expanded = os.path.expandvars(expanded)
        leftover = re.search(r"\$\{(SABER_[A-Z_]+)\}", expanded)
        if leftover:
            raise ConfigError(
                f"Config references ${{{leftover.group(1)}}} but that variable is not "
                "configured. Export it (or add it to .saber_env, see env.example and "
                "docs/DATA.md) and retry."
            )
        return expanded
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_expand_env(v) for v in value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def load_config(path: str | Path) -> SaberConfig:
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}
    raw = _expand_env(raw)
    if not isinstance(raw, dict):
        raise ConfigError(f"Config must be a mapping: {path}")
    _check_keys("top-level", raw, _TOP_KEYS)

    sections: dict[str, dict[str, Any]] = {}
    for name in _TOP_KEYS:
        value = raw.get(name, {}) or {}
        if not isinstance(value, dict):
            raise ConfigError(f"Config section '{name}' must be a mapping")
        _check_keys(name, value, _SECTION_KEYS[name])
        sections[name] = value

    cfg = SaberConfig(
        project=ProjectConfig(**sections["project"]),
        data=DataConfig(**sections["data"]),
        missing=MissingConfig(**sections["missing"]),
        model=ModelConfig(**sections["model"]),
        training=TrainingConfig(**sections["training"]),
        evaluation=EvaluationConfig(**sections["evaluation"]),
        expansion=ExpansionConfig(**sections["expansion"]),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: SaberConfig) -> None:
    if cfg.data.scale_mode not in {"none", "zscore"}:
        raise ConfigError("data.scale_mode must be 'none' or 'zscore'")
    from .missing import FILL_MODES

    if cfg.data.fill_mode not in FILL_MODES:
        raise ConfigError(f"data.fill_mode must be one of {list(FILL_MODES)}")
    if not cfg.data.synthetic and not cfg.data.path:
        raise ConfigError("data.path is required when data.synthetic=false")
    if not 0.0 <= cfg.missing.total_fraction < 1.0:
        raise ConfigError("missing.total_fraction must be in [0, 1)")
    if not 0.0 <= cfg.missing.mnar_fraction <= 1.0:
        raise ConfigError("missing.mnar_fraction must be in [0, 1]")
    if not 0.0 <= cfg.training.val_frac < 1.0:
        raise ConfigError("training.val_frac must be in [0, 1)")
    if cfg.model.latent_dim <= 0:
        raise ConfigError("model.latent_dim must be positive")
    if cfg.model.protein_feature_dim <= 0:
        raise ConfigError("model.protein_feature_dim must be positive")
    if cfg.training.epochs <= 0:
        raise ConfigError("training.epochs must be positive")
    if cfg.training.batch_size <= 0:
        raise ConfigError("training.batch_size must be positive")
    if cfg.expansion.holdout_mode != "explicit" and not 0.0 < cfg.expansion.target_fraction < 1.0:
        raise ConfigError("expansion.target_fraction must be in (0, 1)")
    if cfg.expansion.min_target_observed < 0:
        raise ConfigError("expansion.min_target_observed must be non-negative")
    if cfg.expansion.holdout_mode not in {"random", "cluster", "sequence", "explicit"}:
        raise ConfigError("expansion.holdout_mode must be 'random', 'cluster', 'sequence', or 'explicit'")
    if cfg.expansion.holdout_mode == "explicit" and not cfg.expansion.target_protein_ids and not cfg.expansion.target_protein_ids_path:
        raise ConfigError("expansion.target_protein_ids (or _path) is required when holdout_mode == 'explicit'")
    if cfg.expansion.holdout_mode == "sequence" and not cfg.expansion.cluster_map_path:
        raise ConfigError("expansion.cluster_map_path is required when holdout_mode == 'sequence'")
    if not 0.0 < cfg.expansion.cluster_cosine_threshold <= 1.0:
        raise ConfigError("expansion.cluster_cosine_threshold must be in (0, 1]")
    allowed_refs = {"plm_ridge", "plm_knn"}
    for ref in cfg.expansion.references:
        if ref not in allowed_refs:
            raise ConfigError(
                f"expansion.references entries must be in {sorted(allowed_refs)}; got {ref!r}"
            )
    if cfg.expansion.ridge_lambda <= 0.0:
        raise ConfigError("expansion.ridge_lambda must be positive")
    if "plm_knn" in cfg.expansion.references and not cfg.expansion.knn_ks:
        raise ConfigError("expansion.knn_ks must be non-empty when plm_knn is a reference")
    for k in cfg.expansion.knn_ks:
        if int(k) <= 0:
            raise ConfigError("expansion.knn_ks entries must be positive integers")
