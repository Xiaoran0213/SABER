from __future__ import annotations

from dataclasses import replace

import pytest

from saber.config import ConfigError, ModelConfig, SaberConfig, validate_config


def _valid_config(**expansion_overrides):
    cfg = SaberConfig()
    expansion = replace(cfg.expansion, **expansion_overrides)
    return replace(cfg, data=replace(cfg.data, path="/tmp/dummy.h5ad"), expansion=expansion)


def test_default_protein_features_are_frozen() -> None:
    assert ModelConfig().trainable_protein_features is False
    assert SaberConfig().model.trainable_protein_features is False


def test_expansion_min_target_observed_allows_zero() -> None:
    validate_config(_valid_config(min_target_observed=0))


def test_expansion_min_target_observed_rejects_negative() -> None:
    with pytest.raises(ConfigError, match="non-negative"):
        validate_config(_valid_config(min_target_observed=-1))
