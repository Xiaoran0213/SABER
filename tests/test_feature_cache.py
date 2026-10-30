from __future__ import annotations

import numpy as np
import pytest

from saber.feature_cache import FeatureCacheError, load_protein_feature_cache, save_protein_feature_cache


def test_feature_cache_reorders_by_protein_id(tmp_path) -> None:
    cache_path = tmp_path / "features.npz"
    features = np.asarray([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)
    save_protein_feature_cache(cache_path, features, np.asarray(["B", "A", "C"], dtype=object))

    cache = load_protein_feature_cache(cache_path, protein_ids=np.asarray(["A", "B", "C"], dtype=object), expected_dim=2)

    np.testing.assert_allclose(cache.features, [[2.0, 2.0], [1.0, 1.0], [3.0, 3.0]])
    assert cache.metadata["id_alignment"] == "protein_id"


def test_feature_cache_fails_on_missing_id(tmp_path) -> None:
    cache_path = tmp_path / "features.npz"
    save_protein_feature_cache(cache_path, np.ones((2, 2), dtype=np.float32), np.asarray(["A", "B"], dtype=object))

    with pytest.raises(FeatureCacheError, match="missing"):
        load_protein_feature_cache(cache_path, protein_ids=np.asarray(["A", "C"], dtype=object), expected_dim=2)
