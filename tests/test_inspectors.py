from __future__ import annotations

import numpy as np

from saber.feature_cache import save_protein_feature_cache
from saber.inspectors import inspect_feature_cache


def test_inspect_feature_cache_reports_coverage(tmp_path) -> None:
    cache_path = tmp_path / "features.npz"
    save_protein_feature_cache(
        cache_path,
        np.ones((3, 2), dtype=np.float32),
        np.asarray(["A", "B", "C"], dtype=object),
    )

    summary = inspect_feature_cache(cache_path, expected_dim=2)

    assert summary["n_feature_rows"] == 3
    assert summary["feature_dim"] == 2
    assert summary["expected_dim_match"] is True
