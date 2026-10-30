from __future__ import annotations

import numpy as np
import anndata as ad

from saber.feature_cache import load_protein_feature_cache, save_protein_feature_cache
from saber.prepare import prepare_aligned_inputs


def test_prepare_aligned_inputs_drops_missing_with_report(tmp_path) -> None:
    adata = ad.AnnData(np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32))
    adata.var_names = ["A", "B", "C"]
    data_path = tmp_path / "input.h5ad"
    adata.write_h5ad(data_path)

    cache_path = tmp_path / "legacy_features.npy"
    cache_ids_path = tmp_path / "legacy_ids.npy"
    np.save(cache_path, np.asarray([[10.0, 0.0], [30.0, 0.0], [30.5, 0.0], [40.0, 0.0]], dtype=np.float32))
    np.save(cache_ids_path, np.asarray(["A", "C", "C", "D"], dtype=object))

    result = prepare_aligned_inputs(
        data_path=data_path,
        cache_path=cache_path,
        cache_protein_ids_path=cache_ids_path,
        outdir=tmp_path / "prepared",
        name="toy",
        expected_dim=2,
        duplicate_cache_policy="first",
        drop_missing_proteins=True,
    )

    prepared = ad.read_h5ad(result.h5ad_path)
    cache = load_protein_feature_cache(result.feature_cache_path, protein_ids=np.asarray(prepared.var_names, dtype=object), expected_dim=2)

    assert list(prepared.var_names) == ["A", "C"]
    np.testing.assert_allclose(cache.features, [[10.0, 0.0], [30.0, 0.0]])
    assert result.report["n_dropped_missing_requested_proteins"] == 1
    assert result.report["n_duplicate_cache_protein_ids"] == 1
