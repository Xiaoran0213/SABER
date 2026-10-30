from __future__ import annotations

import numpy as np

from saber.missing import fill_missing_column_mean, inject_missing


def test_inject_missing_masks_requested_fraction() -> None:
    x = np.arange(100, dtype=np.float32).reshape(10, 10)
    x_obs, observed, missing, info = inject_missing(
        x_full=x,
        total_fraction=0.2,
        mnar_fraction=0.5,
        threshold_quantile=0.3,
        threshold_scale=0.1,
        seed=7,
    )

    assert info.n_masked == 20
    assert int(missing.sum()) == 20
    assert int(observed.sum()) == 80
    assert np.isnan(x_obs[missing.astype(bool)]).all()


def test_fill_missing_column_mean_uses_column_values() -> None:
    x = np.asarray([[1.0, np.nan], [3.0, 2.0], [np.nan, 4.0]], dtype=np.float32)
    filled = fill_missing_column_mean(x)

    np.testing.assert_allclose(filled[:, 0], [1.0, 3.0, 2.0])
    np.testing.assert_allclose(filled[:, 1], [3.0, 2.0, 4.0])


def test_fill_missing_column_mean_uses_fallback_for_all_missing_columns() -> None:
    x = np.asarray([[np.nan, 1.0], [np.nan, np.nan]], dtype=np.float32)

    filled = fill_missing_column_mean(x, fallback=-2.5)

    np.testing.assert_allclose(filled[:, 0], [-2.5, -2.5])
    np.testing.assert_allclose(filled[:, 1], [1.0, 1.0])
