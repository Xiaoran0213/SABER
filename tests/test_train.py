from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset

from saber.train import _split_dataset


def test_split_dataset_matches_legacy_randomstate_floor_rule():
    dataset = TensorDataset(torch.arange(20, dtype=torch.float32).view(10, 2))

    train_set, val_set = _split_dataset(dataset, val_frac=0.25, seed=42)

    expected = np.arange(10)
    np.random.RandomState(42).shuffle(expected)
    assert val_set is not None
    assert val_set.indices == expected[:2].tolist()
    assert train_set.indices == expected[2:].tolist()
