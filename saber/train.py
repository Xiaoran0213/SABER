from __future__ import annotations

import resource
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from .config import TrainingConfig
from .data import DataBundle
from .model import SABER


@dataclass(frozen=True)
class TrainResult:
    history: list[dict[str, float]]
    device: str
    resources: dict[str, Any] = field(default_factory=dict)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    numerator = (((pred - target) * mask) ** 2).sum()
    denominator = mask.sum().clamp_min(1.0)
    return numerator / denominator


def _split_dataset(dataset: TensorDataset, val_frac: float, seed: int) -> tuple[Subset, Subset | None]:
    n = len(dataset)
    n_val = max(1, int(np.floor(float(val_frac) * n))) if float(val_frac) > 0.0 and n >= 2 else 0
    if n_val <= 0:
        return Subset(dataset, list(range(n))), None
    if n_val >= n:
        raise ValueError("training.val_frac leaves no training samples")

    rng = np.random.RandomState(int(seed))
    indices = np.arange(n)
    rng.shuffle(indices)
    val_idx = indices[:n_val].tolist()
    train_idx = indices[n_val:].tolist()
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def _make_loader(dataset: Iterable, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator if shuffle else None,
    )


def _epoch_loss(model: SABER, loader: DataLoader, device: torch.device) -> float:
    losses: list[float] = []
    with torch.no_grad():
        for x_batch, mask_batch in loader:
            x_batch = x_batch.to(device)
            mask_batch = mask_batch.to(device)
            pred, _, _ = model(x_batch)
            losses.append(float(masked_mse(pred, x_batch, mask_batch).detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def train_model(
    model: SABER,
    data: DataBundle,
    cfg: TrainingConfig,
    seed: int,
) -> TrainResult:
    set_seed(seed)
    device = resolve_device(cfg.device)
    model.to(device)

    x = torch.as_tensor(data.x_input, dtype=torch.float32)
    observed = torch.as_tensor(data.observed_mask, dtype=torch.float32)
    dataset = TensorDataset(x, observed)
    train_set, val_set = _split_dataset(dataset, cfg.val_frac, seed)
    train_loader = _make_loader(train_set, cfg.batch_size, shuffle=True, seed=seed)
    val_loader = None if val_set is None else _make_loader(val_set, cfg.batch_size, shuffle=False, seed=seed)

    lr = float(cfg.lr)
    feature_param = getattr(model, "protein_features", None)
    feature_ids: set[int] = set()
    param_groups = []
    if isinstance(feature_param, torch.nn.Parameter) and feature_param.requires_grad:
        feature_ids.add(id(feature_param))
        param_groups.append({"params": [feature_param], "lr": lr * float(cfg.lr_mult_protein_features)})

    protein_projection = getattr(model, "protein_projection", model.protein_encoder)
    protein_encoder_params = [
        p for p in protein_projection.parameters()
        if p.requires_grad and id(p) not in feature_ids
    ]
    protein_encoder_ids = {id(p) for p in protein_encoder_params}
    if protein_encoder_params:
        param_groups.append({"params": protein_encoder_params, "lr": lr * float(cfg.lr_mult_protein_encoder)})

    grouped = feature_ids | protein_encoder_ids
    base_params = [p for p in model.parameters() if p.requires_grad and id(p) not in grouped]
    if base_params:
        param_groups.append({"params": base_params, "lr": lr})

    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=float(cfg.weight_decay))

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB (peak so far)
    t0 = time.perf_counter()
    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        train_losses: list[float] = []
        for x_batch, mask_batch in train_loader:
            x_batch = x_batch.to(device)
            mask_batch = mask_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred, _, _ = model(x_batch)
            loss = masked_mse(pred, x_batch, mask_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
        }
        if val_loader is not None:
            model.eval()
            row["val_loss"] = _epoch_loss(model, val_loader, device)
            if row["val_loss"] < best_val_loss:
                best_val_loss = float(row["val_loss"])
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append(row)

    if best_state is not None:
        model.load_state_dict(best_state)

    training_time_sec = time.perf_counter() - t0
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB
    resources: dict[str, Any] = {
        "training_time_sec": float(training_time_sec),
        "n_epochs_run": int(len(history)),
        "peak_cpu_rss_mb": float(max(rss_before, rss_after) / 1024.0),
    }
    if device.type == "cuda":
        resources["peak_gpu_memory_mb"] = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
        resources["gpu_name"] = torch.cuda.get_device_name(device)

    return TrainResult(history=history, device=str(device), resources=resources)


@torch.no_grad()
def predict_all(model: SABER, x_input: np.ndarray, device: str | torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device_obj = torch.device(device)
    model.eval()
    model.to(device_obj)
    x = torch.as_tensor(x_input, dtype=torch.float32, device=device_obj)
    pred, z, e = model(x)
    return (
        pred.detach().cpu().numpy().astype(np.float32),
        z.detach().cpu().numpy().astype(np.float32),
        e.detach().cpu().numpy().astype(np.float32),
    )
