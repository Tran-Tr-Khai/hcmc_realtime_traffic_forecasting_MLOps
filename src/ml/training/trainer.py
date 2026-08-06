from dataclasses import dataclass
import logging
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.ml.datasets.traffic_dataset import TrafficWindowDataset
from src.ml.evaluation.metrics import regression_metrics
from src.ml.models.stgtn import STGTN, build_adjacency_mask, compute_laplacian_positional_encoding


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TimeSeriesSplit:
    train: np.ndarray
    validation: np.ndarray


@dataclass(frozen=True)
class TrainingConfig:
    input_len: int = 12
    output_len: int = 3
    train_ratio: float = 0.8
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-3
    hidden_dim: int = 64
    pe_dim: int = 8
    num_heads: int = 4
    dropout: float = 0.1
    grad_clip: float = 5.0
    seed: int = 42


@dataclass(frozen=True)
class TrainingResult:
    model: STGTN
    train_metrics: dict[str, float]
    validation_metrics: dict[str, float]
    best_epoch: int
    max_flow: float


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_by_time(data: np.ndarray, train_ratio: float) -> TimeSeriesSplit:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    split_index = int(len(data) * train_ratio)
    return TimeSeriesSplit(train=data[:split_index], validation=data[split_index:])


def scale_by_max(data: np.ndarray, max_flow: float) -> np.ndarray:
    divisor = max(max_flow, 1.0)
    return (data / divisor).astype(np.float32)


def build_dataloader(
    data: np.ndarray,
    config: TrainingConfig,
    shuffle: bool,
) -> DataLoader:
    dataset = TrafficWindowDataset(
        data,
        input_len=config.input_len,
        output_len=config.output_len,
    )
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle)


def train_one_epoch(
    model: STGTN,
    dataloader: DataLoader,
    adjacency_mask: torch.Tensor,
    positional_encoding: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    log_interval = max(1, len(dataloader) // 10)

    for batch_index, (x, y) in enumerate(dataloader, start=1):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        prediction = model(x, adjacency_mask, positional_encoding)
        loss = criterion(prediction, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()

        if batch_index == 1 or batch_index % log_interval == 0 or batch_index == len(dataloader):
            logger.info("train batch %s/%s: loss=%.6f", batch_index, len(dataloader), loss.item())

    return total_loss / max(len(dataloader), 1)


def collect_predictions(
    model: STGTN,
    dataloader: DataLoader,
    adjacency_mask: torch.Tensor,
    positional_encoding: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for x, y in dataloader:
            prediction = model(x.to(device), adjacency_mask, positional_encoding).cpu().numpy()
            predictions.append(prediction)
            targets.append(y.numpy())

    return np.concatenate(targets), np.concatenate(predictions)


def evaluate_model(
    model: STGTN,
    dataloader: DataLoader,
    adjacency_mask: torch.Tensor,
    positional_encoding: torch.Tensor,
    device: torch.device,
    max_flow: float,
) -> dict[str, float]:
    y_true, y_pred = collect_predictions(model, dataloader, adjacency_mask, positional_encoding, device)
    return regression_metrics(y_true * max_flow, y_pred * max_flow)


def train_stgtn(
    data: np.ndarray,
    adjacency_matrix: np.ndarray,
    config: TrainingConfig,
    device: torch.device | None = None,
) -> TrainingResult:
    set_seed(config.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = split_by_time(data, train_ratio=config.train_ratio)
    max_flow = float(split.train.max())

    train_loader = build_dataloader(scale_by_max(split.train, max_flow), config, shuffle=True)
    validation_loader = build_dataloader(scale_by_max(split.validation, max_flow), config, shuffle=False)

    adjacency_mask = build_adjacency_mask(torch.from_numpy(adjacency_matrix)).to(device)
    positional_encoding = torch.from_numpy(
        compute_laplacian_positional_encoding(adjacency_matrix, config.pe_dim)
    ).to(device)
    model = STGTN(
        num_nodes=data.shape[1],
        input_dim=1,
        hidden_dim=config.hidden_dim,
        pe_dim=positional_encoding.shape[1],
        num_heads=config.num_heads,
        output_len=config.output_len,
        dropout=config.dropout,
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_state = None
    best_validation_mae = float("inf")
    best_epoch = 0

    logger.info(
        "training start: device=%s, train_samples=%s, validation_samples=%s, train_batches=%s, validation_batches=%s, max_flow=%.2f",
        device,
        f"{len(train_loader.dataset):,}",
        f"{len(validation_loader.dataset):,}",
        f"{len(train_loader):,}",
        f"{len(validation_loader):,}",
        max_flow,
    )

    for epoch in range(1, config.epochs + 1):
        started_at = perf_counter()
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            adjacency_mask=adjacency_mask,
            positional_encoding=positional_encoding,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=config.grad_clip,
        )
        validation_metrics = evaluate_model(
            model, validation_loader, adjacency_mask, positional_encoding, device, max_flow
        )

        is_best = validation_metrics["mae"] < best_validation_mae
        if is_best:
            best_validation_mae = validation_metrics["mae"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        logger.info(
            "epoch %s/%s: train_loss=%.6f, val_mae=%.4f, val_rmse=%.4f, val_mape=%.2f%%, elapsed=%.2fs%s",
            epoch,
            config.epochs,
            train_loss,
            validation_metrics["mae"],
            validation_metrics["rmse"],
            validation_metrics["mape"],
            perf_counter() - started_at,
            " best" if is_best else "",
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    logger.info("training finished: best_epoch=%s, best_val_mae=%.4f", best_epoch, best_validation_mae)

    return TrainingResult(
        model=model,
        train_metrics=evaluate_model(model, train_loader, adjacency_mask, positional_encoding, device, max_flow),
        validation_metrics=evaluate_model(
            model, validation_loader, adjacency_mask, positional_encoding, device, max_flow
        ),
        best_epoch=best_epoch,
        max_flow=max_flow,
    )
