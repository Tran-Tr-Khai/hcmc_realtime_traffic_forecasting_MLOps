from datetime import UTC, datetime
from io import BytesIO
import json
import logging
from pathlib import Path
from typing import Any

import polars as pl
import torch
import yaml

from config.settings import settings
from src.infrastructure.storage.s3_client import S3Client
from src.ml.datasets.traffic_dataset import build_adjacency_matrix, build_traffic_matrix
from src.ml.training.trainer import TrainingConfig, train_stgtn
from src.processing.offline.topology import load_graph_topology


ROOT_DIR = Path(__file__).resolve().parents[3]
MODEL_CONFIG_PATH = ROOT_DIR / "config" / "model_config.yaml"
JSON_CONTENT_TYPE = "application/json"
TORCH_CONTENT_TYPE = "application/octet-stream"
logger = logging.getLogger(__name__)


def read_train_parquet(storage: S3Client) -> pl.DataFrame:
    stream = storage.get_object_stream(settings.offline_pipeline.processed_history_key)
    return pl.read_parquet(BytesIO(stream.read()))


def load_model_config(config_path: Path = MODEL_CONFIG_PATH) -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def to_training_config(config: dict[str, Any]) -> TrainingConfig:
    model_config = config.get("model", {})
    training_config = config.get("training", {})

    return TrainingConfig(
        input_len=training_config.get("input_len", 12),
        output_len=training_config.get("output_len", 3),
        train_ratio=training_config.get("train_ratio", 0.8),
        batch_size=training_config.get("batch_size", 32),
        epochs=training_config.get("epochs", 20),
        learning_rate=training_config.get("learning_rate", 1e-3),
        hidden_dim=model_config.get("hidden_dim", 64),
        pe_dim=model_config.get("pe_dim", 8),
        num_heads=model_config.get("num_heads", 4),
        dropout=model_config.get("dropout", 0.1),
        grad_clip=training_config.get("grad_clip", 5.0),
        seed=training_config.get("seed", 42),
    )


def artifact_prefix(config: dict[str, Any]) -> str:
    return config.get("artifacts", {}).get("prefix", "models/stgtn/latest")


def model_name(config: dict[str, Any]) -> str:
    return config.get("model", {}).get("name", "stgtn")


def to_json_bytes(data: dict[str, Any]) -> bytes:
    return json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")


def model_to_buffer(model: torch.nn.Module) -> BytesIO:
    buffer = BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    return buffer


def upload_json(storage: S3Client, key: str, data: dict[str, Any]) -> None:
    storage.put_object(key=key, data=to_json_bytes(data), content_type=JSON_CONTENT_TYPE)


def train_model(s3_client: S3Client | None = None, config: dict[str, Any] | None = None) -> None:
    storage = s3_client or S3Client()
    model_config = config or load_model_config()
    training_config = to_training_config(model_config)
    output_prefix = artifact_prefix(model_config)

    logger.info("loading training data: %s", settings.offline_pipeline.processed_history_key)
    dataframe = read_train_parquet(storage)
    traffic_matrix = build_traffic_matrix(dataframe)

    logger.info("loading graph: %s", settings.offline_pipeline.graph_key)
    topology = load_graph_topology(storage.get_object_stream(settings.offline_pipeline.graph_key))
    adjacency_matrix = build_adjacency_matrix(topology, traffic_matrix.sensor_columns)

    logger.info(
        "training matrix: timesteps=%s, sensors=%s",
        f"{traffic_matrix.num_timesteps:,}",
        f"{traffic_matrix.num_sensors:,}",
    )
    logger.info(
        "training config: model=%s, input_len=%s, output_len=%s, epochs=%s, batch_size=%s",
        model_name(model_config),
        training_config.input_len,
        training_config.output_len,
        training_config.epochs,
        training_config.batch_size,
    )

    result = train_stgtn(
        data=traffic_matrix.values,
        adjacency_matrix=adjacency_matrix,
        config=training_config,
    )

    metadata = {
        "model_name": model_name(model_config),
        "created_at": datetime.now(UTC).isoformat(),
        "train_data_key": settings.offline_pipeline.processed_history_key,
        "graph_key": settings.offline_pipeline.graph_key,
        "input_len": training_config.input_len,
        "output_len": training_config.output_len,
        "train_ratio": training_config.train_ratio,
        "batch_size": training_config.batch_size,
        "epochs": training_config.epochs,
        "learning_rate": training_config.learning_rate,
        "hidden_dim": training_config.hidden_dim,
        "pe_dim": training_config.pe_dim,
        "num_heads": training_config.num_heads,
        "dropout": training_config.dropout,
        "grad_clip": training_config.grad_clip,
        "seed": training_config.seed,
        "num_timesteps": traffic_matrix.num_timesteps,
        "num_sensors": traffic_matrix.num_sensors,
        "sensor_columns": traffic_matrix.sensor_columns,
        "max_flow": result.max_flow,
        "best_epoch": result.best_epoch,
    }
    metrics = {
        "train": result.train_metrics,
        "validation": result.validation_metrics,
    }

    storage.put_object(
        key=f"{output_prefix}/model.pt",
        data=model_to_buffer(result.model),
        content_type=TORCH_CONTENT_TYPE,
    )
    upload_json(storage, f"{output_prefix}/metadata.json", metadata)
    upload_json(storage, f"{output_prefix}/metrics.json", metrics)

    logger.info(
        "training complete: best_epoch=%s, val_mae=%.4f, val_rmse=%.4f, val_mape=%.2f%%",
        result.best_epoch,
        result.validation_metrics["mae"],
        result.validation_metrics["rmse"],
        result.validation_metrics["mape"],
    )
    logger.info("artifacts uploaded: %s", output_prefix)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[train-model] %(message)s")
    train_model()
