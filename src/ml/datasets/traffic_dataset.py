from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from src.processing.offline.topology import GraphTopology


TIMESTAMP_COLUMN = "timestamp"


@dataclass(frozen=True)
class TrafficMatrix:
    values: np.ndarray
    sensor_columns: list[str]

    @property
    def num_timesteps(self) -> int:
        return self.values.shape[0]

    @property
    def num_sensors(self) -> int:
        return self.values.shape[1]


def get_sensor_columns(df: pl.DataFrame) -> list[str]:
    return [column for column in df.columns if column != TIMESTAMP_COLUMN]


def build_traffic_matrix(df: pl.DataFrame) -> TrafficMatrix:
    sensor_columns = get_sensor_columns(df)
    if not sensor_columns:
        raise ValueError("Traffic dataframe has no sensor columns")

    null_count = int(df.select(sensor_columns).null_count().sum_horizontal().sum())
    if null_count > 0:
        raise ValueError(f"Traffic dataframe still has {null_count:,} null sensor values")

    values = df.select(sensor_columns).to_numpy().astype(np.float32)
    return TrafficMatrix(values=values, sensor_columns=sensor_columns)


def build_adjacency_matrix(topology: GraphTopology, sensor_columns: list[str]) -> np.ndarray:
    """Reorder/filter the graph adjacency matrix to match the training matrix's sensor column order."""
    missing = [column for column in sensor_columns if not topology.has_node(int(column))]
    if missing:
        preview = ", ".join(missing[:10]) + ("..." if len(missing) > 10 else "")
        raise ValueError(f"{len(missing)} sensor column(s) are missing from the graph topology: {preview}")

    indices = [topology.node_to_index[int(column)] for column in sensor_columns]
    adjacency = np.asarray(topology.adjacency_matrix, dtype=np.float32)
    return adjacency[np.ix_(indices, indices)]


class TrafficWindowDataset(Dataset):
    def __init__(self, data: np.ndarray, input_len: int, output_len: int):
        if data.ndim != 2:
            raise ValueError(f"Expected 2D traffic matrix, got shape {data.shape}")
        if input_len <= 0 or output_len <= 0:
            raise ValueError("input_len and output_len must be positive")

        self.data = data.astype(np.float32, copy=False)
        self.input_len = input_len
        self.output_len = output_len
        self.num_samples = len(data) - input_len - output_len + 1

        if self.num_samples <= 0:
            raise ValueError(
                "Traffic matrix is too short for the requested windows: "
                f"timesteps={len(data)}, input_len={input_len}, output_len={output_len}"
            )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= self.num_samples:
            raise IndexError(f"index {index} out of range")

        input_start = index
        input_end = input_start + self.input_len
        output_end = input_end + self.output_len

        x = self.data[input_start:input_end]
        y = self.data[input_end:output_end]
        return torch.from_numpy(x[..., np.newaxis]), torch.from_numpy(y[..., np.newaxis])
