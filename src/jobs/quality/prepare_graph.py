from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from config.settings import settings
from src.infrastructure.storage.s3_client import S3Client


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_LOCAL_GRAPH_PATH = ROOT_DIR / "data" / "hcmc-clustered-graph.json"
JSON_CONTENT_TYPE = "application/json"
logger = logging.getLogger(__name__)


def load_graph_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_square_matrix(matrix: list[list[Any]], name: str) -> None:
    size = len(matrix)
    invalid_rows = [index for index, row in enumerate(matrix) if len(row) != size]
    if invalid_rows:
        preview = ", ".join(str(index) for index in invalid_rows[:10])
        raise ValueError(f"{name} must be square: size={size}, invalid row(s): {preview}")


def trim_camera_dictionary(camera_dictionary: dict[str, Any], matrix_size: int) -> dict[str, Any]:
    return {
        str(node_id): camera_dictionary[str(node_id)]
        for node_id in range(matrix_size)
        if str(node_id) in camera_dictionary
    }


def prepare_graph_topology(graph_data: dict[str, Any]) -> dict[str, Any]:
    adjacency_matrix = graph_data.get("adjacency-matrix")
    if not adjacency_matrix:
        raise ValueError("Graph JSON does not contain adjacency-matrix")

    validate_square_matrix(adjacency_matrix, "adjacency-matrix")
    matrix_size = len(adjacency_matrix)

    camera_dictionary = graph_data.get("camera-dictionary") or {}
    trimmed_camera_dictionary = trim_camera_dictionary(camera_dictionary, matrix_size)
    if len(trimmed_camera_dictionary) != matrix_size:
        raise ValueError(
            "Cannot align camera-dictionary with adjacency-matrix: expected node ids "
            f"0..{matrix_size - 1}, found {len(trimmed_camera_dictionary)} matching camera(s)."
        )

    prepared_graph = dict(graph_data)
    prepared_graph["camera-dictionary"] = trimmed_camera_dictionary

    distance_matrix = prepared_graph.get("distance-matrix")
    if distance_matrix:
        validate_square_matrix(distance_matrix, "distance-matrix")
        if len(distance_matrix) != matrix_size:
            prepared_graph["distance-matrix"] = [
                row[:matrix_size] for row in distance_matrix[:matrix_size]
            ]

    return prepared_graph


def upload_prepared_graph(
    storage: S3Client,
    graph_data: dict[str, Any],
    key: str,
) -> None:
    payload = json.dumps(graph_data, ensure_ascii=False, indent=2).encode("utf-8")
    storage.put_object(key=key, data=payload, content_type=JSON_CONTENT_TYPE)


def prepare_graph(
    local_graph_path: Path = DEFAULT_LOCAL_GRAPH_PATH,
    s3_client: S3Client | None = None,
) -> None:
    storage = s3_client or S3Client()
    raw_graph = load_graph_json(local_graph_path)
    prepared_graph = prepare_graph_topology(raw_graph)

    raw_camera_count = len(raw_graph.get("camera-dictionary") or {})
    prepared_camera_count = len(prepared_graph["camera-dictionary"])
    matrix_size = len(prepared_graph["adjacency-matrix"])

    logger.info(
        "prepared graph: cameras=%s -> %s, adjacency=%sx%s",
        raw_camera_count,
        prepared_camera_count,
        matrix_size,
        matrix_size,
    )
    upload_prepared_graph(storage, prepared_graph, settings.offline_pipeline.graph_key)
    logger.info("uploaded graph: %s", settings.offline_pipeline.graph_key)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[prepare-graph] %(message)s")
    prepare_graph()
