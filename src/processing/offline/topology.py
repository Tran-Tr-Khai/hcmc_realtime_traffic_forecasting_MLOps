import json
from dataclasses import dataclass
from typing import Any, BinaryIO


@dataclass(frozen=True)
class GraphTopology:
    node_ids: list[int]
    adjacency_matrix: list[list[float]]
    distance_matrix: list[list[float]] | None = None

    @property
    def node_to_index(self) -> dict[int, int]:
        return {node_id: index for index, node_id in enumerate(self.node_ids)}

    def has_node(self, node_id: int) -> bool:
        return node_id in self.node_to_index

    def get_neighbors(self, node_id: int) -> list[int]:
        node_index = self.node_to_index.get(node_id)
        if node_index is None:
            return []

        return [
            self.node_ids[index]
            for index, weight in enumerate(self.adjacency_matrix[node_index])
            if index != node_index and weight > 0
        ]

    def distance_between(self, source_id: int, target_id: int) -> float | None:
        if self.distance_matrix is None:
            return None

        source_index = self.node_to_index.get(source_id)
        target_index = self.node_to_index.get(target_id)
        if source_index is None or target_index is None:
            return None

        return float(self.distance_matrix[source_index][target_index])


def load_graph_topology(stream: BinaryIO) -> GraphTopology:
    data: dict[str, Any] = json.loads(stream.read().decode("utf-8"))
    adjacency_matrix = data.get("adjacency-matrix")

    if not adjacency_matrix:
        raise ValueError("Graph data does not contain adjacency-matrix")

    node_ids = sorted(int(node_id) for node_id in data.get("camera-dictionary", {}))
    if not node_ids:
        node_ids = list(range(len(adjacency_matrix)))

    if len(node_ids) != len(adjacency_matrix):
        raise ValueError(
            "Graph topology is inconsistent: camera-dictionary has "
            f"{len(node_ids)} node(s) but adjacency-matrix has {len(adjacency_matrix)} row(s). "
            "Regenerate the clustered graph file so both match before using it for training or imputation."
        )

    return GraphTopology(
        node_ids=node_ids,
        adjacency_matrix=adjacency_matrix,
        distance_matrix=data.get("distance-matrix"),
    )