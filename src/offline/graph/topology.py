"""
Graph topology data structure.
Contains the GraphTopology dataclass and abstract GraphProvider interface.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Set
import numpy as np


@dataclass
class GraphTopology:
    """
    Immutable representation of graph topology.
    
    This structure enforces the constraint that the adjacency matrix
    is the authoritative source of truth for graph connectivity.
    """
    
    # The authoritative adjacency matrix (N x N)
    adjacency_matrix: np.ndarray
    
    # Ordered list of node IDs that correspond to matrix indices
    # node_ids[i] gives the sensor ID for matrix row/col i
    node_ids: List[int]
    
    # Mapping from sensor ID to matrix index
    node_to_index: Dict[int, int]
    
    # Distance matrix (if available)
    distance_matrix: np.ndarray = None
    
    def __post_init__(self):
        """Validate topology consistency."""
        n_nodes = len(self.node_ids)
        
        # Validate adjacency matrix shape
        if self.adjacency_matrix.shape != (n_nodes, n_nodes):
            raise ValueError(
                f"Adjacency matrix shape {self.adjacency_matrix.shape} "
                f"does not match {n_nodes} nodes"
            )
        
        # Validate node_to_index mapping
        if len(self.node_to_index) != n_nodes:
            raise ValueError(
                f"node_to_index has {len(self.node_to_index)} entries "
                f"but expected {n_nodes}"
            )
        
        # Validate distance matrix if provided
        if self.distance_matrix is not None:
            if self.distance_matrix.shape != (n_nodes, n_nodes):
                raise ValueError(
                    f"Distance matrix shape {self.distance_matrix.shape} "
                    f"does not match {n_nodes} nodes"
                )
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        return len(self.node_ids)
    
    def get_neighbors(self, node_id: int) -> List[int]:
        """
        Get neighbors of a node.
        
        Args:
            node_id: Sensor ID
            
        Returns:
            List of neighbor sensor IDs (excluding self-loops)
        """
        if node_id not in self.node_to_index:
            return []
        
        idx = self.node_to_index[node_id]
        neighbor_indices = np.where(self.adjacency_matrix[idx] > 0)[0]
        
        # Exclude self-loop
        neighbor_ids = [
            self.node_ids[i] for i in neighbor_indices
            if i != idx
        ]
        
        return neighbor_ids
    
    def has_node(self, node_id: int) -> bool:
        """Check if a node exists in the graph."""
        return node_id in self.node_to_index
    
    def filter_to_nodes(self, node_ids: Set[int]) -> 'GraphTopology':
        """
        Create a subgraph containing only specified nodes.
        
        This is used to align the graph with available traffic data sensors.
        
        Args:
            node_ids: Set of sensor IDs to keep
            
        Returns:
            New GraphTopology with filtered nodes
        """
        # Find intersection of requested nodes and graph nodes
        valid_nodes = sorted(set(node_ids) & set(self.node_ids))
        
        if not valid_nodes:
            raise ValueError("No valid nodes to filter")
        
        # Get indices of nodes to keep
        indices = [self.node_to_index[nid] for nid in valid_nodes]
        
        # Extract submatrix
        new_adj = self.adjacency_matrix[np.ix_(indices, indices)]
        
        # Build new mapping
        new_node_to_index = {nid: idx for idx, nid in enumerate(valid_nodes)}
        
        # Extract distance submatrix if available
        new_dist = None
        if self.distance_matrix is not None:
            new_dist = self.distance_matrix[np.ix_(indices, indices)]
        
        return GraphTopology(
            adjacency_matrix=new_adj,
            node_ids=valid_nodes,
            node_to_index=new_node_to_index,
            distance_matrix=new_dist
        )


class GraphProvider(ABC):
    """
    Abstract interface for graph topology providers.
    
    This abstraction ensures that downstream components do not depend on:
    - How the graph is stored (JSON, database, etc.)
    - Whether the graph is static, dynamic, or learned
    - The specific format of graph data
    """
    
    @abstractmethod
    def load_topology(self) -> GraphTopology:
        """
        Load and return graph topology.
        
        Returns:
            GraphTopology with adjacency matrix and node mappings
        """
        pass
    
    @abstractmethod
    def get_sensor_metadata(self) -> Dict[int, Dict]:
        """
        Get metadata for all sensors in the graph.
        
        Returns:
            Dictionary mapping sensor ID to metadata
            (e.g., location, name)
        """
        pass
