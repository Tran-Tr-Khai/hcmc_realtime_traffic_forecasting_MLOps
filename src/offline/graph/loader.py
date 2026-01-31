"""
Consolidated graph loaders.
Supports loading from local JSON files or MinIO storage.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Optional
import numpy as np

from src.core.storage import MinIOClient
from .topology import GraphProvider, GraphTopology

logger = logging.getLogger(__name__)


class LocalGraphLoader(GraphProvider):
    """
    Loads graph topology from local JSON file.
    """
    
    def __init__(self, graph_file_path: str):
        self.graph_file_path = Path(graph_file_path)
        
        if not self.graph_file_path.exists():
            raise FileNotFoundError(f"Graph file not found: {self.graph_file_path}")
        
        self._topology = None
        self._sensor_metadata = None
    
    def load_topology(self) -> GraphTopology:
        if self._topology is not None:
            return self._topology
        
        logger.info(f"Loading graph topology from {self.graph_file_path}")
        
        with open(self.graph_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._topology = self._parse_topology_from_dict(data)
        return self._topology
    
    def get_sensor_metadata(self) -> Dict[int, Dict]:
        if self._sensor_metadata is not None:
            return self._sensor_metadata
        
        with open(self.graph_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._sensor_metadata = self._parse_metadata_from_dict(data)
        return self._sensor_metadata
    
    def _parse_topology_from_dict(self, data: dict) -> GraphTopology:
        """Parse topology from dictionary (shared logic)."""
        adj_list = data.get('adjacency-matrix', [])
        if not adj_list:
            raise ValueError("No adjacency-matrix found in graph file")
        
        adj_matrix = np.array(adj_list, dtype=np.float32)
        n_nodes = adj_matrix.shape[0]
        
        logger.info(f"Loaded adjacency matrix: shape {adj_matrix.shape}")
        
        # Extract distance matrix if available
        dist_matrix = None
        if 'distance-matrix' in data:
            dist_list = data['distance-matrix']
            dist_matrix = np.array(dist_list, dtype=np.float32)
            logger.info(f"Loaded distance matrix: shape {dist_matrix.shape}")
        
        # Nodes are implicitly [0, 1, 2, ..., n-1]
        node_ids = list(range(n_nodes))
        node_to_index = {nid: idx for idx, nid in enumerate(node_ids)}
        
        self._validate_adjacency_matrix(adj_matrix)
        
        topology = GraphTopology(
            adjacency_matrix=adj_matrix,
            node_ids=node_ids,
            node_to_index=node_to_index,
            distance_matrix=dist_matrix
        )
        
        logger.info(f"Graph topology loaded: {n_nodes} nodes, density={adj_matrix.mean():.3f}")
        return topology
    
    def _parse_metadata_from_dict(self, data: dict) -> Dict[int, Dict]:
        """Parse sensor metadata from dictionary (shared logic)."""
        camera_dict = data.get('camera-dictionary', {})
        
        metadata = {}
        for sensor_id_str, info in camera_dict.items():
            sensor_id = int(sensor_id_str)
            
            # info is [[lat, lon], "name"]
            if isinstance(info, list) and len(info) >= 2:
                location = info[0] if isinstance(info[0], list) else None
                name = info[1] if len(info) > 1 else "Unknown"
                
                metadata[sensor_id] = {
                    'location': location,
                    'name': name
                }
        
        logger.info(f"Loaded metadata for {len(metadata)} sensors")
        return metadata
    
    def _validate_adjacency_matrix(self, adj_matrix: np.ndarray):
        """Validate adjacency matrix properties."""
        n = adj_matrix.shape[0]
        
        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError(f"Adjacency matrix is not square: {adj_matrix.shape}")
        
        if not np.allclose(adj_matrix, adj_matrix.T, atol=1e-6):
            logger.warning("Adjacency matrix is not symmetric - may represent directed graph")
        
        diagonal = np.diag(adj_matrix)
        if not np.all(diagonal > 0):
            logger.warning(f"Some nodes lack self-loops: {np.sum(diagonal == 0)} nodes")
        
        num_edges = np.sum(adj_matrix > 0) - n
        max_edges = n * (n - 1)
        sparsity = 1.0 - (num_edges / max_edges) if max_edges > 0 else 0.0
        
        logger.info(f"Graph properties: {n} nodes, {num_edges} edges, sparsity={sparsity:.3f}")


class MinIoGraphLoader(MinIOClient, GraphProvider):
    """
    Loads graph topology from MinIO storage.
    Inherits MinIO connection from MinIOClient.
    """
    
    def __init__(self, graph_key: str, bucket_name: Optional[str] = None):
        MinIOClient.__init__(self, bucket_name=bucket_name)
        self.graph_key = graph_key
        self._topology = None
        self._sensor_metadata = None
        self._local_loader_helper = LocalGraphLoader.__new__(LocalGraphLoader)
    
    def load_topology(self) -> GraphTopology:
        if self._topology is not None:
            return self._topology
        
        logger.info(f"Loading graph from s3://{self.bucket_name}/{self.graph_key}")
        
        # Download JSON from MinIO
        stream = self.get_object_stream(self.graph_key)
        data = json.loads(stream.read().decode('utf-8'))
        
        # Reuse parsing logic from LocalGraphLoader
        self._topology = self._local_loader_helper._parse_topology_from_dict(data)
        return self._topology
    
    def get_sensor_metadata(self) -> Dict[int, Dict]:
        if self._sensor_metadata is not None:
            return self._sensor_metadata
        
        stream = self.get_object_stream(self.graph_key)
        data = json.loads(stream.read().decode('utf-8'))
        
        self._sensor_metadata = self._local_loader_helper._parse_metadata_from_dict(data)
        return self._sensor_metadata
