import logging
import numpy as np

from .topology import GraphTopology

logger = logging.getLogger(__name__)


class AdjacencyMatrixBuilder:
    """
    Builds adjacency matrices for graph neural networks.
    
    Ensures consistency between feature matrix and graph structure:
    - Node ordering matches feature matrix
    - Self-loops are included
    - Symmetric (undirected graph)
    """
    
    def __init__(self, add_self_loops: bool = True):
        self.add_self_loops = add_self_loops
    
    def build(self, topology: GraphTopology) -> np.ndarray:
        """
        Build adjacency matrix from topology.
        
        Args:
            topology: Graph topology with adjacency matrix
            
        Returns:
            Adjacency matrix (N x N) ready for GNN input
        """
        adj_matrix = topology.adjacency_matrix.copy()
        
        if self.add_self_loops:
            # Ensure all diagonal elements are 1
            np.fill_diagonal(adj_matrix, 1.0)
        
        # Validate symmetry
        if not np.allclose(adj_matrix, adj_matrix.T, atol=1e-6):
            logger.warning("Adjacency matrix is not symmetric - symmetrizing")
            adj_matrix = (adj_matrix + adj_matrix.T) / 2.0
        
        logger.info(
            f"Built adjacency matrix: shape {adj_matrix.shape}, "
            f"density {adj_matrix.mean():.3f}"
        )
        
        return adj_matrix
    
    def build_normalized(
        self, 
        topology: GraphTopology,
        normalization: str = 'symmetric'
    ) -> np.ndarray:
        """
        Build normalized adjacency matrix for GCN.
        
        Args:
            topology: Graph topology
            normalization: Type of normalization
                - 'symmetric': D^(-1/2) A D^(-1/2)
                - 'row': D^(-1) A
                - 'none': No normalization
                
        Returns:
            Normalized adjacency matrix
        """
        adj_matrix = self.build(topology)
        
        if normalization == 'none':
            return adj_matrix
        
        # Compute degree matrix
        degrees = np.sum(adj_matrix, axis=1)
        
        # Avoid division by zero
        degrees = np.where(degrees == 0, 1, degrees)
        
        if normalization == 'row':
            # Row normalization: D^(-1) A
            D_inv = np.diag(1.0 / degrees)
            adj_normalized = D_inv @ adj_matrix
            
        elif normalization == 'symmetric':
            # Symmetric normalization: D^(-1/2) A D^(-1/2)
            D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
            adj_normalized = D_inv_sqrt @ adj_matrix @ D_inv_sqrt
            
        else:
            raise ValueError(f"Unknown normalization: {normalization}")
        
        logger.info(f"Applied {normalization} normalization to adjacency matrix")
        
        return adj_normalized
