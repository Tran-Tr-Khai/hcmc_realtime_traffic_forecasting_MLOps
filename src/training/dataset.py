"""
Traffic Dataset for STGTN Model Training.

This module provides a PyTorch Dataset class for loading and preprocessing
traffic flow data with sliding window logic for time-series forecasting.

Key Features:
- MinMax Normalization (suitable for non-negative traffic counts)
- Sliding window creation for input-output pairs
- Proper tensor shape handling: (Time, Nodes, Features)
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class TrafficDataset(Dataset):
    """
    PyTorch Dataset for traffic flow time-series data.
    
    Implements sliding window approach for spatial-temporal forecasting:
    - Input: Past `input_len` time steps
    - Output: Next `output_len` time steps
    
    Normalization Strategy:
    - Uses MinMax normalization (division by global max) instead of StandardScaler
    - Rationale: Traffic counts are strictly non-negative. StandardScaler creates
      negative values which are physically meaningless for vehicle counts.
    
    Args:
        data: Traffic flow data with shape (Total_Time, Num_Nodes)
        input_len: Number of historical time steps to use as input (default: 12 = 60 mins)
        output_len: Number of future time steps to predict (default: 3 = 15 mins)
        max_flow: Global maximum value for normalization. If None, computed from data.
        
    Returns:
        x: Input tensor of shape (input_len, num_nodes, 1)
        y: Target tensor of shape (output_len, num_nodes, 1)
    """
    
    def __init__(
        self, 
        data: np.ndarray, 
        input_len: int = 12, 
        output_len: int = 3,
        max_flow: float = None
    ):
        """
        Initialize the traffic dataset.
        
        Args:
            data: Numpy array of shape (Total_Time, Num_Nodes)
            input_len: Length of input window (default: 12 time steps)
            output_len: Length of prediction horizon (default: 3 time steps)
            max_flow: Maximum value for normalization. If None, uses data.max()
        """
        super().__init__()
        
        # Validate input
        if len(data.shape) != 2:
            raise ValueError(f"Expected 2D data (Time, Nodes), got shape {data.shape}")
        
        self.data = data.astype(np.float32)
        self.input_len = input_len
        self.output_len = output_len
        self.total_len, self.num_nodes = data.shape
        
        # MinMax Normalization: Scale by global maximum
        # This ensures all values are in [0, 1] while preserving the non-negativity
        if max_flow is None:
            self.max_flow = float(self.data.max())
            logger.info(f"Computed max_flow from data: {self.max_flow:.2f}")
        else:
            self.max_flow = float(max_flow)
            logger.info(f"Using provided max_flow: {self.max_flow:.2f}")
        
        # Avoid division by zero
        if self.max_flow == 0:
            logger.warning("max_flow is 0, setting to 1.0 to avoid division error")
            self.max_flow = 1.0
        # Normalize data
        self.normalized_data = self.data / self.max_flow
        
        # Calculate number of valid samples
        # Valid window: We need input_len + output_len consecutive time steps
        self.num_samples = self.total_len - self.input_len - self.output_len + 1
        
        if self.num_samples <= 0:
            raise ValueError(
                f"Dataset too short! Total length: {self.total_len}, "
                f"Required: {self.input_len + self.output_len}. "
                f"Cannot create any valid samples."
            )
        
        logger.info(
            f"TrafficDataset initialized: {self.num_samples} samples, "
            f"{self.num_nodes} nodes, input_len={input_len}, output_len={output_len}"
        )
        logger.info(f"Data range: [{self.data.min():.2f}, {self.data.max():.2f}]")
        logger.info(f"Normalized range: [{self.normalized_data.min():.4f}, {self.normalized_data.max():.4f}]")
    
    def __len__(self) -> int:
        """Return the total number of samples."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample (input-output pair) by index.
        
        Args:
            idx: Sample index (0 to num_samples-1)
            
        Returns:
            x: Input tensor of shape (input_len, num_nodes, 1)
            y: Target tensor of shape (output_len, num_nodes, 1)
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.num_samples})")
        
        # Define the sliding window boundaries
        start_idx = idx
        input_end_idx = start_idx + self.input_len
        output_end_idx = input_end_idx + self.output_len
        
        # Extract input and output windows
        # Input: [start_idx : start_idx + input_len]
        # Output: [start_idx + input_len : start_idx + input_len + output_len]
        x_data = self.normalized_data[start_idx:input_end_idx, :]  # (input_len, num_nodes)
        y_data = self.normalized_data[input_end_idx:output_end_idx, :]  # (output_len, num_nodes)
        
        # Add feature dimension: (Time, Nodes) -> (Time, Nodes, 1)
        x = torch.FloatTensor(x_data).unsqueeze(-1)  # (input_len, num_nodes, 1)
        y = torch.FloatTensor(y_data).unsqueeze(-1)  # (output_len, num_nodes, 1)
        
        return x, y
    
    def denormalize(self, normalized_data: np.ndarray) -> np.ndarray:
        """
        Convert normalized predictions back to original scale.
        
        This is crucial for inference: the model outputs values in [0, 1],
        but we need actual vehicle counts for visualization/reporting.
        
        Args:
            normalized_data: Normalized data (values in [0, 1])
            
        Returns:
            Original scale data (vehicle counts)
        """
        return normalized_data * self.max_flow
    
    def get_statistics(self) -> dict:
        """
        Return dataset statistics for logging and validation.
        
        Returns:
            Dictionary containing dataset metadata
        """
        return {
            'num_samples': self.num_samples,
            'num_nodes': self.num_nodes,
            'input_len': self.input_len,
            'output_len': self.output_len,
            'max_flow': self.max_flow,
            'data_min': float(self.data.min()),
            'data_max': float(self.data.max()),
            'data_mean': float(self.data.mean()),
            'data_std': float(self.data.std())
        }
