"""
Production-Grade Training Script for STGTN Model.

This script implements a complete training pipeline for spatial-temporal traffic forecasting:
1. Load traffic data and graph topology from MinIO
2. Time-series train/validation split (no random shuffle to prevent leakage)
3. Pre-compute Laplacian positional encodings for efficiency
4. Train STGTN model with MAE loss and gradient clipping
5. Save best model checkpoint and configuration for inference

Usage:
    python scripts/train.py --epochs 100 --batch_size 32 --lr 0.001
"""

import sys
import os
import json
import logging
import argparse
from pathlib import Path
from typing import Tuple, Dict
import io

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.storage.minio_client import MinIOClient
from src.training.dataset import TrafficDataset
from src.model.stgtn import STGTN, compute_laplacian_positional_encoding

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        #logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train STGTN Model')
    
    # Data parameters
    parser.add_argument('--bucket', type=str, default='hcmc-traffic-data',
                        help='MinIO bucket name')
    parser.add_argument('--traffic_file', type=str, default='processed/traffic_clean.parquet',
                        help='Path to traffic data in MinIO')
    parser.add_argument('--adj_matrix_file', type=str, default='processed/adj_matrix.npy',
                        help='Path to adjacency matrix in MinIO')
    
    # Model parameters
    parser.add_argument('--input_len', type=int, default=12,
                        help='Number of historical time steps (12 = 60 mins)')
    parser.add_argument('--output_len', type=int, default=3,
                        help='Number of future time steps to predict (3 = 15 mins)')
    parser.add_argument('--d_model', type=int, default=64,
                        help='Model hidden dimension')
    parser.add_argument('--nhead', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=2,
                        help='Number of encoder STGT layers')
    parser.add_argument('--num_decoder_layers', type=int, default=2,
                        help='Number of decoder STGT layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--laplacian_k', type=int, default=10,
                        help='Number of Laplacian eigenvectors to use')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='Gradient clipping threshold')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of data to use for training (rest for validation)')
    
    # Scheduler parameters
    parser.add_argument('--scheduler_patience', type=int, default=5,
                        help='Patience for ReduceLROnPlateau scheduler')
    parser.add_argument('--scheduler_factor', type=float, default=0.5,
                        help='Factor to reduce LR by')
    
    # Output parameters
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save model checkpoints')
    parser.add_argument('--config_file', type=str, default='models/config.json',
                        help='Path to save model configuration')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of DataLoader workers')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_data_from_minio(
    minio_client: MinIOClient, 
    bucket: str, 
    traffic_file: str, 
    adj_matrix_file: str
) -> Tuple[np.ndarray, np.ndarray, int, list]:
    """
    Load traffic data and adjacency matrix from MinIO.
    
    Args:
        minio_client: MinIO client instance
        bucket: Bucket name
        traffic_file: Path to traffic parquet file
        adj_matrix_file: Path to adjacency matrix npy file
        
    Returns:
        traffic_data: Numpy array of shape (Total_Time, Num_Nodes)
        adj_matrix: Numpy array of shape (Num_Nodes, Num_Nodes)
        num_nodes: Number of nodes in the graph
        node_ids: Ordered list of sensor ID strings (canonical order for inference)
    """
    logger.info(f"Loading data from MinIO bucket: {bucket}")
    
    try:
        # Load traffic data (Parquet format)
        logger.info(f"Loading traffic data from {traffic_file}...")
        traffic_stream = minio_client.get_object_stream(traffic_file, bucket=bucket)
        traffic_bytes = traffic_stream.read()
        
        # Use Polars for efficient parquet reading
        traffic_df = pl.read_parquet(io.BytesIO(traffic_bytes))
        logger.info(f"Traffic data shape: {traffic_df.shape}")
        
        # Convert to numpy array, excluding timestamp column if present
        # Assume first column is timestamp, rest are sensor readings
        if 'time' in traffic_df.columns or 'timestamp' in traffic_df.columns:
            time_col = 'time' if 'time' in traffic_df.columns else 'timestamp'
            sensor_columns = [c for c in traffic_df.columns if c != time_col]
            traffic_data = traffic_df.drop(time_col).to_numpy()
        else:
            sensor_columns = list(traffic_df.columns)
            traffic_data = traffic_df.to_numpy()
        
        # sensor_columns preserves the canonical column order from offline pipeline
        # This order MUST match the adjacency matrix row/col order
        logger.info(f"Canonical node order: {len(sensor_columns)} sensors, first 5: {sensor_columns[:5]}")
        
        # Load adjacency matrix (NumPy format)
        logger.info(f"Loading adjacency matrix from {adj_matrix_file}...")
        adj_stream = minio_client.get_object_stream(adj_matrix_file, bucket=bucket)
        adj_bytes = adj_stream.read()
        adj_matrix = np.load(io.BytesIO(adj_bytes))
        logger.info(f"Adjacency matrix shape: {adj_matrix.shape}")
        
        # Validate dimensions
        num_nodes = traffic_data.shape[1]
        if adj_matrix.shape[0] != num_nodes or adj_matrix.shape[1] != num_nodes:
            raise ValueError(
                f"Adjacency matrix shape {adj_matrix.shape} does not match "
                f"number of nodes {num_nodes} in traffic data"
            )
        
        logger.info(f"Successfully loaded data: {traffic_data.shape[0]} time steps, {num_nodes} nodes")
        return traffic_data, adj_matrix, num_nodes, sensor_columns
        
    except Exception as e:
        logger.error(f"Failed to load data from MinIO: {e}")
        raise


def create_time_series_split(
    traffic_data: np.ndarray, 
    train_ratio: float = 0.8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split time-series data into train and validation sets.
    
    CRITICAL: Uses chronological split (no random shuffle) to prevent data leakage.
    In time-series forecasting, the model should never see future data during training.
    
    Args:
        traffic_data: Full dataset of shape (Total_Time, Num_Nodes)
        train_ratio: Fraction of data to use for training (default: 0.8)
        
    Returns:
        train_data: Training set (first train_ratio of time steps)
        val_data: Validation set (remaining time steps)
    """
    total_len = traffic_data.shape[0]
    train_len = int(total_len * train_ratio)
    
    train_data = traffic_data[:train_len]
    val_data = traffic_data[train_len:]
    
    logger.info(f"Time-series split: Train={train_data.shape[0]} steps, Val={val_data.shape[0]} steps")
    logger.info(f"Train period: timesteps [0, {train_len})")
    logger.info(f"Val period: timesteps [{train_len}, {total_len})")
    
    return train_data, val_data


def save_config(
    config_path: str, 
    max_flow: float, 
    num_nodes: int, 
    args: argparse.Namespace,
    node_ids: list = None
):
    """
    Save model configuration to JSON file for inference.
    
    This file is CRITICAL for inference: it contains the normalization factor
    needed to convert model predictions (0-1 range) back to vehicle counts,
    AND the canonical node ordering for proper tensor alignment.
    
    Args:
        config_path: Path to save config JSON
        max_flow: Maximum traffic flow value from training set
        num_nodes: Number of nodes in the graph
        args: Command line arguments
        node_ids: Ordered list of sensor ID strings (canonical order)
    """
    config = {
        'max_flow': float(max_flow),
        'num_nodes': int(num_nodes),
        'node_ids': node_ids or [],  # Canonical node ordering for inference alignment
        'input_len': args.input_len,
        'output_len': args.output_len,
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_encoder_layers': args.num_encoder_layers,
        'num_decoder_layers': args.num_decoder_layers,
        'dropout': args.dropout,
        'laplacian_k': args.laplacian_k,
        'train_ratio': args.train_ratio,
        'training_date': pd.Timestamp.now().isoformat()
    }
    
    # Ensure directory exists
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Saved model configuration to {config_path}")
    logger.info(f"max_flow={max_flow:.2f}, num_nodes={num_nodes}")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    adj_matrix: torch.Tensor,
    grad_clip: float,
    epoch: int
) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: STGTN model
        dataloader: Training DataLoader
        criterion: Loss function (MAE)
        optimizer: Optimizer
        device: Device to train on
        adj_matrix: Adjacency matrix tensor
        grad_clip: Gradient clipping threshold
        epoch: Current epoch number
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    for batch_idx, (x, y) in enumerate(dataloader):
        # Move data to device
        x = x.to(device)  # (Batch, input_len, Num_Nodes, 1)
        y = y.to(device)  # (Batch, output_len, Num_Nodes, 1)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        # Note: Model expects (Batch, Time, Nodes, Features)
        y_pred = model(x, adj_matrix)  # (Batch, output_len, Num_Nodes, 1)
        
        # Compute loss
        loss = criterion(y_pred, y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Update weights
        optimizer.step()
        
        total_loss += loss.item()
        
        # Log progress every 10% of batches
        if (batch_idx + 1) % max(1, num_batches // 10) == 0:
            logger.info(
                f"Epoch {epoch} | Batch [{batch_idx + 1}/{num_batches}] | "
                f"Loss: {loss.item():.6f}"
            )
    
    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    adj_matrix: torch.Tensor
) -> float:
    """
    Evaluate the model on validation set.
    
    Args:
        model: STGTN model
        dataloader: Validation DataLoader
        criterion: Loss function (MAE)
        device: Device to evaluate on
        adj_matrix: Adjacency matrix tensor
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            
            # Forward pass
            y_pred = model(x, adj_matrix)
            
            # Compute loss
            loss = criterion(y_pred, y)
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    """Main training pipeline."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")
    
    # Create directories
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path('logs').mkdir(parents=True, exist_ok=True)
    
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # ======================================================================
        # STEP 1: Load Data from MinIO
        # ======================================================================
        logger.info("=" * 80)
        logger.info("STEP 1: Loading data from MinIO")
        logger.info("=" * 80)
        
        minio_client = MinIOClient(bucket_name=args.bucket)
        traffic_data, adj_matrix, num_nodes, sensor_columns = load_data_from_minio(
            minio_client, args.bucket, args.traffic_file, args.adj_matrix_file
        )
        
        # ======================================================================
        # STEP 2: Time-Series Train/Val Split (NO RANDOM SHUFFLE)
        # ======================================================================
        logger.info("=" * 80)
        logger.info("STEP 2: Creating time-series split")
        logger.info("=" * 80)
        
        train_data, val_data = create_time_series_split(traffic_data, args.train_ratio)
        
        # ======================================================================
        # STEP 3: Compute Normalization Factor and Save Config
        # ======================================================================
        logger.info("=" * 80)
        logger.info("STEP 3: Computing normalization factor")
        logger.info("=" * 80)
        
        # CRITICAL: Compute max_flow from TRAINING SET ONLY to prevent data leakage
        max_flow = float(train_data.max())
        logger.info(f"Training set max_flow: {max_flow:.2f}")
        
        # Save configuration for inference (including canonical node ordering)
        save_config(args.config_file, max_flow, num_nodes, args, node_ids=sensor_columns)
        
        # ======================================================================
        # STEP 4: Create PyTorch Datasets and DataLoaders
        # ======================================================================
        logger.info("=" * 80)
        logger.info("STEP 4: Creating datasets and dataloaders")
        logger.info("=" * 80)
        
        train_dataset = TrafficDataset(
            train_data, 
            input_len=args.input_len, 
            output_len=args.output_len,
            max_flow=max_flow
        )
        
        val_dataset = TrafficDataset(
            val_data, 
            input_len=args.input_len, 
            output_len=args.output_len,
            max_flow=max_flow  # Use same max_flow from training set
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,  # Shuffle within training set is OK
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # Log sample shapes for debugging
        sample_x, sample_y = next(iter(train_loader))
        logger.info(f"Sample input shape: {sample_x.shape}")
        logger.info(f"Sample target shape: {sample_y.shape}")
        
        # ======================================================================
        # STEP 5: Pre-compute Laplacian Positional Encoding (OPTIMIZATION)
        # ======================================================================
        logger.info("=" * 80)
        logger.info("STEP 5: Pre-computing Laplacian positional encoding")
        logger.info("=" * 80)
        
        # Convert adjacency matrix to tensor
        adj_matrix_tensor = torch.FloatTensor(adj_matrix).to(device)
        
        # Compute Laplacian eigenvectors once (expensive operation)
        laplacian_pos_enc = compute_laplacian_positional_encoding(
            adj_matrix_tensor, k=args.laplacian_k
        )
        logger.info(f"Laplacian positional encoding shape: {laplacian_pos_enc.shape}")
        
        # ======================================================================
        # STEP 6: Initialize Model
        # ======================================================================
        logger.info("=" * 80)
        logger.info("STEP 6: Initializing STGTN model")
        logger.info("=" * 80)
        
        model = STGTN(
            num_nodes=num_nodes,
            in_dim=1,  # Traffic flow is univariate
            out_len=args.output_len,
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dropout=args.dropout,
            laplacian_k=args.laplacian_k,
            device=device
        ).to(device)
        
        model.laplacian_pos_enc = laplacian_pos_enc
        logger.info("Assigned pre-computed Laplacian encoding to model.")
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # ======================================================================
        # STEP 7: Setup Training Components
        # ======================================================================
        logger.info("=" * 80)
        logger.info("STEP 7: Setting up training components")
        logger.info("=" * 80)
        
        # Loss function: MAE (L1Loss) - Robust for counting tasks
        criterion = nn.L1Loss()
        logger.info("Using L1Loss (MAE) for robust counting predictions")
        
        # Optimizer: Adam
        optimizer = Adam(model.parameters(), lr=args.lr)
        logger.info(f"Using Adam optimizer with lr={args.lr}")
        
        # Scheduler: ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min',  # Minimize validation loss
            factor=args.scheduler_factor,
            patience=args.scheduler_patience
            # verbose=True  <--- XÓA DÒNG NÀY ĐI
        )
        logger.info(f"Using ReduceLROnPlateau scheduler (patience={args.scheduler_patience})")
        
        # ======================================================================
        # STEP 8: Training Loop
        # ======================================================================
        logger.info("=" * 80)
        logger.info("STEP 8: Starting training loop")
        logger.info("=" * 80)
        
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        early_stop_patience = 15  # Stop if no improvement for 15 epochs
        
        for epoch in range(1, args.epochs + 1):
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Epoch {epoch}/{args.epochs}")
            logger.info(f"{'=' * 80}")
            
            # Train one epoch
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, 
                device, adj_matrix_tensor, args.grad_clip, epoch
            )
            
            # Validate
            val_loss = evaluate(model, val_loader, criterion, device, adj_matrix_tensor)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log epoch summary
            logger.info(f"\nEpoch {epoch} Summary:")
            logger.info(f"  Train Loss: {train_loss:.6f}")
            logger.info(f"  Val Loss:   {val_loss:.6f}")
            logger.info(f"  LR:         {current_lr:.8f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                
                # Save checkpoint
                checkpoint_path = Path(args.model_dir) / 'stgtn_best.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'config': vars(args)
                }, checkpoint_path)
                
                logger.info(f"  ✓ New best model saved! (Val Loss: {val_loss:.6f})")
            else:
                patience_counter += 1
                logger.info(f"  No improvement. Patience: {patience_counter}/{early_stop_patience}")
            
            # Early stopping
            if patience_counter >= early_stop_patience:
                logger.info(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        # ======================================================================
        # STEP 9: Training Complete
        # ======================================================================
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Best model from epoch {best_epoch} with Val Loss: {best_val_loss:.6f}")
        logger.info(f"Model saved to: {Path(args.model_dir) / 'stgtn_best.pth'}")
        logger.info(f"Config saved to: {args.config_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
