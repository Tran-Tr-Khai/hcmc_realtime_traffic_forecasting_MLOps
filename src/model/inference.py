"""
Mock Model Inference Engine
============================

This module provides a mock inference engine for testing the
streaming pipeline. In production, this would be replaced with
the actual STGTN model.

Purpose:
- Test data flow through the pipeline
- Validate tensor/array conversions
- Ensure proper shape handling (1, 12, Nodes, 1)
- Simulate prediction latency

CRITICAL: Node Alignment
- The canonical node ordering is loaded from models/config.json
- This ordering MUST match the adjacency matrix used during training
- Missing sensors are filled with 0.0, unknown sensors are ignored

"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Path to model config (contains canonical node ordering)
CONFIG_PATH = Path(__file__).parent.parent.parent / "models" / "config.json"


class MockSTGTNModel:
    """
    Mock Spatio-Temporal Graph Transformer Network for testing.
    
    Simulates the behavior of the actual model without requiring
    heavy computation or GPU resources.
    """
    
    def __init__(self, num_nodes: int = 100, predict_steps: int = 12):
        """
        Initialize mock model.
        
        Args:
            num_nodes: Number of traffic nodes in the graph
            predict_steps: Number of future time steps to predict
        """
        self.num_nodes = num_nodes
        self.predict_steps = predict_steps
        
        logger.info(
            f"Mock STGTN Model initialized: "
            f"nodes={num_nodes}, predict_steps={predict_steps}"
        )
    
    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Generate mock predictions based on input data.
        
        Args:
            input_tensor: Shape (1, 12, num_nodes, 1)
                - Batch size: 1
                - Time steps: 12 (historical window)
                - Nodes: number of traffic nodes
                - Features: 1 (flow)
        
        Returns:
            np.ndarray: Predictions with shape (1, predict_steps, num_nodes, 1)
        """
        # Validate input shape
        expected_shape = (1, 12, self.num_nodes, 1)
        if input_tensor.shape != expected_shape:
            logger.warning(
                f"Input shape mismatch: expected {expected_shape}, "
                f"got {input_tensor.shape}"
            )
        
        # Simulate processing time (100-200ms)
        time.sleep(0.1 + np.random.random() * 0.1)
        
        # Generate mock predictions
        # Strategy: Use last value + small random noise
        last_values = input_tensor[0, -1, :, :]  # Shape: (num_nodes, 1)
        
        predictions = np.zeros((1, self.predict_steps, self.num_nodes, 1))
        
        for t in range(self.predict_steps):
            # Add random walk noise
            noise = np.random.randn(self.num_nodes, 1) * 2.0
            predictions[0, t, :, :] = np.clip(
                last_values + noise,
                a_min=0.0,  # flow cannot be negative
                a_max=120.0,  # Max flow cap
            )
        
        logger.info(
            f"Mock prediction generated: shape={predictions.shape}, "
            f"mean_flow={predictions.mean():.2f}"
        )
        
        return predictions


def load_model_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """
    Load model configuration including canonical node ordering.
    
    Args:
        config_path: Path to config.json
        
    Returns:
        Dict with model config including 'node_ids', 'max_flow', 'num_nodes'
    """
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}. Using dynamic node discovery.")
        return {}
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    node_ids = config.get('node_ids', [])
    if node_ids:
        logger.info(
            f"Loaded canonical node ordering: {len(node_ids)} nodes "
            f"(first 5: {node_ids[:5]})"
        )
    else:
        logger.warning(
            "Config file does not contain 'node_ids'. "
            "Node alignment cannot be guaranteed. Re-run training to fix."
        )
    
    return config


def prepare_input_tensor(
    window: List[Dict[str, Any]],
    canonical_node_ids: Optional[List[str]] = None
) -> np.ndarray:
    """
    Convert Redis window snapshots to model input tensor.
    
    CRITICAL: Uses canonical_node_ids from training config to ensure
    node ordering matches the adjacency matrix and trained model weights.
    
    Args:
        window: List of 12 traffic snapshots from Redis
            Each snapshot format: {
                "timestamp": "...",
                "data": {"nodes": [{"node_id": ..., "flow": ...}, ...]},
                ...
            }
        canonical_node_ids: Ordered list of sensor IDs from training config.
            If provided, tensor dimensions and ordering are fixed.
            If None, falls back to dynamic discovery (NOT RECOMMENDED).
    
    Returns:
        np.ndarray: Tensor with shape (1, 12, num_nodes, 1)
    """
    logger.info(f"Preparing input tensor from {len(window)} snapshots")
    
    # --- Determine node ordering ---
    if canonical_node_ids and len(canonical_node_ids) > 0:
        # USE CANONICAL ORDERING (from training config)
        sorted_sensor_ids = canonical_node_ids
        num_nodes = len(sorted_sensor_ids)
        logger.info(
            f"Using canonical node ordering: {num_nodes} nodes "
            f"(aligned with training adjacency matrix)"
        )
    else:
        # FALLBACK: Dynamic discovery (DANGEROUS - may not match training order)
        logger.warning(
            "No canonical node ordering provided! "
            "Falling back to dynamic discovery. Predictions may be misaligned."
        )
        all_sensor_ids = set()
        for snapshot in window:
            nodes = snapshot.get("data", {}).get("nodes", [])
            for node in nodes:
                if "node_id" in node:
                    all_sensor_ids.add(str(node.get("node_id")))
        sorted_sensor_ids = sorted(list(all_sensor_ids))
        num_nodes = len(sorted_sensor_ids)
    
    if num_nodes == 0:
        logger.warning("No nodes found in window snapshots!")
        return np.zeros((1, len(window), 1, 1), dtype=np.float32)

    logger.debug(f"Aligned tensor will have {num_nodes} nodes.")
    
    # Build sensor ID set for fast lookup
    canonical_set = set(sorted_sensor_ids)

    time_steps = []
    
    # Build tensor: iterate each snapshot and align to canonical order
    for snap_idx, snapshot in enumerate(window):
        # Build lookup map: {node_id_str: flow}
        current_data_map = {}
        unknown_count = 0
        for n in snapshot.get("data", {}).get("nodes", []):
            nid = str(n.get("node_id", ""))
            if nid in canonical_set:
                current_data_map[nid] = n.get("flow", 0.0)
            else:
                unknown_count += 1
        
        if unknown_count > 0:
            logger.debug(
                f"Snapshot {snap_idx}: {unknown_count} sensors not in "
                f"canonical set (ignored)"
            )
        
        # Fill vector in canonical order (missing sensors → 0.0)
        step_vector = []
        missing_count = 0
        for sensor_id in sorted_sensor_ids:
            val = current_data_map.get(sensor_id, 0.0)
            if sensor_id not in current_data_map:
                missing_count += 1
            step_vector.append(val)
        
        if missing_count > 0:
            logger.debug(
                f"Snapshot {snap_idx}: {missing_count}/{num_nodes} sensors "
                f"missing (filled with 0.0)"
            )
            
        time_steps.append(step_vector)
    
    # Convert to numpy array
    # Shape: (time_steps, num_nodes)
    data_array = np.array(time_steps, dtype=np.float32)
    
    # Reshape to (Batch, Time, Nodes, Features) -> (1, 12, N, 1)
    # Lưu ý: len(window) có thể < 12 lúc mới khởi động, code này vẫn chạy tốt
    input_tensor = data_array.reshape(1, len(time_steps), num_nodes, 1)
    
    logger.info(
        f"Input tensor prepared: shape={input_tensor.shape}, "
        f"mean_flow={input_tensor.mean():.2f}"
    )
    
    return input_tensor


def format_predictions(
    predictions: np.ndarray,
    canonical_node_ids: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Convert model predictions to human-readable format.
    
    Args:
        predictions: Shape (1, predict_steps, num_nodes, 1)
        canonical_node_ids: Ordered list of sensor IDs for labeling
    
    Returns:
        List of prediction dictionaries
    """
    batch_size, time_steps, num_nodes, features = predictions.shape
    
    results = []
    
    for t in range(time_steps):
        # Extract predictions for this time step
        step_predictions = predictions[0, t, :, 0]  # Shape: (num_nodes,)
        
        # Create prediction object with actual node IDs if available
        node_predictions = []
        for node_idx, flow in enumerate(step_predictions):
            node_id = (
                canonical_node_ids[node_idx] 
                if canonical_node_ids and node_idx < len(canonical_node_ids)
                else node_idx
            )
            node_predictions.append({
                "node_id": node_id,
                "predicted_flow": float(flow),
            })
        
        pred_obj = {
            "time_step": t + 1,
            "predictions": node_predictions,
            "statistics": {
                "mean_flow": float(step_predictions.mean()),
                "std_flow": float(step_predictions.std()),
                "min_flow": float(step_predictions.min()),
                "max_flow": float(step_predictions.max()),
            }
        }
        
        results.append(pred_obj)
    
    return results


def run_inference(window: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Main inference function - orchestrates the prediction pipeline.
    
    Loads canonical node ordering from config to ensure tensor alignment
    matches the trained model and adjacency matrix.
    
    Args:
        window: List of 12 traffic snapshots from Redis
    
    Returns:
        List of formatted predictions
    """
    logger.info("=" * 60)
    logger.info("Starting inference pipeline")
    logger.info("=" * 60)
    
    try:
        # Step 0: Load model config (canonical node ordering)
        config = load_model_config()
        canonical_node_ids = config.get('node_ids', [])
        
        if not canonical_node_ids:
            logger.warning(
                "Config missing 'node_ids'. Predictions may be misaligned "
                "with adjacency matrix. Re-run training to fix."
            )
        
        # Step 1: Prepare input tensor (aligned to canonical order)
        input_tensor = prepare_input_tensor(
            window, canonical_node_ids=canonical_node_ids or None
        )
        
        # Step 2: Determine number of nodes from tensor
        num_nodes = input_tensor.shape[2]
        
        # Validate against config
        expected_nodes = config.get('num_nodes', num_nodes)
        if num_nodes != expected_nodes:
            logger.error(
                f"Node count mismatch! Tensor has {num_nodes} nodes "
                f"but model expects {expected_nodes}. Check data pipeline."
            )
        
        # Step 3: Initialize model
        model = MockSTGTNModel(num_nodes=num_nodes, predict_steps=12)
        
        # Step 4: Run prediction
        predictions = model.predict(input_tensor)
        
        # Step 5: Format results (with actual node IDs)
        formatted_results = format_predictions(
            predictions, canonical_node_ids=canonical_node_ids or None
        )
        
        logger.info(
            f"Inference completed successfully: "
            f"{len(formatted_results)} time steps predicted"
        )
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise


# =====================================================================
# Testing & Validation
# =====================================================================

def generate_mock_window(num_snapshots: int = 12, num_nodes: int = 50) -> List[Dict]:
    """
    Generate a mock window for testing purposes.
    
    Args:
        num_snapshots: Number of snapshots to generate
        num_nodes: Number of traffic nodes
    
    Returns:
        List of mock snapshots
    """
    from datetime import datetime, timedelta
    
    window = []
    base_time = datetime.utcnow() - timedelta(minutes=5 * num_snapshots)
    
    for i in range(num_snapshots):
        timestamp = base_time + timedelta(minutes=5 * i)
        
        # Generate random flows for all nodes
        nodes = [
            {
                "node_id": node_idx,
                "flow": float(np.random.uniform(20, 80)),
                "num_samples": 3,
            }
            for node_idx in range(num_nodes)
        ]
        
        snapshot = {
            "timestamp": timestamp.isoformat(),
            "data": {
                "nodes": nodes,
                "total_nodes": num_nodes,
            },
            "num_messages_aggregated": 3,
        }
        
        window.append(snapshot)
    
    return window


if __name__ == "__main__":
    # Test the inference pipeline
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "=" * 60)
    print("Testing Mock Inference Pipeline")
    print("=" * 60 + "\n")
    
    # Generate test data
    test_window = generate_mock_window(num_snapshots=12, num_nodes=50)
    
    # Run inference
    results = run_inference(test_window)
    
    # Display sample results
    print("\n" + "=" * 60)
    print("Sample Prediction Results")
    print("=" * 60)
    
    for i, pred in enumerate(results[:3]):  # Show first 3 time steps
        print(f"\nTime Step {pred['time_step']}:")
        print(f"  Mean flow: {pred['statistics']['mean_flow']:.2f} km/h")
        print(f"  Std flow:  {pred['statistics']['std_flow']:.2f} km/h")
        print(f"  Range:      [{pred['statistics']['min_flow']:.2f}, "
              f"{pred['statistics']['max_flow']:.2f}]")
    
    print("\nTest completed successfully!\n")
