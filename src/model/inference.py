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

"""

import logging
import time
from typing import List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


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


def prepare_input_tensor(window: List[Dict[str, Any]]) -> np.ndarray:
    """
    Convert Redis window snapshots to model input tensor.
    
    Args:
        window: List of 12 traffic snapshots from Redis
            Each snapshot format: {
                "timestamp": "...",
                "data": {"nodes": [{"node_id": ..., "flow": ...}, ...]},
                ...
            }
    
    Returns:
        np.ndarray: Tensor with shape (1, 12, num_nodes, 1)
    """
    logger.info(f"Preparing input tensor from {len(window)} snapshots")
    all_sensor_ids = set()
    for snapshot in window:
        nodes = snapshot.get("data", {}).get("nodes", [])
        for node in nodes:
            if "node_id" in node:
                all_sensor_ids.add(node.get("node_id"))
    # Extract node data from each snapshot
    sorted_sensor_ids = sorted(list(all_sensor_ids))
    num_nodes = len(sorted_sensor_ids)
    
    if num_nodes == 0:
        logger.warning("No nodes found in window snapshots!")
        # Trả về dummy tensor để tránh crash pipeline
        return np.zeros((1, len(window), 1, 1), dtype=np.float32)

    logger.debug(f"Aligned tensor will have {num_nodes} nodes.")

    time_steps = []
    
    # BƯỚC 2: Duyệt từng snapshot và điền dữ liệu vào đúng chỗ (Alignment)
    for snapshot in window:
        # Map nhanh: {node_id: flow}
        current_data_map = {
            n["node_id"]: n.get("flow", 0.0) # <--- SỬA KEY: 'flow' thay vì 'flow'
            for n in snapshot.get("data", {}).get("nodes", [])
        }
        
        # Tạo vector cho timestep này
        step_vector = []
        for sensor_id in sorted_sensor_ids:
            # Nếu mất tín hiệu -> Điền 0.0
            val = current_data_map.get(sensor_id, 0.0)
            step_vector.append(val)
            
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


def format_predictions(predictions: np.ndarray) -> List[Dict[str, Any]]:
    """
    Convert model predictions to human-readable format.
    
    Args:
        predictions: Shape (1, predict_steps, num_nodes, 1)
    
    Returns:
        List of prediction dictionaries
    """
    batch_size, time_steps, num_nodes, features = predictions.shape
    
    results = []
    
    for t in range(time_steps):
        # Extract predictions for this time step
        step_predictions = predictions[0, t, :, 0]  # Shape: (num_nodes,)
        
        # Create prediction object
        pred_obj = {
            "time_step": t + 1,
            "predictions": [
                {
                    "node_id": node_idx,
                    "predicted_flow": float(flow),
                }
                for node_idx, flow in enumerate(step_predictions)
            ],
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
    
    Args:
        window: List of 12 traffic snapshots from Redis
    
    Returns:
        List of formatted predictions
    """
    logger.info("=" * 60)
    logger.info("Starting inference pipeline")
    logger.info("=" * 60)
    
    try:
        # Step 1: Prepare input tensor
        input_tensor = prepare_input_tensor(window)
        
        # Step 2: Determine number of nodes from tensor
        num_nodes = input_tensor.shape[2]
        
        # Step 3: Initialize model
        model = MockSTGTNModel(num_nodes=num_nodes, predict_steps=12)
        
        # Step 4: Run prediction
        predictions = model.predict(input_tensor)
        
        # Step 5: Format results
        formatted_results = format_predictions(predictions)
        
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
