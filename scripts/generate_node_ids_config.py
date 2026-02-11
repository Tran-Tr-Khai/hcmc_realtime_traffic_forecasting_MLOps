"""
Utility Script: Generate canonical node_ids in config.json
===========================================================

This script reads the training parquet from MinIO to extract the exact
column order (canonical node ordering) and updates config.json.

Use this script if config.json was created by an older version of train.py
that did not save node_ids.

After running, config.json will contain a 'node_ids' field with the
exact string-sorted sensor IDs matching the adjacency matrix order.

Usage:
    python scripts/generate_node_ids_config.py
"""

import sys
import os
import io
import json
import logging
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

load_dotenv()

from src.core.storage.minio_client import MinIOClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    bucket = os.getenv("MINIO_BUCKET_NAME", "hcmc-traffic-data")
    traffic_file = "processed/traffic_clean.parquet"
    config_path = project_root / "models" / "config.json"

    # --- Step 1: Load parquet from MinIO ---
    logger.info(f"Loading parquet from MinIO: {bucket}/{traffic_file}")
    minio_client = MinIOClient(bucket_name=bucket)
    stream = minio_client.get_object_stream(traffic_file, bucket=bucket)
    traffic_df = pl.read_parquet(io.BytesIO(stream.read()))
    logger.info(f"Parquet shape: {traffic_df.shape}, columns: {traffic_df.columns[:5]}...")

    # --- Step 2: Extract canonical sensor column order ---
    time_col = None
    for col_name in ['timestamp', 'time']:
        if col_name in traffic_df.columns:
            time_col = col_name
            break

    if time_col:
        sensor_columns = [c for c in traffic_df.columns if c != time_col]
    else:
        sensor_columns = list(traffic_df.columns)

    logger.info(f"Found {len(sensor_columns)} sensor columns")
    logger.info(f"First 10: {sensor_columns[:10]}")
    logger.info(f"Last 5: {sensor_columns[-5:]}")

    # --- Step 3: Update config.json ---
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded existing config: num_nodes={config.get('num_nodes')}")
    else:
        config = {}

    # Validate consistency
    if config.get('num_nodes') and config['num_nodes'] != len(sensor_columns):
        logger.warning(
            f"num_nodes mismatch! Config says {config['num_nodes']} "
            f"but parquet has {len(sensor_columns)} sensor columns. Updating."
        )
        config['num_nodes'] = len(sensor_columns)

    config['node_ids'] = sensor_columns

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Updated {config_path} with {len(sensor_columns)} node_ids")
    logger.info("Done! Inference pipeline will now use canonical node ordering.")


if __name__ == '__main__':
    main()
