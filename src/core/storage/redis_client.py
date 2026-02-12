import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import redis
from redis.exceptions import RedisError, ConnectionError, TimeoutError

logger = logging.getLogger(__name__)


class RedisStateManager:
    """
    Manages the sliding window state in Redis for real-time traffic data.
    
    This class provides methods to:
    - Push new snapshots to the sliding window
    - Retrieve the complete window history
    - Check if the window is ready (has enough data)
    - Perform atomic operations to maintain consistency
    """
    
    # Redis Keys
    TRAFFIC_WINDOW_KEY = "traffic:history:window"
    METADATA_KEY = "traffic:metadata"
    
    def __init__(
        self,
        host: str = "redis",
        port: int = 6379,
        db: int = 0,
        window_size: int = 12,
        max_retries: int = 3,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
    ):
        """
        Initialize Redis client with connection pooling.
        
        Args:
            host: Redis server hostname
            port: Redis server port
            db: Redis database number
            window_size: Number of snapshots to maintain in the sliding window
            max_retries: Maximum number of connection retries
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Socket connect timeout in seconds
        """
        self.window_size = window_size
        self.max_retries = max_retries
        
        # Create connection pool for better performance
        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            decode_responses=True,  # Auto-decode bytes to strings
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            socket_keepalive=True,
            max_connections=10,
        )
        
        self.client = redis.Redis(connection_pool=self.pool)
        
        logger.info(
            f"Redis State Manager initialized: {host}:{port}/{db}, "
            f"window_size={window_size}"
        )
    
    def health_check(self) -> bool:
        """
        Check if Redis is healthy and responsive.
        
        Returns:
            bool: True if Redis is healthy, False otherwise
        """
        try:
            return self.client.ping()
        except RedisError as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    def push_snapshot(self, snapshot: Dict[str, Any]) -> bool:
        """
        Push a new traffic snapshot to the sliding window.
        
        This method:
        1. Serializes the snapshot to JSON
        2. Pushes it to the left of the Redis list (LPUSH)
        3. Trims the list to maintain the window size (LTRIM)
        4. Updates metadata (last update time)
        
        Operations are atomic and maintain FIFO ordering.
        
        Args:
            snapshot: Traffic data snapshot as a dictionary
                Expected format: {
                    "timestamp": "2026-02-05T10:35:00",
                    "nodes": [...],  # Traffic data for each node
                    "metadata": {...}
                }
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Add processing timestamp if not present
            if "processed_at" not in snapshot:
                snapshot["processed_at"] = datetime.utcnow().isoformat()
            
            # Serialize to JSON
            snapshot_json = json.dumps(snapshot)
            
            # Use pipeline for atomic operations
            pipe = self.client.pipeline()
            
            # 1. Push new snapshot to the left (most recent)
            pipe.lpush(self.TRAFFIC_WINDOW_KEY, snapshot_json)
            
            # 2. Trim to keep only the latest N records (0 to window_size-1)
            pipe.ltrim(self.TRAFFIC_WINDOW_KEY, 0, self.window_size - 1)
            
            # 3. Update metadata
            metadata = {
                "last_update": datetime.utcnow().isoformat(),
                "window_size": self.window_size,
                "snapshot_timestamp": snapshot.get("timestamp", "unknown"),
            }
            pipe.set(self.METADATA_KEY, json.dumps(metadata))
            
            # Execute all commands atomically
            pipe.execute()
            
            logger.debug(
                f"Pushed snapshot to Redis: timestamp={snapshot.get('timestamp')}"
            )
            return True
            
        except RedisError as e:
            logger.error(f"Failed to push snapshot to Redis: {e}")
            return False
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize snapshot: {e}")
            return False
    
    def get_window(self) -> List[Dict[str, Any]]:
        """
        Retrieve the complete sliding window from Redis.
        
        Returns snapshots in chronological order (oldest to newest).
        
        Returns:
            List[Dict]: List of traffic snapshots. Empty list if error occurs.
        """
        try:
            # LRANGE 0 -1 retrieves all elements
            snapshot_jsons = self.client.lrange(
                self.TRAFFIC_WINDOW_KEY, 0, -1
            )
            
            if not snapshot_jsons:
                logger.warning("No snapshots found in Redis window")
                return []
            
            # Deserialize and reverse (LRANGE returns newest first)
            snapshots = [json.loads(s) for s in snapshot_jsons]
            snapshots.reverse()  # Oldest to newest
            
            logger.debug(f"Retrieved {len(snapshots)} snapshots from Redis")
            return snapshots
            
        except RedisError as e:
            logger.error(f"Failed to retrieve window from Redis: {e}")
            return []
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to deserialize snapshots: {e}")
            return []
    
    def get_window_size_current(self) -> int:
        """
        Get the current number of snapshots in the window.
        
        Returns:
            int: Number of snapshots currently stored
        """
        try:
            return self.client.llen(self.TRAFFIC_WINDOW_KEY)
        except RedisError as e:
            logger.error(f"Failed to get window size: {e}")
            return 0
    
    def is_window_ready(self) -> bool:
        """
        Check if the window has enough data for inference.
        
        Returns:
            bool: True if window has required number of snapshots
        """
        current_size = self.get_window_size_current()
        is_ready = current_size >= self.window_size
        
        logger.debug(
            f"Window readiness check: {current_size}/{self.window_size} "
            f"(ready={is_ready})"
        )
        return is_ready
    
    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata about the current state.
        
        Returns:
            Dict or None: Metadata dictionary or None if not available
        """
        try:
            metadata_json = self.client.get(self.METADATA_KEY)
            if metadata_json:
                return json.loads(metadata_json)
            return None
        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Failed to retrieve metadata: {e}")
            return None
    
    def clear_window(self) -> bool:
        """
        Clear all data in the sliding window (use with caution).
        
        Returns:
            bool: True if successful
        """
        try:
            self.client.delete(self.TRAFFIC_WINDOW_KEY)
            self.client.delete(self.METADATA_KEY)
            logger.info("Cleared Redis sliding window")
            return True
        except RedisError as e:
            logger.error(f"Failed to clear window: {e}")
            return False
    
    def save_predictions(self, predictions: Dict[str, Any]) -> bool:
        """
        Save predictions to Redis in optimized format.
        
        Expected predictions format:
        {
            "timestamp": "2024-01-01T12:00:00+07:00",
            "data": {"node_id": flow, ...}
        }
        
        Args:
            predictions: Dictionary with timestamp and flattened node predictions
            
        Returns:
            bool: True if successful
        """
        PREDICTION_KEY = "traffic:predictions:latest"
        try:
            # Store predictions as-is (already in correct format from inference)
            # DO NOT generate new timestamp - preserve the one from input data
            self.client.set(PREDICTION_KEY, json.dumps(predictions))
            
            timestamp = predictions.get("timestamp", "N/A")
            logger.info(f"Saved predictions to Redis with timestamp: {timestamp}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")
            return False
        
    def close(self):
        """
        Close Redis connection pool gracefully.
        """
        try:
            self.client.close()
            self.pool.disconnect()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")


# =====================================================================
# Utility Functions
# =====================================================================

def create_redis_client(
    host: str,
    port: int = 6379,
    db: int = 0,
    window_size: int = 12,
) -> RedisStateManager:
    """
    Factory function to create and validate a Redis client.
    
    Args:
        host: Redis hostname
        port: Redis port
        db: Redis database number
        window_size: Sliding window size
    
    Returns:
        RedisStateManager: Initialized client
    
    Raises:
        ConnectionError: If Redis is not reachable
    """
    client = RedisStateManager(
        host=host,
        port=port,
        db=db,
        window_size=window_size,
    )
    
    # Validate connection
    if not client.health_check():
        raise ConnectionError(
            f"Failed to connect to Redis at {host}:{port}"
        )
    
    logger.info("Redis client created and validated successfully")
    return client
