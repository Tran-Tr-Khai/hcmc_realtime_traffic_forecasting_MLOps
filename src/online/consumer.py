import os
import json
import logging
import signal
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from kafka import KafkaConsumer
from kafka.errors import KafkaError

from src.core.storage.redis_client import create_redis_client

# Configure logging
# Ensure logs directory exists

logger = logging.getLogger(__name__)


class TrafficDataBuffer:
    def __init__(self, interval_seconds: int = 300):
        """
            interval_seconds: Target snapshot interval (default: 300s = 5min)
        """
        self.interval_seconds = interval_seconds
        self.buffer: List[Dict[str, Any]] = []
        self.current_window_start: Optional[str] = None
        
        logger.info(f"Buffer initialized with {interval_seconds}s interval")
    
    def add_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Add a message to the buffer and check if it's time to flush.
        """
        ts_ms = message.get("ts")
        if not ts_ms: return None
        
        msg_dt = datetime.fromtimestamp(ts_ms / 1000.0)
        msg_window_start = self._get_aligned_timestamp(msg_dt)
        snapshot = None
  
        if self.current_window_start is None:
            self.current_window_start = msg_window_start
            
        # Nếu tin nhắn mới thuộc khung giờ MỚI (lớn hơn khung giờ đang gom)
        elif msg_window_start > self.current_window_start:
            # => Flush dữ liệu của khung giờ cũ
            snapshot = self.flush()
            
            # Cập nhật sang khung giờ mới
            self.current_window_start = msg_window_start
            
        self.buffer.append(message)
        return snapshot
    
    def flush(self) -> Dict[str, Any]:
        """
        Aggregate buffered messages into a single snapshot.
        """
        if not self.buffer:
            logger.warning("Attempted to flush empty buffer")
            return {}
        
        # Aggregate: Average traffic values per node
        aggregated = self._aggregate_traffic_data(self.buffer)
        
        # Create snapshot with aligned timestamp from data (not current time)
        snapshot = {
            "timestamp": self.current_window_start,  # Use aligned timestamp from message data
            "data": aggregated,
            "num_messages_aggregated": len(self.buffer),
            "buffer_flush_time": datetime.utcnow().isoformat(),
        }
        
        logger.info(
            f"Flushed buffer: {len(self.buffer)} messages → snapshot "
            f"at {snapshot['timestamp']}"
        )
        
        # Reset buffer
        self.buffer.clear()
        self.last_flush_time = datetime.utcnow()
        
        return snapshot
    
    def _aggregate_traffic_data(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate multiple traffic messages by summing counts per node.
        """
        # Group values by node_id
        node_values = defaultdict(list)
        
        for msg in messages:
            node_id = msg.get("sensor_id")
            count = msg.get("count")
            if node_id is not None and count is not None:
                try:
                    val = float(count)
                    node_values[node_id].append(val)
                except ValueError:
                    continue
        
        # Compute sum (total flow) for each node
        aggregated_nodes = []
        for node_id, counts in node_values.items():
            avg_flow = round(sum(counts)/len(counts), 2) if counts else 0
            aggregated_nodes.append({
                "node_id": node_id,
                "flow": avg_flow,
                "num_samples": len(counts),
            })
        
        return {
            "nodes": aggregated_nodes,
            "total_nodes": len(aggregated_nodes),
        }
    
    def _get_aligned_timestamp(self, dt: datetime) -> str:
        """
        Get timestamp aligned to 5-minute boundaries.
        
        Example: 10:32:xx → 10:30:00
        """
        aligned = dt.replace(
            minute=(dt.minute // 5) * 5,
            second=0,
            microsecond=0
        )
        return aligned.isoformat()


class RealtimeTrafficConsumer:
    """
    Production-ready Kafka consumer with Redis state management.
    
    Features:
    - Consumes traffic data from Kafka
    - Buffers data into 5-minute snapshots
    - Persists to Redis for fault tolerance
    - Recovers from Redis on cold start
    - Triggers inference when window is ready
    """
    
    def __init__(
        self,
        kafka_broker: str,
        kafka_topic: str,
        kafka_group_id: str,
        redis_host: str,
        redis_port: int = 6379,
        redis_db: int = 0,
        window_size: int = 12,
        buffer_interval_sec: int = 300,
    ):
        """
        Initialize the consumer with Kafka and Redis connections.
        
        Args:
            kafka_broker: Kafka bootstrap servers
            kafka_topic: Kafka topic to consume from
            kafka_group_id: Consumer group ID
            redis_host: Redis hostname
            redis_port: Redis port
            redis_db: Redis database number
            window_size: Sliding window size (number of snapshots)
            buffer_interval_sec: Buffer flush interval in seconds
        """
        self.kafka_broker = kafka_broker
        self.kafka_topic = kafka_topic
        self.kafka_group_id = kafka_group_id
        self.window_size = window_size
        
        # Initialize components
        self.buffer = TrafficDataBuffer(interval_seconds=buffer_interval_sec)
        self.redis_client = create_redis_client(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            window_size=window_size,
        )
        
        # Kafka consumer (lazy init)
        self.consumer: Optional[KafkaConsumer] = None
        
        # Graceful shutdown flag
        self.running = True
        
        logger.info(
            f"Consumer initialized: broker={kafka_broker}, "
            f"topic={kafka_topic}, window_size={window_size}"
        )
    
    def connect_kafka(self):
        """
        Establish connection to Kafka with retry logic.
        """
        try:
            self.consumer = KafkaConsumer(
                self.kafka_topic,
                bootstrap_servers=self.kafka_broker,
                group_id=self.kafka_group_id,
                auto_offset_reset='earliest',  # Start from latest on first run
                enable_auto_commit=False,
                auto_commit_interval_ms=5000,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                max_poll_interval_ms=300000,  # 5 minutes
                session_timeout_ms=30000,
                heartbeat_interval_ms=10000,
            )
            
            logger.info(
                f"Connected to Kafka: {self.kafka_broker} → {self.kafka_topic}"
            )
            
        except KafkaError as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise
    
    def check_cold_start_recovery(self):
        """
        Check if Redis has existing state for cold-start recovery.
        
        If Redis contains sufficient data, the system can immediately
        start making predictions without waiting for new data.
        """
        logger.info("Checking for cold-start recovery from Redis...")
        
        if self.redis_client.is_window_ready():
            window_size = self.redis_client.get_window_size_current()
            metadata = self.redis_client.get_metadata()
            
            logger.info(
                f"Cold-start recovery SUCCESS: Found {window_size} "
                f"snapshots in Redis"
            )
            
            if metadata:
                logger.info(
                    f"   Last update: {metadata.get('last_update')}, "
                    f"Last snapshot: {metadata.get('snapshot_timestamp')}"
                )
            
            return True
        else:
            current_size = self.redis_client.get_window_size_current()
            logger.info(
                f"Cold-start: Window not ready yet "
                f"({current_size}/{self.window_size} snapshots)"
            )
            return False
    
    def run(self):
        """
        Main consumer loop with graceful shutdown support.
        """
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)
        
        # Connect to Kafka
        self.connect_kafka()
        
        # Check for cold-start recovery
        self.check_cold_start_recovery()
        
        logger.info("Consumer started - listening for messages...")
        
        try:
            # Main consumption loop
            while self.running:
                # 1. Thăm dò tin nhắn (Non-blocking), chờ tối đa 1 giây
                message_pack = self.consumer.poll(timeout_ms=1000)
                
                if not message_pack:
                    continue # Nếu không có tin, quay lại kiểm tra self.running
                
                # 2. Xử lý tin nhắn (Phải nằm TRONG vòng while)
                for topic_partition, messages in message_pack.items():
                    for message in messages:
                        raw_data = message.value
                        logger.debug(f"Received message: {message.offset}")
                        
                        # Add to buffer
                        snapshot = self.buffer.add_message(raw_data)
                        
                        if snapshot:
                            # Push snapshot to Redis
                            success = self.redis_client.push_snapshot(snapshot)
                            
                            # Kiểm tra success phải nằm TRONG khối if snapshot
                            if success:
                                logger.info(
                                    f"Snapshot pushed to Redis: "
                                    f"{snapshot['timestamp']}"
                                )
                                self.consumer.commit()
                                logger.debug("Kafka offset committed manually.")
                                # Check window ready
                                if self.redis_client.is_window_ready():
                                    self._trigger_inference()
                            else:
                                logger.error("Failed to push snapshot to Redis")
        
        except Exception as e:
            logger.error(f"Consumer error: {e}", exc_info=True)
            raise
        
        finally:
            self.shutdown()
    
    def _trigger_inference(self):
        """
        Trigger model inference when window is ready.
        
        This retrieves the full window from Redis and passes it
        to the inference engine.
        """
        logger.info("Window ready - triggering inference...")
        
        try:
            # Retrieve window from Redis
            window = self.redis_client.get_window()
            
            if len(window) < self.window_size:
                logger.warning(
                    f"Window incomplete: {len(window)}/{self.window_size}"
                )
                return
            
            # Import and run inference (lazy import to avoid circular dependency)
            from src.model.mock_inference import run_inference
            
            predictions = run_inference(window)
            
            logger.info(
                f"Inference completed: {len(predictions)} predictions generated"
            )
            
            # TODO: Publish predictions to Kafka output topic or store in DB
            
        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
    
    def _shutdown_handler(self, signum, frame):
        """
        Handle shutdown signals gracefully.
        """
        logger.info(f"Received signal {signum} - initiating graceful shutdown...")
        self.running = False
    
    def shutdown(self):
        """
        Clean up resources before exit.
        """
        logger.info("Shutting down consumer...")
        
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer closed")
        
        if self.redis_client:
            self.redis_client.close()
            logger.info("Redis connection closed")
        
        logger.info("Consumer shutdown complete")


# =====================================================================
# Factory Function
# =====================================================================

def create_consumer_from_env() -> RealtimeTrafficConsumer:
    """
    Create consumer instance from environment variables.
    
    Required env vars:
        KAFKA_BROKER, KAFKA_TOPIC, KAFKA_GROUP_ID,
        REDIS_HOST, REDIS_PORT, REDIS_DB,
        WINDOW_SIZE, BUFFER_INTERVAL_SEC
    
    Returns:
        RealtimeTrafficConsumer: Configured consumer instance
    """
    return RealtimeTrafficConsumer(
        kafka_broker=os.getenv("KAFKA_BROKER", "kafka:29092"),
        kafka_topic=os.getenv("KAFKA_TOPIC", "traffic-realtime"),
        kafka_group_id=os.getenv("KAFKA_GROUP_ID", "traffic-consumer-group"),
        redis_host=os.getenv("REDIS_HOST", "redis"),
        redis_port=int(os.getenv("REDIS_PORT", "6379")),
        redis_db=int(os.getenv("REDIS_DB", "0")),
        window_size=int(os.getenv("WINDOW_SIZE", "12")),
        buffer_interval_sec=int(os.getenv("BUFFER_INTERVAL_SEC", "15")),
    )
