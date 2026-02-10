"""
Kafka Client Factory Module

Provides centralized Kafka client creation with consistent configuration
for both producers and consumers across the traffic forecasting system.
"""

import json
import logging
from typing import Optional, Callable, Any
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError

logger = logging.getLogger(__name__)


def create_kafka_producer(
    bootstrap_servers: str = 'localhost:9092',
    acks: str = 'all',
    retries: int = 3,
    compression_type: str = 'gzip',
    batch_size: int = 16384,
    linger_ms: int = 10,
    max_in_flight_requests_per_connection: int = 5,
    request_timeout_ms: int = 30000,
    metadata_max_age_ms: int = 300000,
    value_serializer: Optional[Callable] = None,
    key_serializer: Optional[Callable] = None,
    **kwargs
) -> KafkaProducer:
    """
    Create a configured Kafka producer instance.
    
    Args:
        bootstrap_servers: Kafka broker address(es)
        acks: Number of acknowledgments the producer requires ('all', '1', '0')
        retries: Number of retries for failed requests
        compression_type: Compression algorithm ('gzip', 'snappy', 'lz4', 'zstd', None)
        batch_size: Batch size for batching records
        linger_ms: Time to wait for additional records before sending
        max_in_flight_requests_per_connection: Max unacknowledged requests per connection
        request_timeout_ms: Request timeout in milliseconds
        metadata_max_age_ms: Metadata cache timeout
        value_serializer: Custom value serializer function
        key_serializer: Custom key serializer function
        **kwargs: Additional KafkaProducer configuration
        
    Returns:
        KafkaProducer: Configured Kafka producer instance
        
    Raises:
        KafkaError: If connection to Kafka fails
    """
    # Default serializers if not provided
    if value_serializer is None:
        value_serializer = lambda v: json.dumps(v).encode('utf-8')
    
    if key_serializer is None:
        key_serializer = lambda k: k.encode('utf-8') if k else None
    
    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=value_serializer,
            key_serializer=key_serializer,
            
            # Reliability settings
            acks=acks,
            retries=retries,
            max_in_flight_requests_per_connection=max_in_flight_requests_per_connection,
            
            # Performance tuning
            batch_size=batch_size,
            linger_ms=linger_ms,
            compression_type=compression_type,
            
            # Timeout settings
            request_timeout_ms=request_timeout_ms,
            metadata_max_age_ms=metadata_max_age_ms,
            
            # API version
            api_version=(2, 5, 0),
            
            # Additional user configurations
            **kwargs
        )
        
        logger.info(
            f"Kafka producer created successfully - "
            f"servers={bootstrap_servers}, acks={acks}, compression={compression_type}"
        )
        
        return producer
        
    except KafkaError as e:
        logger.error(f"Failed to create Kafka producer: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating Kafka producer: {e}")
        raise


def create_kafka_consumer(
    topic: str,
    bootstrap_servers: str = 'localhost:9092',
    group_id: str = 'default-consumer-group',
    auto_offset_reset: str = 'earliest',
    enable_auto_commit: bool = False,
    auto_commit_interval_ms: int = 5000,
    max_poll_interval_ms: int = 300000,
    session_timeout_ms: int = 30000,
    heartbeat_interval_ms: int = 10000,
    value_deserializer: Optional[Callable] = None,
    key_deserializer: Optional[Callable] = None,
    **kwargs
) -> KafkaConsumer:
    """
    Create a configured Kafka consumer instance.
    
    Args:
        topic: Kafka topic to subscribe to
        bootstrap_servers: Kafka broker address(es)
        group_id: Consumer group ID
        auto_offset_reset: What to do when no initial offset ('earliest', 'latest', 'none')
        enable_auto_commit: Enable automatic offset commits
        auto_commit_interval_ms: Frequency of automatic offset commits
        max_poll_interval_ms: Max time between polls before consumer is considered dead
        session_timeout_ms: Consumer session timeout
        heartbeat_interval_ms: Heartbeat frequency
        value_deserializer: Custom value deserializer function
        key_deserializer: Custom key deserializer function
        **kwargs: Additional KafkaConsumer configuration
        
    Returns:
        KafkaConsumer: Configured Kafka consumer instance
        
    Raises:
        KafkaError: If connection to Kafka fails
    """
    # Default deserializers if not provided
    if value_deserializer is None:
        value_deserializer = lambda m: json.loads(m.decode('utf-8'))
    
    if key_deserializer is None:
        key_deserializer = lambda k: k.decode('utf-8') if k else None
    
    try:
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            
            # Offset management
            auto_offset_reset=auto_offset_reset,
            enable_auto_commit=enable_auto_commit,
            auto_commit_interval_ms=auto_commit_interval_ms,
            
            # Session management
            max_poll_interval_ms=max_poll_interval_ms,
            session_timeout_ms=session_timeout_ms,
            heartbeat_interval_ms=heartbeat_interval_ms,
            
            # Deserialization
            value_deserializer=value_deserializer,
            key_deserializer=key_deserializer,
            
            # Additional user configurations
            **kwargs
        )
        
        logger.info(
            f"Kafka consumer created successfully - "
            f"topic={topic}, servers={bootstrap_servers}, group_id={group_id}"
        )
        
        return consumer
        
    except KafkaError as e:
        logger.error(f"Failed to create Kafka consumer: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating Kafka consumer: {e}")
        raise


def check_kafka_connection(bootstrap_servers: str = 'localhost:9092') -> bool:
    """
    Check if Kafka broker is reachable.
    
    Args:
        bootstrap_servers: Kafka broker address(es)
        
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        # Create a temporary producer to test connection
        producer = create_kafka_producer(
            bootstrap_servers=bootstrap_servers,
            request_timeout_ms=5000
        )
        producer.close()
        logger.info(f"Kafka connection test successful - {bootstrap_servers}")
        return True
        
    except Exception as e:
        logger.error(f"Kafka connection test failed - {bootstrap_servers}: {e}")
        return False
