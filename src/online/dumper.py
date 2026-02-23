import os
import json
import logging
import signal
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional

from kafka import KafkaConsumer
from kafka.errors import KafkaError


logger = logging.getLogger(__name__)

class KafkaDailyDumper:
    """
    Background service to dump Kafka messages to daily JSON files.
    
    Features:
        - Daily file rotation at midnight (00:00:00)
        - Buffered writes (configurable buffer size)
        - Graceful shutdown handling (SIGINT, SIGTERM)
        - Message enrichment with Kafka metadata
        - Crash recovery (append mode)
        - Periodic statistics logging
    """
    
    def __init__(
        self,
        bootstrap_servers: str = 'kafka:29092',
        topic: str = 'traffic-raw',
        group_id: str = 'kafka-dumper-group',
        output_dir: str = 'data/kafka_dumps',
        auto_offset_reset: str = 'latest',
        buffer_size: int = 100
    ):
        """
        Initialize Kafka dumper.
        
        Args:
            bootstrap_servers: Kafka broker address
            topic: Kafka topic to consume from
            group_id: Consumer group ID
            output_dir: Directory to save daily dump files
            auto_offset_reset: Where to start consuming ('earliest' or 'latest')
            buffer_size: Number of messages to buffer before flushing to disk
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.output_dir = Path(output_dir)
        self.auto_offset_reset = auto_offset_reset
        self.buffer_size = buffer_size
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir.absolute()}")
        
        # Kafka consumer
        self.consumer: Optional[KafkaConsumer] = None
        
        # Daily file management
        self.current_date: Optional[date] = None
        self.current_file_path: Optional[Path] = None
        self.current_file_handle = None
        self.buffer: List[Dict] = []
        
        # Statistics
        self.total_messages = 0
        self.daily_messages = 0
        self.files_created = 0
        
        # Graceful shutdown
        self.running = True
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}. Shutting down gracefully...")
        self.running = False
    
    def _init_consumer(self):
        """Initialize Kafka consumer."""
        try:
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset=self.auto_offset_reset,
                enable_auto_commit=True,
                auto_commit_interval_ms=5000,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda m: m.decode('utf-8') if m else None,
                consumer_timeout_ms=1000,  # Poll timeout
                max_poll_records=500,
                session_timeout_ms=30000,
                heartbeat_interval_ms=10000
            )
            logger.info(f"✓ Kafka consumer initialized")
            logger.info(f"  Bootstrap servers: {self.bootstrap_servers}")
            logger.info(f"  Topic: {self.topic}")
            logger.info(f"  Group ID: {self.group_id}")
            logger.info(f"  Auto offset reset: {self.auto_offset_reset}")
            
        except KafkaError as e:
            logger.error(f"Failed to initialize Kafka consumer: {e}")
            raise
    
    def _get_current_date_file_path(self) -> Path:
        """Get file path for current date."""
        today = date.today()
        filename = f"kafka-dump-{today.strftime('%Y-%m-%d')}.json"
        return self.output_dir / filename
    
    def _open_daily_file(self, force_new: bool = False):
        """
        Open or rotate daily dump file.
        
        Args:
            force_new: Force create new file even if date hasn't changed
        """
        today = date.today()
        
        # Check if date changed or force new file
        if self.current_date != today or force_new:
            # Close previous file if exists
            if self.current_file_handle is not None:
                self._flush_buffer()
                self._close_file()
            
            # Open new file
            self.current_date = today
            self.current_file_path = self._get_current_date_file_path()
            
            # Check if file already exists (append mode for crash recovery)
            file_exists = self.current_file_path.exists()
            
            self.current_file_handle = open(
                self.current_file_path, 
                'a',  # Append mode
                encoding='utf-8'
            )
            
            if file_exists:
                logger.info(f"📝 Appending to existing file: {self.current_file_path.name}")
            else:
                logger.info(f"📄 Created new daily file: {self.current_file_path.name}")
                self.files_created += 1
            
            self.daily_messages = 0
    
    def _close_file(self):
        """Close current file and log statistics."""
        if self.current_file_handle is not None:
            self.current_file_handle.close()
            
            file_size = self.current_file_path.stat().st_size / 1024  # KB
            logger.info(
                f"✓ Closed file: {self.current_file_path.name} "
                f"(Messages: {self.daily_messages}, Size: {file_size:.2f} KB)"
            )
            
            self.current_file_handle = None
    
    def _write_message(self, message: Dict):
        """
        Write message to buffer and flush if needed.
        
        Args:
            message: Kafka message payload
        """
        # Add to buffer
        self.buffer.append(message)
        
        # Flush if buffer is full
        if len(self.buffer) >= self.buffer_size:
            self._flush_buffer()
    
    def _flush_buffer(self):
        """Flush buffer to disk."""
        if not self.buffer or self.current_file_handle is None:
            return
        
        try:
            # Write each message as JSON line
            for msg in self.buffer:
                json_line = json.dumps(msg, ensure_ascii=False)
                self.current_file_handle.write(json_line + '\n')
            
            # Flush to disk
            self.current_file_handle.flush()
            os.fsync(self.current_file_handle.fileno())
            
            logger.debug(f"Flushed {len(self.buffer)} messages to disk")
            self.buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush buffer: {e}")
    
    def _process_message(self, message):
        """
        Process a single Kafka message.
        
        Args:
            message: Kafka ConsumerRecord
        """
        try:
            # Extract message data
            payload = message.value
            
            # Validate required fields
            if not isinstance(payload, dict):
                logger.warning(f"Invalid message format (not dict): {payload}")
                return
            
            required_fields = ['sensor_id', 'count', 'ts']
            missing_fields = [f for f in required_fields if f not in payload]
            if missing_fields:
                logger.warning(f"Missing required fields: {missing_fields} in message: {payload}")
                return
            
            # Enrich with metadata
            enriched_message = {
                **payload,  # Original payload
                'kafka_partition': message.partition,
                'kafka_offset': message.offset,
                'kafka_timestamp': message.timestamp,  # Kafka broker timestamp
                'dumper_timestamp': int(datetime.now().timestamp() * 1000)  # Dumper timestamp
            }
            
            # Write to file
            self._write_message(enriched_message)
            
            # Update statistics
            self.total_messages += 1
            self.daily_messages += 1
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
    
    def _print_statistics(self):
        """Print current statistics."""
        logger.info("=" * 80)
        logger.info("KAFKA DUMPER STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Total messages dumped: {self.total_messages}")
        logger.info(f"Today's messages: {self.daily_messages}")
        logger.info(f"Files created: {self.files_created}")
        logger.info(f"Current file: {self.current_file_path.name if self.current_file_path else 'None'}")
        logger.info(f"Buffer size: {len(self.buffer)}")
        logger.info("=" * 80)
    
    def run(self):
        """
        Main loop: consume Kafka messages and dump to daily files.
        """
        try:
            logger.info("=" * 80)
            logger.info("STARTING KAFKA DAILY DUMPER")
            logger.info("=" * 80)
            
            # Initialize consumer
            self._init_consumer()
            
            # Open initial file
            self._open_daily_file()
            
            logger.info("✓ Ready to consume messages. Press Ctrl+C to stop.")
            logger.info("=" * 80)
            
            # Main consumption loop
            last_stats_time = datetime.now()
            stats_interval = timedelta(minutes=5)  # Print stats every 5 minutes
            
            while self.running:
                try:
                    # Check if date changed (rotate to new file at midnight)
                    if self.current_date != date.today():
                        logger.info("📅 Date changed. Rotating to new daily file...")
                        self._open_daily_file()
                    
                    # Poll messages
                    messages = self.consumer.poll(timeout_ms=1000, max_records=500)
                    
                    if messages:
                        for topic_partition, records in messages.items():
                            for record in records:
                                self._process_message(record)
                                
                                # Log progress every 1000 messages
                                if self.total_messages % 1000 == 0:
                                    logger.info(
                                        f"Progress: {self.total_messages} total messages | "
                                        f"{self.daily_messages} today | "
                                        f"Buffer: {len(self.buffer)}"
                                    )
                    
                    # Print statistics periodically
                    if datetime.now() - last_stats_time > stats_interval:
                        self._print_statistics()
                        last_stats_time = datetime.now()
                
                except Exception as e:
                    logger.error(f"Error in consumption loop: {e}", exc_info=True)
                    continue
            
            # Graceful shutdown
            logger.info("\nShutting down...")
            self._flush_buffer()
            self._close_file()
            
            if self.consumer:
                self.consumer.close()
                logger.info("✓ Kafka consumer closed")
            
            self._print_statistics()
            logger.info("✓ Kafka dumper stopped successfully")
            
        except KeyboardInterrupt:
            logger.info("\nInterrupted by user (Ctrl+C)")
            self._shutdown()
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            self._shutdown()
    
    def _shutdown(self):
        """Clean shutdown procedure."""
        logger.info("Performing cleanup...")
        self._flush_buffer()
        self._close_file()
        
        if self.consumer:
            self.consumer.close()
        
        self._print_statistics()
        logger.info("✓ Cleanup complete")
