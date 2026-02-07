import json
import time
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from kafka import KafkaProducer
from kafka.errors import KafkaError

logger = logging.getLogger(__name__)

class KafkaTrafficProducer:
    """
    Kafka producer for streaming traffic data.
    
    Reads traffic data from a JSON file and sends it to Kafka topic
    with configurable delay to simulate real-time streaming.
    """
    def __init__(
        self,
        bootstrap_servers: str = 'localhost:9092',
        topic_name: str = 'traffic-raw',
        input_file: str = 'data/raw/hcmc-traffic-data-realtime.json',
        speed_factor: float = 20.0
    ):
        """
        Initialize the Kafka producer.
        
        Args:
            bootstrap_servers: Kafka broker address
            topic_name: Name of the Kafka topic to publish to
            input_file: Path to the input JSON file
            speed_factor: Time acceleration factor (20.0 = 20x faster than real-time)
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic_name = topic_name
        self.input_file = Path(input_file)
        self.speed_factor = speed_factor
        
        # Initialize Kafka producer
        self.producer = None
        self._init_producer()
        
        # Statistics
        self.messages_sent = 0
        self.errors_count = 0
        self.start_time = None
    
    def _init_producer(self):
        """Initialize Kafka producer with proper configuration."""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                
                # Performance tuning
                acks='all',  # Wait for all replicas to acknowledge
                retries=3,   # Retry failed sends
                max_in_flight_requests_per_connection=5,
                
                # Batching for efficiency
                batch_size=16384,
                linger_ms=10,
                
                # Compression
                compression_type='gzip',
                
                # Error handling
                api_version=(2, 5, 0),
                request_timeout_ms=30000,
                metadata_max_age_ms=300000
            )
            logger.info(f"Kafka producer initialized successfully")
            logger.info(f"Bootstrap servers: {self.bootstrap_servers}")
            logger.info(f"Target topic: {self.topic_name}")
            
        except KafkaError as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            raise
    
    def load_data(self) -> List[Dict]:
        """
        ETL Strategy: "Flatten & Sort" with Unix Timestamp Precision
        1. Flatten nested structure into list of records
        2. Extract Unix millisecond timestamp from filename (e.g., file_1652056213860.txt)
        3. Sort by unix_timestamp_ms for absolute chronological order
        
        Returns:
            List of flattened and sorted traffic records
            
        """
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        logger.info(f"Loading data from {self.input_file}...")
        
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Flatten nested JSON structure
            logger.info("Flattening nested JSON structure...")
            records = self._flatten_nested_json(data)
            
            if not records:
                raise ValueError(
                    "No valid records found. Expected format: "
                    "{date: {sensor_id: {filename: {count, timestamp}}}}"
                )
            
            # Sort by unix_timestamp_ms (absolute chronological order)
            logger.info("Sorting records by Unix timestamp (absolute chronological order)...")
            records.sort(key=lambda x: x['unix_timestamp_ms'])
            
            # Calculate time statistics
            first_dt = datetime.fromtimestamp(records[0]['unix_timestamp_ms'] / 1000)
            last_dt = datetime.fromtimestamp(records[-1]['unix_timestamp_ms'] / 1000)
            time_span_seconds = (records[-1]['unix_timestamp_ms'] - records[0]['unix_timestamp_ms']) / 1000
            
            logger.info(f"Loaded and sorted {len(records)} traffic records")
            logger.info(f"Time range: {first_dt} to {last_dt}")
            logger.info(f"Total span: {time_span_seconds / 3600:.2f} hours")
            logger.info(f"First record: date={records[0]['date']}, sensor={records[0]['sensor_id']}, count={records[0]['count']}")
            
            return records
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _flatten_nested_json(self, data: Dict) -> List[Dict]:
        """
        Flatten nested JSON structure into a list of records.
        """
        flattened_records = []
        skipped_count = 0
        
        # Iterate through dates (level 1)
        for date, sensors in data.items():
            if not isinstance(sensors, dict):
                logger.warning(f"Skipping invalid date entry: {date}")
                continue
            
            # Iterate through sensor IDs (level 2)
            for sensor_id, files in sensors.items():
                if not isinstance(files, dict):
                    logger.warning(f"Skipping invalid sensor entry: {sensor_id} on {date}")
                    continue
                
                # Iterate through filenames (level 3)
                for filename, record_data in files.items():
                    if not isinstance(record_data, dict):
                        logger.warning(f"Skipping invalid file entry: {filename}")
                        continue
                    
                    # Extract count and timestamp
                    count = record_data.get('count')
                    timestamp = record_data.get('timestamp')
                    
                    # Validate timestamp format
                    if not timestamp or not isinstance(timestamp, list) or len(timestamp) != 2:
                        logger.warning(f"Invalid timestamp for {date}/{sensor_id}/{filename}: {timestamp}")
                        skipped_count += 1
                        continue
                    
                    hour, minute = timestamp[0], timestamp[1]
                    
                    # CRITICAL: Extract Unix timestamp from filename
                    # Expected format: prefix_1652056213860.txt (13-digit Unix ms timestamp)
                    unix_timestamp_ms = self._extract_unix_timestamp(filename)
                    
                    if unix_timestamp_ms is None:
                        logger.warning(f"Could not extract Unix timestamp from filename: {filename}")
                        skipped_count += 1
                        continue
                    
                    # Convert to datetime for human-readable logging
                    try:
                        dt = datetime.fromtimestamp(unix_timestamp_ms / 1000)
                        datetime_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                    except (ValueError, OSError):
                        logger.warning(f"Invalid Unix timestamp {unix_timestamp_ms} in {filename}")
                        skipped_count += 1
                        continue
                    
                    # Create flattened record
                    flattened_record = {
                        'date': date,
                        'sensor_id': str(sensor_id),
                        'filename': filename,
                        'count': count,
                        'timestamp': timestamp,
                        'hour': hour,
                        'minute': minute,
                        'unix_timestamp_ms': unix_timestamp_ms,
                        'datetime': datetime_str
                    }
                    
                    flattened_records.append(flattened_record)
        
        logger.info(f"  Flattened {len(flattened_records)} records from nested structure")
        if skipped_count > 0:
            logger.warning(f"  Skipped {skipped_count} records due to missing/invalid timestamps")
        return flattened_records
    
    def _extract_unix_timestamp(self, filename: str) -> Optional[int]:
        # Pattern: Look for 13-digit number (Unix ms timestamp)
        # It should be preceded by underscore or start of string
        # and followed by file extension or end of string
        pattern = r'[_]?(\d{13})(?:\.\w+)?$'
        
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))
        
        # Fallback: Try to find any 13-digit number
        pattern_fallback = r'(\d{13})'
        match_fallback = re.search(pattern_fallback, filename)
        if match_fallback:
            return int(match_fallback.group(1))
        
        return None
    
    def send_message(self, record: Dict, record_index: int) -> bool:
        try:
            # Extract key information for message key (for partitioning)
            sensor_id = record.get('sensor_id', f'unknown_{record_index}')
            message_key = sensor_id
            
            # LEAN PAYLOAD: Transform to strict schema
            # Rule: Single Source of Truth (Time) - use unix_timestamp_ms as 'ts'
            # Rule: No Derived Data - exclude date, hour, minute, datetime, timestamp array
            # Rule: Traceability - include source filename as 'src'
            lean_payload = {
                'sensor_id': sensor_id,
                'count': record.get('count'),
                'ts': record.get('unix_timestamp_ms'),  # Unix ms timestamp - source of truth
                'src': record.get('filename'),  # Source filename for traceability
                'ingest_ts': int(time.time() * 1000)  # Producer-side timestamp (current time in ms)
            }
            
            # Send to Kafka asynchronously
            future = self.producer.send(
                topic=self.topic_name,
                key=message_key,
                value=lean_payload
            )
            
            # Wait for send confirmation (optional, but recommended for reliability)
            # record_metadata = future.get(timeout=10)
            
            # Log success (using internal record for readable datetime)
            self.messages_sent += 1
            logger.debug(
                f"Sent message #{self.messages_sent}: "
                f"sensor_id={sensor_id}, "
                # f"ts={lean_payload['ts']}, "
                # f"datetime={record.get('datetime', 'N/A')}, "
                # f"partition={record_metadata.partition}, "
                # f"offset={record_metadata.offset}"
            )
            
            return True
            
        except KafkaError as e:
            self.errors_count += 1
            logger.error(f"✗ Failed to send message #{record_index}: {e}")
            return False
        except Exception as e:
            self.errors_count += 1
            logger.error(f"✗ Unexpected error sending message #{record_index}: {e}")
            return False
    
    def stream_data(self):
        try:
            # Load and sort data
            records = self.load_data()
            total_records = len(records)
            
            if total_records == 0:
                logger.warning("No records to stream. Exiting.")
                return
            
            # Calculate total real-time span using Unix timestamps
            time_span_ms = records[-1]['unix_timestamp_ms'] - records[0]['unix_timestamp_ms']
            time_span_seconds = time_span_ms / 1000
            estimated_runtime_seconds = time_span_seconds / self.speed_factor
            
            first_dt = datetime.fromtimestamp(records[0]['unix_timestamp_ms'] / 1000)
            last_dt = datetime.fromtimestamp(records[-1]['unix_timestamp_ms'] / 1000)
            
            logger.info("=" * 80)
            logger.info("STARTING REAL-TIME DATA STREAMING (Unix Timestamp Precision)")
            logger.info("=" * 80)
            logger.info(f"Total records to stream: {total_records}")
            logger.info(f"Time range: {first_dt} to {last_dt}")
            logger.info(f"Real-time span: {time_span_seconds / 3600:.2f} hours")
            logger.info(f"Speed factor: {self.speed_factor}x")
            logger.info(f"Estimated streaming time: {estimated_runtime_seconds / 60:.2f} minutes")
            logger.info("=" * 80)
            
            self.start_time = time.time()
            prev_unix_timestamp_ms = records[0]['unix_timestamp_ms']
            
            # Stream each record
            for idx, record in enumerate(records, start=1):
                current_unix_timestamp_ms = record['unix_timestamp_ms']
                
                # Calculate dynamic sleep time based on Unix timestamp difference
                if idx > 1:  # Skip sleep before first message
                    time_diff_ms = current_unix_timestamp_ms - prev_unix_timestamp_ms
                    time_diff_seconds = time_diff_ms / 1000
                    
                    # Apply speed factor
                    sleep_seconds = time_diff_seconds / self.speed_factor
                    
                    # Sleep to simulate real-time flow
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
                
                # Send message
                success = self.send_message(record, idx)
                
                # Progress update every 50 messages
                if idx % 50 == 0:
                    elapsed = time.time() - self.start_time
                    rate = self.messages_sent / elapsed if elapsed > 0 else 0
                    progress = (idx / total_records) * 100
                    
                    # Calculate simulated time progress
                    simulated_seconds = (current_unix_timestamp_ms - records[0]['unix_timestamp_ms']) / 1000
                    current_dt = datetime.fromtimestamp(current_unix_timestamp_ms / 1000)
                    
                    logger.info(
                        f"Progress: {idx}/{total_records} ({progress:.1f}%) | "
                        f"Simulated time: {current_dt.strftime('%H:%M:%S')} | "
                        f"Rate: {rate:.2f} msg/sec | "
                        f"Errors: {self.errors_count}"
                    )
                
                # Update previous timestamp
                prev_unix_timestamp_ms = current_unix_timestamp_ms
            
            # Final statistics
            self._print_statistics()
            
        except KeyboardInterrupt:
            logger.warning("\nStreaming interrupted by user (Ctrl+C)")
            self._print_statistics()
        except Exception as e:
            logger.error(f"Streaming failed: {e}", exc_info=True)
        finally:
            self.close()
    
    def _print_statistics(self):
        """Print streaming statistics."""
        if self.start_time:
            elapsed = time.time() - self.start_time
            rate = self.messages_sent / elapsed if elapsed > 0 else 0
            
            logger.info("\n" + "=" * 80)
            logger.info("STREAMING STATISTICS")
            logger.info("=" * 80)
            logger.info(f"Total messages sent: {self.messages_sent}")
            logger.info(f"Total errors: {self.errors_count}")
            logger.info(f"Success rate: {(self.messages_sent / (self.messages_sent + self.errors_count) * 100):.2f}%")
            logger.info(f"Total time: {elapsed:.2f}s")
            logger.info(f"Average rate: {rate:.2f} messages/sec")
            logger.info("=" * 80)
    
    def close(self):
        """Close the Kafka producer and cleanup resources."""
        if self.producer:
            logger.info("Closing Kafka producer...")
            self.producer.flush()  # Ensure all messages are sent
            self.producer.close()
            logger.info("Kafka producer closed successfully")

