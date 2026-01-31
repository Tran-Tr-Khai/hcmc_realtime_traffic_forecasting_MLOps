import logging
from typing import Generator, Tuple, Optional

import polars as pl
import ijson

from src.core.storage import MinIOClient

logger = logging.getLogger(__name__)


class TrafficExtractor(MinIOClient):
    def __init__(self, data_key: str, bucket_name: Optional[str] = None):
        """
        Initialize traffic extractor.
        
        Args:
            data_key: Object key in MinIO bucket (e.g., 'hcmc-traffic-data.json')
            bucket_name: Override default bucket (optional)
        """
        super().__init__(bucket_name=bucket_name)
        self.data_key = data_key
        
    def _parse_traffic_stream(self, stream) -> Generator[Tuple[int, int, int], None, None]:
        try:
            for date_str, sensors in ijson.kvitems(stream, ""):
                for sensor_id, files in sensors.items():
                    for filename, details in files.items():
                        ts = self._extract_timestamp_from_filename(filename)
                        if ts is not None:
                            yield (
                                ts,
                                int(sensor_id),
                                int(details.get("count", 0))
                            )
        except Exception as e:
            logger.error(f"Error parsing traffic stream: {e}")
            raise

    def _extract_timestamp_from_filename(self, filename: str) -> Optional[int]:
        """
        Extract Unix timestamp (milliseconds) from filename.
        
        Format: {hash}_{timestamp_ms}.txt
        Example: 5d8cd..._1649154313885.txt -> 1649154313885
        """
        try:
            return int(filename.split("_")[-1].replace('.txt', ''))
        except (IndexError, ValueError):
            return None

    def extract(self, chunk_size: int=100000) -> pl.LazyFrame:
        """
        Extract traffic data from MinIO as a Polars LazyFrame.
        
        Returns:
            LazyFrame with columns: [timestamp, sensor_id, count]
            Timestamp is localized to Asia/Ho_Chi_Minh timezone
        """
        logger.info(f"Streaming data from s3://{self.bucket_name}/{self.data_key}")

        # Get stream from MinIO (using inherited method)
        stream = self.get_object_stream(self.data_key)

        # Parse stream
        record_generator = self._parse_traffic_stream(stream)

        # Create Polars DataFrame from generator
        schema = {
            "timestamp": pl.Int64,
            "sensor_id": pl.Int64,
            "count": pl.Int64
        }

        chunks = []
        current_batch = []
        
        for record in record_generator:
            current_batch.append(record)
            
            # Khi đủ 100k dòng, chuyển ngay sang Polars DataFrame (rất nhẹ)
            if len(current_batch) >= chunk_size:
                chunks.append(pl.from_records(current_batch, schema=schema))
                current_batch = [] # Giải phóng List Python
        
        # Xử lý phần dư còn lại
        if current_batch:
            chunks.append(pl.from_records(current_batch, schema=schema))

        if not chunks:
            logger.warning("No data extracted! Check JSON structure.")
            return pl.DataFrame(schema=schema).lazy()

        # 3. Nối các chunks lại thành 1 DataFrame duy nhất
        # Polars concat cực nhanh và tiết kiệm RAM
        df = pl.concat(chunks)
        
        logger.info(f"Extracted {df.height} records. Peak RAM will stay low.")
        # Convert to LazyFrame with timezone handling
        return (
            df.lazy()
            .with_columns(
                pl.from_epoch(pl.col("timestamp"), time_unit="ms")
                .alias("timestamp_utc")
            )
            .with_columns(
                pl.col("timestamp_utc")
                .dt.replace_time_zone("UTC")
                .dt.convert_time_zone("Asia/Ho_Chi_Minh")
                .alias("timestamp")
            )
            .select(["timestamp", "sensor_id", "count"])
        )


# Test script
if __name__ == "__main__":
    import time
    import tracemalloc
    
    path = "hcmc-traffic-data.json"
    
    print("-" * 60)
    print(f"Testing TrafficExtractor with MEMORY PROFILING")
    print(f"File: {path}")
    print("-" * 60)

    try:
        tracemalloc.start()
        start_time = time.time()
        
        extractor = TrafficExtractor(path)
        df = extractor.extract().collect()
        
        end_time = time.time()
        duration = end_time - start_time
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        peak_mb = peak / 1024 / 1024
        
        print("\n" + "=" * 60)
        print("EXTRACTION SUCCESSFUL")
        print("=" * 60)
        print(f"Time taken:      {duration:.4f} seconds")
        print(f"Peak RAM Usage:  {peak_mb:.2f} MB")
        print(f"Total Rows:      {df.height}")
        print("-" * 60)
        
    except Exception as e:
        print("\nTEST FAILED")
        print(f"Error: {e}")
