import os
import logging
import time
from typing import Generator, Tuple, Optional

import polars as pl
from dotenv import load_dotenv
import boto3
import ijson  # BẮT BUỘC: pip install ijson

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class MinIoTrafficExtractor:
    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path
        self.endpoint = os.getenv("MINIO_ENDPOINT_URL")
        self.access_key = os.getenv("MINIO_ACCESS_KEY")
        self.secret_key = os.getenv("MINIO_SECRET_KEY")
        self.default_bucket = os.getenv("MINIO_BUCKET_NAME")
        
        if not all([self.endpoint, self.access_key, self.secret_key]):
            raise ValueError("Missing MinIO credentials.")
            
        self._s3_client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )

    def _get_s3_stream(self, bucket: str, key: str):
        """Lấy luồng dữ liệu (stream) thay vì tải cả file."""
        try:
            response = self._s3_client.get_object(Bucket=bucket, Key=key)
            return response["Body"]
        except Exception as e:
            logger.error(f"Failed to get object s3://{bucket}/{key}: {e}")
            raise

    def _parse_traffic_stream(self, stream) -> Generator[Tuple[int, int, int], None, None]:
        """
        Dùng ijson để duyệt qua file JSON mà không load vào RAM.
        Cấu trúc: { "YYYY-MM-DD": { "sensor_id": { "filename": { "count": X, ... } } } }
        """
        try:
            # ijson.kvitems(stream, "") giúp duyệt qua level cao nhất (Date)
            for date_str, sensors in ijson.kvitems(stream, ""):
                for sensor_id, files in sensors.items():
                    for filename, details in files.items():
                        ts = self._extract_timestamp_from_filename(filename)
                        if ts is not None:
                            # Yield tuple (nhẹ hơn dict) để tiết kiệm memory
                            yield (
                                ts,
                                int(sensor_id),
                                int(details.get("count", 0))
                            )
        except Exception as e:
            logger.error(f"Error parsing stream: {e}")
            raise

    def extract(self) -> pl.LazyFrame:
        # Xử lý đường dẫn
        if self.raw_data_path.startswith("s3://"):
            parts = self.raw_data_path.replace("s3://", "").split("/", 1)
            bucket, key = parts[0], parts[1]
        elif self.default_bucket:
            bucket, key = self.default_bucket, self.raw_data_path
        else:
            raise ValueError("Invalid path configuration")

        logger.info(f"🚀 Streaming data from {bucket}/{key}")

        # 1. Lấy Stream
        stream = self._get_s3_stream(bucket, key)

        # 2. Tạo Generator
        record_generator = self._parse_traffic_stream(stream)

        # 3. Tạo Polars DataFrame từ Generator
        # Schema tường minh giúp Polars cấp phát bộ nhớ hiệu quả
        schema = {
            "timestamp": pl.Int64,
            "sensor_id": pl.Int64,
            "count": pl.Int64
        }

        data_list = list(record_generator)
        if not data_list:
            logger.warning("⚠️ Generator yielded no data! Check your JSON structure or File Path.")
        # from_records tiêu thụ generator trực tiếp
        df = pl.from_records(data_list, schema=schema, orient="row")
        
        logger.info(f"✅ Extracted {df.height} records via streaming.")

        # 4. Trả về LazyFrame với xử lý Timezone chuẩn
        return (
            df.lazy()
            .with_columns(
                pl.from_epoch(pl.col("timestamp"), time_unit="ms")
                .alias("timestamp_utc")
            )
            .with_columns(
                pl.col("timestamp_utc")
                .dt.replace_time_zone("UTC")              # Đánh dấu gốc là UTC
                .dt.convert_time_zone("Asia/Ho_Chi_Minh") # Chuyển sang giờ VN
                .alias("timestamp")
            )
            .select(["timestamp", "sensor_id", "count"])
        )

    def _extract_timestamp_from_filename(self, filename: str) -> Optional[int]:
        try:
            return int(filename.split("_")[-1].replace('.txt', ''))
        except (IndexError, ValueError):
            return None

# --- TEST SCRIPT ĐỂ SO SÁNH ---
if __name__ == "__main__":
    import time
    import tracemalloc  # <--- Thư viện đo bộ nhớ
    
    # Path file (bạn sửa lại cho đúng file đang test)
    path = "hcmc-traffic-data.json"
    
    print("-" * 60)
    print(f"🚀 Testing Extraction with MEMORY PROFILING")
    print(f"📁 File: {path}")
    print("-" * 60)

    try:
        # 1. Bắt đầu theo dõi RAM
        tracemalloc.start()
        
        # 2. Bắt đầu bấm giờ
        start_time = time.time()
        
        # --- CHẠY EXTRACTOR ---
        extractor = MinIoTrafficExtractor(path)
        # collect() là lúc data thực sự được load vào RAM
        df = extractor.extract().collect() 
        # ----------------------

        # 3. Kết thúc bấm giờ
        end_time = time.time()
        duration = end_time - start_time
        
        # 4. Lấy thông số RAM (current, peak)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop() # Dừng theo dõi
        
        # Chuyển đổi sang MB
        peak_mb = peak / 1024 / 1024
        
        # 5. In kết quả
        print("\n" + "=" * 60)
        print("✅ EXTRACTION SUCCESSFUL")
        print("=" * 60)
        print(f"⏱️  Time taken:      {duration:.4f} seconds")
        print(f"🧠 Peak RAM Usage:  {peak_mb:.2f} MB")  # <--- SỰ KHÁC BIỆT LÀ ĐÂY
        print(f"📊 Total Rows:      {df.height}")
        print("-" * 60)
        
    except Exception as e:
        print("\n❌ TEST FAILED")
        print(f"Error details: {e}")