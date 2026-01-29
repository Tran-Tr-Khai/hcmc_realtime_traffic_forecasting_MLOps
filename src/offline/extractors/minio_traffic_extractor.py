import os
import json
import logging
from typing import Set, Optional

import polars as pl
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
load_dotenv()

class MinIoTrafficExtractor:
    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path

        # MinIO / S3 credentials
        self.endpoint = os.getenv("MINIO_ENDPOINT_URL")
        self.access_key = os.getenv("MINIO_ACCESS_KEY")
        self.secret_key = os.getenv("MINIO_SECRET_KEY")
        self.default_bucket = os.getenv("MINIO_BUCKET_NAME")

        self._s3_client: Optional[boto3.client] = None

    def _ensure_s3_client(self):
        if self._s3_client is None:
            if not all([self.endpoint, self.access_key, self.secret_key]):
                raise ValueError("Missing MinIO credentials in environment.")
            self._s3_client = boto3.client(
                "s3",
                endpoint_url=self.endpoint,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
            )
        return self._s3_client

    def _read_json_from_s3(self, bucket: str, key: str):
        client = self._ensure_s3_client()
        try:
            # Lưu ý: Với file >500MB, cách đọc này sẽ tốn RAM.
            # Nhưng với JSON nested, đây là cách đơn giản nhất.
            resp = client.get_object(Bucket=bucket, Key=key)
            raw = resp["Body"].read().decode("utf-8")
            return json.loads(raw)
        except ClientError as e:
            logger.exception("Failed to read s3://%s/%s: %s", bucket, key, e)
            raise

    def _load_data(self):
        # Check if full S3 path provided
        if self.raw_data_path.startswith("s3://"):
            parts = self.raw_data_path.replace("s3://", "").split("/", 1)
            if len(parts) != 2:
                raise ValueError("Invalid S3 path format. Use s3://bucket/key")
            return self._read_json_from_s3(bucket=parts[0], key=parts[1])

        # Check if key provided + default bucket exists
        if self.default_bucket:
            return self._read_json_from_s3(bucket=self.default_bucket, key=self.raw_data_path)

        raise ValueError(
            "MinIO extractor requires a full 's3://' path OR 'MINIO_BUCKET_NAME' env var set."
        )

    def extract(self) -> pl.LazyFrame:
        logger.info(f"Loading data from {self.raw_data_path}")
        data = self._load_data()

        rows = []
        # Parsing logic giữ nguyên vì đúng với cấu trúc Data của bạn
        for date_str, sensors in data.items():
            for sensor_id, files in sensors.items():
                for filename, details in files.items():
                    timestamp_ms = self._extract_timestamp_from_filename(filename)
                    if timestamp_ms is not None:
                        rows.append({
                            "timestamp": timestamp_ms,
                            "sensor_id": int(sensor_id),
                            "count": int(details.get("count", 0)),
                        })

        logger.info(f"Extracted {len(rows)} records.")
        
        return (
            pl.DataFrame(rows)
            .lazy()
            .with_columns(
                pl.from_epoch(pl.col("timestamp"), time_unit="ms")
                .alias("timestamp")
            )
            # Nếu server chạy khác múi giờ VN, hãy kiểm tra lại logic này.
            .with_columns(
                (pl.col("timestamp") + pl.duration(hours=7)).alias("timestamp")
            )
        )

    def _extract_timestamp_from_filename(self, filename: str) -> Optional[int]:
        try:
            return int(filename.split("_")[-1].replace('.txt', ''))
        except (IndexError, ValueError):
            return None

# Test script
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