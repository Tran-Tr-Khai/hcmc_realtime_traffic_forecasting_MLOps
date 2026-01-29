import os
import json
import time
import shutil
import pytest
import polars as pl
from unittest.mock import patch

# Import các class "xịn" của bạn
from src.core.ingestors.minio_ingestor import MinIOIngestor
from src.offline.extractors.minio_traffic_extractor import MinIoTrafficExtractor

# --- CẤU HÌNH MÔI TRƯỜNG TEST ---
TEST_ENV = {
    "MINIO_ENDPOINT_URL": "http://localhost:9000",
    "MINIO_ACCESS_KEY": "minioadmin",
    "MINIO_SECRET_KEY": "minioadmin",
    "MINIO_BUCKET_NAME": "traffic-bronze-test" # Dùng bucket riêng cho test
}

# Tạo thư mục tạm để chứa file giả
TEMP_DATA_DIR = "tests/temp_data"

@pytest.fixture(scope="module", autouse=True)
def setup_teardown_env():
    """
    Fixture này chạy 1 lần cho cả file test:
    1. Thiết lập biến môi trường.
    2. Tạo folder dữ liệu giả.
    3. Xóa folder sau khi test xong.
    """
    # 1. Setup Environment
    with patch.dict(os.environ, TEST_ENV):
        # 2. Tạo folder tạm và file JSON mẫu
        if os.path.exists(TEMP_DATA_DIR):
            shutil.rmtree(TEMP_DATA_DIR)
        os.makedirs(TEMP_DATA_DIR)

        # Cấu trúc JSON giả lập (phải khớp logic của Extractor)
        # Date -> SensorID -> Filename (có timestamp) -> Data
        mock_data = {
            "2024-01-01": {
                "12345": {
                    "sensor_1704067200000.txt": {"count": 50}, # Timestamp: 2024-01-01 00:00:00 UTC
                    "sensor_1704067500000.txt": {"count": 30}  # Timestamp: 2024-01-01 00:05:00 UTC
                }
            }
        }
        
        # Lưu thành file
        file_path = os.path.join(TEMP_DATA_DIR, "test-traffic.json")
        with open(file_path, "w") as f:
            json.dump(mock_data, f)
        
        yield # Chạy test ở đây
        
        # 3. Cleanup (Dọn dẹp sau khi test xong)
        if os.path.exists(TEMP_DATA_DIR):
            shutil.rmtree(TEMP_DATA_DIR)

def test_full_pipeline_ingest_and_extract():
    """
    Kịch bản Integration Test:
    Bước 1: Ingestor upload file từ máy lên MinIO.
    Bước 2: Extractor đọc từ MinIO về xử lý.
    """
    print("\n🚀 [INTEGRATION] Starting End-to-End Test...")

    # --- BƯỚC 1: INGESTION ---
    print("1️⃣ Testing Ingestion...")
    # Cần patch biến môi trường vì class MinIOIngestor load .env ngay khi init
    with patch.dict(os.environ, TEST_ENV):
        ingestor = MinIOIngestor()
        ingestor.setup_bucket() # Tạo bucket thật trên MinIO Container
        ingestor.ingest_folder(TEMP_DATA_DIR)
    
    print("✅ Ingestion finished. Data uploaded to MinIO.")

    # --- BƯỚC 2: EXTRACTION ---
    print("2️⃣ Testing Extraction...")
    file_key = "test-traffic.json" # Do ingestor giữ nguyên tên file
    
    with patch.dict(os.environ, TEST_ENV):
        # Lưu ý: Class của bạn xử lý logic path hơi đặc thù, ta truyền đúng key
        extractor = MinIoTrafficExtractor(file_key)
        
        # Thực hiện Extract
        df = extractor.extract().collect()

    # --- BƯỚC 3: VERIFICATION ---
    print("3️⃣ Verifying Data...")
    print(df)

    # Kiểm tra số lượng dòng (có 2 file con trong json -> 2 dòng)
    assert df.height == 2, f"Expected 2 rows, got {df.height}"
    
    # Kiểm tra cột
    assert "timestamp" in df.columns
    assert "count" in df.columns
    
    # Kiểm tra giá trị tổng (50 + 30 = 80)
    total_count = df["count"].sum()
    assert total_count == 80, f"Expected total count 80, got {total_count}"
    
    print("✅ Integration Test Passed! Full cycle verified.")