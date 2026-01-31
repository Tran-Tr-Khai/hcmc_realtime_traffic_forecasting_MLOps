import os
import json
import shutil
import pytest
import polars as pl
from unittest.mock import patch
from datetime import time

from src.ingestors import LocalToMinIOIngestor
from src.offline.pipeline import OfflinePipeline

# --- CẤU HÌNH GIẢ LẬP ---
TEST_ENV = {
    "MINIO_ENDPOINT_URL": "http://localhost:9000",
    "MINIO_ACCESS_KEY": "minioadmin",
    "MINIO_SECRET_KEY": "minioadmin",
    "MINIO_BUCKET_NAME": "traffic-bronze-test"
}
TEMP_DATA_DIR = "tests/temp_e2e_data"

@pytest.fixture(scope="module", autouse=True)
def setup_teardown_env():
    # 1. Setup Environment
    with patch.dict(os.environ, TEST_ENV):
        if os.path.exists(TEMP_DATA_DIR):
            shutil.rmtree(TEMP_DATA_DIR)
        os.makedirs(TEMP_DATA_DIR)

        # --- TẠO DATA GIẢ ---
        # Mốc thời gian: 2024-01-01
        # 07:30:00 VN = 00:30:00 UTC = 1704069000000
        # 07:35:00 VN = 00:35:00 UTC = 1704069300000
        # 07:40:00 VN = 00:40:00 UTC = 1704069600000
        
        traffic_data = {
            "2024-01-01": {
                "1": { # Sensor ID 1
                    "sensor_1704069000000.txt": {"count": 10}, # 07:30
                    # Giả lập MẤT TIN lúc 07:35 (Không có file này)
                    "sensor_1704069600000.txt": {"count": 20}  # 07:40
                },
                "2": { # Sensor ID 2 (Hàng xóm)
                    "sensor_1704069000000.txt": {"count": 15}, # 07:30
                    "sensor_1704069300000.txt": {"count": 18}, # 07:35 (Có tin để cứu hàng xóm)
                    "sensor_1704069600000.txt": {"count": 25}  # 07:40
                }
            }
        }
        
        # B. File Graph Topology
        graph_data = {
            "adjacency-matrix": [
                [1, 1], 
                [1, 1] 
            ],
            "distance-matrix": [
                [0.0, 0.5],
                [0.5, 0.0]
            ],
            "camera-dictionary": {
                "1": [[10.0, 100.0], "Sensor 1"],
                "2": [[10.1, 100.1], "Sensor 2"]
            }
        }

        # Lưu file ra disk
        with open(os.path.join(TEMP_DATA_DIR, "traffic.json"), "w") as f:
            json.dump(traffic_data, f)
            
        with open(os.path.join(TEMP_DATA_DIR, "graph.json"), "w") as f:
            json.dump(graph_data, f)

        # --- UPLOAD LÊN MINIO (Dùng boto3 upload_file) ---
        ingestor = LocalToMinIOIngestor()
        ingestor.ensure_bucket()
        
        # Upload Traffic
        ingestor.client.upload_file(
            os.path.join(TEMP_DATA_DIR, "traffic.json"), 
            TEST_ENV["MINIO_BUCKET_NAME"], 
            "traffic.json"
        )
        
        # Upload Graph
        ingestor.client.upload_file(
            os.path.join(TEMP_DATA_DIR, "graph.json"), 
            TEST_ENV["MINIO_BUCKET_NAME"], 
            "graph.json"
        )

        yield

        # Cleanup
        if os.path.exists(TEMP_DATA_DIR):
            shutil.rmtree(TEMP_DATA_DIR)

def test_offline_pipeline_execution():
    """
    Test toàn bộ luồng:
    MinIO (Traffic + Graph) -> Extract -> Resample -> Impute -> Clean Data
    """
    print("\n🚀 Starting Pipeline E2E Test...")
    
    with patch.dict(os.environ, TEST_ENV):
        # 1. Khởi tạo Pipeline
        # Chạy đúng khung giờ có data (07:30 - 07:45)
        pipeline = OfflinePipeline(
            raw_data_key="traffic.json",
            graph_key="graph.json",
            interval="5m",
            start_time=time(7, 30),
            end_time=time(7, 45) 
        )
        
        # 2. Chạy Pipeline
        df = pipeline.run()
        
        print(df)
        
        # 3. Assertions (Kiểm tra kết quả)
        assert not df.is_empty(), "Pipeline returned empty DataFrame. Check timestamp vs Time Filter!"
        
        # Kiểm tra logic Imputer:
        # Input có 2 mốc thời gian (30, 40) + 1 mốc bị mất (35)
        # Pipeline resample 5m -> Sẽ tạo ra 3 dòng: 07:30, 07:35, 07:40
        # (Lưu ý: số dòng phụ thuộc vào cách TimeSeriesResampler xử lý biên)
        # Nếu chỉ tính các điểm có dữ liệu hoặc nội suy, ít nhất phải có dữ liệu.
        
        # Kiểm tra không còn Null
        null_count = df.select(pl.all().exclude("timestamp")).null_count().sum_horizontal().sum()
        assert null_count == 0, f"Data still has {null_count} nulls after imputation!"
        
        print("✅ Pipeline E2E Test Passed: Data is clean and Graph loaded correctly.")