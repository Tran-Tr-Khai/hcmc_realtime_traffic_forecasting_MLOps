import argparse
import logging
import os
import sys

# Thêm thư mục gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Tạo folder logs trước khi config logging
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("logs/offline_pipeline.log"),
        # logging.StreamHandler() # Bỏ comment nếu muốn in ra màn hình
    ],
    force=True
)
logger = logging.getLogger(__name__)

from src.offline.pipeline import OfflinePipeline

def main():
    print("Pipeline triggered! Check 'logs/offline_pipeline.log' for details.")
    
    parser = argparse.ArgumentParser(description="Run Offline Traffic Data Pipeline (MinIO -> MinIO)")
    
    # Input Args
    parser.add_argument("--data-key", type=str, default="raw/hcmc-traffic-data.json", help="Input Raw JSON key")
    parser.add_argument("--graph-key", type=str, default="raw/hcmc-clustered-graph.json", help="Input Graph Topology key")
    parser.add_argument("--bucket", type=str, help="MinIO bucket name (override env var)")
    
    # Output Args (SỬA ĐỔI: Dùng MinIO Key thay vì Local Path)
    parser.add_argument("--output-key", type=str, default="processed/traffic_clean.parquet", help="Output MinIO key (Parquet)")
    
    # Config Args
    parser.add_argument("--interval", type=str, default="5m", help="Resampling interval")
    
    args = parser.parse_args()
    
    try:
        # 1. Khởi tạo Pipeline
        pipeline = OfflinePipeline(
            raw_data_key=args.data_key,
            graph_key=args.graph_key,
            bucket_name=args.bucket,
            output_key=args.output_key, # <--- Truyền Output Key vào đây
            interval=args.interval
        )
        
        # 2. Chạy Pipeline (Việc lưu lên MinIO đã được xử lý bên trong hàm run)
        clean_df = pipeline.run()
        
        # 3. Kết thúc
        if not clean_df.is_empty():
            logger.info(f"JOB SUCCESSFUL. Data saved to MinIO bucket '{pipeline.bucket}' at key '{args.output_key}'")
            print(f"Job Done! Data saved to MinIO at: {args.output_key}")
        else:
            logger.warning("Job finished but returned empty data.")
            print("Job finished but returned empty data.")
            
    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()