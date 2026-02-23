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
    
    # Output Args
    parser.add_argument("--output-key", type=str, default="processed/traffic_clean.parquet", help="Output MinIO key (Parquet)")
    
    # Kafka Dumps Args (NEW)
    parser.add_argument("--kafka-dumps-prefix", type=str, default="raw/kafka-dumps", 
                        help="MinIO prefix for Kafka dump files")
    parser.add_argument("--window-days", type=int, default=30, 
                        help="Rolling window size (days) for historical data. Use 0 to disable windowing.")
    
    # Config Args
    parser.add_argument("--interval", type=str, default="5m", help="Resampling interval")
    
    args = parser.parse_args()
    
    try:
        # 1. Initialize Pipeline with "Concat Raw First" strategy
        logger.info("="*80)
        logger.info("OFFLINE PIPELINE - CONCAT RAW DATA FIRST STRATEGY")
        logger.info("="*80)
        logger.info(f"Windowing: {args.window_days} days" if args.window_days > 0 else "Windowing: DISABLED")
        logger.info(f"Kafka dumps prefix: {args.kafka_dumps_prefix}")
        logger.info("="*80)
        
        pipeline = OfflinePipeline(
            raw_data_key=args.data_key,
            graph_key=args.graph_key,
            bucket_name=args.bucket,
            kafka_dumps_prefix=args.kafka_dumps_prefix,
            output_key=args.output_key,
            window_days=args.window_days if args.window_days > 0 else None,
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