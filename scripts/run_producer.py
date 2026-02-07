import sys 
import os
import argparse
import logging
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.online.producer import KafkaTrafficProducer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("logs/producer.log"),
        # logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    # 1. Dùng argparse để nhận tham số từ bên ngoài (Cực tiện cho Airflow/Docker)
    parser = argparse.ArgumentParser(description="Run Traffic Kafka Producer")
    parser.add_argument("--speed", type=float, default=20.0, help="Speed factor simulation")
    parser.add_argument("--file", type=str, default="data/raw/hcmc-traffic-data-realtime.json", help="Path to input file")
    parser.add_argument("--broker", type=str, default="localhost:9092", help="Kafka Broker URL")
    
    args = parser.parse_args()
    
    logger.info(f"Starting Producer with Speed: {args.speed}x")
    logger.info(f"Reading file: {args.file}")

    # 3. Gọi Class từ src ra chạy
    try:
        producer = KafkaTrafficProducer(
            bootstrap_servers=args.broker,
            topic_name='traffic-raw',
            input_file=args.file,
            speed_factor=args.speed
        )
        producer.stream_data()
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()