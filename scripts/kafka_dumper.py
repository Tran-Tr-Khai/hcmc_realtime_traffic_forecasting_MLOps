import os
import sys
import logging
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.online.dumper import KafkaDailyDumper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/kafka_dumper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Kafka Daily Dumper - Archive streaming data to daily files")
    parser.add_argument('--broker', default=os.getenv('KAFKA_BROKER', 'kafka:29092'), help='Kafka bootstrap servers (default: kafka:29092)')
    parser.add_argument('--topic', default=os.getenv('KAFKA_TOPIC', 'traffic-realtime'), help='Kafka topic to consume from (default: traffic-raw)')
    parser.add_argument('--group-id', default='dumper-recovery-group', help='Consumer group ID (default: kafka-dumper-group)')
    parser.add_argument('--output-dir', default='data/kafka_dumps', help='Directory to save daily dumps (default: data/kafka_dumps)')
    parser.add_argument(
        '--auto-offset-reset',
        choices=['earliest', 'latest'],
        default='earliest',
        help='Where to start consuming: earliest (from beginning) or latest (from now)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("KAFKA DAILY DUMPER - STARTING")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Broker: {args.broker}")
    logger.info(f"  Topic: {args.topic}")
    logger.info(f"  Group ID: {args.group_id}")
    logger.info(f"  Output Directory: {args.output_dir}")
    logger.info(f"  Auto Offset Reset: {args.auto_offset_reset}")
    logger.info("=" * 80)
    
    # Create and run dumper instance
    dumper = KafkaDailyDumper(
        bootstrap_servers=args.broker,
        topic=args.topic,
        group_id=args.group_id,
        output_dir=args.output_dir,
        auto_offset_reset=args.auto_offset_reset
    )
    
    dumper.run()


if __name__ == "__main__":
    main()
