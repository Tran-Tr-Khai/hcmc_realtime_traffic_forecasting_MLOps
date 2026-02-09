"""
Real-time Traffic Consumer - Main Entry Point
==============================================

This script starts the real-time traffic data consumer with
Kafka and Redis integration.

Usage:
    python scripts/run_consumer.py

Environment Variables:
    KAFKA_BROKER          - Kafka bootstrap servers (default: kafka:29092)
    KAFKA_TOPIC           - Kafka topic name (default: traffic-realtime)
    KAFKA_GROUP_ID        - Consumer group ID (default: traffic-consumer-group)
    REDIS_HOST            - Redis hostname (default: redis)
    REDIS_PORT            - Redis port (default: 6379)
    REDIS_DB              - Redis database number (default: 0)
    WINDOW_SIZE           - Sliding window size (default: 12)
    BUFFER_INTERVAL_SEC   - Buffer flush interval in seconds (default: 300)

Environment Loading Priority:
    1. System environment variables (highest priority)
    2. Variables from .env file in project root
    3. Default values (lowest priority)
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Load environment variables from .env file (if exists)
# override=False ensures system env vars take precedence over .env
env_file = os.path.join(project_root, '.env')
if os.path.exists(env_file):
    load_dotenv(dotenv_path=env_file, override=False)
    print(f"✓ Loaded environment variables from: {env_file}")
else:
    print(f"ℹ No .env file found at: {env_file}")
    print("  Using system environment variables and defaults")

from src.online.consumer import create_consumer_from_env

# Configure logging
# Ensure logs directory exists
log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'consumer.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print startup banner with configuration."""
    print("\n Configuration:")
    print(f"   Kafka Broker:      {os.getenv('KAFKA_BROKER', 'kafka:29092')}")
    print(f"   Kafka Topic:       {os.getenv('KAFKA_TOPIC', 'traffic-realtime')}")
    print(f"   Kafka Group ID:    {os.getenv('KAFKA_GROUP_ID', 'traffic-consumer-group')}")
    print(f"   Redis Host:        {os.getenv('REDIS_HOST', 'redis')}")
    print(f"   Redis Port:        {os.getenv('REDIS_PORT', '6379')}")
    print(f"   Window Size:       {os.getenv('WINDOW_SIZE', '12')} snapshots")
    print(f"   Buffer Interval:   {os.getenv('BUFFER_INTERVAL_SEC', '300')} seconds")
    print()


def validate_environment():
    """Validate required environment variables and connections."""
    logger.info("Validating environment...")
    
    # Check critical environment variables
    required_vars = ['KAFKA_BROKER', 'REDIS_HOST']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(
            f"Missing environment variables (using defaults): {missing_vars}"
        )
    
    logger.info("Environment validation passed")


def main():
    """
    Main entry point for the consumer application.
    """
    try:
        # Print startup banner
        print_banner()
        
        # Validate environment
        validate_environment()
        
        # Create consumer from environment variables
        logger.info("Initializing consumer...")
        consumer = create_consumer_from_env()
        
        # Start consuming
        logger.info("Starting consumer...")
        consumer.run()
        
    except KeyboardInterrupt:
        logger.info("\nReceived keyboard interrupt - shutting down...")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
