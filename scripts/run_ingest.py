import sys
import os
import argparse
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.offline.ingestors import LocalToMinIOIngestor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("logs/ingestion.log"),
        logging.StreamHandler()
    ]
)


def main():
    parser = argparse.ArgumentParser(description="Ingest local raw data folder to MinIO")
    parser.add_argument("--local-dir", default="data/raw", help="Local folder containing raw files")
    parser.add_argument("--prefix", default="raw", help="S3 prefix to write files under (optional)")
    args = parser.parse_args()

    # Instantiate ingestor (uses env vars for credentials)
    ingestor = LocalToMinIOIngestor()
    ingestor.ensure_bucket()
    
    # Ingest folder
    ingestor.ingest_folder(args.local_dir, s3_prefix=args.prefix)


if __name__ == "__main__":
    main()
