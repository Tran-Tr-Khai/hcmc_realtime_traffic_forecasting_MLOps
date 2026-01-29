import argparse
import logging
import os

from src.core.ingestors.minio_ingestor import MinIOIngestor
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("logs/ingestion.log"), # Ghi vào file này
        logging.StreamHandler()                # Vẫn hiện ra màn hình
    ]
)

def main():
    parser = argparse.ArgumentParser(description="Ingest local raw data folder to MinIO/S3")
    parser.add_argument("--local-dir", default="data/raw", help="Local folder containing raw files")
    parser.add_argument("--prefix", default="", help="S3 prefix to write files under (optional)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(f"DEBUG: Type of MinIOIngestor is {type(MinIOIngestor)}")
    # Instantiate using env vars (or pass explicit values)
    ingestor = MinIOIngestor()
    ingestor.setup_bucket()

    local_dir = os.path.abspath(args.local_dir)
    if not os.path.isdir(local_dir):
        raise SystemExit(f"Local directory not found: {local_dir}")

    ingestor.ingest_folder(local_dir, s3_prefix=args.prefix)


if __name__ == "__main__":
    main()
