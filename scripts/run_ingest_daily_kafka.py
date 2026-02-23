import sys
import os
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.offline.ingestors import LocalToMinIOIngestor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/kafka_ingestion.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_yesterday_date() -> str:
    """
    Get yesterday's date in YYYY-MM-DD format.
    
    Returns:
        Date string (e.g., "2026-02-22")
    """
    yesterday = datetime.now() - timedelta(days=1)
    return yesterday.strftime("%Y-%m-%d")


def find_dump_files_by_date(dump_dir: str, target_date: str) -> List[Path]:
    """
    Find Kafka dump files matching the target date.
    
    Expected filename pattern: kafka-dump-YYYY-MM-DD.json
    
    Args:
        dump_dir: Directory containing dump files
        target_date: Date to filter (YYYY-MM-DD)
    
    Returns:
        List of file paths matching the date
    """
    dump_path = Path(dump_dir)
    
    if not dump_path.exists():
        logger.error(f"Dump directory not found: {dump_dir}")
        return []
    
    # Find files matching the date pattern
    pattern = f"kafka-dump-{target_date}.json"
    matching_files = list(dump_path.glob(pattern))
    
    # Also check for any files containing the date
    if not matching_files:
        # Fallback: Find any files containing the date
        for file in dump_path.glob("*.json"):
            if target_date in file.name:
                matching_files.append(file)
                logger.info(f"Found file with date pattern: {file.name}")
    
    if matching_files:
        logger.info(f"Found {len(matching_files)} dump file(s) for {target_date}")
        for file in matching_files:
            file_size = file.stat().st_size / 1024  # KB
            logger.info(f"  - {file.name} ({file_size:.2f} KB)")
    else:
        logger.warning(f"No Kafka dump files found for date: {target_date}")
    
    return matching_files


def validate_dump_file(file_path: Path) -> bool:
    """
    Validate dump file integrity.
    
    Args:
        file_path: Path to dump file
    
    Returns:
        True if file is valid, False otherwise
    """
    try:
        # Check file exists
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        # Check file is not empty
        if file_path.stat().st_size == 0:
            logger.error(f"File is empty: {file_path}")
            return False
        
        # Try to read first line (JSONL format)
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line:
                json.loads(first_line)  # Validate JSON format
        
        logger.debug(f"File validated: {file_path.name}")
        return True
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error validating file {file_path}: {e}")
        return False
    
def ingest_daily_kafka_dumps(
    dump_dir: str = "data/kafka_dumps",
    target_date: Optional[str] = None,
    mode: str = "daily"
):
    """
    Ingest Kafka dump files to MinIO.
    
    Args:
        dump_dir: Directory containing dump files
        target_date: Specific date to ingest (YYYY-MM-DD), None for yesterday
        mode: 'daily' (one date) or 'all' (all files)
    """
    logger.info("=" * 80)
    logger.info("KAFKA DUMPS INGESTION TO MINIO")
    logger.info("=" * 80)
    logger.info(f"Mode: {mode}")
    logger.info(f"Dump directory: {dump_dir}")
    
    # Initialize MinIO ingestor
    try:
        ingestor = LocalToMinIOIngestor()
        ingestor.ensure_bucket()
        logger.info(f"MinIO client initialized")
        logger.info(f"Endpoint: {os.getenv('MINIO_ENDPOINT_URL', 'N/A')}")
        logger.info(f"Bucket: {os.getenv('MINIO_BUCKET_NAME', 'N/A')}")
    except Exception as e:
        logger.error(f"Failed to initialize MinIO client: {e}")
        sys.exit(1)
    
    # Determine files to upload
    files_to_upload: List[Path] = []
    
    if mode == "all":
        # Upload all dump files
        logger.info("Scanning for all Kafka dump files...")
        dump_path = Path(dump_dir)
        if dump_path.exists():
            files_to_upload = list(dump_path.glob("kafka-dump-*.json"))
            logger.info(f"Found {len(files_to_upload)} total dump files")
        else:
            logger.error(f"Dump directory not found: {dump_dir}")
            sys.exit(1)
    
    else:  # mode == "daily"
        # Get target date
        if target_date is None:
            target_date = get_yesterday_date()
            logger.info(f"Target date (yesterday): {target_date}")
        else:
            logger.info(f"Target date (specified): {target_date}")
        
        # Find files for target date
        files_to_upload = find_dump_files_by_date(dump_dir, target_date)
        
        if not files_to_upload:
            logger.error(
                f"  CRITICAL: No Kafka dump files found for {target_date}\n"
                f"  This usually means:\n"
                f"   1. kafka_dumper.py service is not running\n"
                f"   2. Kafka producer is not sending data\n"
                f"   3. Dump directory path is incorrect\n"
                f"\n"
                f"   Expected file: {dump_dir}/kafka-dump-{target_date}.json"
            )
            sys.exit(1)  # Fail the Airflow task to trigger alert
    
    # Validate files
    logger.info("Validating dump files...")
    valid_files = [f for f in files_to_upload if validate_dump_file(f)]
    
    if not valid_files:
        logger.error("No valid dump files to upload")
        sys.exit(1)
    
    logger.info(f"✓ {len(valid_files)} valid file(s) ready for upload")
    
    # Upload files to MinIO
    logger.info("=" * 80)
    logger.info("UPLOADING TO MINIO")
    logger.info("=" * 80)
    
    upload_count = 0
    error_count = 0
    total_size = 0
    
    for file_path in valid_files:
        try:
            # Extract date from filename (kafka-dump-YYYY-MM-DD.json)
            file_date = None
            if "kafka-dump-" in file_path.name:
                # Extract date part
                date_part = file_path.name.replace("kafka-dump-", "").replace(".json", "")
                # Validate date format
                try:
                    datetime.strptime(date_part, "%Y-%m-%d")
                    file_date = date_part
                except ValueError:
                    logger.warning(f"Could not extract date from filename: {file_path.name}")
            
            # Construct S3 path with date partitioning
            # Structure: raw/kafka-dumps/YYYY-MM-DD/kafka-dump-YYYY-MM-DD.json
            if file_date:
                s3_prefix = f"raw/kafka-dumps/{file_date}"
            else:
                s3_prefix = "raw/kafka-dumps/unknown"
            
            # Upload file
            logger.info(f"Uploading: {file_path.name} -> {s3_prefix}/")
            
            # Use MinIO client to upload
            s3_key = f"{s3_prefix}/{file_path.name}"
            ingestor.client.upload_file(
                str(file_path),
                ingestor.bucket_name,
                s3_key
            )
            
            file_size = file_path.stat().st_size
            total_size += file_size
            upload_count += 1
            
            logger.info(
                f"Uploaded: {file_path.name} "
                f"({file_size / 1024:.2f} KB) -> s3://{ingestor.bucket_name}/{s3_key}"
            )
            
        except Exception as e:
            error_count += 1
            logger.error(f"✗ Failed to upload {file_path.name}: {e}")
    
    # Summary
    logger.info("=" * 80)
    logger.info("INGESTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total files processed: {len(valid_files)}")
    logger.info(f"Successfully uploaded: {upload_count}")
    logger.info(f"Upload errors: {error_count}")
    logger.info(f"Total data uploaded: {total_size / 1024 / 1024:.2f} MB")
    logger.info("=" * 80)
    
    # Fail if no files were uploaded
    if upload_count == 0:
        logger.error("CRITICAL: No files were uploaded to MinIO")
        sys.exit(1)
    
    # Fail if there were errors
    if error_count > 0:
        logger.error(f"WARNING: {error_count} files failed to upload")
        # Don't exit with error if at least some files uploaded
        # sys.exit(1)
    
    logger.info("Daily Kafka dumps ingestion completed successfully")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Ingest daily Kafka dumps to MinIO for Continuous Training")
    parser.add_argument('--dump-dir', default='data/kafka_dumps', help='Directory containing Kafka dump files (default: data/kafka_dumps)')
    parser.add_argument('--date', default=None, help='Specific date to ingest (YYYY-MM-DD). If not specified, uses yesterday.')
    
    parser.add_argument('--mode', choices=['daily', 'all'], default='daily',
        help=(
            "'daily' uploads only files for target date (default), "
            "'all' uploads all dump files (backfill)"
        )
    )
    
    args = parser.parse_args()
    
    # Run ingestion
    ingest_daily_kafka_dumps(
        dump_dir=args.dump_dir,
        target_date=args.date,
        mode=args.mode
    )


if __name__ == "__main__":
    main()
