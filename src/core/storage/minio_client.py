import os
import logging
import mimetypes  # 1. Move to top
from typing import Optional, Any, IO

import boto3
from botocore.exceptions import ClientError
from botocore.config import Config  # 2. Import Config
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class MinIOClient:
    """
    Wrapper class for MinIO interactions using boto3.
    Handles connection, bucket management, and streaming.
    """

    def __init__(self, bucket_name: Optional[str] = None):
        self.endpoint = os.getenv("MINIO_ENDPOINT_URL")
        self.access_key = os.getenv("MINIO_ACCESS_KEY")
        self.secret_key = os.getenv("MINIO_SECRET_KEY")
        
        self.bucket_name = bucket_name or os.getenv("MINIO_BUCKET_NAME")

        if not all([self.endpoint, self.access_key, self.secret_key]):
            raise ValueError(
                "Missing MinIO credentials. Check your .env file."
            )

        
        my_config = Config(
            retries={
                'max_attempts': 3,
                'mode': 'standard'
            },
            connect_timeout=10,  # Timeout kết nối (giây)
            read_timeout=30      # Timeout đọc dữ liệu (giây)
        )

        try:
            self.client = boto3.client(
                "s3",
                endpoint_url=self.endpoint,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                config=my_config  # Apply config
            )
            logger.info(f"MinIO client initialized at {self.endpoint}")
        except Exception as e:
            logger.critical(f"Failed to initialize MinIO client: {e}")
            raise

    def ensure_bucket(self, bucket: Optional[str] = None) -> None:
        """
        Check if bucket exists, create if not (Idempotent).
        """
        target_bucket = bucket or self.bucket_name
        if not target_bucket:
            raise ValueError("No bucket name provided.")

        try:
            self.client.head_bucket(Bucket=target_bucket)
            # logger.debug(f"Bucket '{target_bucket}' exists.") # Debug đỡ rác log
        except ClientError:
            logger.warning(f"Bucket '{target_bucket}' not found. Creating...")
            try:
                self.client.create_bucket(Bucket=target_bucket)
                logger.info(f"Created bucket '{target_bucket}'")
            except ClientError as e:
                logger.error(f"Failed to create bucket '{target_bucket}': {e}")
                raise

    def get_object_stream(self, key: str, bucket: Optional[str] = None) -> Any:
        target_bucket = bucket or self.bucket_name
        try:
            response = self.client.get_object(Bucket=target_bucket, Key=key)
            return response["Body"]
        except ClientError as e:
            logger.error(f"Failed to stream object s3://{target_bucket}/{key}: {e}")
            raise

    def upload_file(self, local_path: str, key: str, bucket: Optional[str] = None, 
                    content_type: Optional[str] = None) -> None:
        """
        Upload a local file to MinIO with auto MIME-type detection.
        """
        target_bucket = bucket or self.bucket_name
        
        if content_type is None:
            # guess_type trả về (type, encoding), ta chỉ cần type
            guessed_type, _ = mimetypes.guess_type(local_path)
            content_type = guessed_type or "application/octet-stream"
        
        try:
            logger.info(f"Uploading {os.path.basename(local_path)} -> s3://{target_bucket}/{key}")
            self.client.upload_file(
                Filename=local_path,
                Bucket=target_bucket,
                Key=key,
                ExtraArgs={'ContentType': content_type}
            )
        except ClientError as e:
            logger.error(f"Failed to upload {local_path}: {e}")
            raise

    def object_exists(self, key: str, bucket: Optional[str] = None) -> bool:
        """
        Check if an object exists in the bucket (Lightweight check).
        """
        target_bucket = bucket or self.bucket_name
        try:
            self.client.head_object(Bucket=target_bucket, Key=key)
            return True
        except ClientError:
            return False