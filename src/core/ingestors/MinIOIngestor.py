import os
import logging

from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError


load_dotenv()

class MinIOIngestor:
    def __init__(self) -> None:
        self.endpoint = os.getenv("MINIO_ENDPOINT_URL")
        self.access_key = os.getenv("MINIO_ACCESS_KEY")
        self.secret_key = os.getenv("MINIO_SECRET_KEY")
        self.bucket_name = os.getenv("MINIO_BUCKET_NAME")

        if not all([self.endpoint, self.access_key, self.secret_key, self.bucket_name]):
            raise ValueError(
                "Missing MinIO configuration. Ensure MINIO_ENDPOINT, ACCESS_KEY, SECRET_KEY and BUCKET_NAME are set."
            )

        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )

        self.logger = logging.getLogger(self.__class__.__name__)

    def setup_bucket(self) -> None:
        """Check whether the bucket exists; create it if missing."""
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
            self.logger.info("Bucket '%s' exists.", self.bucket_name)
        except ClientError as exc:
            error_code = None
            try:
                error_code = exc.response.get("Error", {}).get("Code")
            except Exception:
                pass

            self.logger.info("Bucket '%s' not found or inaccessible (%s). Creating...", self.bucket_name, error_code)
            try:
                # For MinIO the CreateBucketConfiguration / LocationConstraint is typically not required.
                self.client.create_bucket(Bucket=self.bucket_name)
                self.logger.info("Created bucket '%s'.", self.bucket_name)
            except ClientError:
                self.logger.exception("Failed to create bucket '%s'.", self.bucket_name)
                raise

    def upload_file(self, local_path: str, s3_path: str) -> None:
        """Upload a single file to the configured bucket.

        local_path: path to local file
        s3_path: key/path inside the bucket (use forward slashes)
        """
        self.logger.info("Uploading %s to s3://%s/%s", local_path, self.bucket_name, s3_path)
        try:
            self.client.upload_file(local_path, self.bucket_name, s3_path)
        except ClientError:
            self.logger.exception("Failed to upload %s to %s/%s", local_path, self.bucket_name, s3_path)
            raise

    def ingest_folder(self, local_folder_path: str, s3_prefix: str = "") -> None:
        """Recursively upload all files from a local folder, preserving relative paths.

        s3_prefix: optional prefix inside the bucket (no leading slash required)
        """
        local_folder_path = os.path.abspath(local_folder_path)
        if not os.path.isdir(local_folder_path):
            raise ValueError(f"Local folder not found: {local_folder_path}")

        # Normalize prefix (no leading/trailing slashes)
        s3_prefix = (s3_prefix or "").strip("/")

        for root, _dirs, files in os.walk(local_folder_path):
            for fname in files:
                local_path = os.path.join(root, fname)
                rel_path = os.path.relpath(local_path, local_folder_path)
                # Ensure forward slashes in S3 keys
                rel_posix = rel_path.replace(os.sep, "/")
                if s3_prefix:
                    key = f"{s3_prefix}/{rel_posix}"
                else:
                    key = rel_posix

                self.upload_file(local_path, key)
