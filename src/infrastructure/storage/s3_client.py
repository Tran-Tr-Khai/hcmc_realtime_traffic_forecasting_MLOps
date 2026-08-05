from typing import BinaryIO

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from config.settings import S3Config, settings


class S3Client:
    def __init__(self, config: S3Config | None = None):
        self.config = config or settings.s3
        self.bucket_name = self.config.bucket_name
        self.client = boto3.client(
            "s3",
            region_name=self.config.region_name,
            aws_access_key_id=self.config.access_key_id,
            aws_secret_access_key=self.config.secret_access_key,
            config=Config(
                retries={"max_attempts": 3, "mode": "standard"},
                connect_timeout=10,
                read_timeout=60,
            ),
        )

    def object_exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError as error:
            error_code = error.response.get("Error", {}).get("Code")
            if error_code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def get_object_stream(self, key: str) -> BinaryIO:
        response = self.client.get_object(Bucket=self.bucket_name, Key=key)
        return response["Body"]

    def put_object(
        self,
        key: str,
        data: bytes | BinaryIO,
        content_type: str = "application/octet-stream",
    ) -> None:
        self.client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=data,
            ContentType=content_type,
        )