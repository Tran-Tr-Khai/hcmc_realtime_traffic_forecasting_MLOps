import os
import logging
from pathlib import Path

from src.core.storage import MinIOClient

logger = logging.getLogger(__name__)


class LocalToMinIOIngestor(MinIOClient):
    """
    Ingests local files/folders to MinIO.
    """
    def ingest_folder(self, local_folder_path: str, s3_prefix: str = ""):
        """
        Recursively upload all files from a local folder to MinIO.
        
        Args:
            local_folder_path: Path to local folder
            s3_prefix: Optional prefix inside the bucket
        """
        local_folder_path = os.path.abspath(local_folder_path)
        if not os.path.isdir(local_folder_path):
            raise ValueError(f"Local folder not found: {local_folder_path}")
        
        # Normalize prefix
        s3_prefix = (s3_prefix or "").strip("/")
        
        uploaded_count = 0
        for root, _dirs, files in os.walk(local_folder_path):
            for fname in files:
                local_path = os.path.join(root, fname)
                rel_path = os.path.relpath(local_path, local_folder_path)
                # Use forward slashes for S3 keys
                rel_posix = rel_path.replace(os.sep, "/")
                
                if s3_prefix:
                    key = f"{s3_prefix}/{rel_posix}"
                else:
                    key = rel_posix
                
                self.upload_file(local_path, key)
                uploaded_count += 1
        
        logger.info(f"Ingestion complete: {uploaded_count} files uploaded to s3://{self.bucket_name}/")
