from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT_DIR / ".env"


class S3Config(BaseModel):
    bucket_name: str
    region_name: str = "ap-southeast-1"
    access_key_id: str | None = None
    secret_access_key: str | None = None


class OfflinePipelineConfig(BaseModel):
    raw_history_key: str = "raw/hcmc-traffic-history.json"
    processed_history_key: str = "processed/train.parquet"


class Settings(BaseSettings):
    s3: S3Config
    offline_pipeline: OfflinePipelineConfig = OfflinePipelineConfig()

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )


settings = Settings()