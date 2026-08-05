from io import BytesIO

from config.settings import OfflinePipelineConfig, settings
from src.infrastructure.storage.s3_client import S3Client
from src.pipelines.offline.extract import extract_traffic_dataframe
from src.pipelines.offline.transform import resample_traffic


PARQUET_CONTENT_TYPE = "application/vnd.apache.parquet"


def dataframe_to_parquet_buffer(dataframe) -> BytesIO:
    buffer = BytesIO()
    dataframe.write_parquet(buffer)
    buffer.seek(0)
    return buffer


def run_etl(
    s3_client: S3Client | None = None,
    config: OfflinePipelineConfig | None = None,
) -> None:
    pipeline_config = config or settings.offline_pipeline
    storage = s3_client or S3Client()

    raw_stream = storage.get_object_stream(pipeline_config.raw_history_key)
    raw_dataframe = extract_traffic_dataframe(raw_stream)
    processed_dataframe = resample_traffic(raw_dataframe)

    parquet_buffer = dataframe_to_parquet_buffer(processed_dataframe)
    storage.put_object(
        key=pipeline_config.processed_history_key,
        data=parquet_buffer,
        content_type=PARQUET_CONTENT_TYPE,
    )


def main() -> None:
    run_etl()