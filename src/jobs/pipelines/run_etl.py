from io import BytesIO
import logging
import resource
from time import perf_counter
from typing import Any

import polars as pl

from config.settings import OfflinePipelineConfig, settings
from src.infrastructure.storage.s3_client import S3Client
from src.processing.offline.extract import extract_traffic_dataframe
from src.processing.offline.resample import resample_traffic
from src.processing.offline.topology import load_graph_topology
from src.processing.offline.transform import transform_traffic_data


PARQUET_CONTENT_TYPE = "application/vnd.apache.parquet"
logger = logging.getLogger(__name__)


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def dataframe_to_parquet_buffer(dataframe: pl.DataFrame) -> BytesIO:
    buffer = BytesIO()
    dataframe.write_parquet(buffer)
    buffer.seek(0)
    return buffer


def format_metric(name: str, value: Any) -> str:
    if name == "elapsed":
        return f"{value:.2f}s"
    if name in {"peak_rss", "parquet_size"}:
        return f"{value:.1f}MB"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def log_event(event: str, **metrics: Any) -> None:
    if not metrics:
        logger.info(event)
        return

    detail = ", ".join(
        f"{name}={format_metric(name, value)}" for name, value in metrics.items()
    )
    logger.info("%s: %s", event, detail)


def dataframe_metrics(dataframe: pl.DataFrame) -> dict[str, int]:
    rows, columns = dataframe.shape
    return {"rows": rows, "cols": columns}


def run_etl(
    s3_client: S3Client | None = None,
    config: OfflinePipelineConfig | None = None,
) -> None:
    pipeline_config = config or settings.offline_pipeline
    storage = s3_client or S3Client()
    pipeline_started_at = perf_counter()

    log_event("starting pipeline", peak_rss=peak_rss_mb())

    started_at = perf_counter()
    log_event("loading graph", key=pipeline_config.graph_key)
    topology = load_graph_topology(storage.get_object_stream(pipeline_config.graph_key))
    log_event(
        "graph loaded",
        nodes=len(topology.node_ids),
        elapsed=perf_counter() - started_at,
        peak_rss=peak_rss_mb(),
    )

    started_at = perf_counter()
    log_event("loading raw traffic", key=pipeline_config.raw_history_key)
    raw_dataframe = extract_traffic_dataframe(storage.get_object_stream(pipeline_config.raw_history_key))
    log_event(
        "extract complete",
        **dataframe_metrics(raw_dataframe),
        elapsed=perf_counter() - started_at,
        peak_rss=peak_rss_mb(),
    )

    started_at = perf_counter()
    resampled_dataframe = resample_traffic(raw_dataframe)
    log_event(
        "resample complete",
        **dataframe_metrics(resampled_dataframe),
        elapsed=perf_counter() - started_at,
        peak_rss=peak_rss_mb(),
    )

    started_at = perf_counter()
    training_dataframe = transform_traffic_data(resampled_dataframe, topology)
    log_event(
        "transform complete",
        **dataframe_metrics(training_dataframe),
        elapsed=perf_counter() - started_at,
        peak_rss=peak_rss_mb(),
    )

    started_at = perf_counter()
    log_event("uploading parquet", key=pipeline_config.processed_history_key)
    parquet_buffer = dataframe_to_parquet_buffer(training_dataframe)
    storage.put_object(
        key=pipeline_config.processed_history_key,
        data=parquet_buffer,
        content_type=PARQUET_CONTENT_TYPE,
    )
    log_event(
        "upload complete",
        parquet_size=parquet_buffer.getbuffer().nbytes / (1024 * 1024),
        elapsed=perf_counter() - started_at,
        peak_rss=peak_rss_mb(),
    )

    log_event(
        "done",
        elapsed=perf_counter() - pipeline_started_at,
        peak_rss=peak_rss_mb(),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[run-etl] %(message)s")
    run_etl()
