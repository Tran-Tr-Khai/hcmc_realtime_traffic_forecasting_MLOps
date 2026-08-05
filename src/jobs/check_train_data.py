from pathlib import Path
from tempfile import NamedTemporaryFile

import duckdb

from config.settings import settings
from src.infrastructure.storage.s3_client import S3Client


SQL_PATH = Path(__file__).with_name("sql") / "train_quality.sql"
EXPECTED_INTERVAL_MINUTES = 5
TIMESTAMP_COLUMN = "timestamp"


def download_s3_object_to_tempfile(s3_client: S3Client, key: str) -> Path:
    stream = s3_client.get_object_stream(key)

    with NamedTemporaryFile(delete=False, suffix=".parquet") as file:
        while chunk := stream.read(1024 * 1024):
            file.write(chunk)
        return Path(file.name)


def load_quality_sql(parquet_path: Path) -> str:
    return SQL_PATH.read_text(encoding="utf-8").format(
        parquet_path=parquet_path.as_posix(),
        expected_interval_minutes=EXPECTED_INTERVAL_MINUTES,
    )


def get_sensor_count(connection: duckdb.DuckDBPyConnection, parquet_path: Path) -> int:
    query = f"select * from read_parquet('{parquet_path.as_posix()}') limit 0"
    columns = [column[0] for column in connection.execute(query).description]
    return len([column for column in columns if column != TIMESTAMP_COLUMN])


def validate_report(report: dict) -> None:
    problems = []

    if report["row_count"] == 0:
        problems.append("train parquet has no rows")
    if report["sensor_count"] == 0:
        problems.append("train parquet has no sensor columns")
    if report["null_timestamps"] > 0:
        problems.append(f"timestamp has {report['null_timestamps']} null values")
    if report["duplicate_timestamps"] > 0:
        problems.append(f"found {report['duplicate_timestamps']} duplicated timestamps")

    if problems:
        raise ValueError("Data quality check failed: " + "; ".join(problems))


def print_report(report: dict) -> None:
    print("Train data quality report")
    print(f"- rows: {report['row_count']}")
    print(f"- sensor columns: {report['sensor_count']}")
    print(f"- time range: {report['min_timestamp']} -> {report['max_timestamp']}")
    print(f"- duplicate timestamps: {report['duplicate_timestamps']}")
    print(f"- timestamp nulls: {report['null_timestamps']}")
    print(f"- unexpected intervals: {report['unexpected_intervals']} (warning only)")
    print("- status: passed")


def check_train_data(s3_client: S3Client | None = None) -> None:
    storage = s3_client or S3Client()
    parquet_path = download_s3_object_to_tempfile(
        storage,
        settings.offline_pipeline.processed_history_key,
    )

    try:
        with duckdb.connect() as connection:
            result = connection.execute(load_quality_sql(parquet_path))
            values = result.fetchone()
            columns = [column[0] for column in result.description]
            report = dict(zip(columns, values, strict=True))
            report["sensor_count"] = get_sensor_count(connection, parquet_path)

        validate_report(report)
        print_report(report)
    finally:
        parquet_path.unlink(missing_ok=True)


def main() -> None:
    check_train_data()