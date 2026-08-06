from pathlib import Path
from tempfile import NamedTemporaryFile

import duckdb

from config.settings import settings
from src.infrastructure.storage.s3_client import S3Client


SQL_PATH = Path(__file__).with_name("sql") / "train_gaps.sql"
EXPECTED_INTERVAL_MINUTES = 5


def download_s3_object_to_tempfile(s3_client: S3Client, key: str) -> Path:
    stream = s3_client.get_object_stream(key)

    with NamedTemporaryFile(delete=False, suffix=".parquet") as file:
        while chunk := stream.read(1024 * 1024):
            file.write(chunk)
        return Path(file.name)


def load_gaps_sql(parquet_path: Path) -> str:
    return SQL_PATH.read_text(encoding="utf-8").format(
        parquet_path=parquet_path.as_posix(),
        expected_interval_minutes=EXPECTED_INTERVAL_MINUTES,
    )


def print_gap_rows(rows: list[tuple]) -> None:
    if not rows:
        print("No train timeline gaps found")
        return

    print("Train timeline gaps")
    for previous_timestamp, current_timestamp, gap_minutes in rows:
        print(f"- {previous_timestamp} -> {current_timestamp} ({gap_minutes} minutes)")


def show_train_gaps(s3_client: S3Client | None = None) -> None:
    storage = s3_client or S3Client()
    parquet_path = download_s3_object_to_tempfile(
        storage,
        settings.offline_pipeline.processed_history_key,
    )

    try:
        with duckdb.connect() as connection:
            rows = connection.execute(load_gaps_sql(parquet_path)).fetchall()
        print_gap_rows(rows)
    finally:
        parquet_path.unlink(missing_ok=True)


def main() -> None:
    show_train_gaps()