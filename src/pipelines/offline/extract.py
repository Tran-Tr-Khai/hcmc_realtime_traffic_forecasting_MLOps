from collections.abc import Iterator, Mapping
from typing import Any, BinaryIO

import ijson
import polars as pl


TIMESTAMP_COLUMN = "timestamp_ms"
SENSOR_COLUMN = "sensor_id"
COUNT_COLUMN = "count"

TRAFFIC_SCHEMA = {
    TIMESTAMP_COLUMN: pl.Int64,
    SENSOR_COLUMN: pl.Int64,
    COUNT_COLUMN: pl.Int64,
}

TrafficRecord = tuple[int, int, int]


def extract_timestamp_from_filename(filename: str) -> int | None:
    """Extract millisecond Unix timestamp from '<hash>_<timestamp_ms>.txt'."""
    if not filename.endswith(".txt"):
        return None

    parts = filename.rsplit("_", maxsplit=1)
    if len(parts) != 2:
        return None

    timestamp_text = parts[1].removesuffix(".txt")

    try:
        return int(timestamp_text)
    except ValueError:
        return None


def parse_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def iter_traffic_records(stream: BinaryIO) -> Iterator[TrafficRecord]:
    """Yield traffic records from raw HCMC nested JSON."""
    for _date_key, sensors in ijson.kvitems(stream, ""):
        if not isinstance(sensors, Mapping):
            continue

        for sensor_id, files in sensors.items():
            parsed_sensor_id = parse_int(sensor_id)
            if parsed_sensor_id is None or not isinstance(files, Mapping):
                continue

            for filename, payload in files.items():
                if not isinstance(payload, Mapping):
                    continue

                timestamp_ms = extract_timestamp_from_filename(filename)
                count = parse_int(payload.get(COUNT_COLUMN))

                if timestamp_ms is None or count is None:
                    continue

                yield timestamp_ms, parsed_sensor_id, count


def build_traffic_dataframe(
    timestamps: list[int],
    sensors: list[int],
    counts: list[int],
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            TIMESTAMP_COLUMN: timestamps,
            SENSOR_COLUMN: sensors,
            COUNT_COLUMN: counts,
        },
        schema=TRAFFIC_SCHEMA,
    )


def extract_traffic_dataframe(
    stream: BinaryIO,
    chunk_size: int = 100_000,
) -> pl.DataFrame:
    """Parse raw JSON stream into long-format traffic data."""
    chunks: list[pl.DataFrame] = []
    timestamps: list[int] = []
    sensors: list[int] = []
    counts: list[int] = []

    for timestamp_ms, sensor_id, count in iter_traffic_records(stream):
        timestamps.append(timestamp_ms)
        sensors.append(sensor_id)
        counts.append(count)

        if len(timestamps) >= chunk_size:
            chunks.append(build_traffic_dataframe(timestamps, sensors, counts))
            timestamps, sensors, counts = [], [], []

    if timestamps:
        chunks.append(build_traffic_dataframe(timestamps, sensors, counts))

    if not chunks:
        return pl.DataFrame(schema=TRAFFIC_SCHEMA)

    return pl.concat(chunks)