from datetime import time
import polars as pl

TIMESTAMP_MS_COLUMN = "timestamp_ms"
TIMESTAMP_COLUMN = "timestamp"
TIMESTAMP_BUCKET_COLUMN = "timestamp_bucket"
SENSOR_COLUMN = "sensor_id"
COUNT_COLUMN = "count"
DEFAULT_TIMEZONE = "Asia/Ho_Chi_Minh"
DEFAULT_INTERVAL = "5m"
DEFAULT_START_TIME = time(7, 30)
DEFAULT_END_TIME = time(22, 30)

REQUIRED_COLUMNS = {
    TIMESTAMP_MS_COLUMN,
    SENSOR_COLUMN,
    COUNT_COLUMN,
}


def validate_columns(df: pl.DataFrame, required_columns: set[str] = REQUIRED_COLUMNS) -> None:
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s): {missing}")


def add_local_timestamp(
    df: pl.DataFrame,
    timezone: str = DEFAULT_TIMEZONE,
) -> pl.DataFrame:
    return df.with_columns(
        pl.from_epoch(TIMESTAMP_MS_COLUMN, time_unit="ms")
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone(timezone)
        .alias(TIMESTAMP_COLUMN)
    )


def aggregate_by_interval(df: pl.DataFrame, interval: str) -> pl.DataFrame:
    return (
        df.with_columns(
            pl.col(TIMESTAMP_COLUMN).dt.truncate(interval).alias(TIMESTAMP_BUCKET_COLUMN)
        )
        .group_by([TIMESTAMP_BUCKET_COLUMN, SENSOR_COLUMN])
        .agg(pl.col(COUNT_COLUMN).mean().alias(COUNT_COLUMN))
    )


def filter_time_range(
    df: pl.DataFrame,
    start_time: time | None,
    end_time: time | None,
) -> pl.DataFrame:
    if start_time is None or end_time is None:
        return df

    return df.filter(
        (pl.col(TIMESTAMP_BUCKET_COLUMN).dt.time() >= start_time)
        & (pl.col(TIMESTAMP_BUCKET_COLUMN).dt.time() <= end_time)
    )


def sensor_sort_key(column_name: str) -> tuple[int, int | str]:
    """Sort numeric sensor names numerically, fall back to string sorting."""
    try:
        return (0, int(column_name))
    except ValueError:
        return (1, column_name)


def pivot_by_sensor(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df

    wide_df = df.pivot(
        on=SENSOR_COLUMN,
        index=TIMESTAMP_BUCKET_COLUMN,
        values=COUNT_COLUMN,
        aggregate_function="mean",
    ).rename({TIMESTAMP_BUCKET_COLUMN: TIMESTAMP_COLUMN})

    sensor_columns = sorted(
        (col for col in wide_df.columns if col != TIMESTAMP_COLUMN),
        key=sensor_sort_key,
    )
    return wide_df.sort(TIMESTAMP_COLUMN).select([TIMESTAMP_COLUMN, *sensor_columns])


def resample_traffic(
    df: pl.DataFrame,
    interval: str = DEFAULT_INTERVAL,
    timezone: str = DEFAULT_TIMEZONE,
    start_time: time | None = DEFAULT_START_TIME,
    end_time: time | None = DEFAULT_END_TIME,
) -> pl.DataFrame:
    """Convert long traffic data into a resampled wide time series."""
    validate_columns(df)

    if df.is_empty():
        return pl.DataFrame()

    return (
        df.pipe(add_local_timestamp, timezone=timezone)
        .pipe(aggregate_by_interval, interval=interval)
        .pipe(filter_time_range, start_time, end_time)
        .pipe(pivot_by_sensor)
    )