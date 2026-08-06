import logging
from time import perf_counter

import polars as pl

from src.processing.offline.topology import GraphTopology


TIMESTAMP_COLUMN = "timestamp"
TEMP_COLUMNS = {"date", "weekday", "time"}
DEFAULT_TEMPORAL_GAP_LIMIT = 12
logger = logging.getLogger(__name__)


def sensor_columns(df: pl.DataFrame) -> list[str]:
    return [column for column in df.columns if column != TIMESTAMP_COLUMN and column not in TEMP_COLUMNS]


def add_time_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col(TIMESTAMP_COLUMN).dt.date().alias("date"),
        pl.col(TIMESTAMP_COLUMN).dt.weekday().alias("weekday"),
        pl.col(TIMESTAMP_COLUMN).dt.time().alias("time"),
    )


def drop_time_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.drop([column for column in TEMP_COLUMNS if column in df.columns])


def count_nulls(df: pl.DataFrame, columns: list[str]) -> int:
    if not columns:
        return 0
    return int(df.select(columns).null_count().sum_horizontal().sum())


def count_missing_rows(df: pl.DataFrame, columns: list[str]) -> int:
    if not columns:
        return 0
    return int(df.select(pl.any_horizontal([pl.col(column).is_null() for column in columns]).sum()).item())


def build_gap_mask(column: str, limit_rows: int) -> pl.Expr:
    return (
        pl.col(column)
        .is_null()
        .rle()
        .map_batches(
            lambda series: series.struct.field("value") & (series.struct.field("len") <= limit_rows)
        )
        .repeat_by(pl.col(column).is_null().rle().struct.field("len"))
        .explode()
    )


def temporal_fill(
    df: pl.DataFrame,
    columns: list[str],
    limit_rows: int = DEFAULT_TEMPORAL_GAP_LIMIT,
) -> pl.DataFrame:
    df = df.sort(["date", TIMESTAMP_COLUMN])

    expressions = []
    for column in columns:
        interpolated = pl.col(column).interpolate().over("date")
        small_gap_mask = build_gap_mask(column, limit_rows)

        expressions.append(
            pl.when(pl.col(column).is_not_null())
            .then(pl.col(column))
            .when(small_gap_mask)
            .then(interpolated)
            .otherwise(None)
            .alias(column)
        )

    return df.with_columns(expressions)


def spatial_fill(df: pl.DataFrame, columns: list[str], topology: GraphTopology) -> pl.DataFrame:
    available_columns = set(columns)
    expressions = []

    for column in columns:
        try:
            sensor_id = int(column)
        except ValueError:
            continue

        if not topology.has_node(sensor_id):
            continue

        neighbors = [
            neighbor_id
            for neighbor_id in topology.get_neighbors(sensor_id)
            if str(neighbor_id) in available_columns
        ]
        if not neighbors:
            continue

        expressions.append(
            pl.coalesce([
                pl.col(column),
                neighbor_value_expr(sensor_id, neighbors, topology),
            ]).alias(column)
        )

    if not expressions:
        return df

    return df.with_columns(expressions)


def neighbor_value_expr(
    sensor_id: int,
    neighbors: list[int],
    topology: GraphTopology,
) -> pl.Expr:
    weighted_values = []
    weights = []

    for neighbor_id in neighbors:
        column = str(neighbor_id)
        distance = topology.distance_between(sensor_id, neighbor_id)
        weight = 1.0 / max(distance or 1.0, 1e-6)

        weighted_values.append(pl.col(column).fill_null(0) * weight)
        weights.append(pl.when(pl.col(column).is_not_null()).then(weight).otherwise(0.0))

    total_weight = pl.sum_horizontal(weights)
    return (
        pl.when(total_weight > 0)
        .then(pl.sum_horizontal(weighted_values) / total_weight)
        .otherwise(None)
    )


def historical_fill(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    mean_columns = [f"__{column}_historical_mean" for column in columns]
    profile = df.group_by(["weekday", "time"]).agg(
        [pl.col(column).mean().alias(mean_column) for column, mean_column in zip(columns, mean_columns, strict=True)]
    )

    filled = df.join(profile, on=["weekday", "time"], how="left").with_columns(
        [
            pl.coalesce([pl.col(column), pl.col(mean_column)]).alias(column)
            for column, mean_column in zip(columns, mean_columns, strict=True)
        ]
    )
    return filled.drop(mean_columns)


def global_fill(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    return df.with_columns([pl.col(column).fill_null(pl.col(column).mean()).alias(column) for column in columns])


def drop_unresolved_sensors(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    if not columns:
        return df

    null_counts = df.select([pl.col(column).null_count().alias(column) for column in columns]).row(0)
    unresolved_columns = [column for column, null_count in zip(columns, null_counts, strict=True) if null_count > 0]

    if not unresolved_columns:
        return df

    logger.info("dropping unresolved sensors: count=%s", len(unresolved_columns))
    return df.drop(unresolved_columns)


def run_fill_layer(
    df: pl.DataFrame,
    columns: list[str],
    layer_name: str,
    fill_fn,
) -> tuple[pl.DataFrame, int]:
    before_nulls = count_nulls(df, columns)
    before_missing_rows = count_missing_rows(df, columns)
    started_at = perf_counter()

    filled_df = fill_fn(df)

    after_nulls = count_nulls(filled_df, columns)
    after_missing_rows = count_missing_rows(filled_df, columns)
    elapsed = perf_counter() - started_at

    logger.info(
        "%s: fixed_nulls=%s, fixed_rows=%s, remaining_nulls=%s, remaining_rows=%s, elapsed=%.2fs",
        layer_name,
        f"{before_nulls - after_nulls:,}",
        f"{before_missing_rows - after_missing_rows:,}",
        f"{after_nulls:,}",
        f"{after_missing_rows:,}",
        elapsed,
    )
    return filled_df, after_nulls


def transform_traffic_data(df: pl.DataFrame, topology: GraphTopology) -> pl.DataFrame:
    if df.is_empty():
        return df

    df = add_time_features(df)
    columns = sensor_columns(df)
    current_nulls = count_nulls(df, columns)
    logger.info(
        "missing transform start: sensor_columns=%s, initial_nulls=%s, initial_missing_rows=%s",
        len(columns),
        f"{current_nulls:,}",
        f"{count_missing_rows(df, columns):,}",
    )

    df, current_nulls = run_fill_layer(
        df,
        columns,
        "temporal fill",
        lambda current_df: temporal_fill(current_df, columns),
    )

    if current_nulls > 0:
        df, current_nulls = run_fill_layer(
            df,
            columns,
            "spatial fill",
            lambda current_df: spatial_fill(current_df, columns, topology),
        )

    if current_nulls > 0:
        df, current_nulls = run_fill_layer(
            df,
            columns,
            "historical fill",
            lambda current_df: historical_fill(current_df, columns),
        )

    if current_nulls > 0:
        df, current_nulls = run_fill_layer(
            df,
            columns,
            "global fill",
            lambda current_df: global_fill(current_df, columns),
        )

    df = drop_unresolved_sensors(df, columns)
    final_columns = sensor_columns(df)
    logger.info(
        "missing transform done: final_nulls=%s, final_missing_rows=%s, final_sensor_columns=%s",
        f"{count_nulls(df, final_columns):,}",
        f"{count_missing_rows(df, final_columns):,}",
        len(final_columns),
    )
    return drop_time_features(df)



