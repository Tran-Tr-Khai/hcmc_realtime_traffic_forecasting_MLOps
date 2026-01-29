import logging
from datetime import time
from typing import List
import polars as pl

logger = logging.getLogger(__name__)

class TimeSeriesResampler:
    def __init__(self, interval: str = "5m", start_time: time = None, end_time: time = None):
        self.interval = interval
        self.start_time = start_time
        self.end_time = end_time

    def transform(self, lazy_df: pl.LazyFrame) -> pl.DataFrame:
        """
        Resamples time series data using Static Window Truncation.
        Strategy: Truncate Timestamp -> GroupBy (Static) -> Pivot
        Benefit: Faster and strictly robust against unsorted data.
        """
        # dt.truncate: Làm tròn thời gian xuống (ví dụ 08:03 -> 08:00)
        resampled_lazy = (
            lazy_df
            .with_columns(
                pl.col("timestamp")
                .dt.truncate(self.interval)
                .alias("timestamp_bucket")
            )
            .group_by(["timestamp_bucket", "sensor_id"]) # Group tĩnh
            .agg(pl.col("count").mean().alias("count"))
        )

        if self.start_time and self.end_time:
            time_filter = (
                (pl.col('timestamp_bucket').dt.time() >= self.start_time) &
                (pl.col('timestamp_bucket').dt.time() <= self.end_time)
            )
            resampled_lazy = resampled_lazy.filter(time_filter)

        df_long = resampled_lazy.collect()
        
        if df_long.is_empty():
            logger.warning("Resampled data is empty!")
            return df_long

        wide_df = df_long.pivot(
            on="sensor_id",
            index="timestamp_bucket", # Dùng cột bucket làm index
            values="count",
            aggregate_function="mean"
        )

        wide_df = wide_df.rename({"timestamp_bucket": "timestamp"})
        
        sensor_cols = sorted([c for c in wide_df.columns if c != "timestamp"])
        
        final_df = (
            wide_df
            .sort("timestamp")
            .select(["timestamp"] + sensor_cols)
        )

        logger.info(f"Resampled V2 (Truncate method): {final_df.shape}")
        return final_df

    def get_sensor_columns(self, df: pl.DataFrame) -> List[str]:
        return [col for col in df.columns if col != "timestamp"]