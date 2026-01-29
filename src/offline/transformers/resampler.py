import logging
from datetime import time
from typing import List
import polars as pl

logger = logging.getLogger(__name__)


class TimeSeriesResampler:
    def __init__(self, interval: str = "5m", start_time: time = None, end_time: time = None):
        """
        Initialize resampler.
        
        Args:
            interval: Resample interval (Polars duration string, e.g., "5m")
            start_time: Optional time filter after resampling
            end_time: Optional time filter after resampling
        """
        self.interval = interval
        self.start_time = start_time
        self.end_time = end_time
    
    def transform(self, df: pl.LazyFrame) -> pl.DataFrame:
        """
        Resample time series to fixed intervals.
        
        Args:
            df: LazyFrame with [timestamp, sensor_id, count]
            
        Returns:
            DataFrame in wide format with timestamp index and sensor columns
        """
        # Collect to eager for pivoting (Polars pivot requires eager)
        df_eager = df.collect()
        
        logger.info(f"Collected {df_eager.shape[0]} records for resampling")
        
        # Pivot to wide format
        pivot_df = df_eager.pivot(
            values='count',
            index='timestamp',
            on='sensor_id',
            aggregate_function='mean'
        )
        
        # Sort by timestamp for group_by_dynamic
        pivot_df = pivot_df.sort('timestamp')
        
        # Get all sensor columns (exclude timestamp)
        sensor_cols = [col for col in pivot_df.columns if col != 'timestamp']
        
        # Resample to fixed intervals using group_by_dynamic
        resampled = pivot_df.group_by_dynamic(
            'timestamp',
            every=self.interval,
            period=self.interval,
            closed='left'
        ).agg([
            pl.col(sensor_col).mean().alias(sensor_col) 
            for sensor_col in sensor_cols
        ])
        
        # Apply time filter again after resampling if specified
        if self.start_time is not None and self.end_time is not None:
            resampled = resampled.filter(
                (pl.col('timestamp').dt.time() >= pl.time(
                    self.start_time.hour, 
                    self.start_time.minute, 
                    self.start_time.second
                )) &
                (pl.col('timestamp').dt.time() <= pl.time(
                    self.end_time.hour, 
                    self.end_time.minute, 
                    self.end_time.second
                ))
            )
        
        logger.info(
            f"Resampled to {self.interval} intervals: shape {resampled.shape}"
        )
        
        return resampled
    
    def get_sensor_columns(self, df: pl.DataFrame) -> List[str]:
        """
        Extract sensor column names from resampled DataFrame.
        
        Args:
            df: Resampled DataFrame
            
        Returns:
            List of sensor column names (as strings)
        """
        return [col for col in df.columns if col != 'timestamp']
