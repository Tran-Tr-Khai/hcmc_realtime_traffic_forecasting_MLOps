import logging
import os
import numpy as np 
import io  # <--- MỚI: Dùng để xử lý file trong bộ nhớ
import boto3  # <--- MỚI: Dùng để kết nối MinIO
from datetime import time, datetime, timedelta
from typing import Union, Optional, List
import posixpath
import polars as pl
from dotenv import load_dotenv

from src.offline.extractors import TrafficExtractor
from src.offline.transformers import TimeSeriesResampler, CausalImputer
from src.core.graph import MinIoGraphLoader, AdjacencyMatrixBuilder

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class OfflinePipeline:
    def __init__(
        self, 
        raw_data_key: str, 
        graph_key: str,
        bucket_name: Optional[str] = None,
        kafka_dumps_prefix: str = "raw/kafka-dumps",  # <--- MỚI: Prefix cho Kafka dumps
        output_key: str = "processed/traffic_clean.parquet",
        interval: str = "5m",
        start_time: Optional[time] = time(7, 30),
        end_time: Optional[time] = time(22, 30),
        window_days: Optional[int] = 30  # <--- MỚI: Windowing strategy (None = load all)
    ):
        """
        Initializes the offline traffic processing pipeline.
        
        Args:
            raw_data_key: Historical raw data file key (e.g., 'raw/hcmc-traffic-data.json')
            graph_key: Graph topology file key
            bucket_name: MinIO bucket name
            kafka_dumps_prefix: Prefix for Kafka dump files (e.g., 'raw/kafka-dumps')
            output_key: Output file key for cleaned data
            interval: Resampling interval
            start_time: Daily start time filter
            end_time: Daily end time filter
            window_days: Rolling window size (days) for historical data. None = load all.
                        Production recommendation: 30 days (enough context for imputation)
        """
        self.bucket = bucket_name or os.getenv("MINIO_BUCKET_NAME")
        self.raw_data_key = raw_data_key
        self.kafka_dumps_prefix = kafka_dumps_prefix
        self.output_key = output_key
        self.interval = interval
        self.window_days = window_days
        
        logger.info(f"Initializing OfflinePipeline using MinIO bucket: {self.bucket}")
        logger.info(f"Windowing strategy: {window_days} days" if window_days else "Windowing: DISABLED (load all historical data)")
        
        # --- MỚI: Khởi tạo S3 Client để ghi dữ liệu ---
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=os.getenv("MINIO_ENDPOINT_URL"),
            aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("MINIO_SECRET_KEY")
        )
        
        # 1. Extractor (for historical data)
        self.extractor = TrafficExtractor(raw_data_key, bucket_name=self.bucket)
        
        # 2. Graph Loader
        self.graph_loader = MinIoGraphLoader(graph_key, bucket_name=self.bucket)
        self.storage = self.graph_loader
        self.topology = self.graph_loader.load_topology()
        
        # 3. Resampler
        self.resampler = TimeSeriesResampler(
            interval=interval,
            start_time=start_time,
            end_time=end_time
        )
        
        # 4. Imputer
        self.imputer = CausalImputer(self.topology)

    def run(self) -> pl.DataFrame:
        """
        Executes the pipeline with "Concat Raw First, Then Impute" strategy.
        
        Flow:
            1. Load historical raw data (with optional windowing)
            2. Load all Kafka dumps
            3. Concatenate all raw data (full context for imputation)
            4. Resample
            5. Impute (with sufficient historical context)
            6. Save to MinIO
        """
        logger.info("="*80)
        logger.info("STARTING OFFLINE PIPELINE - CONCAT RAW FIRST STRATEGY")
        logger.info("="*80)
        
        # Step 1: Load historical raw data (with windowing)
        logger.info("Step 1/6: Loading historical raw data...")
        historical_df = self._load_historical_data()
        
        # Step 2: Load Kafka dumps
        logger.info("Step 2/6: Loading Kafka dump files...")
        kafka_df = self._load_kafka_dumps()
        
        # Step 3: Concatenate raw data
        logger.info("Step 3/6: Concatenating raw data (historical + Kafka dumps)...")
        combined_df = self._concat_raw_data(historical_df, kafka_df)
        
        # Step 4: Resampling
        logger.info("Step 4/6: Resampling and Pivoting...")
        resampled_df = self.resampler.transform(combined_df)
        
        # Validate data
        self.validate_data(resampled_df)

        if resampled_df.is_empty():
            logger.error("Resampling returned an empty DataFrame. Aborting pipeline.")
            return resampled_df
            
        # Step 5: Imputation (now with full context!)
        logger.info("Step 5/6: Performing Causal Imputation with full historical context...")
        clean_df = self.imputer.transform(resampled_df)
        
        # Step 6: Save to MinIO
        logger.info("Step 6/6: Saving processed data and graph to MinIO...")
        self.save_to_minio(clean_df, self.output_key) 

        # Process and Save Graph Topology
        final_adj_matrix = self._process_topology(clean_df.columns)
        
        folder = posixpath.dirname(self.output_key)
        graph_key = posixpath.join(folder, "adj_matrix.npy")
        self.save_to_minio(final_adj_matrix, graph_key)

        logger.info(f"Pipeline completed successfully. Final shape: {clean_df.shape}")
        logger.info("="*80)
        
        return clean_df
    
    def _load_historical_data(self) -> pl.LazyFrame:
        """
        Load historical raw data with optional windowing.
        
        Returns:
            LazyFrame with columns: [timestamp, sensor_id, count]
        """
        lazy_df = self.extractor.extract()
        
        # Apply windowing if configured
        if self.window_days:
            cutoff_date = datetime.now() - timedelta(days=self.window_days)
            cutoff_ts = cutoff_date.replace(tzinfo=None)  # Remove timezone for comparison
            
            logger.info(f"⏱  Applying {self.window_days}-day window. Cutoff: {cutoff_ts}")
            
            lazy_df = lazy_df.filter(
                pl.col("timestamp").dt.replace_time_zone(None) >= cutoff_ts
            )
        
        # Collect to check size
        collected = lazy_df.collect()
        logger.info(f"✓ Historical data: {collected.height:,} records")
        
        return collected.lazy()
    
    def _load_kafka_dumps(self) -> pl.LazyFrame:
        """
        Load all Kafka dump files from MinIO.
        
        Returns:
            LazyFrame with columns: [timestamp, sensor_id, count]
            Empty LazyFrame if no dumps found
        """
        try:
            # List all objects under kafka_dumps_prefix
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=self.kafka_dumps_prefix
            )
            
            if 'Contents' not in response:
                logger.warning(f"⚠️  No Kafka dumps found at {self.kafka_dumps_prefix}")
                return self._empty_traffic_lazyframe()
            
            # Filter for JSON files only
            dump_files = [
                obj['Key'] for obj in response['Contents']
                if obj['Key'].endswith('.json') and obj['Size'] > 0
            ]
            
            if not dump_files:
                logger.warning("⚠️  No valid JSON dump files found")
                return self._empty_traffic_lazyframe()
            
            logger.info(f"📁 Found {len(dump_files)} Kafka dump files")
            
            # Load each dump file
            dump_dfs = []
            for dump_key in dump_files:
                try:
                    logger.debug(f"  Loading {dump_key}...")
                    
                    # Get file stream
                    stream = self.storage.get_object_stream(dump_key)
                    
                    # Read JSONL format (one JSON per line)
                    df = pl.read_ndjson(stream)
                    
                    if df.is_empty():
                        logger.warning(f"  Empty: {dump_key}")
                        continue
                    
                    # Normalize schema to match historical data
                    # sensor_id might be string "node_001" -> extract int
                    processed_df = (
                        df.lazy()
                        .select([
                            pl.col("ts").alias("timestamp_ms"),
                            pl.when(pl.col("sensor_id").cast(pl.Utf8).str.contains("node_"))
                                .then(pl.col("sensor_id").cast(pl.Utf8).str.extract(r"(\d+)", 1).cast(pl.Int64))
                                .otherwise(pl.col("sensor_id").cast(pl.Int64))
                                .alias("sensor_id"),
                            pl.col("count").cast(pl.Int64)
                        ])
                        .with_columns(
                            pl.from_epoch(pl.col("timestamp_ms"), time_unit="ms")
                            .alias("timestamp_utc")
                        )
                        .with_columns(
                            pl.col("timestamp_utc")
                            .dt.replace_time_zone("UTC")
                            .dt.convert_time_zone("Asia/Ho_Chi_Minh")
                            .alias("timestamp")
                        )
                        .select(["timestamp", "sensor_id", "count"])
                        .collect()
                    )
                    
                    dump_dfs.append(processed_df)
                    logger.debug(f"    ✓ {processed_df.height:,} records")
                    
                except Exception as e:
                    logger.error(f"  ❌ Failed to load {dump_key}: {e}")
                    continue
            
            if not dump_dfs:
                logger.warning("⚠️  No Kafka dumps successfully loaded")
                return self._empty_traffic_lazyframe()
            
            # Concatenate all dumps
            combined_dumps = pl.concat(dump_dfs)
            logger.info(f"✓ Kafka dumps total: {combined_dumps.height:,} records")
            
            return combined_dumps.lazy()
            
        except Exception as e:
            logger.error(f"❌ Error loading Kafka dumps: {e}")
            return self._empty_traffic_lazyframe()
    
    def _empty_traffic_lazyframe(self) -> pl.LazyFrame:
        """Return empty LazyFrame with correct schema."""
        return pl.DataFrame({
            "timestamp": [],
            "sensor_id": [],
            "count": []
        }).cast({
            "timestamp": pl.Datetime("ms", "Asia/Ho_Chi_Minh"),
            "sensor_id": pl.Int64,
            "count": pl.Int64
        }).lazy()
    
    def _concat_raw_data(
        self, 
        historical_df: pl.LazyFrame, 
        kafka_df: pl.LazyFrame
    ) -> pl.LazyFrame:
        """
        Concatenate historical and Kafka dump data.
        
        Args:
            historical_df: Historical raw data
            kafka_df: Kafka dump data
            
        Returns:
            Combined LazyFrame
        """
        # Collect both to concatenate
        hist_collected = historical_df.collect()
        kafka_collected = kafka_df.collect()
        
        logger.info("-" * 60)
        logger.info("📊 RAW DATA CONCATENATION SUMMARY")
        logger.info("-" * 60)
        logger.info(f"Historical records: {hist_collected.height:,}")
        logger.info(f"Kafka dump records: {kafka_collected.height:,}")
        
        # Concatenate
        if kafka_collected.height == 0:
            logger.warning("⚠️  No Kafka dumps to concatenate. Using historical data only.")
            combined = hist_collected
        else:
            combined = pl.concat([hist_collected, kafka_collected])
        
        # Remove duplicates (same timestamp + sensor_id)
        # Keep last occurrence (Kafka dumps override historical)
        original_height = combined.height
        combined = combined.unique(subset=["timestamp", "sensor_id"], keep="last")
        duplicates_removed = original_height - combined.height
        
        if duplicates_removed > 0:
            logger.info(f"🔄 Removed {duplicates_removed:,} duplicate records (kept latest)")
        
        logger.info(f"✅ Combined total: {combined.height:,} unique records")
        logger.info(f"📅 Time range: {combined['timestamp'].min()} → {combined['timestamp'].max()}")
        logger.info("-" * 60)
        
        return combined.lazy()
    
    def save_to_minio(self, data: Union[pl.DataFrame, np.ndarray], key: str):
        """
        Tự động phát hiện kiểu dữ liệu để serialize phù hợp.
        """
        buffer = io.BytesIO()
        
        try:
            # --- Logic chuyển đổi dữ liệu (Serialization) ---
            if isinstance(data, pl.DataFrame):
                logger.info(f"Saving DataFrame to {key}...")
                data.write_parquet(buffer)
                
            elif isinstance(data, np.ndarray):
                logger.info(f"Saving Numpy Array to {key}...")
                np.save(buffer, data)
                
            else:
                raise TypeError(f"Unsupported data type for MinIO upload: {type(data)}")

            # --- Logic Upload (IO) ---
            buffer.seek(0)
            self.storage.put_object(
                bucket_name=self.bucket,
                object_name=key,
                data=buffer,
                length=buffer.getbuffer().nbytes
            )
            logger.info(f"Upload successful: {key}")
            
        except Exception as e:
            logger.error(f"Failed to save to MinIO at key '{key}': {e}")
            raise e # Bắt buộc raise lỗi để Pipeline biết mà dừng

    def _process_topology(self, valid_columns: list[str]) -> np.ndarray:
        """
        Build Adjacency Matrix aligned with DataFrame columns.
        Includes a SANITY CHECK to prove alignment.
        """
        # 1. Lấy danh sách cột từ DataFrame (Ground Truth)
        # valid_columns đã được sort trong resampler (thường là string sort)
        ordered_sensor_cols = [c for c in valid_columns if c != 'timestamp']
        
        # Chuyển sang int để tìm trong Graph
        target_node_ids = [int(col) for col in ordered_sensor_cols]
        
        logger.info(f"Aligning graph to {len(target_node_ids)} sensors.")

        # 2. Load Topology gốc
        raw_topology = self.graph_loader.load_topology()
        
        # --- [START] SANITY CHECK (KIỂM TRA ĐỐI CHIẾU) ---
        logger.info("-" * 40)
        logger.info("🕵️ ALIGNMENT SANITY CHECK (First 5 items)")
        logger.info(f"{'Index':<5} | {'DF Column (Name)':<15} | {'Matrix Row (ID)':<15}")
        logger.info("-" * 40)
        
        # In ra 5 phần tử đầu tiên để bạn soi
        for i in range(min(5, len(target_node_ids))):
            df_col = ordered_sensor_cols[i]  # Tên cột trong Data
            matrix_id = target_node_ids[i]   # ID sẽ dùng để xếp hàng ma trận
            
            # Nếu 2 cái này lệch nhau về mặt ý nghĩa -> BÁO ĐỘNG
            logger.info(f"{i:<5} | {df_col:<15} | {matrix_id:<15}")
            
        logger.info("-" * 40)
        # --- [END] SANITY CHECK ---

        # 3. Re-index Logic (Bắt buộc Graph theo thứ tự target_node_ids)
        # Tìm xem node 10 nằm ở đâu trong graph cũ, node 100 nằm ở đâu...
        try:
            index_map = [raw_topology.node_to_index[nid] for nid in target_node_ids]
        except KeyError as e:
            logger.error(f"Graph is missing a sensor ID present in Data: {e}")
            raise e

        # 4. Sắp xếp lại ma trận (Advanced Indexing)
        raw_adj = raw_topology.adjacency_matrix
        # Reorder rows and columns
        aligned_adj = raw_adj[index_map, :][:, index_map]
        
        logger.info(f"Matrix re-indexed. Shape: {aligned_adj.shape}")

        # 5. Normalize (Symmetric)
        # D^(-1/2) * (A+I) * D^(-1/2)
        adj_with_loop = aligned_adj + np.eye(aligned_adj.shape[0])
        row_sum = np.array(adj_with_loop.sum(1))
        
        # Tránh chia cho 0
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        
        norm_adj = d_mat_inv_sqrt.dot(adj_with_loop).dot(d_mat_inv_sqrt)
        
        return norm_adj
        
    def validate_data(self, df: pl.DataFrame):
        """Kiểm tra sức khỏe dữ liệu (Data Health Check)."""
        if df.is_empty():
            return

        total_cells = df.height * (df.width - 1)
        null_count = df.select(pl.all().exclude("timestamp")).null_count().sum_horizontal().sum()
        sparsity = (null_count / total_cells) * 100 if total_cells > 0 else 0

        logger.info("-" * 60)
        logger.info("📊 DATA HEALTH REPORT")
        logger.info("-" * 60)
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Time Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"Sparsity: {sparsity:.2f}% ({null_count} missing values)")

        try:
            null_per_sensor = (
                df.select(pl.all().exclude("timestamp"))
                .null_count()
                .unpivot()
                .with_columns((pl.col("value") / df.height * 100).alias("null_pct"))
                .sort("null_pct", descending=True)
            )
        except AttributeError:
            null_per_sensor = (
                df.select(pl.all().exclude("timestamp"))
                .null_count()
                .melt()
                .with_columns((pl.col("value") / df.height * 100).alias("null_pct"))
                .sort("null_pct", descending=True)
            )

        logger.info("Top 5 Sensors with most missing data:")
        logger.info("\n" + str(null_per_sensor.head(5)))
        
        # Kiểm tra Timeline Gap
        try:
            interval_minutes = int(self.interval.replace("m", ""))
            seconds_per_step = interval_minutes * 60
            
            actual_duration = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
            expected_rows = int(actual_duration / seconds_per_step) + 1
            
            if df.height < expected_rows - 2:
                logger.warning(f"Timeline Gap detected! Expected ~{expected_rows} rows, got {df.height}")
        except Exception:
            pass
        
        logger.info("-" * 30)