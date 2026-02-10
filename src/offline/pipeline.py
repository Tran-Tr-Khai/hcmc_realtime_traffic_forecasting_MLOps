import logging
import os
import numpy as np 
import io  # <--- MỚI: Dùng để xử lý file trong bộ nhớ
import boto3  # <--- MỚI: Dùng để kết nối MinIO
from datetime import time
from typing import Union, Optional
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
        output_key: str = "processed/traffic_clean.parquet", # <--- MỚI: Đường dẫn đầu ra mặc định trên MinIO
        interval: str = "5m",
        start_time: Optional[time] = time(7, 30),
        end_time: Optional[time] = time(22, 30)
    ):
        """
        Initializes the offline traffic processing pipeline.
        """
        self.bucket = bucket_name or os.getenv("MINIO_BUCKET_NAME")
        self.output_key = output_key # Lưu đường dẫn output
        self.interval = interval 
        
        logger.info(f"Initializing OfflinePipeline using MinIO bucket: {self.bucket}")
        
        # --- MỚI: Khởi tạo S3 Client để ghi dữ liệu ---
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=os.getenv("MINIO_ENDPOINT_URL"),
            aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("MINIO_SECRET_KEY")
        )
        
        # 1. Extractor
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
        Executes the pipeline: Extract -> Resample -> Impute -> Save to MinIO.
        """
        logger.info("Starting Offline Pipeline execution...")
        
        # Step 1: Extraction
        logger.info("Step 1/5 Extracting raw data...")
        lazy_df = self.extractor.extract()
        
        # Step 2: Resampling
        logger.info("Step 2/5: Resampling and Pivoting...")
        resampled_df = self.resampler.transform(lazy_df)
        
        # Validate data
        self.validate_data(resampled_df)

        if resampled_df.is_empty():
            logger.error("Resampling returned an empty DataFrame. Aborting pipeline.")
            return resampled_df
            
        # Step 3: Imputation
        logger.info("Step 3/5: Performing Causal Imputation...")
        clean_df = self.imputer.transform(resampled_df)
        
        # Step 4: Save to MinIO ---
        logger.info("Step 4/5: Saving processed data to MinIO...")
        self.save_to_minio(clean_df, self.output_key) 

        # Step 5: Process and Save Graph Topology
        logger.info("Step 5/5: Aligning and Saving Graph Topology...")
        final_adj_matrix = self._process_topology(clean_df.columns)
        
        folder = posixpath.dirname(self.output_key)
        graph_key = posixpath.join(folder, "adj_matrix.npy")
        self.save_to_minio(final_adj_matrix, graph_key)

        logger.info(f"Pipeline completed successfully. Final shape: {clean_df.shape}")
        
        return clean_df
    
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

        logger.info("-" * 30)
        logger.info("DATA HEALTH REPORT")
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