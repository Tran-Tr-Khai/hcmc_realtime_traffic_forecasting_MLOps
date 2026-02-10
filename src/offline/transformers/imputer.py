import logging
from typing import List, Set
import polars as pl
import numpy as np

from src.core.graph import GraphTopology

logger = logging.getLogger(__name__)

class CausalImputer:
    """
    Imputes missing values using a 3-Layer Causal Strategy:
    
    1. Temporal (Nội suy): Lấp lỗ hổng nhỏ trong ngày.
    2. Spatial (Không gian): Lấp lỗ hổng lớn bằng hàng xóm (IDW).
    3. Historical (Lịch sử): Lấp lỗ hổng "thảm họa" bằng chu kỳ tuần (Weekly Pattern).
    """
    
    def __init__(self, graph_topology: GraphTopology):
        self.topology = graph_topology
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        logger.info("Starting 3-Layer Imputation Strategy...")
        
        # 0. Chuẩn bị số liệu
        sensor_cols = [c for c in df.columns if c not in ['timestamp', 'date', 'weekday', 'time']]
        total_cells = df.height * len(sensor_cols)
        
        initial_nulls = self._count_total_nulls(df, sensor_cols)
        current_sparsity = (initial_nulls / total_cells) * 100
        
        logger.info(f"Initial State: {initial_nulls:,} missing values ({current_sparsity:.2f}% sparsity)")

        # Tạo feature thời gian hỗ trợ tính toán
        df = df.with_columns([
            pl.col('timestamp').dt.date().alias('date'),
            pl.col('timestamp').dt.weekday().alias('weekday'),
            pl.col('timestamp').dt.time().alias('time')
        ])
    
        # --- LAYER 1: TEMPORAL IMPUTATION ---
        df = self._temporal_imputation(df, sensor_cols)
        
        nulls_l1 = self._count_total_nulls(df, sensor_cols)
        fixed_l1 = initial_nulls - nulls_l1
        sparsity_l1 = (nulls_l1 / total_cells) * 100
        logger.info(f"Layer 1 (Temporal) fixed: {fixed_l1:,}. Remaining: {nulls_l1:,} ({sparsity_l1:.2f}%)")
        
        # --- LAYER 2: SPATIAL IMPUTATION (IDW) ---
        if nulls_l1 > 0:
            df = self._spatial_imputation(df, sensor_cols)
            
            nulls_l2 = self._count_total_nulls(df, sensor_cols)
            fixed_l2 = nulls_l1 - nulls_l2
            sparsity_l2 = (nulls_l2 / total_cells) * 100
            logger.info(f"Layer 2 (Spatial)  fixed: {fixed_l2:,}. Remaining: {nulls_l2:,} ({sparsity_l2:.2f}%)")
        else:
            nulls_l2 = 0
            logger.info("Layer 2 skipped.")
        
        # --- LAYER 3: HISTORICAL IMPUTATION (Weekly Profile) ---
        if nulls_l2 > 0:
            df = self._historical_imputation(df, sensor_cols)
            
            nulls_l3 = self._count_total_nulls(df, sensor_cols)
            fixed_l3 = nulls_l2 - nulls_l3
            sparsity_l3 = (nulls_l3 / total_cells) * 100
            logger.info(f"Layer 3 (History)  fixed: {fixed_l3:,}. Remaining: {nulls_l3:,} ({sparsity_l3:.2f}%)")
        else:
            logger.info("Layer 3 skipped.")
        
        # --- LAYER 4: GLOBAL MEAN RESCUE ---
        if nulls_l3 > 0:
            df = self._global_imputation(df, sensor_cols, threshold=0.05)
            
            # Tính lại thống kê sau Layer 4
            nulls_l4 = self._count_total_nulls(df, sensor_cols)
            fixed_l4 = nulls_l3 - nulls_l4
            sparsity_l4 = (nulls_l4 / total_cells) * 100
            logger.info(f"Layer 4 (Global)   fixed: {fixed_l4:,}. Remaining: {nulls_l4:,} ({sparsity_l4:.2f}%)")
        else:
            logger.info("Layer 4 skipped.")

        # CLEANUP
        df = self._drop_unresolved_sensors(df, sensor_cols)
        df = df.drop(['date', 'weekday', 'time'])
        
        # Final Check
        final_nulls = self._count_total_nulls(df, sensor_cols)
        if final_nulls == 0:
            logger.info("SUCCESS: Data is 100% clean.")
        else:
            logger.warning(f"Warning: {final_nulls} nulls remain.")
            
        return df
    
    def _count_total_nulls(self, df: pl.DataFrame, cols: List[str] = None) -> int:
        """Helper để đếm tổng số ô Null trong DataFrame."""
        if cols is None:
            # Tự động loại bỏ các cột không phải sensor
            cols = [c for c in df.columns if c not in ['timestamp', 'date', 'weekday', 'time']]
        
        # Dùng sum_horizontal để đếm nhanh
        return df.select(cols).null_count().sum_horizontal().sum()   
     
    def _temporal_imputation(self, df: pl.DataFrame, sensor_cols: List[str], limit_rows: int = 12) -> pl.DataFrame:
        """
        Layer 1: Linear Interpolation within the same day.
        
        UPGRADE: Chỉ nội suy nếu khoảng trống nhỏ hơn 'limit_rows'.
        Mặc định limit_rows = 12 (tương đương 60 phút với interval 5p).
        Các khoảng trống lớn hơn sẽ bị bỏ qua để Layer 2/3 xử lý.
        """
        df = df.sort(['date', 'timestamp'])
        
        exprs = []
        for col in sensor_cols:
            interpolated = (
                pl.col(col)
                .interpolate()
                .forward_fill()
                .backward_fill()
                .over('date')
            )
            
            # Tạo mặt nạ (Mask) để phát hiện gap lớn
            # Logic: Tìm các chuỗi Null liên tiếp > limit_rows
            # rle() trả về struct: {len: int, value: bool}
            rle_mask = pl.col(col).is_null().rle()
            
            # Nếu là Null (value=True) VÀ độ dài > limit -> Trả về True (là Big Gap)
            is_big_gap = (pl.col("value")) & (pl.col("len") > limit_rows)
            
            # Bung ngược (explode) cái RLE này ra để khớp độ dài với DataFrame gốc
            # repeat_by sẽ nhân bản giá trị True/False theo độ dài 'len'
            big_gap_mask_expanded = (
                is_big_gap.repeat_by(pl.col("len")).explode()
            )
            
            # 3. Kết hợp: Nếu là Big Gap -> Giữ Null. Nếu không -> Lấy giá trị nội suy
            # Lưu ý: Cần dùng rle_mask làm gốc để tính toán expanded mask
            final_expr = (
                pl.when(
                    pl.col(col).is_null().rle().map_batches(
                        lambda s: s.struct.field("value") & (s.struct.field("len") > limit_rows)
                    ).repeat_by(
                        pl.col(col).is_null().rle().struct.field("len")
                    ).explode()
                )
                .then(None)  # Giữ nguyên Null
                .otherwise(interpolated) # Dùng giá trị nội suy
                .alias(col)
            )
            
            exprs.append(final_expr)

        # Áp dụng logic
        df = df.with_columns(exprs)
        
        logger.info(f"Layer 1: Smart Temporal Interpolation (Limit={limit_rows} rows) done.")
        return df

    def _spatial_imputation(self, df: pl.DataFrame, sensor_cols: List[str]) -> pl.DataFrame:
        """Layer 2: Spatial IDW using Neighbors."""
        impute_exprs = []
        has_dist = self.topology.distance_matrix is not None
        
        for col in sensor_cols:
            try:
                sensor_id = int(col)
            except ValueError: continue

            if not self.topology.has_node(sensor_id): continue
            
            neighbors = self.topology.get_neighbors(sensor_id)
            valid_neighbors = [n for n in neighbors if str(n) in sensor_cols]
            
            if not valid_neighbors: continue
            
            # Tính giá trị hàng xóm (Weighted hoặc Mean)
            if has_dist:
                curr_idx = self.topology.node_to_index[sensor_id]
                weighted_vals = []
                total_weight = 0.0
                for n_id in valid_neighbors:
                    n_idx = self.topology.node_to_index[n_id]
                    dist = max(self.topology.distance_matrix[curr_idx, n_idx], 1e-6)
                    weight = 1.0 / dist
                    weighted_vals.append(pl.col(str(n_id)) * weight)
                    total_weight += weight
                neighbor_val = pl.sum_horizontal(weighted_vals) / total_weight
            else:
                neighbor_val = pl.mean_horizontal([pl.col(str(n)) for n in valid_neighbors])

            # Chỉ điền vào chỗ còn Null
            expr = pl.coalesce([pl.col(col), neighbor_val]).alias(col)
            impute_exprs.append(expr)
        
        if impute_exprs:
            df = df.with_columns(impute_exprs)
            logger.info("Layer 2: Spatial Imputation done.")
        return df

    def _historical_imputation(self, df: pl.DataFrame, sensor_cols: List[str]) -> pl.DataFrame:
        """
        Layer 3: Historical Profile (Điền theo chu kỳ tuần).
        Logic: Tính trung bình lưu lượng của (Thứ 2, 08:00) trong quá khứ để điền cho (Thứ 2, 08:00) hiện tại.
        """
        # 1. Tính Profile trung bình cho từng Sensor theo (Weekday, Time)
        # Group by Thứ và Giờ -> Tính Mean
        profile_df = (
            df.group_by(['weekday', 'time'])
            .agg([
                pl.col(c).mean().alias(f"{c}_mean") 
                for c in sensor_cols
            ])
        )
        
        # 2. Join Profile này ngược lại vào bảng gốc
        # Left Join để giữ nguyên số dòng
        df_joined = df.join(profile_df, on=['weekday', 'time'], how='left')
        
        # 3. Điền khuyết (Coalesce: Gốc -> Hàng xóm -> Lịch sử)
        fill_exprs = []
        for col in sensor_cols:
            mean_col = f"{col}_mean"
            # Nếu cột gốc vẫn Null (sau 2 bước trên), lấy giá trị Mean lịch sử
            expr = pl.coalesce([pl.col(col), pl.col(mean_col)]).alias(col)
            fill_exprs.append(expr)
            
        df_final = df_joined.with_columns(fill_exprs)
        
        # Xóa các cột tạm (_mean) sinh ra do join
        cols_to_drop = [f"{c}_mean" for c in sensor_cols]
        df_final = df_final.drop(cols_to_drop)
        
        logger.info("Layer 3: Historical Pattern Imputation done.")
        return df_final
    
    def _global_imputation(self, df: pl.DataFrame, sensor_cols: List[str], threshold: float = 0.5) -> pl.DataFrame:
        """
        Layer 4: Global Mean Rescue (Last Resort).
        Chiến thuật: Cứu các sensor còn thiếu ít dữ liệu (< 5%) bằng giá trị trung bình toàn cục của chính nó.
        """
        total_rows = df.height
        rescue_exprs = []
        rescued_sensors = []

        # 1. Tính nhanh số lượng null của tất cả sensor (1 lệnh duy nhất)
        # Trả về 1 row chứa null count của từng cột
        null_stats = df.select([
            pl.col(c).null_count().alias(c) for c in sensor_cols
        ]).row(0)
        
        # 2. Duyệt qua kết quả thống kê để quyết định cứu hay bỏ
        for idx, col in enumerate(sensor_cols):
            missing_count = null_stats[idx]
            
            if missing_count > 0:
                missing_rate = missing_count / total_rows
                
                # CHỈ CỨU NẾU TỶ LỆ LỖI < THRESHOLD (Ví dụ 5%)
                if missing_rate < threshold:
                    # Fill bằng Mean của toàn bộ cột đó
                    expr = pl.col(col).fill_null(pl.col(col).mean())
                    rescue_exprs.append(expr)
                    rescued_sensors.append(col)

        # 3. Thực thi (Chỉ chạy 1 lần with_columns cho tất cả sensor cần cứu)
        if rescue_exprs:
            df = df.with_columns(rescue_exprs)
            logger.info(f"Layer 4 (Global Rescue): Saved {len(rescued_sensors)} sensors using Global Mean (Threshold < {threshold*100}%).")
            # logger.debug(f"Rescued sensors: {rescued_sensors}")
        else:
            logger.info("Layer 4 skipped (No eligible sensors to rescue).")
            
        return df        

    def _drop_unresolved_sensors(self, df: pl.DataFrame, sensor_cols: List[str]) -> pl.DataFrame:
        """Loại bỏ sensor chết hẳn (không cứu được bằng cả 3 cách)."""
        null_counts = df.select([
            pl.col(c).null_count() for c in sensor_cols if c in df.columns
        ]).row(0)
        
        cols_to_drop = [sensor_cols[i] for i, cnt in enumerate(null_counts) if cnt > 0]
        
        if cols_to_drop:
            df = df.drop(cols_to_drop)
            logger.warning(f"Dropped {len(cols_to_drop)} dead sensors: {cols_to_drop[:5]}...")
            
        return df
    
    def get_valid_sensors(self, df: pl.DataFrame) -> Set[int]:
        cols = [col for col in df.columns if col not in ['timestamp', 'date', 'weekday', 'time']]
        return set(int(c) for c in cols)
    

    