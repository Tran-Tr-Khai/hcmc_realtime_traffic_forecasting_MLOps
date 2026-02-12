import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import Redis Client Factory từ module core
from src.core.storage.redis_client import RedisStateManager, create_redis_client

# --- 1. CẤU HÌNH LOGGING & ENV ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Lấy cấu hình từ biến môi trường
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT"))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
PREDICTION_KEY = "traffic:predictions:latest"

# Biến toàn cục lưu kết nối Redis
# Lưu ý: Trong môi trường thực tế có thể dùng Dependency Injection, 
# nhưng dùng biến global với lifespan là cách đơn giản và hiệu quả.
redis_manager: Optional[RedisStateManager] = None


# --- 2. DATA MODELS (PYDANTIC) ---
class PredictionResponse(BaseModel):
    """Schema trả về cho Frontend - Cấu trúc tối ưu"""
    timestamp: str = Field(..., description="Thời gian dự đoán (ISO Format, UTC+7)")
    data: Dict[str, float] = Field(
        default_factory=dict, 
        description="Dictionary mapping node_id -> predicted_flow"
    )


# --- 3. LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Quản lý vòng đời ứng dụng:
    - Startup: Kết nối Redis
    - Shutdown: Đóng kết nối để tránh rò rỉ tài nguyên
    """
    global redis_manager
    logger.info(f"Đang khởi tạo kết nối Redis tới {REDIS_HOST}:{REDIS_PORT}...")
    
    try:
        # Sử dụng factory function có sẵn
        redis_manager = create_redis_client(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB
        )
        logger.info("Kết nối Redis thành công.")
        yield
    except Exception as e:
        logger.error(f"Lỗi kết nối Redis: {e}")
        # Vẫn cho app chạy lên, nhưng các endpoint gọi Redis sẽ lỗi 503
        yield
    finally:
        # Clean up khi tắt app
        if redis_manager:
            redis_manager.close()
            logger.info("Đã đóng kết nối Redis.")


# --- 4. KHỞI TẠO APP ---
app = FastAPI(
    title="Traffic Forecasting API",
    description="API cung cấp dữ liệu dự báo giao thông realtime từ Redis",
    version="1.0.0",
    lifespan=lifespan
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong môi trường Prod nên đổi thành domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 5. ENDPOINTS ---

@app.get("/health", tags=["System"])
def health_check():
    """Kiểm tra trạng thái hệ thống"""
    is_redis_connected = False
    if redis_manager:
        is_redis_connected = redis_manager.health_check()
    
    return {
        "status": "active",
        "redis_connected": is_redis_connected
    }


@app.get(
    "/predictions/latest", 
    response_model=PredictionResponse,
    tags=["Predictions"],
    summary="Lấy dữ liệu dự báo mới nhất"
)
def get_latest_predictions():
    """
    Lấy kết quả dự đoán giao thông mới nhất từ Redis.
    
    - Nếu Redis chưa sẵn sàng -> Trả về 503.
    - Nếu chưa có dữ liệu -> Trả về danh sách rỗng.
    """
    # Kiểm tra connection instance
    if not redis_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis client chưa được khởi tạo"
        )

    try:
        # Lấy dữ liệu thô từ Redis (dạng bytes hoặc string)
        # Lưu ý: Dùng def thường (không async) vì thư viện redis-py là đồng bộ
        raw_data = redis_manager.client.get(PREDICTION_KEY)
        
        if not raw_data:
            logger.warning("Không tìm thấy dữ liệu dự đoán trong Redis.")
            return PredictionResponse(timestamp="", data={})
        
        # Parse JSON
        parsed_data = json.loads(raw_data)
        
        # Validate data format
        data = parsed_data.get("data")
        timestamp = parsed_data.get("timestamp", "")
        
        # Check if data is in old format (list) and needs migration
        if isinstance(data, list):
            logger.error(
                "Redis contains predictions in OLD format (list). "
                "Please clear Redis and restart consumer to regenerate predictions."
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "Predictions in Redis are in outdated format. "
                    "Run: redis-cli DEL traffic:predictions:latest && restart consumer"
                )
            )
        
        # Check if data is in correct format (dict)
        if not isinstance(data, dict):
            logger.error(f"Invalid data format in Redis: {type(data)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Invalid prediction data format in Redis"
            )
        
        # Log timestamp để kiểm tra múi giờ
        logger.info(f"Returning predictions with timestamp: {timestamp}")
        
        # Return validated data
        return PredictionResponse(timestamp=timestamp, data=data)

    except json.JSONDecodeError:
        logger.error("Dữ liệu trong Redis không phải JSON hợp lệ.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Lỗi định dạng dữ liệu"
        )
    except HTTPException:
        # Re-raise HTTPException as-is
        raise
    except Exception as e:
        logger.error(f"Lỗi không xác định khi lấy dữ liệu: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi server nội bộ: {str(e)}"
        )

# Block này giúp bạn chạy file trực tiếp để test: python src/api/main.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)