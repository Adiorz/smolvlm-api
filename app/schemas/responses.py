from typing import Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    device: str
    cuda_available: bool
    model_loaded: bool
    loading_status: str
    loading_progress: float
    download_info: Optional[dict] = None
    gpu_info: Optional[dict] = None
