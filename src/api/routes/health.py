"""Health check endpoints."""

import logging
import time
import torch
try:
    import psutil
except ImportError:
    psutil = None
from fastapi import APIRouter, Request
from src.api.models import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Health check endpoint."""
    
    model_manager = getattr(request.app.state, 'model_manager', None)
    
    # Check model status
    models_loaded = {}
    if model_manager:
        if model_manager.current_model_name:
            models_loaded[model_manager.current_model_name] = True
    
    # Check CUDA status
    cuda_available = torch.cuda.is_available()
    
    status = "healthy" if model_manager and model_manager.current_model else "degraded"
    
    return HealthResponse(
        status=status,
        models_loaded=models_loaded,
        version="1.0.0"
    )

@router.get("/health/ready")
async def readiness_check(request: Request):
    """Readiness check for Kubernetes."""
    model_manager = getattr(request.app.state, 'model_manager', None)
    
    if not model_manager or not model_manager.current_model:
        return {"status": "not_ready", "reason": "no_model_loaded"}
    
    return {"status": "ready"}

@router.get("/health/live")
async def liveness_check():
    """Liveness check for Kubernetes."""
    return {"status": "alive"}

@router.get("/health/detailed")
async def detailed_health_check(request: Request):
    """Detailed health check with system metrics."""
    try:
        model_manager = getattr(request.app.state, 'model_manager', None)
        
        # System metrics (if psutil available)
        system_info = {}
        if psutil:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)  # Shorter interval
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                system_info = {
                    "cpu_percent": cpu_percent,
                    "memory": {
                        "total": memory.total,
                        "available": memory.available,
                        "percent": memory.percent,
                        "used": memory.used
                    },
                    "disk": {
                        "total": disk.total,
                        "free": disk.free,
                        "used": disk.used,
                        "percent": (disk.used / disk.total) * 100
                    }
                }
            except Exception as e:
                system_info = {"error": f"Failed to get system info: {str(e)}"}
        else:
            system_info = {"error": "psutil not available"}
        
        # GPU metrics
        gpu_info = {}
        try:
            if torch.cuda.is_available():
                gpu_info = {
                    "cuda_available": True,
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name(0),
                    "memory_allocated": torch.cuda.memory_allocated(0),
                    "memory_reserved": torch.cuda.memory_reserved(0),
                    "memory_total": torch.cuda.get_device_properties(0).total_memory
                }
            else:
                gpu_info = {"cuda_available": False}
        except Exception as e:
            gpu_info = {"error": f"Failed to get GPU info: {str(e)}"}
        
        # Model status
        model_status = {}
        try:
            if model_manager:
                model_status = {
                    "current_model": model_manager.current_model_name,
                    "model_loaded": model_manager.current_model is not None,
                    "tokenizer_loaded": model_manager.current_tokenizer is not None
                }
            else:
                model_status = {"error": "Model manager not available"}
        except Exception as e:
            model_status = {"error": f"Failed to get model status: {str(e)}"}
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "system": system_info,
            "gpu": gpu_info,
            "model": model_status
        }
        
    except Exception as e:
        logging.error(f"Detailed health check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }
