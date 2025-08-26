"""Health check endpoints."""

import logging
import torch
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
