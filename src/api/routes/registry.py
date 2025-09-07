"""Model registry management routes (Wave 2)."""

import logging
from fastapi import APIRouter, HTTPException, Depends, Request
from src.api.middleware.auth import require_permission

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/model-registry/stats")
async def get_registry_stats(
    request: Request,
    admin_user: dict = Depends(require_permission("admin"))
):
    """Get model registry statistics - admin only."""
    model_registry = getattr(request.app.state, 'model_registry', None)
    
    if not model_registry:
        raise HTTPException(status_code=500, detail="Model registry not available")
    
    stats = model_registry.get_registry_stats()
    return {
        "status": "success",
        "data": stats
    }


@router.post("/model-registry/preload")
async def preload_model(
    request: Request,
    model_name: str,
    admin_user: dict = Depends(require_permission("admin"))
):
    """Preload a specific model - admin only."""
    model_registry = getattr(request.app.state, 'model_registry', None)
    config_manager = getattr(request.app.state, 'config_manager', None)
    
    if not all([model_registry, config_manager]):
        raise HTTPException(status_code=500, detail="Service components not available")
    
    # Validate model exists
    available_models = config_manager.get_available_models()
    if model_name not in available_models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    # Preload model
    model_manager = await model_registry.get_model(model_name)
    if not model_manager:
        raise HTTPException(status_code=500, detail=f"Failed to preload model {model_name}")
    
    return {
        "status": "success",
        "message": f"Model {model_name} preloaded successfully",
        "registry_stats": model_registry.get_registry_stats()
    }


@router.delete("/model-registry/evict/{model_name}")
async def evict_model(
    model_name: str,
    request: Request,
    admin_user: dict = Depends(require_permission("admin"))
):
    """Evict a specific model from registry - admin only."""
    model_registry = getattr(request.app.state, 'model_registry', None)
    
    if not model_registry:
        raise HTTPException(status_code=500, detail="Model registry not available")
    
    # Check if model is loaded
    if model_name not in model_registry.loaded_models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not currently loaded")
    
    # Evict model
    await model_registry._evict_model(model_name)
    
    return {
        "status": "success",
        "message": f"Model {model_name} evicted successfully",
        "registry_stats": model_registry.get_registry_stats()
    }


@router.post("/model-registry/clear")
async def clear_registry(
    request: Request,
    admin_user: dict = Depends(require_permission("admin"))
):
    """Clear all models from registry - admin only."""
    model_registry = getattr(request.app.state, 'model_registry', None)
    
    if not model_registry:
        raise HTTPException(status_code=500, detail="Model registry not available")
    
    await model_registry.clear_registry()
    
    return {
        "status": "success",
        "message": "Model registry cleared successfully",
        "registry_stats": model_registry.get_registry_stats()
    }
