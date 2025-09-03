"""Model management endpoints."""

import logging
from fastapi import APIRouter, Request, HTTPException
from ..models import ModelsResponse, ModelInfo, HealthResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/v1/models", response_model=ModelsResponse)
async def list_models(request: Request):
    """List available embedding models.
    
    Returns a list of all available models that can be used for embedding generation.
    """
    try:
        config_manager = getattr(request.app.state, 'config_manager', None)
        if not config_manager:
            raise HTTPException(status_code=500, detail="Configuration manager not available")
        
        # Get available models
        available_models = config_manager.get_available_models()
        
        # Create model info objects
        model_data = []
        for model_id, display_name in available_models.items():
            model_info = ModelInfo(
                id=model_id,
                object="model",
                created=0,  # We don't track creation time
                owned_by="organization-owner"
            )
            model_data.append(model_info)
        
        return ModelsResponse(
            object="list",
            data=model_data
        )
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@router.get("/v1/models/{model_id}")
async def get_model_info(model_id: str, request: Request):
    """Get detailed information about a specific model.
    
    Args:
        model_id: The ID of the model to get information about
    """
    try:
        model_manager = getattr(request.app.state, 'model_manager', None)
        if not model_manager:
            raise HTTPException(status_code=500, detail="Model manager not available")
        
        # Parse model name to handle suffixes
        config_manager = getattr(request.app.state, 'config_manager', None)
        base_model_name, _ = config_manager.parse_model_name(model_id)
        
        # Get model info
        model_info = model_manager.get_model_info(base_model_name)
        
        if 'error' in model_info:
            raise HTTPException(status_code=404, detail=model_info['error'])
        
        # Return detailed model information
        return {
            "id": model_id,
            "object": "model",
            "created": 0,
            "owned_by": "organization-owner",
            "details": {
                "display_name": model_info.get('display_name'),
                "family": model_info.get('family'),
                "is_loaded": model_info.get('is_loaded', False),
                "max_seq_length": model_info.get('max_seq_length'),
                "embedding_dimension": model_info.get('embedding_dimension'),
                "supports_input_type": model_info.get('supports_input_type', False),
                "supported_embedding_types": model_info.get('supported_embedding_types', []),
                "supported_modalities": model_info.get('supported_modalities', []),
                "memory_usage": model_info.get('memory_usage', {})
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info for {model_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@router.post("/v1/models/{model_id}/load", response_model=HealthResponse)
async def load_model(model_id: str, request: Request):
    """Load a specific model.
    
    Args:
        model_id: The ID of the model to load
    """
    try:
        model_manager = getattr(request.app.state, 'model_manager', None)
        if not model_manager:
            raise HTTPException(status_code=500, detail="Model manager not available")
        
        # Parse model name to handle suffixes
        config_manager = getattr(request.app.state, 'config_manager', None)
        base_model_name, _ = config_manager.parse_model_name(model_id)
        
        # Check if model exists
        available_models = config_manager.get_available_models()
        if base_model_name not in available_models:
            raise HTTPException(
                status_code=404, 
                detail=f"Model {base_model_name} not found. Available: {list(available_models.keys())}"
            )
        
        # Load the model
        success = model_manager.load_model(base_model_name)
        
        if success:
            return HealthResponse(
                message=f"Model {model_id} loaded successfully.",
                status="loaded"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model {model_id}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading model {model_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@router.post("/v1/models/{model_id}/unload", response_model=HealthResponse)
async def unload_model(model_id: str, request: Request):
    """Unload a specific model.
    
    Args:
        model_id: The ID of the model to unload
    """
    try:
        model_manager = getattr(request.app.state, 'model_manager', None)
        if not model_manager:
            raise HTTPException(status_code=500, detail="Model manager not available")
        
        # Parse model name to handle suffixes
        config_manager = getattr(request.app.state, 'config_manager', None)
        base_model_name, _ = config_manager.parse_model_name(model_id)
        
        # Check if model is currently loaded
        if not model_manager.is_model_loaded(base_model_name):
            return HealthResponse(
                message=f"Model {model_id} is not currently loaded.",
                status="not_loaded"
            )
        
        # Unload the model
        model_manager.unload_model()
        
        return HealthResponse(
            message=f"Model {model_id} unloaded successfully.",
            status="unloaded"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unloading model {model_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(e)}")


@router.get("/v1/models/{model_id}/status")
async def get_model_status(model_id: str, request: Request):
    """Get the loading status of a specific model.
    
    Args:
        model_id: The ID of the model to check
    """
    try:
        model_manager = getattr(request.app.state, 'model_manager', None)
        if not model_manager:
            raise HTTPException(status_code=500, detail="Model manager not available")
        
        # Parse model name to handle suffixes
        config_manager = getattr(request.app.state, 'config_manager', None)
        base_model_name, _ = config_manager.parse_model_name(model_id)
        
        # Get model status
        is_loaded = model_manager.is_model_loaded(base_model_name)
        current_model = model_manager.get_current_model_name()
        
        status = {
            "model_id": model_id,
            "base_model_name": base_model_name,
            "is_loaded": is_loaded,
            "is_current": current_model == base_model_name,
            "current_model": current_model
        }
        
        if is_loaded:
            # Add memory usage info
            model_info = model_manager.get_model_info(base_model_name)
            status["memory_usage"] = model_info.get("memory_usage", {})
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model status for {model_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")
