"""Model management endpoints."""

import logging
import time
from fastapi import APIRouter, Request, HTTPException, Depends
from src.api.models import ModelsResponse, ModelInfo
from src.api.middleware.auth import optional_auth, require_permission

router = APIRouter()

@router.get("/models", response_model=ModelsResponse)
async def list_models(request: Request):
    """List available models - OpenAI compatible."""
    
    config_manager = getattr(request.app.state, 'config_manager', None)
    if not config_manager:
        raise HTTPException(status_code=500, detail="Configuration manager not available")
    
    try:
        available_models = config_manager.get_available_models()
        
        model_list = []
        for model_name, display_name in available_models.items():
            model_info = ModelInfo(
                id=model_name,
                owned_by="simple-llm-service",
                created=int(time.time())
            )
            model_list.append(model_info)
        
        return ModelsResponse(data=model_list)
        
    except Exception as e:
        logging.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list models")

@router.get("/models/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str, request: Request):
    """Get specific model information - OpenAI compatible."""
    
    config_manager = getattr(request.app.state, 'config_manager', None)
    if not config_manager:
        raise HTTPException(status_code=500, detail="Configuration manager not available")
    
    try:
        available_models = config_manager.get_available_models()
        
        if model_id not in available_models:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        return ModelInfo(
            id=model_id,
            owned_by="simple-llm-service",
            created=int(time.time())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting model {model_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model information")

@router.post("/models/{model_id}/load")
async def load_model(
    model_id: str, 
    request: Request,
    current_user: dict = Depends(require_permission("write"))
):
    """Load a specific model (requires write permission)."""
    
    model_manager = getattr(request.app.state, 'model_manager', None)
    config_manager = getattr(request.app.state, 'config_manager', None)
    conversation_manager = getattr(request.app.state, 'conversation_manager', None)
    
    if not all([model_manager, config_manager, conversation_manager]):
        raise HTTPException(status_code=500, detail="Service components not available")
    
    try:
        # Check if model exists
        available_models = config_manager.get_available_models()
        if model_id not in available_models:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        # Unload current model if different
        if model_manager.current_model_name != model_id:
            if model_manager.current_model:
                model_manager.unload_model()
            
            # Load new model
            if not model_manager.load_model(model_id):
                raise HTTPException(status_code=500, detail=f"Failed to load model {model_id}")
            
            # Update conversation manager
            conversation_manager.set_current_model(model_id)
        
        return {
            "status": "success",
            "model": model_id,
            "message": f"Model {model_id} loaded successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error loading model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model {model_id}")
