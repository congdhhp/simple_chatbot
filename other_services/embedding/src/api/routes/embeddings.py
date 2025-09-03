"""Embedding generation endpoints."""

import logging
from typing import Union, List
from fastapi import APIRouter, Request, HTTPException
from ..models import (
    EmbeddingRequest, 
    EmbeddingResponse, 
    ErrorResponse,
    validate_model_name,
    validate_input_length,
    validate_batch_size,
    create_error_response
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest, http_request: Request):
    """Generate embeddings for input text(s).
    
    This endpoint is compatible with OpenAI's embeddings API and NVIDIA NIM specification.
    Supports various input types, embedding types, and modalities.
    
    Args:
        request: Embedding request with input texts and parameters
        http_request: FastAPI request object for accessing app state
        
    Returns:
        EmbeddingResponse with generated embeddings and metadata
    """
    try:
        # Get service components
        embedding_manager = getattr(http_request.app.state, 'embedding_manager', None)
        config_manager = getattr(http_request.app.state, 'config_manager', None)
        
        if not embedding_manager or not config_manager:
            raise HTTPException(
                status_code=500, 
                detail="Service not properly initialized"
            )
        
        # Validate request parameters
        await _validate_request(request, config_manager)
        
        # Normalize input to list format
        if isinstance(request.input, str):
            inputs = [request.input]
        else:
            inputs = request.input
        
        # Parse model name for input_type suffix
        base_model_name, suffix_input_type = validate_model_name(request.model)
        
        # Determine effective input_type
        effective_input_type = request.input_type or suffix_input_type
        
        # Log request
        logger.info(
            f"Embedding request: model={request.model}, "
            f"inputs={len(inputs)}, input_type={effective_input_type}, "
            f"embedding_type={request.embedding_type}"
        )
        
        # Generate embeddings
        try:
            response = await embedding_manager.generate_embeddings(
                inputs=inputs,
                model=request.model,
                input_type=effective_input_type,
                modality=request.modality,
                embedding_type=request.embedding_type,
                dimensions=request.dimensions,
                normalize=request.normalize
            )
            
            # Convert to API response format
            return EmbeddingResponse(**response)
            
        except ValueError as e:
            # Handle validation errors
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            # Handle runtime errors (model loading, etc.)
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error in embedding generation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in create_embeddings endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


async def _validate_request(request: EmbeddingRequest, config_manager) -> None:
    """Validate embedding request parameters.
    
    Args:
        request: Embedding request to validate
        config_manager: Configuration manager for validation
        
    Raises:
        HTTPException: If validation fails
    """
    try:
        # Validate model exists
        base_model_name, _ = validate_model_name(request.model)
        available_models = config_manager.get_available_models()
        
        if base_model_name not in available_models:
            available = ', '.join(available_models.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Model '{base_model_name}' not found. Available models: {available}"
            )
        
        # Validate input length
        max_input_length = config_manager.get_max_input_length()
        validate_input_length(request.input, max_input_length)
        
        # Validate batch size
        max_batch_size = config_manager.get_max_batch_size()
        validate_batch_size(request.input, max_batch_size)
        
        # Validate embedding type for model
        supported_types = config_manager.get_supported_embedding_types(base_model_name)
        if request.embedding_type not in supported_types:
            supported = ', '.join(supported_types)
            raise HTTPException(
                status_code=400,
                detail=f"Embedding type '{request.embedding_type}' not supported by model '{base_model_name}'. Supported: {supported}"
            )
        
        # Validate dimensions for model
        if request.dimensions:
            supported_dims = config_manager.get_supported_dimensions(base_model_name)
            if supported_dims and request.dimensions not in supported_dims:
                supported = ', '.join(map(str, supported_dims))
                raise HTTPException(
                    status_code=400,
                    detail=f"Dimensions {request.dimensions} not supported by model '{base_model_name}'. Supported: {supported}"
                )
        
        # Validate modality for model
        if request.modality:
            modalities = [request.modality] if isinstance(request.modality, str) else request.modality
            supported_modalities = config_manager.get_supported_modalities(base_model_name)
            
            for modality in modalities:
                if modality not in supported_modalities:
                    supported = ', '.join(supported_modalities)
                    raise HTTPException(
                        status_code=400,
                        detail=f"Modality '{modality}' not supported by model '{base_model_name}'. Supported: {supported}"
                    )
        
        # Validate input_type for model
        supports_input_type = config_manager.supports_input_type(base_model_name)
        base_model_name_parsed, suffix_input_type = validate_model_name(request.model)
        effective_input_type = request.input_type or suffix_input_type
        
        if supports_input_type and effective_input_type is None:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{base_model_name}' requires input_type parameter or model name suffix (-query/-passage)"
            )
        
        if not supports_input_type and effective_input_type is not None:
            logger.warning(f"Model '{base_model_name}' ignores input_type parameter")
        
        # Validate parameter combinations
        if request.dimensions and request.embedding_type != 'float':
            raise HTTPException(
                status_code=400,
                detail="The 'dimensions' parameter cannot be used with compressed embedding types. Use either 'dimensions' or 'embedding_type', not both."
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Request validation failed: {str(e)}")


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings_alt(request: EmbeddingRequest, http_request: Request):
    """Alternative endpoint for embedding generation (without v1 prefix).
    
    This provides backward compatibility with some clients.
    """
    return await create_embeddings(request, http_request)


@router.get("/v1/embeddings/models")
async def list_embedding_models(http_request: Request):
    """List models specifically for embedding generation.
    
    Returns detailed information about embedding capabilities.
    """
    try:
        config_manager = getattr(http_request.app.state, 'config_manager', None)
        model_manager = getattr(http_request.app.state, 'model_manager', None)
        
        if not config_manager or not model_manager:
            raise HTTPException(status_code=500, detail="Service not properly initialized")
        
        available_models = config_manager.get_available_models()
        detailed_models = []
        
        for model_name in available_models.keys():
            try:
                model_info = model_manager.get_model_info(model_name)
                
                # Add embedding-specific information
                detailed_info = {
                    "id": model_name,
                    "display_name": model_info.get("display_name", model_name),
                    "family": model_info.get("family"),
                    "embedding_dimension": model_info.get("embedding_dimension"),
                    "max_seq_length": model_info.get("max_seq_length"),
                    "supports_input_type": model_info.get("supports_input_type", False),
                    "supported_embedding_types": model_info.get("supported_embedding_types", []),
                    "supported_modalities": model_info.get("supported_modalities", []),
                    "supports_dimensions": config_manager.get_supported_dimensions(model_name),
                    "is_loaded": model_info.get("is_loaded", False)
                }
                
                detailed_models.append(detailed_info)
                
            except Exception as e:
                logger.warning(f"Failed to get info for model {model_name}: {str(e)}")
                continue
        
        return {
            "object": "list",
            "data": detailed_models,
            "total": len(detailed_models)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing embedding models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list embedding models: {str(e)}")
