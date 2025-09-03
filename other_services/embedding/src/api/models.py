"""Pydantic models for the embedding API."""

from typing import List, Optional, Union, Any, Dict
from pydantic import BaseModel, Field, validator
import re


class EmbeddingRequest(BaseModel):
    """Request model for embedding generation."""
    
    input: Union[str, List[str]] = Field(
        ...,
        description="Input text(s) to embed. Can be a single string or list of strings.",
        example=["Hello world", "This is a test"]
    )
    
    model: str = Field(
        ...,
        description="Model to use for embedding generation",
        example="e5-large-v2"
    )
    
    input_type: Optional[str] = Field(
        None,
        description="Type of input for retrieval models",
        pattern="^(query|passage)$",
        example="query"
    )
    
    modality: Optional[Union[str, List[str]]] = Field(
        None,
        description="Modality of input data",
        example="text"
    )
    
    embedding_type: Optional[str] = Field(
        "float",
        description="Type of embeddings to return",
        pattern="^(float|int8|uint8|binary|ubinary)$",
        example="float"
    )
    
    dimensions: Optional[int] = Field(
        None,
        description="Number of dimensions for output embeddings (for supported models)",
        gt=0,
        example=512
    )
    
    normalize: Optional[bool] = Field(
        True,
        description="Whether to normalize embeddings"
    )
    
    @validator('input')
    def validate_input(cls, v):
        """Validate input field."""
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("Input text cannot be empty")
        elif isinstance(v, list):
            if not v:
                raise ValueError("Input list cannot be empty")
            for i, item in enumerate(v):
                if not isinstance(item, str):
                    raise ValueError(f"Input[{i}] must be a string")
                if not item.strip():
                    raise ValueError(f"Input[{i}] cannot be empty")
        else:
            raise ValueError("Input must be a string or list of strings")
        
        return v
    
    @validator('modality')
    def validate_modality(cls, v):
        """Validate modality field."""
        valid_modalities = {'text', 'image', 'text_image'}
        
        if isinstance(v, str):
            if v not in valid_modalities:
                raise ValueError(f"Invalid modality '{v}'. Must be one of: {', '.join(valid_modalities)}")
        elif isinstance(v, list):
            for modality in v:
                if modality not in valid_modalities:
                    raise ValueError(f"Invalid modality '{modality}'. Must be one of: {', '.join(valid_modalities)}")
        
        return v


class EmbeddingObject(BaseModel):
    """Individual embedding object."""
    
    index: int = Field(
        ...,
        description="Index of the embedding in the input list"
    )
    
    embedding: List[Union[float, int]] = Field(
        ...,
        description="The embedding vector"
    )
    
    object: str = Field(
        "embedding",
        description="Object type"
    )


class EmbeddingUsage(BaseModel):
    """Usage statistics for embedding request."""
    
    prompt_tokens: int = Field(
        ...,
        description="Number of tokens in the input"
    )
    
    total_tokens: int = Field(
        ...,
        description="Total number of tokens processed"
    )


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""
    
    object: str = Field(
        "list",
        description="Object type"
    )
    
    data: List[EmbeddingObject] = Field(
        ...,
        description="List of embedding objects"
    )
    
    model: str = Field(
        ...,
        description="Model used for generation"
    )
    
    usage: EmbeddingUsage = Field(
        ...,
        description="Usage statistics"
    )


class ModelInfo(BaseModel):
    """Information about a model."""
    
    id: str = Field(
        ...,
        description="Model identifier"
    )
    
    object: str = Field(
        "model",
        description="Object type"
    )
    
    created: int = Field(
        0,
        description="Creation timestamp"
    )
    
    owned_by: str = Field(
        "organization-owner",
        description="Owner of the model"
    )


class ModelsResponse(BaseModel):
    """Response model for listing models."""
    
    object: str = Field(
        "list",
        description="Object type"
    )
    
    data: List[ModelInfo] = Field(
        ...,
        description="List of available models"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    
    object: str = Field(
        "health-response",
        description="Object type"
    )
    
    message: str = Field(
        ...,
        description="Health status message"
    )
    
    status: Optional[str] = Field(
        None,
        description="Detailed status"
    )


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: Dict[str, Any] = Field(
        ...,
        description="Error details"
    )


class ErrorDetail(BaseModel):
    """Detailed error information."""
    
    message: str = Field(
        ...,
        description="Error message"
    )
    
    type: str = Field(
        ...,
        description="Error type"
    )
    
    param: Optional[str] = Field(
        None,
        description="Parameter that caused the error"
    )
    
    code: Optional[str] = Field(
        None,
        description="Error code"
    )


# Request validation utilities
def validate_model_name(model_name: str) -> tuple[str, Optional[str]]:
    """Validate and parse model name.
    
    Args:
        model_name: Model name to validate
        
    Returns:
        Tuple of (base_model_name, input_type_suffix)
        
    Raises:
        ValueError: If model name is invalid
    """
    if not model_name or not isinstance(model_name, str):
        raise ValueError("Model name must be a non-empty string")
    
    # Parse model name for input type suffixes
    if model_name.endswith('-query'):
        return model_name[:-6], 'query'
    elif model_name.endswith('-passage'):
        return model_name[:-8], 'passage'
    else:
        return model_name, None


def validate_input_length(inputs: Union[str, List[str]], max_length: int) -> None:
    """Validate input length constraints.
    
    Args:
        inputs: Input data to validate
        max_length: Maximum allowed length
        
    Raises:
        ValueError: If inputs exceed length limits
    """
    if isinstance(inputs, str):
        inputs = [inputs]
    
    total_length = sum(len(text) for text in inputs)
    
    if total_length > max_length:
        raise ValueError(f"Total input length {total_length} exceeds maximum {max_length}")


def validate_batch_size(inputs: Union[str, List[str]], max_batch_size: int) -> None:
    """Validate batch size constraints.
    
    Args:
        inputs: Input data to validate  
        max_batch_size: Maximum allowed batch size
        
    Raises:
        ValueError: If batch size exceeds limits
    """
    if isinstance(inputs, str):
        batch_size = 1
    else:
        batch_size = len(inputs)
    
    if batch_size > max_batch_size:
        raise ValueError(f"Batch size {batch_size} exceeds maximum {max_batch_size}")


def create_error_response(
    message: str, 
    error_type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None
) -> ErrorResponse:
    """Create a standardized error response.
    
    Args:
        message: Error message
        error_type: Type of error
        param: Parameter that caused the error
        code: Error code
        
    Returns:
        ErrorResponse object
    """
    return ErrorResponse(
        error={
            "message": message,
            "type": error_type,
            "param": param,
            "code": code
        }
    )
