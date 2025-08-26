"""Text completions endpoints."""

import logging
import uuid
import os
from fastapi import APIRouter, Request, HTTPException, Depends
from src.api.models import (
    CompletionRequest, 
    CompletionResponse, 
    CompletionChoice,
    Usage
)
from src.api.middleware.auth import optional_auth
from src.api.middleware.monitoring import metrics_collector

router = APIRouter()

def estimate_tokens(text: str) -> int:
    """Rough token estimation (4 characters ≈ 1 token)."""
    return len(text) // 4

@router.post("/completions", response_model=CompletionResponse)
async def create_completion(
    request_data: CompletionRequest, 
    request: Request,
    current_user: dict = Depends(optional_auth)
):
    """Create text completion - OpenAI compatible."""
    
    model_manager = getattr(request.app.state, 'model_manager', None)
    config_manager = getattr(request.app.state, 'config_manager', None)
    
    if not all([model_manager, config_manager]):
        raise HTTPException(status_code=500, detail="Service components not available")
    
    try:
        # Check if requested model is available
        available_models = config_manager.get_available_models()
        if request_data.model not in available_models:
            raise HTTPException(status_code=404, detail=f"Model {request_data.model} not found")
        
        # Load model if different from current
        if model_manager.current_model_name != request_data.model:
            if not model_manager.load_model(request_data.model):
                raise HTTPException(status_code=500, detail=f"Failed to load model {request_data.model}")
        
        # Check if model is loaded
        if not model_manager.current_model:
            raise HTTPException(status_code=500, detail="No model loaded")
        
        # Handle single prompt or list of prompts
        if isinstance(request_data.prompt, str):
            prompts = [request_data.prompt]
        else:
            prompts = request_data.prompt
        
        # Update generation config with request parameters
        generation_config = model_manager.generation_config
        if generation_config:
            if request_data.max_tokens:
                generation_config.max_new_tokens = request_data.max_tokens
            if request_data.temperature is not None:
                generation_config.temperature = request_data.temperature
            if request_data.top_p is not None:
                generation_config.top_p = request_data.top_p
        
        # Generate completions
        choices = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        for i, prompt in enumerate(prompts):
            # Generate response
            response_text = model_manager.generate_response(prompt)
            
            if not response_text:
                raise HTTPException(status_code=500, detail=f"Failed to generate completion for prompt {i}")
            
            # Record model usage for metrics
            metrics_collector.record_model_usage(request_data.model)
            
            # Store user info for metrics
            if current_user:
                request.state.user = current_user
            
            # Estimate tokens
            prompt_tokens = estimate_tokens(prompt)
            completion_tokens = estimate_tokens(response_text)
            
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            
            # Create choice
            choice = CompletionChoice(
                index=i,
                text=response_text,
                finish_reason="stop"
            )
            choices.append(choice)
        
        # Create response
        completion_response = CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:8]}",
            model=request_data.model,
            choices=choices,
            usage=Usage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens
            )
        )
        
        return completion_response
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in text completion: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
