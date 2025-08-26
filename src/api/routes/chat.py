"""Chat completions endpoints."""

import logging
import uuid
import time
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from src.api.models import (
    ChatCompletionRequest, 
    ChatCompletionResponse, 
    ChatChoice, 
    ChatMessage,
    Usage
)

router = APIRouter()

def format_chat_messages(messages: list) -> str:
    """Convert chat messages to a single prompt string."""
    formatted_parts = []
    
    for message in messages:
        role = message.role
        content = message.content
        
        if role == "system":
            formatted_parts.append(f"System: {content}")
        elif role == "user":
            formatted_parts.append(f"User: {content}")
        elif role == "assistant":
            formatted_parts.append(f"Assistant: {content}")
    
    return "\n".join(formatted_parts)

def estimate_tokens(text: str) -> int:
    """Rough token estimation (4 characters ≈ 1 token)."""
    return len(text) // 4

@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request_data: ChatCompletionRequest, request: Request):
    """Create chat completion - OpenAI compatible."""
    
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
        
        # Convert messages to prompt
        prompt = format_chat_messages(request_data.messages)
        
        # Update generation config with request parameters
        generation_config = model_manager.generation_config
        if generation_config:
            if request_data.max_tokens:
                generation_config.max_new_tokens = request_data.max_tokens
            if request_data.temperature is not None:
                generation_config.temperature = request_data.temperature
            if request_data.top_p is not None:
                generation_config.top_p = request_data.top_p
            if request_data.top_k is not None:
                generation_config.top_k = request_data.top_k
        
        # Get system prompt from last system message or model config
        system_prompt = None
        for message in reversed(request_data.messages):
            if message.role == "system":
                system_prompt = message.content
                break
        
        # Generate response
        response_text = model_manager.generate_response(prompt, system_prompt)
        
        if not response_text:
            raise HTTPException(status_code=500, detail="Failed to generate response")
        
        # Estimate token usage
        prompt_tokens = estimate_tokens(prompt)
        completion_tokens = estimate_tokens(response_text)
        
        # Create response
        chat_response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            model=request_data.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=response_text
                    ),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
        
        return chat_response
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in chat completion: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
