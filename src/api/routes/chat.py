"""Chat completions endpoints."""

import logging
import uuid
import time
import os
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
from src.api.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatChoice,
    ChatMessage,
    Usage
)
from src.api.middleware.auth import optional_auth, require_auth
from prometheus_client import Counter, Histogram, Gauge
import time

CHAT_LATENCY = Histogram('llm_chat_latency_seconds', 'Latency for chat completions')
CHAT_FIRST_TOKEN = Histogram('llm_chat_time_to_first_token_seconds', 'Time to first token (simulated)')
CHAT_REQUESTS = Counter('llm_chat_requests_total', 'Chat completion requests', ['model'])
from src.api.middleware.monitoring import metrics_collector

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
async def create_chat_completion(
    request_data: ChatCompletionRequest, 
    request: Request,
    current_user: dict = Depends(optional_auth)
):
    """Create chat completion - OpenAI compatible."""
    model_manager = getattr(request.app.state, 'model_manager', None)
    config_manager = getattr(request.app.state, 'config_manager', None)

    if not all([model_manager, config_manager]):
        raise HTTPException(status_code=500, detail="Service components not available")

    # Validate model
    available_models = config_manager.get_available_models()
    if request_data.model not in available_models:
        raise HTTPException(status_code=404, detail=f"Model {request_data.model} not found")

    # Load if needed
    if model_manager.current_model_name != request_data.model:
        if not model_manager.load_model(request_data.model):
            raise HTTPException(status_code=500, detail=f"Failed to load model {request_data.model}")

    if not model_manager.current_model:
        raise HTTPException(status_code=500, detail="No model loaded")

    prompt = format_chat_messages(request_data.messages)

    system_prompt = None
    for message in reversed(request_data.messages):
        if message.role == "system":
            system_prompt = message.content
            break

    override_params = {
        'max_new_tokens': request_data.max_tokens,
        'temperature': request_data.temperature,
        'top_p': request_data.top_p,
        'top_k': request_data.top_k
    }

    start_time = time.time()

    # Streaming mode
    if request_data.stream:
        def event_stream():
            CHAT_REQUESTS.labels(model=request_data.model).inc()
            first_chunk_time = None
            try:
                for chunk in model_manager.stream_response(prompt, system_prompt, override_params):
                    if chunk.get("event") == "chunk":
                        if first_chunk_time is None:
                            first_chunk_time = time.time()
                            CHAT_FIRST_TOKEN.observe(first_chunk_time - start_time)
                        yield f"data: {{\"id\": \"stream\", \"object\": \"chat.completion.chunk\", \"choices\":[{{\"delta\":{{\"content\":{chunk['text']!r}}}, \"index\":0, \"finish_reason\":null}}]}}\n\n"
                    elif chunk.get("event") == "error":
                        yield f"data: {{\"error\": {chunk['text']!r}}}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    elif chunk.get("event") == "end":
                        usage_obj = {
                            "prompt_tokens": chunk['prompt_tokens'],
                            "completion_tokens": chunk['completion_tokens'],
                            "total_tokens": chunk['total_tokens']
                        }
                        final_payload = {
                            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                            "object": "chat.completion",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", "content": chunk['text']},
                                    "finish_reason": "stop"
                                }
                            ],
                            "usage": usage_obj
                        }
                        import json
                        yield "data: " + json.dumps(final_payload) + "\n\n"
                        yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                CHAT_LATENCY.observe(time.time() - start_time)
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming path
    gen = model_manager.generate_response(prompt, system_prompt, override_params)
    response_text = gen['text']
    if not response_text or response_text.startswith("Error generating"):
        raise HTTPException(status_code=500, detail="Failed to generate response")

    metrics_collector.record_model_usage(request_data.model)
    if current_user:
        request.state.user = current_user

    prompt_tokens = gen['prompt_tokens']
    completion_tokens = gen['completion_tokens']

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

    CHAT_REQUESTS.labels(model=request_data.model).inc()
    CHAT_LATENCY.observe(time.time() - start_time)
    return chat_response
