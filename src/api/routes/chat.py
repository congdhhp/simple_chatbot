"""Chat completions endpoints."""

import json
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

# Prometheus metrics (optional)
try:
    from prometheus_client import Counter, Histogram
    CHAT_LATENCY = Histogram('llm_chat_latency_seconds', 'Latency for chat completions')
    CHAT_FIRST_TOKEN = Histogram('llm_chat_time_to_first_token_seconds', 'Time to first token (simulated)')
    CHAT_REQUESTS = Counter('llm_chat_requests_total', 'Chat completion requests', ['model'])
    CHAT_TOKENS = Counter('llm_chat_tokens_total', 'Chat tokens processed', ['type', 'model'])  # label type values: 'prompt' | 'completion'
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    CHAT_LATENCY = None
    CHAT_FIRST_TOKEN = None
    CHAT_REQUESTS = None
    CHAT_TOKENS = None

from src.api.middleware.monitoring import metrics_collector

# Warn once if cost env vars not set (to prevent silent zero cost)
_COST_ENV_WARNED = False
def _warn_cost_env_once():
    global _COST_ENV_WARNED
    if _COST_ENV_WARNED:
        return
    import os, logging
    try:
        p = float(os.getenv('COST_PER_1K_PROMPT_TOKENS', '0'))
        c = float(os.getenv('COST_PER_1K_COMPLETION_TOKENS', '0'))
    except ValueError:
        p = c = 0.0
    if p <= 0 and c <= 0:
        logging.getLogger(__name__).warning("Cost env vars unset or zero; cost_usd will remain 0. Set COST_PER_1K_PROMPT_TOKENS and COST_PER_1K_COMPLETION_TOKENS to enable cost estimation.")
    _COST_ENV_WARNED = True

_warn_cost_env_once()

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
    """Create chat completion - OpenAI compatible with Wave 2 enhancements."""
    model_registry = getattr(request.app.state, 'model_registry', None)
    config_manager = getattr(request.app.state, 'config_manager', None)
    safety_pipeline = getattr(request.app.state, 'safety_pipeline', None)

    if not all([model_registry, config_manager]):
        raise HTTPException(status_code=500, detail="Service components not available")

    # Validate model
    available_models = config_manager.get_available_models()
    if request_data.model not in available_models:
        raise HTTPException(status_code=404, detail=f"Model {request_data.model} not found")

    # Get model from registry (handles loading/caching/eviction)
    model_manager = await model_registry.get_model(request_data.model)
    if not model_manager:
        raise HTTPException(status_code=500, detail=f"Failed to load model {request_data.model}")

    # Format messages and extract system prompt
    prompt = format_chat_messages(request_data.messages)
    system_prompt = None
    for message in reversed(request_data.messages):
        if message.role == "system":
            system_prompt = message.content
            break

    # Apply safety pipeline to input if available
    user_id = current_user.get('username') if current_user else None
    if safety_pipeline:
        prompt, safety_report = safety_pipeline.process_input(prompt, user_id)
        if safety_report.get("actions_taken") and "content_blocked" in safety_report["actions_taken"]:
            raise HTTPException(status_code=400, detail="Input content violates safety guidelines")

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
            import os  # Import trong function để tránh scope issues
            if PROMETHEUS_AVAILABLE and CHAT_REQUESTS:
                CHAT_REQUESTS.labels(model=request_data.model).inc()
            first_chunk_time = None
            heartbeat_interval = float(os.getenv('SSE_HEARTBEAT_INTERVAL', '5'))  # seconds
            last_heartbeat = 0.0
            try:
                # Initial heartbeat (Wave 2)
                yield 'data: {"event":"heartbeat"}' + "\n\n"
                last_heartbeat = time.time()
                for chunk in model_manager.stream_response(prompt, system_prompt, override_params):
                    now = time.time()
                    if now - last_heartbeat >= heartbeat_interval:
                        yield 'data: {"event":"heartbeat"}' + "\n\n"
                        last_heartbeat = now
                    if chunk.get("event") == "chunk":
                        if first_chunk_time is None:
                            first_chunk_time = time.time()
                            if PROMETHEUS_AVAILABLE and CHAT_FIRST_TOKEN:
                                CHAT_FIRST_TOKEN.observe(first_chunk_time - start_time)
                        
                        # Apply safety pipeline to output chunk if available
                        chunk_text = chunk['text']
                        if safety_pipeline:
                            chunk_text, _ = safety_pipeline.process_output(chunk_text, user_id)
                        
                        # Create proper JSON response
                        chunk_response = {
                            "id": "stream",
                            "object": "chat.completion.chunk",
                            "choices": [{
                                "delta": {"content": chunk_text},
                                "index": 0,
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(chunk_response)}\n\n"
                    elif chunk.get("event") == "error":
                        error_response = {"error": chunk['text']}
                        yield f"data: {json.dumps(error_response)}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    elif chunk.get("event") == "end":
                        usage_obj = {
                            "prompt_tokens": chunk['prompt_tokens'],
                            "completion_tokens": chunk['completion_tokens'],
                            "total_tokens": chunk['total_tokens']
                        }
                        if PROMETHEUS_AVAILABLE and CHAT_TOKENS:
                            CHAT_TOKENS.labels(type='prompt', model=request_data.model).inc(chunk['prompt_tokens'])
                            CHAT_TOKENS.labels(type='completion', model=request_data.model).inc(chunk['completion_tokens'])
                        ttfb_ms = (first_chunk_time - start_time) * 1000 if first_chunk_time else None
                        usage_obj["time_to_first_token_ms"] = ttfb_ms
                        duration = time.time() - start_time
                        tokens_per_second = chunk['completion_tokens'] / duration if duration > 0 else 0
                        
                        # Apply safety pipeline to final output if available
                        final_text = chunk['text']
                        if safety_pipeline:
                            final_text, safety_report = safety_pipeline.process_output(final_text, user_id)
                        
                        final_payload = {
                            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                            "object": "chat.completion",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", "content": final_text},
                                    "finish_reason": "stop"
                                }
                            ],
                            "usage": {**usage_obj, "tokens_per_second": round(tokens_per_second, 2)}
                        }
                        yield "data: " + json.dumps(final_payload) + "\n\n"
                        yield "data: [DONE]\n\n"
            except Exception as e:
                error_response = {"error": str(e)}
                yield f"data: {json.dumps(error_response)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                if PROMETHEUS_AVAILABLE and CHAT_LATENCY:
                    CHAT_LATENCY.observe(time.time() - start_time)
        return StreamingResponse(
            event_stream(), 
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )

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
    CHAT_TOKENS.labels(type='prompt', model=request_data.model).inc(prompt_tokens)
    CHAT_TOKENS.labels(type='completion', model=request_data.model).inc(completion_tokens)
    duration = time.time() - start_time
    
    # Calculate tokens per second
    tokens_per_second = completion_tokens / duration if duration > 0 else 0

    # Time to first token for non-stream path approximated as full latency (Wave 2 fallback)
    ttfb_ms = duration * 1000

    # Cost estimation (env configurable). Defaults 0 if unset.
    import os
    try:
        cost_prompt_rate = float(os.getenv('COST_PER_1K_PROMPT_TOKENS', '0'))
        cost_completion_rate = float(os.getenv('COST_PER_1K_COMPLETION_TOKENS', '0'))
    except ValueError:
        cost_prompt_rate = cost_completion_rate = 0.0
    cost_usd = ((prompt_tokens / 1000.0) * cost_prompt_rate) + ((completion_tokens / 1000.0) * cost_completion_rate)

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
            total_tokens=prompt_tokens + completion_tokens,
            time_to_first_token_ms=round(ttfb_ms, 2),
            latency_ms=round(duration * 1000, 2),
            tokens_per_second=round(tokens_per_second, 2),
            cost_usd=round(cost_usd, 6)
        ),
        system_fingerprint=f"simple-llm-v1.0-{request_data.model}"
    )

    CHAT_REQUESTS.labels(model=request_data.model).inc()
    CHAT_LATENCY.observe(time.time() - start_time)
    return chat_response
