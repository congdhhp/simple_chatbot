"""Text completions endpoints."""

import logging
import uuid
import os
import json
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
from src.api.models import (
    CompletionRequest, 
    CompletionResponse, 
    CompletionChoice,
    Usage
)
from src.api.middleware.auth import optional_auth
from prometheus_client import Counter, Histogram
import time
from src.api.middleware.monitoring import metrics_collector

# Warn once if cost env vars not set
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

COMPL_LATENCY = Histogram('llm_completion_latency_seconds', 'Latency for text completions')
COMPL_REQUESTS = Counter('llm_completion_requests_total', 'Text completion requests', ['model'])
COMPL_TOKENS = Counter('llm_completion_tokens_total', 'Completion tokens processed', ['type', 'model'])  # label type values: 'prompt' | 'completion'

def _override_params(req: CompletionRequest):
    return {
        'max_new_tokens': req.max_tokens,
        'temperature': req.temperature,
        'top_p': req.top_p
    }

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
    
    start_time = time.time()

    # Check if requested model is available
    available_models = config_manager.get_available_models()
    if request_data.model not in available_models:
        raise HTTPException(status_code=404, detail=f"Model {request_data.model} not found")

    # Load model if different from current (async)
    if model_manager.current_model_name != request_data.model:
        loaded = await model_manager.async_load_model(request_data.model)
        if not loaded:
            raise HTTPException(status_code=500, detail=f"Failed to load model {request_data.model}")

    if not model_manager.current_model:
        raise HTTPException(status_code=500, detail="No model loaded")

    # Normalize prompts to list
    if isinstance(request_data.prompt, str):
        prompts = [request_data.prompt]
    else:
        prompts = request_data.prompt

    params = _override_params(request_data)

    # Streaming path
    if request_data.stream:
        if len(prompts) != 1:
            raise HTTPException(status_code=400, detail="Streaming only supports single prompt")
        prompt = prompts[0]

        def event_stream():
            COMPL_REQUESTS.labels(model=request_data.model).inc()
            first_chunk_time = None
            heartbeat_interval = float(os.getenv('SSE_HEARTBEAT_INTERVAL', '5'))  # seconds
            last_heartbeat = 0.0
            try:
                # Initial heartbeat
                yield 'data: {"event":"heartbeat"}' + "\n\n"
                last_heartbeat = time.time()
                for chunk in model_manager.stream_response(prompt, None, params):
                    now = time.time()
                    if now - last_heartbeat >= heartbeat_interval:
                        yield 'data: {"event":"heartbeat"}' + "\n\n"
                        last_heartbeat = now
                    if chunk.get('event') == 'chunk':
                        if first_chunk_time is None:
                            first_chunk_time = time.time()
                        payload = {
                            "id": f"cmpl-{uuid.uuid4().hex[:8]}",
                            "object": "text_completion.chunk",
                            "choices": [
                                {"index": 0, "text": chunk['text'], "finish_reason": None}
                            ]
                        }
                        yield "data: " + json.dumps(payload) + "\n\n"
                    elif chunk.get('event') == 'error':
                        yield f"data: {{\"error\": {chunk['text']!r}}}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    elif chunk.get('event') == 'end':
                        duration = time.time() - start_time
                        tokens_per_second = chunk['completion_tokens'] / duration if duration > 0 else 0
                        usage_obj = {
                            "prompt_tokens": chunk['prompt_tokens'],
                            "completion_tokens": chunk['completion_tokens'],
                            "total_tokens": chunk['total_tokens']
                        }
                        COMPL_TOKENS.labels(type='prompt', model=request_data.model).inc(chunk['prompt_tokens'])
                        COMPL_TOKENS.labels(type='completion', model=request_data.model).inc(chunk['completion_tokens'])
                        if first_chunk_time:
                            usage_obj["time_to_first_token_ms"] = (first_chunk_time - start_time) * 1000
                        usage_obj["tokens_per_second"] = round(tokens_per_second, 2)
                        # Cost estimation
                        import os
                        try:
                            cost_prompt_rate = float(os.getenv('COST_PER_1K_PROMPT_TOKENS', '0'))
                            cost_completion_rate = float(os.getenv('COST_PER_1K_COMPLETION_TOKENS', '0'))
                        except ValueError:
                            cost_prompt_rate = cost_completion_rate = 0.0
                        cost_usd = ((chunk['prompt_tokens'] / 1000.0) * cost_prompt_rate) + ((chunk['completion_tokens'] / 1000.0) * cost_completion_rate)
                        usage_obj['cost_usd'] = round(cost_usd, 6)
                        final_payload = {
                            "id": f"cmpl-{uuid.uuid4().hex[:8]}",
                            "object": "text_completion",
                            "choices": [
                                {"index": 0, "text": chunk['text'], "finish_reason": "stop"}
                            ],
                            "usage": usage_obj
                        }
                        yield "data: " + json.dumps(final_payload) + "\n\n"
                        yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                COMPL_LATENCY.observe(time.time() - start_time)
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming completions
    choices = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for i, prompt in enumerate(prompts):
        gen = model_manager.generate_response(prompt, None, params)
        text = gen['text']
        if not text or text.startswith("Error generating"):
            raise HTTPException(status_code=500, detail=f"Failed to generate completion for prompt {i}")

        metrics_collector.record_model_usage(request_data.model)
        if current_user:
            request.state.user = current_user

        total_prompt_tokens += gen['prompt_tokens']
        total_completion_tokens += gen['completion_tokens']
        COMPL_TOKENS.labels(type='prompt', model=request_data.model).inc(gen['prompt_tokens'])
        COMPL_TOKENS.labels(type='completion', model=request_data.model).inc(gen['completion_tokens'])

        choices.append(CompletionChoice(
            index=i,
            text=text,
            finish_reason="stop"
        ))

    # Calculate metrics
    duration = time.time() - start_time
    tokens_per_second = total_completion_tokens / duration if duration > 0 else 0

    # Cost estimation env driven
    import os
    try:
        cost_prompt_rate = float(os.getenv('COST_PER_1K_PROMPT_TOKENS', '0'))
        cost_completion_rate = float(os.getenv('COST_PER_1K_COMPLETION_TOKENS', '0'))
    except ValueError:
        cost_prompt_rate = cost_completion_rate = 0.0
    cost_usd = ((total_prompt_tokens / 1000.0) * cost_prompt_rate) + ((total_completion_tokens / 1000.0) * cost_completion_rate)

    # For non-stream path use full latency as time_to_first_token
    ttfb_ms = duration * 1000

    completion_response = CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:8]}",
        model=request_data.model,
        choices=choices,
        usage=Usage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens, 
            total_tokens=total_prompt_tokens + total_completion_tokens,
            tokens_per_second=round(tokens_per_second, 2),
            latency_ms=round(duration * 1000, 2),
            time_to_first_token_ms=round(ttfb_ms, 2),
            cost_usd=round(cost_usd, 6)
        ),
        system_fingerprint=f"simple-llm-v1.0-{request_data.model}"
    )

    COMPL_REQUESTS.labels(model=request_data.model).inc()
    COMPL_LATENCY.observe(time.time() - start_time)
    return completion_response
