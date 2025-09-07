"""FastAPI server for Simple LLM Service."""

import sys
import os
import logging
import uuid
import click
import uvicorn
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, Response
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
parent_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_path))

from src.config_manager import ConfigManager
from src.model_manager import ModelManager
from src.conversation_manager import ConversationManager
from src.api.models import ErrorResponse
from src.api.routes import chat, completions, models, health, auth, monitoring
from src.api.middleware.logging import setup_logging
from src.api.middleware.monitoring import metrics_collector  # legacy collector (middleware removed)
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from src.api.middleware.error_handling import ErrorHandlingMiddleware

# Prometheus metrics definitions (guard against double import via different module names)
if 'METRICS_REGISTERED' not in globals():
    REQUEST_COUNTER = Counter('llm_requests_total', 'Total LLM API requests', ['endpoint', 'model'])
    TOKEN_COUNTER = Counter('llm_tokens_total', 'Tokens processed', ['type', 'model'])
    LATENCY_HIST = Histogram('llm_request_latency_seconds', 'Request latency seconds', ['endpoint'])
    INFLIGHT = Gauge('llm_inflight_requests', 'In-flight LLM requests')
    METRICS_REGISTERED = True
from src.api.middleware.rate_limit import check_rate_limits

# Global instances
config_manager = None
model_manager = None
conversation_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with model preload & degraded flag."""
    global config_manager, model_manager, conversation_manager

    app.state.degraded = False
    app.state.degraded_reason = None
    preload_all = os.getenv('PRELOAD_ALL_MODELS', 'false').lower() == 'true'

    try:
        config_manager = ConfigManager()
        model_manager = ModelManager(config_manager)
        conversation_manager = ConversationManager(config_manager)

        default_model = config_manager.get_default_model()
        logging.info(f"Loading default model: {default_model}")
        if not await model_manager.async_load_model(default_model):
            logging.error(f"Failed to load default model: {default_model}")
            app.state.degraded = True
            app.state.degraded_reason = 'default_model_load_failed'
        else:
            logging.info("Default model loaded successfully")
            conversation_manager.set_current_model(default_model)

        # Optionally preload other models
        if preload_all and not app.state.degraded:
            for name in config_manager.get_available_models().keys():
                if name == default_model:
                    continue
                logging.info(f"Preloading model: {name}")
                success = await model_manager.async_load_model(name)
                if not success:
                    logging.warning(f"Failed to preload model {name}")

        app.state.config_manager = config_manager
        app.state.model_manager = model_manager
        app.state.conversation_manager = conversation_manager
        logging.info("Simple LLM Service startup completed")
    except Exception as e:
        logging.error(f"Failed to initialize service: {e}")
        app.state.degraded = True
        app.state.degraded_reason = f"startup_exception:{e}"
        raise

    yield

    try:
        if model_manager and model_manager.current_model:
            model_manager.unload_model()
            logging.info("Models unloaded")
        logging.info("Simple LLM Service shutdown completed")
    except Exception as e:
        logging.error(f"Error during shutdown: {e}")

# Create FastAPI app
app = FastAPI(
    title="Simple LLM Service",
    description="OpenAI-compatible LLM API service powered by Hugging Face Transformers",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure as needed
)

# Central error handling (Wave 2)
app.add_middleware(ErrorHandlingMiddleware)

# Request ID middleware (placed early)
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    req_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    request.state.request_id = req_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response

# Structured access logging middleware (Wave 2)
@app.middleware("http")
async def access_logging_middleware(request: Request, call_next):
    import time as _t
    start = _t.time()
    path = request.url.path
    model = None
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception as e:  # let error middleware wrap, still log here
        status = getattr(e, 'status_code', 500)
        raise
    finally:
        duration_ms = ( _t.time() - start) * 1000
        # Extract token usage if set on state by routes later (could be extended)
        extra = {
            'request_id': getattr(request.state, 'request_id', None),
            'endpoint': path,
            'model': getattr(request.state, 'model', None),
            'latency_ms': round(duration_ms,2),
            'user': getattr(getattr(request.state, 'user', None) or {}, 'get', lambda *_: None)("username") if hasattr(getattr(request.state, 'user', None), 'get') else None,
            'status_code': status
        }
        import logging as _l
        _l.getLogger("access").info(f"{path} {status} {duration_ms:.2f}ms", extra=extra)
    return response

# Add metrics middleware
@app.middleware("http")
async def inflight_middleware(request: Request, call_next):
    """Track inflight requests only (legacy in-memory metrics middleware removed)."""
    INFLIGHT.inc()
    try:
        response = await call_next(request)
        return response
    finally:
        INFLIGHT.dec()

# Add rate limiting middleware
@app.middleware("http") 
async def add_rate_limiting_middleware(request: Request, call_next):
    """Add rate limiting middleware."""
    # Skip rate limiting for health endpoints
    if request.url.path in ["/health", "/health/live", "/health/ready"]:
        return await call_next(request)
    
    # Check if rate limiting is enabled
    if os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true":
        await check_rate_limits(request)
    
    return await call_next(request)

# (Removed per Wave 2: centralized error middleware handles exceptions)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(auth.router, tags=["Authentication"])
app.include_router(monitoring.router, prefix="/admin", tags=["Monitoring"])
app.include_router(models.router, prefix="/v1", tags=["Models"])
app.include_router(completions.router, prefix="/v1", tags=["Completions"])
app.include_router(chat.router, prefix="/v1", tags=["Chat"])

@app.get('/metrics')
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Simple LLM Service",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "authentication": "/auth/login",
        "monitoring": "/admin/status"
    }

@click.command()
@click.option('--host', default='127.0.0.1', help='Host to bind to')
@click.option('--port', default=8000, type=int, help='Port to bind to')
@click.option('--workers', default=1, type=int, help='Number of worker processes')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
@click.option('--log-level', default='info', help='Log level')
@click.option('--config', default='config/models.yaml', help='Path to config file')
def main(host, port, workers, reload, log_level, config):
    """Start the Simple LLM Service."""
    
    # Setup logging
    setup_logging(log_level.upper())
    
    # Log startup info
    logging.info(f"Starting Simple LLM Service on {host}:{port}")
    logging.info(f"Config file: {config}")
    logging.info(f"Workers: {workers}")
    logging.info(f"Reload: {reload}")
    
    # Start server without module string to avoid double import (prevents duplicate Prometheus metric registration)
    # Note: reload with an app instance is not supported; if reload requested, fall back to module string
    if reload:
        uvicorn.run(
            "src.api.server:app",
            host=host,
            port=port,
            workers=1,  # reload incompatible with multiple workers
            reload=True,
            log_level=log_level,
            access_log=True
        )
    else:
        uvicorn.run(
            app,
            host=host,
            port=port,
            workers=workers,
            reload=False,
            log_level=log_level,
            access_log=True
        )

if __name__ == "__main__":
    main()
