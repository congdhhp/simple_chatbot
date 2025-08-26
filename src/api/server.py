"""FastAPI server for Simple LLM Service."""

import sys
import os
import logging
import click
import uvicorn
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
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
from src.api.middleware.monitoring import metrics_middleware, metrics_collector
from src.api.middleware.rate_limit import check_rate_limits

# Global instances
config_manager = None
model_manager = None
conversation_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    global config_manager, model_manager, conversation_manager
    
    try:
        # Initialize core components
        config_manager = ConfigManager()
        model_manager = ModelManager(config_manager)
        conversation_manager = ConversationManager(config_manager)
        
        # Load default model
        default_model = config_manager.get_default_model()
        logging.info(f"Loading default model: {default_model}")
        
        if not model_manager.load_model(default_model):
            logging.error(f"Failed to load default model: {default_model}")
        else:
            logging.info("Default model loaded successfully")
            
        # Set model for conversation manager
        conversation_manager.set_current_model(default_model)
        
        # Store in app state
        app.state.config_manager = config_manager
        app.state.model_manager = model_manager
        app.state.conversation_manager = conversation_manager
        
        logging.info("Simple LLM Service startup completed")
        
    except Exception as e:
        logging.error(f"Failed to initialize service: {e}")
        raise
    
    yield
    
    # Shutdown
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

# Add metrics middleware
@app.middleware("http")
async def add_metrics_middleware(request: Request, call_next):
    """Add metrics collection middleware."""
    return await metrics_middleware(request, call_next)

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

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error={
                "message": exc.detail,
                "type": "http_error",
                "code": exc.status_code
            }
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logging.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error={
                "message": "Internal server error",
                "type": "internal_error",
                "code": 500
            }
        ).dict()
    )

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(auth.router, tags=["Authentication"])
app.include_router(monitoring.router, prefix="/admin", tags=["Monitoring"])
app.include_router(models.router, prefix="/v1", tags=["Models"])
app.include_router(completions.router, prefix="/v1", tags=["Completions"])
app.include_router(chat.router, prefix="/v1", tags=["Chat"])

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
    
    # Start server
    uvicorn.run(
        "src.api.server:app",
        host=host,
        port=port,
        workers=workers if not reload else 1,
        reload=reload,
        log_level=log_level,
        access_log=True
    )

if __name__ == "__main__":
    main()
