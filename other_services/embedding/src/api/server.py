"""FastAPI server application for the embedding service."""

import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn

from ..config_manager import ConfigManager
from ..model_manager import ModelManager
from ..embedding_manager import EmbeddingManager
from .routes import health, models, embeddings
from .models import ErrorResponse, create_error_response


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown tasks."""
    
    # Startup
    logger.info("Starting embedding service...")
    
    try:
        # Initialize configuration
        config_manager = ConfigManager()
        app.state.config_manager = config_manager
        
        # Initialize model manager
        model_manager = ModelManager(config_manager)
        app.state.model_manager = model_manager
        
        # Initialize embedding manager
        embedding_manager = EmbeddingManager(config_manager, model_manager)
        app.state.embedding_manager = embedding_manager
        
        # Load default model if specified
        default_model = config_manager.get_default_model()
        if default_model:
            logger.info(f"Loading default model: {default_model}")
            try:
                success = model_manager.load_model(default_model)
                if success:
                    logger.info(f"Default model {default_model} loaded successfully")
                else:
                    logger.warning(f"Failed to load default model {default_model}")
            except Exception as e:
                logger.error(f"Failed to load default model {default_model}: {str(e)}")
                # Continue without default model - can be loaded later via API
        
        app.state.service_ready = True
        logger.info("Embedding service started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start embedding service: {str(e)}")
        app.state.service_ready = False
        raise
    
    # Shutdown
    logger.info("Shutting down embedding service...")
    
    try:
        # Unload all models
        if hasattr(app.state, 'model_manager'):
            await app.state.model_manager.shutdown()
        
        logger.info("Embedding service shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Embedding Service",
        description="NVIDIA NIM-compatible embedding service supporting multiple model families",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure appropriately for production
    )
    
    # Include routers
    app.include_router(health.router, tags=["health"])
    app.include_router(models.router, tags=["models"])
    app.include_router(embeddings.router, tags=["embeddings"])
    
    # Add exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions with consistent error format."""
        return JSONResponse(
            status_code=exc.status_code,
            content=create_error_response(
                message=exc.detail,
                error_type="http_error"
            ).dict()
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        error_details = []
        for error in exc.errors():
            error_details.append({
                "field": " -> ".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"]
            })
        
        return JSONResponse(
            status_code=422,
            content=create_error_response(
                message="Request validation failed",
                error_type="validation_error"
            ).dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=create_error_response(
                message="An unexpected error occurred",
                error_type="internal_error"
            ).dict()
        )
    
    # Add middleware to check service readiness
    @app.middleware("http")
    async def service_readiness_middleware(request: Request, call_next):
        """Middleware to check if service is ready before processing requests."""
        
        # Skip readiness check for health endpoints
        if request.url.path.startswith("/health") or request.url.path in ["/docs", "/redoc", "/openapi.json"]:
            response = await call_next(request)
            return response
        
        # Check if service is ready
        if not getattr(request.app.state, 'service_ready', False):
            return JSONResponse(
                status_code=503,
                content=create_error_response(
                    message="Service is not ready",
                    error_type="service_unavailable"
                ).dict()
            )
        
        response = await call_next(request)
        return response
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with service information."""
        return {
            "service": "embedding-service",
            "version": "1.0.0",
            "description": "NVIDIA NIM-compatible embedding service",
            "docs": "/docs",
            "health": "/health/ready"
        }
    
    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8009,
    workers: int = 1,
    log_level: str = "info",
    reload: bool = False
):
    """Run the embedding service server.
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to
        workers: Number of worker processes
        log_level: Logging level
        reload: Enable auto-reload for development
    """
    
    logger.info(f"Starting embedding service on {host}:{port}")
    
    # Configure uvicorn logging
    uvicorn_config = {
        "app": "src.api.server:create_app",
        "factory": True,
        "host": host,
        "port": port,
        "log_level": log_level,
        "access_log": True,
        "loop": "asyncio"
    }
    
    if reload:
        uvicorn_config["reload"] = True
        uvicorn_config["reload_dirs"] = ["src"]
    else:
        uvicorn_config["workers"] = workers
    
    uvicorn.run(**uvicorn_config)


if __name__ == "__main__":
    # Development server
    run_server(reload=True)
