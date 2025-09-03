"""Health check endpoints."""

import logging
from fastapi import APIRouter, Request
from ..models import HealthResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/v1/health/ready", response_model=HealthResponse)
async def readiness_check(request: Request):
    """Kubernetes readiness probe endpoint.
    
    Checks if the service is ready to accept requests.
    """
    try:
        # Check if embedding manager is available
        embedding_manager = getattr(request.app.state, 'embedding_manager', None)
        if not embedding_manager:
            return HealthResponse(
                message="Service not ready - embedding manager not initialized",
                status="not_ready"
            )
        
        # Check if model manager is available
        model_manager = getattr(request.app.state, 'model_manager', None)
        if not model_manager:
            return HealthResponse(
                message="Service not ready - model manager not initialized", 
                status="not_ready"
            )
        
        # Check if service is properly initialized
        service_ready = getattr(request.app.state, 'service_ready', False)
        if not service_ready:
            return HealthResponse(
                message="Service not ready - still initializing",
                status="not_ready"
            )
        
        # Get loaded models status (don't try to load, just check)
        loaded_models = []
        if model_manager:
            try:
                loaded_models = model_manager.get_loaded_models()
            except Exception:
                pass
        
        return HealthResponse(
            message="Service is ready.",
            status="ready"
        )
        
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        return HealthResponse(
            message=f"Service not ready - {str(e)}",
            status="error"
        )


@router.get("/v1/health/startup", response_model=HealthResponse)
async def startup_check(request: Request):
    """Kubernetes startup probe endpoint.
    
    Checks if the service has finished starting up.
    This can be slow during model loading.
    """
    try:
        # Check if basic services are initialized
        embedding_manager = getattr(request.app.state, 'embedding_manager', None)
        model_manager = getattr(request.app.state, 'model_manager', None)
        
        if not embedding_manager or not model_manager:
            return HealthResponse(
                message="Service still starting - managers not initialized",
                status="starting"
            )
        
        # Check if service startup is complete
        service_ready = getattr(request.app.state, 'service_ready', False)
        if not service_ready:
            return HealthResponse(
                message="Service still starting - loading default model",
                status="starting"
            )
        
        return HealthResponse(
            message="Service startup complete.",
            status="ready"
        )
        
    except Exception as e:
        logger.error(f"Startup check failed: {str(e)}")
        return HealthResponse(
            message=f"Service startup error - {str(e)}",
            status="error"
        )


@router.get("/v1/health/live", response_model=HealthResponse)
async def liveness_check(request: Request):
    """Kubernetes liveness probe endpoint.
    
    Checks if the service is alive and responding.
    """
    try:
        # Basic liveness check - just verify the service is responding
        return HealthResponse(
            message="Service is live.",
            status="alive"
        )
        
    except Exception as e:
        logger.error(f"Liveness check failed: {str(e)}")
        return HealthResponse(
            message=f"Service error - {str(e)}",
            status="error"
        )


@router.get("/health", response_model=HealthResponse) 
async def basic_health_check(request: Request):
    """Basic health check endpoint.
    
    Provides general health status information.
    """
    try:
        # Get service state
        embedding_manager = getattr(request.app.state, 'embedding_manager', None)
        model_manager = getattr(request.app.state, 'model_manager', None)
        config_manager = getattr(request.app.state, 'config_manager', None)
        
        status_parts = []
        
        if config_manager:
            available_models = len(config_manager.get_available_models())
            status_parts.append(f"{available_models} models configured")
        
        if model_manager:
            current_model = model_manager.get_current_model_name()
            if current_model:
                status_parts.append(f"model '{current_model}' loaded")
            else:
                status_parts.append("no model loaded")
                
            # GPU memory info
            gpu_info = model_manager.get_gpu_memory_info()
            if gpu_info['total'] > 0:
                gpu_usage = gpu_info['allocated'] / gpu_info['total'] * 100
                status_parts.append(f"GPU memory: {gpu_usage:.1f}% used")
        
        if embedding_manager:
            stats = embedding_manager.get_stats()
            if stats['enable_dynamic_batching']:
                status_parts.append("dynamic batching enabled")
        
        status_message = "Service is healthy. " + ", ".join(status_parts) if status_parts else "Service is healthy."
        
        return HealthResponse(
            message=status_message,
            status="healthy"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            message=f"Service health check error - {str(e)}",
            status="unhealthy"
        )
