"""Monitoring and metrics endpoints."""

import logging
from fastapi import APIRouter, Depends, Query, Request
from src.api.middleware.auth import require_permission, optional_auth
from src.api.middleware.monitoring import metrics_collector

router = APIRouter()

@router.get("/metrics")
async def get_metrics(
    current_user: dict = Depends(require_permission("admin"))
):
    """Get comprehensive metrics (admin only)."""
    system_metrics = metrics_collector.get_system_metrics()
    app_metrics = metrics_collector.get_application_metrics()
    
    return {
        "system": system_metrics,
        "application": app_metrics,
        "timestamp": metrics_collector.start_time
    }

@router.get("/metrics/system")
async def get_system_metrics(
    current_user: dict = Depends(require_permission("admin"))
):
    """Get system resource metrics (admin only)."""
    return metrics_collector.get_system_metrics()

@router.get("/metrics/application")
async def get_application_metrics(
    current_user: dict = Depends(require_permission("admin"))
):
    """Get application metrics (admin only)."""
    return metrics_collector.get_application_metrics()

@router.get("/metrics/requests")
async def get_request_metrics(
    limit: int = Query(default=100, le=1000),
    current_user: dict = Depends(require_permission("admin"))
):
    """Get recent request activity (admin only)."""
    return {
        "recent_requests": metrics_collector.get_recent_activity(limit),
        "total_requests": sum(metrics_collector.request_count.values()),
        "total_errors": sum(metrics_collector.request_errors.values())
    }

@router.get("/status")
async def get_service_status(request: Request, current_user: dict = Depends(optional_auth)):
    """Public service status with degraded reason, GPU + model info."""
    app_metrics = metrics_collector.get_application_metrics()
    degraded = getattr(request.app.state, 'degraded', False)
    degraded_reason = getattr(request.app.state, 'degraded_reason', None)
    config_manager = getattr(request.app.state, 'config_manager', None)
    model_manager = getattr(request.app.state, 'model_manager', None)

    default_model = None
    models_loaded = []
    if config_manager:
        try:
            default_model = config_manager.get_default_model()
            models_loaded = list(config_manager.get_available_models().keys())
        except Exception:  # pragma: no cover
            pass

    gpu = {}
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            alloc = torch.cuda.memory_allocated(0)
            gpu = {
                "device": torch.cuda.get_device_name(0),
                "memory_total": total,
                "memory_allocated": alloc,
                "memory_usage_percent": round((alloc / total) * 100, 2) if total else 0
            }
    except Exception:  # pragma: no cover
        gpu = {"available": False}

    return {
        "status": "operational" if not degraded else "degraded",
        "degraded": degraded,
        "degraded_reason": degraded_reason,
        "uptime_seconds": round(app_metrics["uptime_seconds"], 2),
        "total_requests": app_metrics["total_requests"],
        "requests_per_second": round(app_metrics["requests_per_second"], 2),
        "error_rate_percent": round(app_metrics["error_rate"] * 100, 2),
        "models_in_use": len(app_metrics["model_usage"]),
        "model_usage": app_metrics["model_usage"],
        "default_model": default_model,
        "configured_models": models_loaded,
        "current_model": getattr(model_manager, 'current_model_name', None),
        "gpu": gpu
    }

@router.get("/config/validate")
async def validate_config(request: Request, current_user: dict = Depends(require_permission("admin"))):
    """Validate configuration and return summary of models (admin)."""
    config_manager = getattr(request.app.state, 'config_manager', None)
    if not config_manager:
        return {"status": "error", "reason": "config_manager_unavailable"}
    issues = []
    models_summary = []
    config = config_manager.config
    default_model = config.get('default_model')
    for name, cfg in config.get('models', {}).items():
        missing = [k for k in ['model_id', 'display_name', 'generation_config', 'settings'] if k not in cfg]
        if missing:
            issues.append({"model": name, "missing_keys": missing})
        models_summary.append({
            "name": name,
            "display_name": cfg.get('display_name'),
            "model_id": cfg.get('model_id'),
            "default": name == default_model,
            "quantization": {
                "use_4bit": cfg.get('settings', {}).get('use_4bit_quantization', False),
                "use_8bit": cfg.get('settings', {}).get('use_8bit_quantization', False)
            },
            "flash_attention": cfg.get('settings', {}).get('use_flash_attention', False)
        })
    return {
        "status": "ok" if not issues else "issues",
        "default_model": default_model,
        "models": models_summary,
        "issues": issues
    }

@router.get("/health/detailed")
async def get_detailed_health(
    current_user: dict = Depends(require_permission("read"))
):
    """Get detailed health information (requires authentication)."""
    system_metrics = metrics_collector.get_system_metrics()
    app_metrics = metrics_collector.get_application_metrics()
    
    # Determine health status
    health_status = "healthy"
    issues = []
    
    # Check system resources
    if system_metrics["memory"]["percent"] > 90:
        health_status = "degraded"
        issues.append("High memory usage")
    
    if system_metrics["cpu"]["usage_percent"] > 90:
        health_status = "degraded"
        issues.append("High CPU usage")
    
    if "gpu" in system_metrics and system_metrics["gpu"]["memory_usage_percent"] > 95:
        health_status = "degraded"
        issues.append("High GPU memory usage")
    
    # Check error rate
    if app_metrics["error_rate"] > 0.1:  # More than 10% error rate
        health_status = "degraded"
        issues.append("High error rate")
    
    return {
        "status": health_status,
        "issues": issues,
        "system_resources": {
            "memory_usage": system_metrics["memory"]["percent"],
            "cpu_usage": system_metrics["cpu"]["usage_percent"],
            "gpu_memory_usage": system_metrics.get("gpu", {}).get("memory_usage_percent", 0)
        },
        "application_stats": {
            "uptime": app_metrics["uptime_seconds"],
            "total_requests": app_metrics["total_requests"],
            "error_rate": app_metrics["error_rate"],
            "requests_per_second": app_metrics["requests_per_second"]
        }
    }
