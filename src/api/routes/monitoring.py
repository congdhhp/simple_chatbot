"""Monitoring and metrics endpoints."""

import logging
from fastapi import APIRouter, Depends, Query
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
async def get_service_status(current_user: dict = Depends(optional_auth)):
    """Get basic service status (public endpoint)."""
    app_metrics = metrics_collector.get_application_metrics()
    
    # Basic status info (safe for public)
    return {
        "status": "operational",
        "uptime_seconds": app_metrics["uptime_seconds"],
        "total_requests": app_metrics["total_requests"],
        "requests_per_second": round(app_metrics["requests_per_second"], 2),
        "error_rate": round(app_metrics["error_rate"] * 100, 2),
        "models_in_use": len(app_metrics["model_usage"])
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
