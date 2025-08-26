"""Monitoring and metrics middleware."""

import time
import logging
import psutil
import torch
from typing import Dict, Any, Optional
from collections import defaultdict, deque
from fastapi import Request, Response
from datetime import datetime, timedelta

class MetricsCollector:
    """Collect and store application metrics."""
    
    def __init__(self):
        self.request_count = defaultdict(int)
        self.request_duration = defaultdict(list)
        self.request_errors = defaultdict(int)
        self.model_usage = defaultdict(int)
        self.start_time = time.time()
        
        # Keep last 1000 requests for analysis
        self.recent_requests = deque(maxlen=1000)
    
    def record_request(self, method: str, path: str, status_code: int, duration: float, user_info: Optional[Dict] = None):
        """Record request metrics."""
        endpoint = f"{method} {path}"
        
        # Basic counters
        self.request_count[endpoint] += 1
        self.request_duration[endpoint].append(duration)
        
        if status_code >= 400:
            self.request_errors[endpoint] += 1
        
        # Detailed request info
        request_info = {
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration": duration,
            "user": user_info.get("username") if user_info else "anonymous"
        }
        self.recent_requests.append(request_info)
    
    def record_model_usage(self, model_name: str):
        """Record model usage."""
        self.model_usage[model_name] += 1
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system resource metrics."""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = {
            "cpu": {
                "usage_percent": cpu_percent,
                "count": psutil.cpu_count()
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used
            },
            "disk": {
                "total": disk.total,
                "free": disk.free,
                "used": disk.used,
                "percent": (disk.used / disk.total) * 100
            }
        }
        
        # GPU metrics if available
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_allocated = torch.cuda.memory_allocated(0)
            gpu_memory_reserved = torch.cuda.memory_reserved(0)
            
            metrics["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "total_memory": gpu_memory,
                "allocated_memory": gpu_memory_allocated,
                "reserved_memory": gpu_memory_reserved,
                "memory_usage_percent": (gpu_memory_allocated / gpu_memory) * 100
            }
        
        return metrics
    
    def get_application_metrics(self) -> Dict[str, Any]:
        """Get application-specific metrics."""
        uptime = time.time() - self.start_time
        
        # Request statistics
        total_requests = sum(self.request_count.values())
        total_errors = sum(self.request_errors.values())
        
        # Calculate average response times
        avg_response_times = {}
        for endpoint, durations in self.request_duration.items():
            if durations:
                avg_response_times[endpoint] = {
                    "avg": sum(durations) / len(durations),
                    "min": min(durations),
                    "max": max(durations),
                    "count": len(durations)
                }
        
        return {
            "uptime_seconds": uptime,
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": (total_errors / total_requests) if total_requests > 0 else 0,
            "requests_per_second": total_requests / uptime if uptime > 0 else 0,
            "endpoint_stats": dict(self.request_count),
            "response_times": avg_response_times,
            "model_usage": dict(self.model_usage)
        }
    
    def get_recent_activity(self, limit: int = 100) -> list:
        """Get recent request activity."""
        return list(self.recent_requests)[-limit:]

# Global metrics collector
metrics_collector = MetricsCollector()

async def metrics_middleware(request: Request, call_next):
    """Middleware to collect metrics."""
    start_time = time.time()
    
    # Get user info if available
    user_info = getattr(request.state, 'user', None)
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Record metrics
    metrics_collector.record_request(
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration=duration,
        user_info=user_info
    )
    
    # Add response headers
    response.headers["X-Response-Time"] = f"{duration:.3f}s"
    
    return response

def log_request_info(request: Request, response: Response, duration: float):
    """Log detailed request information."""
    user_info = getattr(request.state, 'user', None)
    user_id = "anonymous"
    
    if user_info:
        if user_info.get("type") == "api_key":
            user_id = f"api_key:{user_info.get('name')}"
        else:
            user_id = f"user:{user_info.get('username')}"
    
    logging.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Duration: {duration:.3f}s - "
        f"User: {user_id} - "
        f"IP: {request.client.host if request.client else 'unknown'}"
    )
