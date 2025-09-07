"""Rate limiting middleware for API service."""

import time
import logging
from typing import Dict, Optional
from collections import defaultdict, deque
from fastapi import Request, HTTPException, status
from slowapi import Limiter, _rate_limit_exceeded_handler
from prometheus_client import Counter

# Prometheus metric for rate limit exceeded events
if 'RATE_LIMIT_EXCEEDED' not in globals():
    RATE_LIMIT_EXCEEDED = Counter('llm_rate_limit_exceeded_total', 'Number of requests rejected due to rate limiting', ['period'])
    RATE_LIMIT_EXCEEDED_REGISTERED = True
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# In-memory rate limiting store (replace with Redis in production)
class InMemoryRateLimitStore:
    """In-memory rate limiting storage."""
    
    def __init__(self):
        self.store: Dict[str, deque] = defaultdict(lambda: deque())
    
    def hit(self, key: str, window: int, limit: int) -> bool:
        """Record a hit and check if limit is exceeded."""
        now = time.time()
        window_start = now - window
        
        # Clean old entries
        while self.store[key] and self.store[key][0] < window_start:
            self.store[key].popleft()
        
        # Check if limit exceeded
        if len(self.store[key]) >= limit:
            return False
        
        # Record hit
        self.store[key].append(now)
        return True
    
    def get_window_stats(self, key: str, window: int) -> Dict[str, int]:
        """Get current window statistics."""
        now = time.time()
        window_start = now - window
        
        # Clean old entries
        while self.store[key] and self.store[key][0] < window_start:
            self.store[key].popleft()
        
        return {
            "hits": len(self.store[key]),
            "window_start": int(window_start),
            "window_end": int(now)
        }

# Global rate limit store
rate_limit_store = InMemoryRateLimitStore()

def get_client_id(request: Request) -> str:
    """Get client identifier for rate limiting."""
    # Try to get user ID from auth
    if hasattr(request.state, 'user') and request.state.user:
        user = request.state.user
        if user.get("type") == "api_key":
            return f"api_key:{user.get('name', 'unknown')}"
        elif user.get("type") == "jwt":
            return f"user:{user.get('username', 'unknown')}"
    
    # Fallback to IP address
    return f"ip:{get_remote_address(request)}"

def get_rate_limits_for_user(user: Optional[Dict] = None) -> Dict[str, Dict[str, int]]:
    """Get rate limits based on user type."""
    if not user:
        # Anonymous users - restrictive limits
        return {
            "per_minute": {"window": 60, "limit": 10},
            "per_hour": {"window": 3600, "limit": 100},
            "per_day": {"window": 86400, "limit": 1000}
        }
    
    user_type = user.get("type")
    permissions = user.get("permissions", [])
    
    if user_type == "api_key":
        if "admin" in permissions:
            # Admin API keys - very high limits
            return {
                "per_minute": {"window": 60, "limit": 1000},
                "per_hour": {"window": 3600, "limit": 10000},
                "per_day": {"window": 86400, "limit": 100000}
            }
        else:
            # Regular API keys - high limits
            return {
                "per_minute": {"window": 60, "limit": 100},
                "per_hour": {"window": 3600, "limit": 1000},
                "per_day": {"window": 86400, "limit": 10000}
            }
    
    elif user_type == "jwt":
        if "admin" in permissions:
            # Admin users - high limits
            return {
                "per_minute": {"window": 60, "limit": 200},
                "per_hour": {"window": 3600, "limit": 2000},
                "per_day": {"window": 86400, "limit": 20000}
            }
        else:
            # Regular users - moderate limits
            return {
                "per_minute": {"window": 60, "limit": 50},
                "per_hour": {"window": 3600, "limit": 500},
                "per_day": {"window": 86400, "limit": 5000}
            }
    
    # Default to anonymous limits
    return {
        "per_minute": {"window": 60, "limit": 10},
        "per_hour": {"window": 3600, "limit": 100},
        "per_day": {"window": 86400, "limit": 1000}
    }

async def check_rate_limits(request: Request) -> None:
    """Check rate limits for the request."""
    client_id = get_client_id(request)
    user = getattr(request.state, 'user', None)
    
    # Get rate limits for user
    limits = get_rate_limits_for_user(user)
    
    # Check each rate limit
    for period, config in limits.items():
        window = config["window"]
        limit = config["limit"]
        key = f"{client_id}:{period}"
        
        if not rate_limit_store.hit(key, window, limit):
            # Rate limit exceeded
            stats = rate_limit_store.get_window_stats(key, window)
            try:
                RATE_LIMIT_EXCEEDED.labels(period=period).inc()
            except Exception:  # pragma: no cover - guard against duplicate metric registration edge cases
                pass
            
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "period": period,
                    "limit": limit,
                    "window": window,
                    "hits": stats["hits"],
                    "retry_after": window
                },
                headers={
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": str(max(0, limit - stats["hits"])),
                    "X-RateLimit-Reset": str(stats["window_end"] + window),
                    "Retry-After": str(window)
                }
    )

# Create slowapi limiter instance
limiter = Limiter(key_func=get_remote_address)

# Basic rate limits for public endpoints
@limiter.limit("60/minute")
async def rate_limit_public(request: Request):
    """Basic rate limit for public endpoints."""
    pass

@limiter.limit("600/hour") 
async def rate_limit_authenticated(request: Request):
    """Rate limit for authenticated endpoints."""
    pass

def get_rate_limit_headers(request: Request) -> Dict[str, str]:
    """Get rate limit headers for response."""
    client_id = get_client_id(request)
    user = getattr(request.state, 'user', None)
    limits = get_rate_limits_for_user(user)
    
    headers = {}
    
    # Add headers for each period
    for period, config in limits.items():
        window = config["window"]
        limit = config["limit"]
        key = f"{client_id}:{period}"
        
        stats = rate_limit_store.get_window_stats(key, window)
        
        headers.update({
            f"X-RateLimit-{period.replace('_', '-')}-Limit": str(limit),
            f"X-RateLimit-{period.replace('_', '-')}-Remaining": str(max(0, limit - stats["hits"])),
            f"X-RateLimit-{period.replace('_', '-')}-Reset": str(stats["window_end"] + window)
        })
    
    return headers
