"""Enhanced rate limiting middleware with token-based quotas (Wave 2)."""

import time
import logging
from typing import Dict, Optional
from collections import defaultdict, deque
from fastapi import Request, HTTPException, status
from slowapi import Limiter, _rate_limit_exceeded_handler

# Prometheus metric for rate limit exceeded events (optional)
try:
    from prometheus_client import Counter
    RATE_LIMIT_EXCEEDED = Counter('llm_rate_limit_exceeded_total', 'Number of requests rejected due to rate limiting', ['period'])
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    RATE_LIMIT_EXCEEDED = None

from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# In-memory rate limiting store with token quota support (replace with Redis in production)
class TokenAwareRateLimitStore:
    """Enhanced rate limiting storage with token quota tracking."""
    
    def __init__(self):
        self.request_store: Dict[str, deque] = defaultdict(lambda: deque())
        self.token_store: Dict[str, deque] = defaultdict(lambda: deque())
    
    def hit_request(self, key: str, window: int, limit: int) -> bool:
        """Record a request hit and check if limit is exceeded."""
        now = time.time()
        window_start = now - window
        
        # Clean old entries
        while self.request_store[key] and self.request_store[key][0][0] < window_start:
            self.request_store[key].popleft()
        
        # Check if limit exceeded
        if len(self.request_store[key]) >= limit:
            return False
        
        # Record hit
        self.request_store[key].append((now, 1))
        return True
    
    def hit_tokens(self, key: str, window: int, token_limit: int, tokens_used: int) -> bool:
        """Record token usage and check if quota is exceeded."""
        now = time.time()
        window_start = now - window
        
        # Clean old entries
        while self.token_store[key] and self.token_store[key][0][0] < window_start:
            self.token_store[key].popleft()
        
        # Calculate current token usage
        current_tokens = sum(entry[1] for entry in self.token_store[key])
        
        # Check if adding new tokens would exceed limit
        if current_tokens + tokens_used > token_limit:
            return False
        
        # Record token usage
        self.token_store[key].append((now, tokens_used))
        return True
    
    def get_request_stats(self, key: str, window: int) -> Dict[str, int]:
        """Get current request window statistics."""
        now = time.time()
        window_start = now - window
        
        # Clean old entries
        while self.request_store[key] and self.request_store[key][0][0] < window_start:
            self.request_store[key].popleft()
        
        return {
            "hits": len(self.request_store[key]),
            "window_start": int(window_start),
            "window_end": int(now)
        }
    
    def get_token_stats(self, key: str, window: int) -> Dict[str, int]:
        """Get current token window statistics."""
        now = time.time()
        window_start = now - window
        
        # Clean old entries
        while self.token_store[key] and self.token_store[key][0][0] < window_start:
            self.token_store[key].popleft()
        
        current_tokens = sum(entry[1] for entry in self.token_store[key])
        
        return {
            "tokens_used": current_tokens,
            "window_start": int(window_start),
            "window_end": int(now)
        }

# Global rate limit store
rate_limit_store = TokenAwareRateLimitStore()

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
    """Get rate limits based on user type including token quotas."""
    if not user:
        # Anonymous users - restrictive limits
        return {
            "per_minute": {"window": 60, "limit": 10, "token_limit": 1000},
            "per_hour": {"window": 3600, "limit": 100, "token_limit": 10000},
            "per_day": {"window": 86400, "limit": 1000, "token_limit": 100000}
        }
    
    user_type = user.get("type")
    permissions = user.get("permissions", [])
    
    if user_type == "api_key":
        if "admin" in permissions:
            # Admin API keys - very high limits
            return {
                "per_minute": {"window": 60, "limit": 1000, "token_limit": 100000},
                "per_hour": {"window": 3600, "limit": 10000, "token_limit": 1000000},
                "per_day": {"window": 86400, "limit": 100000, "token_limit": 10000000}
            }
        else:
            # Regular API keys - high limits
            return {
                "per_minute": {"window": 60, "limit": 100, "token_limit": 10000},
                "per_hour": {"window": 3600, "limit": 1000, "token_limit": 100000},
                "per_day": {"window": 86400, "limit": 10000, "token_limit": 1000000}
            }
    
    elif user_type == "jwt":
        if "admin" in permissions:
            # Admin users - high limits
            return {
                "per_minute": {"window": 60, "limit": 200, "token_limit": 20000},
                "per_hour": {"window": 3600, "limit": 2000, "token_limit": 200000},
                "per_day": {"window": 86400, "limit": 20000, "token_limit": 2000000}
            }
        else:
            # Regular users - moderate limits
            return {
                "per_minute": {"window": 60, "limit": 50, "token_limit": 5000},
                "per_hour": {"window": 3600, "limit": 500, "token_limit": 50000},
                "per_day": {"window": 86400, "limit": 5000, "token_limit": 500000}
            }
    
    # Default to anonymous limits
    return {
        "per_minute": {"window": 60, "limit": 10, "token_limit": 1000},
        "per_hour": {"window": 3600, "limit": 100, "token_limit": 10000},
        "per_day": {"window": 86400, "limit": 1000, "token_limit": 100000}
    }

async def check_rate_limits(request: Request, tokens_used: int = 0) -> None:
    """Check rate limits for the request including token quotas.
    
    Args:
        request: FastAPI request object
        tokens_used: Number of tokens used in this request (for post-request checking)
    """
    # Skip rate limiting for health and metrics endpoints
    skip_paths = ["/health", "/metrics", "/docs", "/openapi.json", "/"]
    if any(request.url.path.startswith(path) for path in skip_paths):
        return
        
    client_id = get_client_id(request)
    user = getattr(request.state, 'user', None)
    
    # Get rate limits for user
    limits = get_rate_limits_for_user(user)
    
    # For non-API endpoints, use relaxed limits
    if not request.url.path.startswith("/v1/"):
        # Only check basic rate limit for non-API calls
        basic_limit = {"window": 60, "limit": 100}  # 100/minute for admin/auth calls
        key = f"{client_id}:basic"
        
        if not rate_limit_store.hit_request(key, basic_limit["window"], basic_limit["limit"]):
            stats = rate_limit_store.get_request_stats(key, basic_limit["window"])
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={"error": "Rate limit exceeded for administrative endpoints"}
            )
        return
    
    # Check each rate limit for API endpoints (requests)
    for period, config in limits.items():
        window = config["window"]
        limit = config["limit"]
        key = f"{client_id}:{period}"
        
        if not rate_limit_store.hit_request(key, window, limit):
            # Rate limit exceeded
            stats = rate_limit_store.get_request_stats(key, window)
            if PROMETHEUS_AVAILABLE and RATE_LIMIT_EXCEEDED:
                RATE_LIMIT_EXCEEDED.labels(period=period).inc()
            
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "type": "requests",
                    "period": period,
                    "limit": limit,
                    "current": stats["hits"],
                    "reset_time": stats["window_end"] + window
                },
                headers={
                    f"X-RateLimit-{period.replace('_', '-').title()}-Limit": str(limit),
                    f"X-RateLimit-{period.replace('_', '-').title()}-Remaining": str(max(0, limit - stats["hits"])),
                    f"X-RateLimit-{period.replace('_', '-').title()}-Reset": str(stats["window_end"] + window)
                }
            )
    
    # Check token quotas if tokens_used > 0
    if tokens_used > 0:
        for period, config in limits.items():
            window = config["window"]
            token_limit = config.get("token_limit", 0)
            if token_limit > 0:
                key = f"{client_id}:tokens:{period}"
                
                if not rate_limit_store.hit_tokens(key, window, token_limit, tokens_used):
                    # Token quota exceeded
                    stats = rate_limit_store.get_token_stats(key, window)
                    if PROMETHEUS_AVAILABLE and RATE_LIMIT_EXCEEDED:
                        RATE_LIMIT_EXCEEDED.labels(period=f"tokens_{period}").inc()
                    
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail={
                            "error": "Token quota exceeded",
                            "type": "tokens",
                            "period": period,
                            "token_limit": token_limit,
                            "tokens_used": stats["tokens_used"],
                            "tokens_requested": tokens_used,
                            "reset_time": stats["window_end"] + window
                        },
                        headers={
                            f"X-RateLimit-Tokens-{period.replace('_', '-').title()}-Limit": str(token_limit),
                            f"X-RateLimit-Tokens-{period.replace('_', '-').title()}-Remaining": str(max(0, token_limit - stats["tokens_used"])),
                            f"X-RateLimit-Tokens-{period.replace('_', '-').title()}-Reset": str(stats["window_end"] + window)
                        }
                    )

# Create slowapi limiter instance
limiter = Limiter(key_func=get_remote_address)

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
        token_limit = config.get("token_limit", 0)
        
        # Request stats
        key = f"{client_id}:{period}"
        stats = rate_limit_store.get_request_stats(key, window)
        
        headers.update({
            f"X-RateLimit-{period.replace('_', '-').title()}-Limit": str(limit),
            f"X-RateLimit-{period.replace('_', '-').title()}-Remaining": str(max(0, limit - stats["hits"])),
            f"X-RateLimit-{period.replace('_', '-').title()}-Reset": str(stats["window_end"] + window)
        })
        
        # Token stats if applicable
        if token_limit > 0:
            token_key = f"{client_id}:tokens:{period}"
            token_stats = rate_limit_store.get_token_stats(token_key, window)
            headers.update({
                f"X-RateLimit-Tokens-{period.replace('_', '-').title()}-Limit": str(token_limit),
                f"X-RateLimit-Tokens-{period.replace('_', '-').title()}-Remaining": str(max(0, token_limit - token_stats["tokens_used"])),
                f"X-RateLimit-Tokens-{period.replace('_', '-').title()}-Reset": str(token_stats["window_end"] + window)
            })
    
    return headers
