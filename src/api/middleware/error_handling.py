"""Centralized error handling & structured problem responses (Wave 2).

Enhancements (Wave 2):
 - Capture HTTPException explicitly and wrap detail in standardized envelope
 - Always emit 'error' object (even if original response provided different shape)
 - Include request_id when available
 - Optional stack trace inclusion controlled by init flag
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import traceback
import logging


ERROR_CODE_MAP = {
    400: ("bad_request", "The request is invalid"),
    401: ("unauthorized", "Authentication required or failed"),
    403: ("forbidden", "You do not have access to this resource"),
    404: ("not_found", "The requested resource was not found"),
    405: ("method_not_allowed", "HTTP method not allowed"),
    408: ("timeout", "The request timed out"),
    409: ("conflict", "Conflict with current state"),
    413: ("payload_too_large", "Payload too large"),
    415: ("unsupported_media_type", "Unsupported media type"),
    422: ("unprocessable_entity", "Validation failed"),
    429: ("rate_limited", "Too many requests"),
    500: ("internal_error", "Internal server error"),
    503: ("unavailable", "Service temporarily unavailable"),
}


def build_error_payload(status_code: int, message: str, request: Request, *, error_id: str | None = None):
    code, default_msg = ERROR_CODE_MAP.get(status_code, ("error", "Unexpected error"))
    return {
        "error": {
            "id": error_id,
            "code": code,
            "message": message or default_msg,
            "status": status_code,
            "path": request.url.path,
            "method": request.method,
            "request_id": getattr(request.state, 'request_id', None)
        }
    }


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, include_trace: bool = False):  # pragma: no cover - initialization
        super().__init__(app)
        self.include_trace = include_trace

    async def dispatch(self, request: Request, call_next):  # pragma: no cover - middleware path
        try:
            return await call_next(request)
        except HTTPException as http_exc:
            # Wrap HTTPException into standardized envelope
            detail = http_exc.detail
            if isinstance(detail, dict):
                # Pull message field if present else stringify
                message = detail.get('error') or detail.get('message') or str(detail)
            else:
                message = str(detail)
            payload = build_error_payload(http_exc.status_code, message, request)
            return JSONResponse(status_code=http_exc.status_code, content=payload)
        except Exception as exc:  # noqa: broad-except
            logging.error(f"Unhandled exception: {exc}")
            if self.include_trace:
                logging.debug("Traceback:\n" + ''.join(traceback.format_exc()))
            payload = build_error_payload(500, str(exc), request)
            return JSONResponse(status_code=500, content=payload)


__all__ = ["ErrorHandlingMiddleware", "build_error_payload", "ERROR_CODE_MAP"]
