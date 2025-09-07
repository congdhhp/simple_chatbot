"""Logging setup with optional structured JSON output and request context enrichment (Wave 2)."""

import logging
import sys
import json
import time
from pathlib import Path
from typing import Any, Dict

class JsonFormatter(logging.Formatter):
    """Structured JSON log formatter."""
    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - formatting
        payload = {
            "ts": time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Inject extra fields if present
        for attr in ["request_id", "endpoint", "model", "latency_ms", "prompt_tokens", "completion_tokens", "status_code", "user"]:
            if hasattr(record, attr):
                payload[attr] = getattr(record, attr)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

def _build_handlers(level: str, json_logs: bool):
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    handlers = []
    if json_logs:
        json_formatter = JsonFormatter()
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(json_formatter)
        file_handler = logging.FileHandler(logs_dir / "api_service.jsonl")
        file_handler.setFormatter(json_formatter)
        handlers.extend([stream_handler, file_handler])
    else:
        handlers.append(logging.FileHandler(logs_dir / "api_service.log"))
        handlers.append(logging.StreamHandler(sys.stdout))
    return handlers

def setup_logging(level: str = "INFO"):
    """Setup logging configuration for the API service.

    Env flags:
      JSON_LOGS=true -> enable structured JSON logs
    """
    json_logs = (os.getenv('JSON_LOGS', 'false').lower() == 'true')
    handlers = _build_handlers(level, json_logs)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)

try:
    import os  # placed after to keep imports minimal for original baseline
except ImportError:  # pragma: no cover
    os = None
