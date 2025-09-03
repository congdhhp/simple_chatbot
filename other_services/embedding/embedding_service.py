#!/usr/bin/env python3
"""Main entry point for the embedding service."""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from src.api.server import run_server


def setup_logging(log_level: str):
    """Set up logging configuration."""
    
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('embedding_service.log') if os.getenv('LOG_TO_FILE') else logging.NullHandler()
        ]
    )


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description="NVIDIA NIM-compatible Embedding Service")
    
    parser.add_argument(
        "--host", 
        type=str, 
        default=os.getenv("HOST", "0.0.0.0"),
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=int(os.getenv("PORT", "8009")),
        help="Port to bind the server to (default: 8009)"
    )
    
    parser.add_argument(
        "--workers", 
        type=int, 
        default=int(os.getenv("WORKERS", "1")),
        help="Number of worker processes (default: 1)"
    )
    
    parser.add_argument(
        "--log-level", 
        type=str, 
        choices=["debug", "info", "warning", "error", "critical"],
        default=os.getenv("LOG_LEVEL", "info").lower(),
        help="Logging level (default: info)"
    )
    
    parser.add_argument(
        "--reload", 
        action="store_true",
        default=os.getenv("RELOAD", "false").lower() == "true",
        help="Enable auto-reload for development (default: false)"
    )
    
    parser.add_argument(
        "--config-path",
        type=str,
        default=os.getenv("CONFIG_PATH", "config/models.yaml"),
        help="Path to the configuration file (default: config/models.yaml)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Set environment variables for components to use
    os.environ["CONFIG_PATH"] = args.config_path
    
    logger = logging.getLogger(__name__)
    logger.info("Starting NVIDIA NIM-compatible Embedding Service")
    logger.info(f"Configuration: host={args.host}, port={args.port}, workers={args.workers}")
    logger.info(f"Log level: {args.log_level}, Config path: {args.config_path}")
    
    try:
        # Run the server
        run_server(
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level=args.log_level,
            reload=args.reload
        )
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Failed to start service: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
