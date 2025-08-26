#!/usr/bin/env python3
"""
Simple LLM Service - NIM Compatible API
A FastAPI-based LLM service compatible with NVIDIA NIM APIs.

Usage:
    python nim_service.py                    # Start with default config
    python nim_service.py --port 8080       # Custom port
    python nim_service.py --host 0.0.0.0    # Custom host
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from api.server import main

if __name__ == '__main__':
    main()
