#!/usr/bin/env python3
"""
Simple CLI Chatbot
A flexible command-line chatbot powered by Hugging Face Transformers.

Usage:
    python chatbot.py                    # Start with default model
    python chatbot.py -m llama-3.2-1b   # Start with specific model
    python chatbot.py -c custom.yaml    # Use custom config file
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from cli import main

if __name__ == '__main__':
    main()
