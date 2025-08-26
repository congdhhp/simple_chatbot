#!/usr/bin/env python3
"""
Simple CLI Chatbot
A flexible command-line chatbot powered by Hugging Face Transformers.

Features:
- Support for multiple LLM models from Hugging Face
- Flash Attention 2 for improved performance  
- Quantization support (4-bit, 8-bit)
- LoRA adapter support
- Conversation management and persistence
- Rich CLI interface with helpful commands

Usage:
    python chatbot.py                    # Start with default config
    python chatbot.py -m model_name     # Load specific model
    python chatbot.py -c custom_config.yaml  # Use custom config
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

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from cli import main

if __name__ == '__main__':
    main()
