#!/usr/bin/env python3
"""Simple test to verify Flash Attention integration without interactive input."""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config_manager import ConfigManager
from model_manager import ModelManager
import logging

# Setup minimal logging
logging.basicConfig(level=logging.INFO)

def main():
    """Test Flash Attention with model info display."""
    print("Flash Attention Integration Test")
    print("=" * 50)
    
    try:
        # Initialize components
        config_manager = ConfigManager()
        model_manager = ModelManager(config_manager)
        
        # Load a small model
        model_name = "llama-3.2-1b-instruct"
        print(f"Loading model: {model_name}")
        
        success = model_manager.load_model(model_name)
        if not success:
            print("Failed to load model!")
            return False
        
        # Get and display model info
        info = model_manager.get_current_model_info()
        print("\nModel Information:")
        print(f"  Name: {info['display_name']}")
        print(f"  Device: {info['device']}")
        print(f"  Flash Attention: {info.get('flash_attention', 'Unknown')}")
        print(f"  Flash Attention Available: {info.get('flash_attention_available', False)}")
        print(f"  Flash Attention Config: {info.get('use_flash_attention_config', False)}")
        
        # Test a simple generation
        print("\nTesting generation...")
        response = model_manager.generate_response("Hello!")
        print(f"Response: {response[:100]}...")
        
        # Cleanup
        model_manager.unload_model()
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
