#!/usr/bin/env python3
"""Test script to verify Flash Attention integration."""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config_manager import ConfigManager
from model_manager import ModelManager
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_flash_attention():
    """Test Flash Attention functionality."""
    print("=" * 60)
    print("Testing Flash Attention Integration")
    print("=" * 60)
    
    try:
        # Initialize managers
        config_manager = ConfigManager()
        model_manager = ModelManager(config_manager)
        
        # Test with a small model that has flash attention enabled
        model_name = "llama-3.2-1b-instruct"  # Small model for testing
        
        print(f"\nTesting with model: {model_name}")
        print("-" * 40)
        
        # Check model config
        model_config = config_manager.get_model_config(model_name)
        model_settings = config_manager.get_settings(model_name)
        
        print(f"Model ID: {model_config['model_id']}")
        print(f"Flash Attention in config: {model_settings.get('use_flash_attention', False)}")
        print(f"Flash Attention available: {getattr(model_manager, 'FLASH_ATTN_AVAILABLE', False)}")
        
        # Load model
        print(f"\nLoading model...")
        success = model_manager.load_model(model_name)
        
        if success:
            # Get model info
            model_info = model_manager.get_current_model_info()
            print(f"\nModel loaded successfully!")
            print(f"Flash Attention Status: {model_info.get('flash_attention', 'Unknown')}")
            print(f"Flash Attention Available: {model_info.get('flash_attention_available', False)}")
            print(f"Use Flash Attention (config): {model_info.get('use_flash_attention_config', False)}")
            print(f"Device: {model_info.get('device', 'Unknown')}")
            
            # Test generation
            print(f"\nTesting generation...")
            test_prompt = "Hello! How are you today?"
            response = model_manager.generate_response(test_prompt)
            print(f"Prompt: {test_prompt}")
            print(f"Response: {response[:100]}..." if len(response) > 100 else f"Response: {response}")
            
            # Unload model
            model_manager.unload_model()
            print(f"\nModel unloaded successfully!")
            
        else:
            print("Failed to load model")
            return False
            
    except Exception as e:
        print(f"Error during test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("Flash Attention test completed successfully!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_flash_attention()
    sys.exit(0 if success else 1)
