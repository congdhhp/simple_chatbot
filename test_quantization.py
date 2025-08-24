#!/usr/bin/env python3
"""
Test script for quantization support
Tests different quantization settings and models.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from config_manager import ConfigManager
from model_manager import ModelManager
import torch

def test_quantization_settings():
    """Test quantization settings for different models."""
    print("🔍 Testing Quantization Settings for RTX 5060 Ti")
    print("=" * 60)
    
    # Initialize managers
    config_manager = ConfigManager()
    model_manager = ModelManager(config_manager)
    
    # Get all available models
    models = config_manager.get_available_models()
    
    print(f"\n📊 GPU Info:")
    if torch.cuda.is_available():
        print(f"   - Device: {torch.cuda.get_device_name()}")
        print(f"   - VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("   - CUDA not available")
    
    print(f"\n📋 Model Quantization Settings:")
    print("-" * 60)
    
    for model_name, display_name in models.items():
        settings = config_manager.get_settings(model_name)
        model_config = config_manager.get_model_config(model_name)
        
        # Check quantization settings
        use_4bit = settings.get('use_4bit_quantization', False)
        use_8bit = settings.get('use_8bit_quantization', False)
        max_memory = settings.get('max_memory_gb', 'Not set')
        model_id = model_config['model_id']
        
        if use_4bit:
            status = "🟢 4-bit"
            quant_type = "4-bit quantization"
        elif use_8bit:
            status = "🟡 8-bit"
            quant_type = "8-bit quantization"
        else:
            status = "🔵 FP16"
            quant_type = "Full precision"
        
        print(f"{status} {display_name}")
        print(f"    Model ID: {model_id}")
        print(f"    Quantization: {quant_type}")
        print(f"    Max Memory: {max_memory} GB")
        print()

def test_model_loading(model_name):
    """Test loading a specific model with quantization."""
    print(f"\n🚀 Testing Model: {model_name}")
    print("-" * 40)
    
    try:
        config_manager = ConfigManager()
        model_manager = ModelManager(config_manager)
        
        # Get model info
        model_config = config_manager.get_model_config(model_name)
        settings = config_manager.get_settings(model_name)
        
        print(f"Display Name: {model_config['display_name']}")
        print(f"4-bit Quantization: {settings.get('use_4bit_quantization', False)}")
        print(f"8-bit Quantization: {settings.get('use_8bit_quantization', False)}")
        print(f"Max Memory: {settings.get('max_memory_gb', 'Auto')} GB")
        
        # Try to load the model
        print("\n⏳ Loading model...")
        success = model_manager.load_model(model_name)
        
        if success:
            print("✅ Model loaded successfully!")
            
            # Get model info
            info = model_manager.get_current_model_info()
            print(f"Device: {info.get('device', 'Unknown')}")
            
            # Test a simple generation
            print("\n🧪 Testing generation...")
            response = model_manager.generate_response("Hello, how are you?")
            print(f"Response: {response[:100]}...")
            
            # Clean up
            model_manager.unload_model()
            print("🔄 Model unloaded")
            
        else:
            print("❌ Failed to load model")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Test quantization settings
    test_quantization_settings()
    
    # Test specific model if provided
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        test_model_loading(model_name)
    else:
        print("💡 To test a specific model, run:")
        print("   python test_quantization.py codellama-13b-instruct-4bit")
        print("   python test_quantization.py codellama-13b-instruct-8bit")
        print("   python test_quantization.py llama-3.1-8b-instruct-8bit")
