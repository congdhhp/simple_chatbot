#!/usr/bin/env python3
"""
Test script for Simple CLI Chatbot
Demonstrates basic functionality without interactive input.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from config_manager import ConfigManager
from model_manager import ModelManager
from conversation_manager import ConversationManager


def test_configuration():
    """Test configuration loading."""
    print("🔧 Testing configuration...")
    try:
        config_manager = ConfigManager()
        models = config_manager.get_available_models()
        print(f"✅ Found {len(models)} configured models:")
        for name, display_name in models.items():
            print(f"   - {display_name} ({name})")
        
        default_model = config_manager.get_default_model()
        print(f"✅ Default model: {default_model}")
        return config_manager
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return None


def test_model_manager(config_manager):
    """Test model manager initialization."""
    print("\n🤖 Testing model manager...")
    try:
        model_manager = ModelManager(config_manager)
        print(f"✅ Model manager initialized")
        print(f"✅ Device: {model_manager.device}")
        
        # Test model info without loading
        info = model_manager.get_current_model_info()
        print(f"✅ Current status: {info['status']}")
        return model_manager
    except Exception as e:
        print(f"❌ Model manager test failed: {e}")
        return None


def test_conversation_manager(config_manager):
    """Test conversation manager."""
    print("\n💬 Testing conversation manager...")
    try:
        conv_manager = ConversationManager(config_manager)
        
        # Test adding messages
        conv_manager.add_message("user", "Hello, this is a test message")
        conv_manager.add_message("assistant", "Hello! This is a test response.")
        
        # Test conversation summary
        summary = conv_manager.get_conversation_summary()
        print(f"✅ Conversation manager working")
        print(f"✅ Messages in history: {summary['total_messages']}")
        
        # Test saving conversation
        if conv_manager.save_conversation("test_conversation"):
            print("✅ Conversation save test passed")
        else:
            print("⚠️  Conversation save test failed")
        
        return conv_manager
    except Exception as e:
        print(f"❌ Conversation manager test failed: {e}")
        return None


def test_model_loading(model_manager, model_name="llama-3.2-1b-instruct"):
    """Test model loading (optional - requires download)."""
    print(f"\n🚀 Testing model loading ({model_name})...")
    print("⚠️  This will download the model if not cached (may take time)")
    
    try:
        success = model_manager.load_model(model_name)
        if success:
            print("✅ Model loaded successfully!")
            
            # Test model info
            info = model_manager.get_current_model_info()
            print(f"✅ Loaded model: {info['display_name']}")
            print(f"✅ Device: {info['device']}")
            
            # Test simple generation
            print("\n🧠 Testing text generation...")
            response = model_manager.generate_response("Hello! How are you?")
            print(f"✅ Generated response: {response[:100]}...")
            
            return True
        else:
            print("❌ Model loading failed")
            return False
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Simple CLI Chatbot - Test Suite")
    print("=" * 50)
    
    # Test configuration
    config_manager = test_configuration()
    if not config_manager:
        sys.exit(1)
    
    # Test model manager
    model_manager = test_model_manager(config_manager)
    if not model_manager:
        sys.exit(1)
    
    # Test conversation manager
    conv_manager = test_conversation_manager(config_manager)
    if not conv_manager:
        sys.exit(1)
    
    print("\n🎯 Core functionality tests passed!")
    
    # Ask user if they want to test model loading
    print("\n" + "="*50)
    print("🤖 Model Loading Test (Optional)")
    print("This will download and test the 1B model (~2.5GB)")
    
    try:
        choice = input("Test model loading? (y/N): ").strip().lower()
        if choice in ['y', 'yes']:
            success = test_model_loading(model_manager)
            if success:
                print("\n🎉 All tests passed! Chatbot is ready to use.")
            else:
                print("\n⚠️  Model loading failed, but core functionality works.")
        else:
            print("\n✅ Skipped model loading test.")
            print("🎉 Core tests passed! Run 'python chatbot.py' to start.")
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user.")
    
    print("\n📋 Next steps:")
    print("  1. Run: python chatbot.py")
    print("  2. Use /help to see available commands")
    print("  3. Use /models to see available models")


if __name__ == "__main__":
    main()
