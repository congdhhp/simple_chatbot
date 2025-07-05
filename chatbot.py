#!/usr/bin/env python3
"""
CLI Chatbot with LLM Model Support
A flexible chatbot that can work with different LLM models and utilize GPU acceleration
"""

import argparse
import sys
import os
from typing import List, Dict
import logging
from datetime import datetime

from model_manager import LLMModelManager
from config import ChatbotConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CLIChatbot:
    """
    Command-line interface chatbot with LLM support
    """
    
    def __init__(self, config_file: str = "chatbot_config.json"):
        """
        Initialize the chatbot
        
        Args:
            config_file (str): Path to configuration file
        """
        self.config = ChatbotConfig(config_file)
        self.model_manager = None
        self.conversation_history = []
        
        # Initialize model manager
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model manager with current configuration"""
        current_model = self.config.get_current_model()
        use_gpu = self.config.use_gpu()
        
        self.model_manager = LLMModelManager(
            model_name=current_model,
            use_gpu=use_gpu
        )
        
        # Update generation parameters
        gen_params = self.config.get_generation_params()
        self.model_manager.update_generation_params(**gen_params)
    
    def load_model(self) -> bool:
        """
        Load the current model
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.model_manager is None:
            self._initialize_model()
        
        return self.model_manager.load_model()
    
    def chat_loop(self):
        """Main chat loop"""
        print("🤖 LLM Chatbot - GPU Accelerated")
        print("=" * 50)
        
        # Load model
        print("Loading model...")
        if not self.load_model():
            print("❌ Failed to load model. Exiting.")
            return
        
        # Show model info
        if self.config.get_chat_settings().get("show_model_info", True):
            self._show_model_info()
        
        print("\n💬 Chat started! Type 'help' for commands or 'quit' to exit.")
        print("-" * 50)
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'info':
                    self._show_model_info()
                    continue
                elif user_input.lower() == 'models':
                    self._show_available_models()
                    continue
                elif user_input.lower().startswith('switch '):
                    model_name = user_input[7:].strip()
                    self._switch_model(model_name)
                    continue
                elif user_input.lower() == 'clear':
                    self.conversation_history.clear()
                    print("🧹 Conversation history cleared.")
                    continue
                elif user_input.lower() == 'history':
                    self._show_history()
                    continue
                
                # Generate response
                print("🤖 Bot: ", end="", flush=True)
                response = self._generate_response(user_input)
                print(response)
                
                # Add to history
                self._add_to_history(user_input, response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {str(e)}")
                print(f"❌ Error: {str(e)}")
    
    def _generate_response(self, user_input: str) -> str:
        """
        Generate response to user input
        
        Args:
            user_input (str): User's message
            
        Returns:
            str: Generated response
        """
        # Prepare prompt with context
        prompt = self._prepare_prompt(user_input)
        
        # Generate response
        gen_params = self.config.get_generation_params()
        response = self.model_manager.generate_response(
            prompt, 
            max_new_tokens=gen_params.get("max_new_tokens", 100)
        )
        
        return response
    
    def _prepare_prompt(self, user_input: str) -> str:
        """
        Prepare prompt with conversation context

        Args:
            user_input (str): Current user input

        Returns:
            str: Formatted prompt
        """
        # Simpler prompt formatting to avoid multiple assistant responses
        # Only include minimal context to prevent confusion
        if len(self.conversation_history) > 0:
            # Include only the last exchange for context
            last_exchange = self.conversation_history[-1]
            context = f"Human: {last_exchange['user']}\nAssistant: {last_exchange['bot']}\n"
        else:
            context = ""

        # Format the current prompt with clear single response expectation
        prompt = f"{context}Human: {user_input}\nAssistant:"
        return prompt
    
    def _add_to_history(self, user_input: str, response: str):
        """Add exchange to conversation history"""
        max_history = self.config.get_chat_settings().get("max_history", 10)
        
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "bot": response
        })
        
        # Keep only recent history
        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]
    
    def _show_help(self):
        """Show help information"""
        print("\n📖 Available Commands:")
        print("  help     - Show this help message")
        print("  info     - Show current model information")
        print("  models   - Show available models")
        print("  switch <model> - Switch to a different model")
        print("  clear    - Clear conversation history")
        print("  history  - Show conversation history")
        print("  quit/exit/q - Exit the chatbot")
    
    def _show_model_info(self):
        """Show current model information"""
        if self.model_manager:
            info = self.model_manager.get_model_info()
            print(f"\n🔧 Current Model Info:")
            print(f"  Model: {info['model_name']}")
            print(f"  Device: {info['device']}")
            print(f"  GPU: {info.get('gpu_name', 'N/A')}")
            if info.get('gpu_memory_allocated'):
                print(f"  GPU Memory: {info['gpu_memory_allocated']} / {info['gpu_memory_reserved']}")
    
    def _show_available_models(self):
        """Show available models"""
        models = self.config.get_available_models()
        print(f"\n📋 Available Models:")
        for alias, model_info in models.items():
            current = "✓" if model_info["name"] == self.config.get_current_model() else " "
            print(f"  {current} {alias}: {model_info['name']}")
            print(f"    {model_info['description']}")
    
    def _switch_model(self, model_name: str):
        """Switch to a different model"""
        print(f"🔄 Switching to model: {model_name}")
        
        if self.config.set_current_model(model_name):
            self._initialize_model()
            if self.load_model():
                print(f"✅ Successfully switched to {model_name}")
                self._show_model_info()
            else:
                print(f"❌ Failed to load model {model_name}")
        else:
            print(f"❌ Failed to switch to model {model_name}")
    
    def _show_history(self):
        """Show conversation history"""
        if not self.conversation_history:
            print("📝 No conversation history.")
            return
        
        print(f"\n📝 Conversation History ({len(self.conversation_history)} exchanges):")
        for i, exchange in enumerate(self.conversation_history, 1):
            print(f"\n{i}. 👤 {exchange['user']}")
            print(f"   🤖 {exchange['bot']}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="LLM Chatbot with GPU Support")
    parser.add_argument("--config", default="chatbot_config.json", 
                       help="Configuration file path")
    parser.add_argument("--model", help="Model to use (overrides config)")
    parser.add_argument("--no-gpu", action="store_true", 
                       help="Disable GPU usage")
    
    args = parser.parse_args()
    
    # Create chatbot
    chatbot = CLIChatbot(args.config)
    
    # Override model if specified
    if args.model:
        chatbot.config.set_current_model(args.model)
        chatbot._initialize_model()
    
    # Override GPU setting if specified
    if args.no_gpu:
        chatbot.config.set_use_gpu(False)
        chatbot._initialize_model()
    
    # Start chat
    chatbot.chat_loop()

if __name__ == "__main__":
    main()
