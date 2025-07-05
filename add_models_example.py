#!/usr/bin/env python3
"""
Example script showing how to add different types of LLM models to the chatbot
This demonstrates the flexibility of the architecture
"""

from config import ChatbotConfig

def add_popular_models():
    """Add various popular LLM models to the configuration"""
    
    config = ChatbotConfig()
    
    print("🔧 Adding popular LLM models to chatbot configuration...")
    
    # LLaMA family models
    config.add_model(
        alias="llama2-13b",
        model_name="meta-llama/Llama-2-13b-chat-hf",
        description="LLaMA 2 13B Chat model (requires HF access token)"
    )
    
    config.add_model(
        alias="code-llama",
        model_name="codellama/CodeLlama-7b-Python-hf",
        description="Code Llama 7B Python specialist"
    )
    
    # Mistral family
    config.add_model(
        alias="mistral-7b-instruct",
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        description="Mistral 7B Instruct v0.2"
    )
    
    config.add_model(
        alias="mixtral-8x7b",
        model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
        description="Mixtral 8x7B MoE model (large, requires significant GPU memory)"
    )
    
    # Microsoft models
    config.add_model(
        alias="phi-3-mini",
        model_name="microsoft/Phi-3-mini-4k-instruct",
        description="Microsoft Phi-3 Mini 4K context"
    )
    
    # Google models
    config.add_model(
        alias="gemma-2b",
        model_name="google/gemma-2b-it",
        description="Google Gemma 2B Instruction Tuned"
    )
    
    config.add_model(
        alias="gemma-7b",
        model_name="google/gemma-7b-it",
        description="Google Gemma 7B Instruction Tuned"
    )
    
    # Specialized models
    config.add_model(
        alias="starcoder",
        model_name="bigcode/starcoder",
        description="StarCoder - Code generation specialist (15B params)"
    )
    
    config.add_model(
        alias="wizardcoder",
        model_name="WizardLM/WizardCoder-Python-7B-V1.0",
        description="WizardCoder Python specialist"
    )
    
    # Smaller efficient models
    config.add_model(
        alias="tinyllama",
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        description="TinyLlama 1.1B - Very fast and lightweight"
    )
    
    # Instruction-tuned models
    config.add_model(
        alias="vicuna-7b",
        model_name="lmsys/vicuna-7b-v1.5",
        description="Vicuna 7B v1.5 - LLaMA fine-tuned for conversation"
    )
    
    config.add_model(
        alias="alpaca-7b",
        model_name="chavinlo/alpaca-native",
        description="Alpaca 7B - Instruction-following model"
    )
    
    # Quantized models (smaller memory footprint)
    config.add_model(
        alias="llama2-7b-gptq",
        model_name="TheBloke/Llama-2-7B-Chat-GPTQ",
        description="LLaMA 2 7B Chat GPTQ quantized (lower memory usage)"
    )
    
    print("✅ Successfully added popular models!")
    print("\n📋 Available models now include:")
    
    models = config.get_available_models()
    for alias, info in models.items():
        print(f"  • {alias}: {info['description']}")
    
    print(f"\n💡 To use any model, run:")
    print(f"   python chatbot.py --model <model_alias>")
    print(f"   or use 'switch <model_alias>' command in the chat")

def add_local_model_example():
    """Example of how to add a local model"""
    
    config = ChatbotConfig()
    
    # Example: Add a local model you've downloaded or fine-tuned
    config.add_model(
        alias="my-local-model",
        model_name="/path/to/your/local/model",  # Local path
        description="My custom fine-tuned model"
    )
    
    print("📁 Added example for local model")
    print("   Edit the path to point to your actual local model directory")

def show_model_requirements():
    """Show memory and access requirements for different models"""
    
    print("\n💾 Model Memory Requirements (approximate):")
    print("  • GPT-2 (124M):     ~500MB GPU memory")
    print("  • GPT-2 Large (774M): ~3GB GPU memory") 
    print("  • Phi-2 (2.7B):     ~6GB GPU memory")
    print("  • LLaMA 2 7B:       ~14GB GPU memory")
    print("  • Mistral 7B:       ~14GB GPU memory")
    print("  • LLaMA 2 13B:      ~26GB GPU memory")
    print("  • Mixtral 8x7B:     ~90GB GPU memory (requires multiple GPUs)")
    
    print("\n🔑 Access Requirements:")
    print("  • Most models: No special access needed")
    print("  • LLaMA 2 models: Require HuggingFace access token")
    print("  • Some Gemma models: May require access approval")
    
    print("\n⚡ Performance Tips:")
    print("  • Use quantized models (GPTQ/AWQ) for lower memory usage")
    print("  • Start with smaller models (GPT-2, Phi-2, TinyLlama)")
    print("  • Use half precision (fp16) for GPU efficiency")

if __name__ == "__main__":
    print("🤖 LLM Chatbot - Model Configuration Examples")
    print("=" * 50)
    
    add_popular_models()
    add_local_model_example()
    show_model_requirements()
    
    print("\n🚀 Your chatbot now supports many different model types!")
    print("   The architecture automatically handles different model formats.")
