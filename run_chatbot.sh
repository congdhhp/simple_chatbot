#!/bin/bash
# Simple CLI Chatbot - Ubuntu/WSL2 Launcher
# This script activates the virtual environment and starts the chatbot
# Optimized for Ubuntu 24.04 WSL2 + RTX 5060 Ti + CUDA 12.0

echo "🤖 Starting Simple CLI Chatbot..."

# Check if virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run setup.py first:"
    echo "  python3 setup.py"
    exit 1
fi

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "✅ CUDA detected"
    # Check GPU memory
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ "$GPU_MEM" -ge 15000 ]; then
        echo "🎯 RTX 5060 Ti (16GB) detected - optimal performance expected"
    else
        echo "⚠️  GPU memory: ${GPU_MEM}MB (expected 16GB for RTX 5060 Ti)"
    fi
else
    echo "⚠️  CUDA not detected - will use CPU mode"
fi

# Activate virtual environment and start chatbot
echo "🚀 Activating virtual environment..."
source venv/bin/activate

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

echo "🎮 Starting chatbot..."
python3 chatbot.py "$@"

# Check exit status
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Chatbot exited with an error"
    echo "Press Enter to continue..."
    read
fi
