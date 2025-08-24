#!/usr/bin/env python3
"""
Setup script for Simple CLI Chatbot
Automates the setup process for the chatbot environment.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        # Use bash explicitly for Linux/WSL2 to support source command
        system = platform.system().lower()
        if system == "windows":
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        else:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, executable='/bin/bash')
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10 or higher is required (recommended: Python 3.12.3)")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    if version.major == 3 and version.minor == 12:
        print("🎯 Perfect! Using recommended Python 3.12.x")
    return True


def check_cuda():
    """Check CUDA availability and version for Ubuntu WSL2."""
    try:
        # Check nvidia-smi
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA is available")
            print("🔍 Checking CUDA version...")

            # Check nvcc version
            nvcc_result = subprocess.run("nvcc --version", shell=True, capture_output=True, text=True)
            if nvcc_result.returncode == 0:
                print("✅ CUDA toolkit detected")
                if "release 12.0" in nvcc_result.stdout:
                    print("🎯 Perfect! CUDA 12.0 detected - optimal for RTX 5060 Ti")
                elif "release 12." in nvcc_result.stdout:
                    print("✅ CUDA 12.x detected - compatible")
                else:
                    print("⚠️  CUDA version may not be optimal (recommended: 12.0+)")
            else:
                print("⚠️  nvcc not found - CUDA toolkit may not be installed")
            return True
        else:
            print("⚠️  CUDA not detected - will use CPU mode")
            return False
    except FileNotFoundError:
        print("⚠️  nvidia-smi not found - CUDA may not be available")
        print("💡 For WSL2: Ensure NVIDIA drivers are installed on Windows host")
        return False


def setup_virtual_environment():
    """Create and setup virtual environment."""
    venv_path = Path("venv")

    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True

    # Use python3 for Ubuntu/Linux systems
    system = platform.system().lower()
    if system == "windows":
        python_cmd = "python"
    else:
        python_cmd = "python3"

    return run_command(f"{python_cmd} -m venv venv", "Creating virtual environment")


def install_dependencies():
    """Install required dependencies optimized for Ubuntu WSL2 + CUDA 12.0."""
    system = platform.system().lower()

    if system == "windows":
        activate_cmd = "venv\\Scripts\\activate"
    else:
        activate_cmd = "source venv/bin/activate"

    # Install PyTorch with CUDA 12.1 support (compatible with CUDA 12.0)
    print("🚀 Installing PyTorch with CUDA 12.1 support for RTX 5060 Ti...")
    pytorch_cmd = f"{activate_cmd} && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    if not run_command(pytorch_cmd, "Installing PyTorch with CUDA 12.1 support"):
        return False

    # Install flash-attention (requires compilation, may take time)
    print("⚡ Installing flash-attention (this may take several minutes)...")
    flash_attn_cmd = f"{activate_cmd} && pip install flash-attn --no-build-isolation"
    if not run_command(flash_attn_cmd, "Installing flash-attention"):
        print("⚠️  Flash-attention installation failed, continuing without it")
        print("💡 You can install it later with: pip install flash-attn --no-build-isolation")

    # Install other dependencies
    deps_cmd = f"{activate_cmd} && pip install -r requirements.txt"
    return run_command(deps_cmd, "Installing other dependencies")


def create_directories():
    """Create necessary directories."""
    directories = ["conversations"]  # Only create conversations dir, use HF cache for models

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

    print("✅ Using Hugging Face default cache for models (~/.cache/huggingface)")
    return True


def test_installation():
    """Test if the installation works."""
    system = platform.system().lower()

    if system == "windows":
        test_cmd = "venv\\Scripts\\activate && python chatbot.py --help"
    else:
        test_cmd = "source venv/bin/activate && python3 chatbot.py --help"

    return run_command(test_cmd, "Testing installation")


def main():
    """Main setup function."""
    print("🤖 Simple CLI Chatbot Setup - Ubuntu 24.04 WSL2 + RTX 5060 Ti")
    print("=" * 65)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    check_cuda()
    
    # Setup steps
    steps = [
        (setup_virtual_environment, "Setting up virtual environment"),
        (install_dependencies, "Installing dependencies"),
        (create_directories, "Creating directories"),
        (test_installation, "Testing installation")
    ]
    
    for step_func, step_name in steps:
        print(f"\n📋 {step_name}...")
        if not step_func():
            print(f"\n❌ Setup failed at: {step_name}")
            sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nTo start the chatbot:")

    system = platform.system().lower()
    if system == "windows":
        print("  venv\\Scripts\\activate")
        print("  python chatbot.py")
    else:
        print("  ./run_chatbot.sh")
        print("  # Or manually:")
        print("  source venv/bin/activate")
        print("  python3 chatbot.py")

    print("\nFor help:")
    if system == "windows":
        print("  python chatbot.py --help")
    else:
        print("  python3 chatbot.py --help")


if __name__ == "__main__":
    main()
