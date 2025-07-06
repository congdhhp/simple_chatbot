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
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 3.9 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def check_cuda():
    """Check CUDA availability."""
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA is available")
            return True
        else:
            print("⚠️  CUDA not detected - will use CPU mode")
            return False
    except FileNotFoundError:
        print("⚠️  nvidia-smi not found - CUDA may not be available")
        return False


def setup_virtual_environment():
    """Create and setup virtual environment."""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    return run_command("python -m venv venv", "Creating virtual environment")


def install_dependencies():
    """Install required dependencies."""
    system = platform.system().lower()
    
    if system == "windows":
        activate_cmd = "venv\\Scripts\\activate"
    else:
        activate_cmd = "source venv/bin/activate"
    
    # Install PyTorch with CUDA support
    pytorch_cmd = f"{activate_cmd} && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    if not run_command(pytorch_cmd, "Installing PyTorch with CUDA support"):
        return False
    
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
        test_cmd = "source venv/bin/activate && python chatbot.py --help"
    
    return run_command(test_cmd, "Testing installation")


def main():
    """Main setup function."""
    print("🤖 Simple CLI Chatbot Setup")
    print("=" * 50)
    
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
    else:
        print("  source venv/bin/activate")
    
    print("  python chatbot.py")
    print("\nFor help:")
    print("  python chatbot.py --help")


if __name__ == "__main__":
    main()
