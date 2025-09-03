#!/bin/bash

# NVIDIA NIM-compatible Embedding Service Startup Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8009}
WORKERS=${WORKERS:-1}
LOG_LEVEL=${LOG_LEVEL:-"info"}
CONFIG_PATH=${CONFIG_PATH:-"config/models.yaml"}
RELOAD=${RELOAD:-false}

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if GPU is available
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            print_success "NVIDIA GPU detected"
            nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
        else
            print_warning "nvidia-smi found but GPU not accessible"
        fi
    else
        print_warning "nvidia-smi not found - running in CPU mode"
    fi
}

# Function to check Python environment
check_python() {
    if ! command -v python &> /dev/null; then
        print_error "Python not found"
        exit 1
    fi
    
    python_version=$(python --version 2>&1)
    print_info "Using $python_version"
    
    # Check if required packages are installed
    if ! python -c "import torch" &> /dev/null; then
        print_error "PyTorch not installed. Please install requirements: pip install -r requirements.txt"
        exit 1
    fi
    
    if ! python -c "import fastapi" &> /dev/null; then
        print_error "FastAPI not installed. Please install requirements: pip install -r requirements.txt"
        exit 1
    fi
    
    print_success "Python environment check passed"
}

# Function to check configuration
check_config() {
    if [ ! -f "$CONFIG_PATH" ]; then
        print_error "Configuration file not found: $CONFIG_PATH"
        exit 1
    fi
    
    print_success "Configuration file found: $CONFIG_PATH"
}

# Function to create necessary directories
create_directories() {
    directories=("logs" "cache" "models")
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_info "Created directory: $dir"
        fi
    done
}

# Function to show usage
show_usage() {
    cat << EOF
NVIDIA NIM-compatible Embedding Service

Usage: $0 [OPTIONS]

Options:
    --host HOST         Host to bind the server to (default: 0.0.0.0)
    --port PORT         Port to bind the server to (default: 8009)
    --workers WORKERS   Number of worker processes (default: 1)
    --log-level LEVEL   Logging level: debug|info|warning|error|critical (default: info)
    --config CONFIG     Path to configuration file (default: config/models.yaml)
    --reload            Enable auto-reload for development
    --dev               Development mode (implies --reload --log-level debug)
    --help, -h          Show this help message

Environment Variables:
    HOST, PORT, WORKERS, LOG_LEVEL, CONFIG_PATH, RELOAD

Examples:
    $0                                    # Start with default settings
    $0 --port 8080 --workers 4          # Custom port and workers
    $0 --dev                             # Development mode
    $0 --config my_config.yaml           # Custom config file

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --reload)
            RELOAD=true
            shift
            ;;
        --dev)
            RELOAD=true
            LOG_LEVEL="debug"
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_info "Starting NVIDIA NIM-compatible Embedding Service"
    print_info "================================"
    
    # System checks
    check_gpu
    check_python
    check_config
    create_directories
    
    print_info "================================"
    print_info "Configuration:"
    print_info "  Host: $HOST"
    print_info "  Port: $PORT"
    print_info "  Workers: $WORKERS"
    print_info "  Log Level: $LOG_LEVEL"
    print_info "  Config: $CONFIG_PATH"
    print_info "  Reload: $RELOAD"
    print_info "================================"
    
    # Export environment variables
    export HOST PORT WORKERS LOG_LEVEL CONFIG_PATH RELOAD
    
    # Build command arguments
    args=(
        "--host" "$HOST"
        "--port" "$PORT"
        "--workers" "$WORKERS"
        "--log-level" "$LOG_LEVEL"
        "--config-path" "$CONFIG_PATH"
    )
    
    if [ "$RELOAD" = true ]; then
        args+=("--reload")
    fi
    
    # Start the service
    print_success "Starting embedding service..."
    exec python embedding_service.py "${args[@]}"
}

# Set trap to handle signals
trap 'print_info "Shutting down embedding service..."; exit 0' SIGINT SIGTERM

# Run main function
main
