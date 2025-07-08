#!/bin/bash
# PerfAI Setup Script for Development Environment
# Sets up development environment with proper dependencies and tools

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}PerfAI Development Environment Setup${NC}"
echo "=================================="

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
else
    echo -e "${YELLOW}Unknown OS: $OSTYPE${NC}"
    OS="unknown"
fi

echo "Detected OS: $OS"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo ""
echo -e "${BLUE}Checking prerequisites...${NC}"

# Check for git
if command_exists git; then
    echo -e "${GREEN}✓${NC} Git found: $(git --version | head -n1)"
else
    echo -e "${RED}✗${NC} Git not found - please install Git"
    exit 1
fi

# Check for Python
if command_exists python3; then
    PYTHON_CMD=python3
    echo -e "${GREEN}✓${NC} Python found: $(python3 --version)"
elif command_exists python; then
    PYTHON_CMD=python
    echo -e "${GREEN}✓${NC} Python found: $(python --version)"
else
    echo -e "${RED}✗${NC} Python not found - please install Python 3.8+"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ $(echo "$PYTHON_VERSION >= 3.8" | bc -l 2>/dev/null) == "1" ]]; then
    echo -e "${GREEN}✓${NC} Python version $PYTHON_VERSION is supported"
else
    echo -e "${YELLOW}⚠${NC} Python version $PYTHON_VERSION may not be fully supported (recommended: 3.8+)"
fi

# Check for pip
if $PYTHON_CMD -m pip --version >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} pip found"
else
    echo -e "${RED}✗${NC} pip not found - please install pip"
    exit 1
fi

# Check for CUDA (optional)
if command_exists nvcc; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo -e "${GREEN}✓${NC} CUDA found: version $CUDA_VERSION"
    CUDA_AVAILABLE=true
else
    echo -e "${YELLOW}⚠${NC} CUDA not found - GPU features will not be available"
    CUDA_AVAILABLE=false
fi

# Check for GPU
if command_exists nvidia-smi; then
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    echo -e "${GREEN}✓${NC} $GPU_COUNT GPU(s) detected"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits | head -3
else
    echo -e "${YELLOW}⚠${NC} nvidia-smi not found - GPU status unknown"
fi

# Install system dependencies
echo ""
echo -e "${BLUE}Installing system dependencies...${NC}"

case $OS in
    linux)
        if command_exists apt-get; then
            echo "Installing dependencies with apt-get..."
            sudo apt-get update
            sudo apt-get install -y build-essential cmake git pkg-config
            if [ "$CUDA_AVAILABLE" = true ]; then
                sudo apt-get install -y cuda-toolkit-$(echo $CUDA_VERSION | tr '.' '-')
            fi
        elif command_exists yum; then
            echo "Installing dependencies with yum..."
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y cmake git
        else
            echo -e "${YELLOW}⚠${NC} Package manager not detected, please install build tools manually"
        fi
        ;;
    macos)
        if command_exists brew; then
            echo "Installing dependencies with Homebrew..."
            brew install cmake git
        else
            echo -e "${YELLOW}⚠${NC} Homebrew not found, please install Xcode Command Line Tools"
        fi
        ;;
    windows)
        echo -e "${YELLOW}⚠${NC} Windows detected - please ensure Visual Studio Build Tools are installed"
        ;;
esac

# Create Python virtual environment
echo ""
echo -e "${BLUE}Setting up Python virtual environment...${NC}"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
else
    echo -e "${GREEN}✓${NC} Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo -e "${RED}✗${NC} Could not find virtual environment activation script"
    exit 1
fi

echo -e "${GREEN}✓${NC} Virtual environment activated"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo ""
echo -e "${BLUE}Installing Python dependencies...${NC}"

if [ -f "requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    pip install -r requirements.txt
    echo -e "${GREEN}✓${NC} Python dependencies installed"
else
    echo -e "${YELLOW}⚠${NC} requirements.txt not found"
fi

# Install development tools
echo ""
echo -e "${BLUE}Installing development tools...${NC}"

pip install pre-commit black isort flake8 mypy pytest pytest-cov
echo -e "${GREEN}✓${NC} Development tools installed"

# Set up pre-commit hooks
if command_exists pre-commit; then
    echo "Setting up pre-commit hooks..."
    pre-commit install
    echo -e "${GREEN}✓${NC} Pre-commit hooks installed"
fi

# Create necessary directories
echo ""
echo -e "${BLUE}Creating project directories...${NC}"

mkdir -p data/runs data/baselines logs build bin
echo -e "${GREEN}✓${NC} Project directories created"

# Build CUDA components (if available)
if [ "$CUDA_AVAILABLE" = true ]; then
    echo ""
    echo -e "${BLUE}Building CUDA components...${NC}"
    
    if [ -f "Makefile" ]; then
        make clean
        if make all; then
            echo -e "${GREEN}✓${NC} CUDA components built successfully"
            
            # Test CUDA executable
            if [ -f "bin/cuda_benchmark" ]; then
                echo "Testing CUDA executable..."
                if ./bin/cuda_benchmark --test-mode; then
                    echo -e "${GREEN}✓${NC} CUDA benchmark test passed"
                else
                    echo -e "${YELLOW}⚠${NC} CUDA benchmark test failed"
                fi
            fi
        else
            echo -e "${YELLOW}⚠${NC} CUDA build failed - check build dependencies"
        fi
    else
        echo -e "${YELLOW}⚠${NC} Makefile not found - skipping CUDA build"
    fi
fi

# Create sample configuration
if [ ! -f "config.local.json" ]; then
    echo ""
    echo -e "${BLUE}Creating local configuration...${NC}"
    cp config.json config.local.json
    echo -e "${GREEN}✓${NC} Local configuration created at config.local.json"
fi

# Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    echo ""
    echo -e "${BLUE}Creating .gitignore...${NC}"
    cat > .gitignore << EOF
# Build artifacts
build/
bin/
*.o
*.so
*.dll

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt

# Data and logs
data/runs/*.json
logs/*.log
*.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Local configuration
config.local.json
.env
EOF
    echo -e "${GREEN}✓${NC} .gitignore created"
fi

# Summary
echo ""
echo -e "${GREEN}Setup completed successfully!${NC}"
echo ""
echo "Environment summary:"
echo "  Python: $PYTHON_VERSION"
if [ "$CUDA_AVAILABLE" = true ]; then
    echo "  CUDA: $CUDA_VERSION"
else
    echo "  CUDA: Not available"
fi
echo "  Virtual environment: $(pwd)/venv"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
if [ -f "venv/bin/activate" ]; then
    echo "   source venv/bin/activate"
else
    echo "   venv\\Scripts\\activate"
fi
echo "2. Review configuration: config.local.json"
echo "3. Run a test: ./scripts/run_benchmark.sh --benchmark-only --verbose"
echo "4. Start developing!"
echo ""
