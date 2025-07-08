#!/bin/bash
# PerfAI Build Script
# Builds CUDA components and sets up Python environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}PerfAI Build Script${NC}"
echo "=================="

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}ERROR: CUDA not found. Please install CUDA Toolkit${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} CUDA found: $(nvcc --version | grep "release")"

# Create directories
echo "Creating build directories..."
mkdir -p build bin logs

# Build CUDA components
echo "Building CUDA components..."
if make clean && make all; then
    echo -e "${GREEN}✓${NC} CUDA build successful"
else
    echo -e "${RED}✗${NC} CUDA build failed"
    exit 1
fi

# Test CUDA executable
echo "Testing CUDA executable..."
if ./bin/cuda_benchmark --test-mode; then
    echo -e "${GREEN}✓${NC} CUDA benchmark test passed"
else
    echo -e "${RED}✗${NC} CUDA benchmark test failed"
    exit 1
fi

# Set up Python environment
echo "Setting up Python environment..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo -e "${RED}ERROR: Python not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Python found: $($PYTHON_CMD --version)"

# Install Python dependencies
echo "Installing Python dependencies..."
if $PYTHON_CMD -m pip install -r requirements.txt; then
    echo -e "${GREEN}✓${NC} Python dependencies installed"
else
    echo -e "${YELLOW}⚠${NC} Some Python dependencies failed to install"
fi

# Run Python tests
echo "Running Python tests..."
if $PYTHON_CMD -m pytest tests/ -v; then
    echo -e "${GREEN}✓${NC} Python tests passed"
else
    echo -e "${YELLOW}⚠${NC} Some Python tests failed"
fi

# Create sample configuration
if [ ! -f "config.local.json" ]; then
    echo "Creating local configuration..."
    cp config.json config.local.json
    echo -e "${GREEN}✓${NC} Local configuration created"
fi

echo ""
echo -e "${GREEN}Build completed successfully!${NC}"
echo ""
echo "Next steps:"
echo "1. Review configuration in config.local.json"
echo "2. Run: ./scripts/run_benchmark.sh"
echo "3. Or run full pipeline: python src/pipeline/perfai_pipeline.py"
echo ""
