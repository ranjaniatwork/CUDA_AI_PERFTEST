# PerfAI: CUDA-Accelerated Performance Regression Detection
# Makefile for building CUDA components

# CUDA Configuration
CUDA_PATH ?= /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc
CUDA_ARCH ?= sm_70

# Compiler flags
NVCCFLAGS = -arch=$(CUDA_ARCH) -O3 -std=c++14 -Xcompiler -fPIC
CPPFLAGS = -Isrc/kernel -I$(CUDA_PATH)/include
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lnvToolsExt

# Directories
SRC_DIR = src/kernel
BUILD_DIR = build
BIN_DIR = bin

# Source files
CUDA_SOURCES = $(wildcard $(SRC_DIR)/*.cu)
CUDA_OBJECTS = $(CUDA_SOURCES:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)

# Main executable
TARGET = $(BIN_DIR)/cuda_benchmark

# Default target
all: directories $(TARGET)

# Create necessary directories
directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

# Build CUDA objects
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@echo "Compiling $<..."
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -c $< -o $@

# Link executable
$(TARGET): $(CUDA_OBJECTS) $(BUILD_DIR)/main.o
	@echo "Linking $(TARGET)..."
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

# Build main.cpp
$(BUILD_DIR)/main.o: main.cpp
	@echo "Compiling main.cpp..."
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	rm -rf $(BIN_DIR)

# Install dependencies (Ubuntu/Debian)
install-deps:
	@echo "Installing dependencies..."
	sudo apt-get update
	sudo apt-get install -y build-essential cmake
	@echo "Please ensure CUDA Toolkit is installed separately"

# Run tests
test: $(TARGET)
	@echo "Running CUDA benchmark tests..."
	./$(TARGET) --test-mode

# Run benchmark suite
benchmark: $(TARGET)
	@echo "Running performance benchmark..."
	./$(TARGET) --matrix-sizes 512,1024,2048 --iterations 10 --output-format json

# Profile with nsight
profile: $(TARGET)
	@echo "Running with NVIDIA Nsight Compute profiling..."
	ncu --target-processes all --set full ./$(TARGET) --matrix-sizes 1024 --iterations 5

# Help target
help:
	@echo "PerfAI CUDA Build System"
	@echo "Available targets:"
	@echo "  all        - Build all components (default)"
	@echo "  clean      - Clean build artifacts"
	@echo "  test       - Run unit tests"
	@echo "  benchmark  - Run performance benchmark"
	@echo "  profile    - Run with NVIDIA profiling"
	@echo "  help       - Show this help message"
	@echo ""
	@echo "Configuration:"
	@echo "  CUDA_PATH  - Path to CUDA installation (default: /usr/local/cuda)"
	@echo "  CUDA_ARCH  - CUDA architecture (default: sm_70)"

# Development targets
dev-setup: install-deps
	@echo "Setting up development environment..."
	pip install -r requirements.txt
	pre-commit install

# CI targets
ci-build: directories $(TARGET)
	@echo "CI build completed"

ci-test: $(TARGET) test
	@echo "CI tests completed"

.PHONY: all clean install-deps test benchmark profile help dev-setup ci-build ci-test directories
