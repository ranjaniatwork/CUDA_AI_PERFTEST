#!/bin/bash
# PerfAI Benchmark Runner Script
# Runs CUDA benchmarks and collects performance data

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default configuration
CONFIG_FILE="config.json"
OUTPUT_DIR=""
BENCHMARK_ONLY=false
VERBOSE=false
MATRIX_SIZES="512,1024,2048"
ITERATIONS=10

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --benchmark-only)
            BENCHMARK_ONLY=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --matrix-sizes)
            MATRIX_SIZES="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --help)
            echo "PerfAI Benchmark Runner"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config FILE          Configuration file (default: config.json)"
            echo "  --output-dir DIR       Output directory for results"
            echo "  --benchmark-only       Run benchmark only (no analysis)"
            echo "  --verbose              Enable verbose output"
            echo "  --matrix-sizes SIZES   Comma-separated matrix sizes (default: 512,1024,2048)"
            echo "  --iterations N         Number of iterations (default: 10)"
            echo "  --help                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --verbose"
            echo "  $0 --benchmark-only --matrix-sizes 1024,2048"
            echo "  $0 --config custom_config.json --output-dir results/"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}PerfAI Benchmark Runner${NC}"
echo "======================"

# Check if executable exists
if [ ! -f "bin/cuda_benchmark" ]; then
    echo -e "${RED}ERROR: CUDA benchmark executable not found${NC}"
    echo "Please run ./scripts/build.sh first"
    exit 1
fi

# Check GPU availability
if ! nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}WARNING: nvidia-smi not available, GPU status unknown${NC}"
fi

# Create output directory if specified
if [ -n "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    OUTPUT_ARGS="--output-dir $OUTPUT_DIR"
else
    OUTPUT_ARGS=""
fi

# Generate session ID
SESSION_ID="session_$(date +%Y%m%d_%H%M%S)"
echo "Session ID: $SESSION_ID"

# Set verbose flag
VERBOSE_FLAG=""
if [ "$VERBOSE" = true ]; then
    VERBOSE_FLAG="--verbose"
fi

# Prepare benchmark arguments
BENCHMARK_ARGS="--matrix-sizes $MATRIX_SIZES --iterations $ITERATIONS --session-id $SESSION_ID --output-format json"

if [ "$VERBOSE" = true ]; then
    BENCHMARK_ARGS="$BENCHMARK_ARGS --verbose"
fi

echo ""
echo -e "${GREEN}Starting benchmark...${NC}"
echo "Configuration:"
echo "  Matrix sizes: $MATRIX_SIZES"
echo "  Iterations: $ITERATIONS"
echo "  Session ID: $SESSION_ID"
echo "  Benchmark only: $BENCHMARK_ONLY"
echo ""

# Record start time
START_TIME=$(date +%s)

if [ "$BENCHMARK_ONLY" = true ]; then
    # Run benchmark only
    echo -e "${BLUE}Running CUDA benchmark...${NC}"
    
    if [ -n "$OUTPUT_DIR" ]; then
        ./bin/cuda_benchmark $BENCHMARK_ARGS --output-file "$OUTPUT_DIR/benchmark_results_${SESSION_ID}.json"
    else
        ./bin/cuda_benchmark $BENCHMARK_ARGS
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Benchmark completed successfully${NC}"
    else
        echo -e "${RED}✗ Benchmark failed${NC}"
        exit 1
    fi
    
else
    # Run full pipeline
    echo -e "${BLUE}Running full PerfAI pipeline...${NC}"
    
    PIPELINE_ARGS="--config $CONFIG_FILE $OUTPUT_ARGS $VERBOSE_FLAG"
    
    if [ "$VERBOSE" = true ]; then
        echo "Command: python src/pipeline/perfai_pipeline.py $PIPELINE_ARGS"
    fi
    
    python src/pipeline/perfai_pipeline.py $PIPELINE_ARGS
    PIPELINE_EXIT_CODE=$?
    
    echo ""
    case $PIPELINE_EXIT_CODE in
        0)
            echo -e "${GREEN}✓ Pipeline completed successfully - No issues detected${NC}"
            ;;
        1)
            echo -e "${YELLOW}⚠ Pipeline completed - Medium severity issues detected${NC}"
            ;;
        2)
            echo -e "${RED}✗ Pipeline completed - High severity performance regression detected${NC}"
            ;;
        3)
            echo -e "${RED}✗ Pipeline failed due to error${NC}"
            exit 1
            ;;
        *)
            echo -e "${RED}✗ Pipeline failed with unknown exit code: $PIPELINE_EXIT_CODE${NC}"
            exit 1
            ;;
    esac
fi

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo -e "${GREEN}Benchmark completed in ${ELAPSED} seconds${NC}"

# Show results location if output directory was specified
if [ -n "$OUTPUT_DIR" ]; then
    echo -e "${BLUE}Results saved to: $OUTPUT_DIR${NC}"
    
    if [ -f "$OUTPUT_DIR/performance_analysis_report.txt" ]; then
        echo ""
        echo -e "${BLUE}Performance Analysis Summary:${NC}"
        head -n 20 "$OUTPUT_DIR/performance_analysis_report.txt"
        echo ""
        echo -e "Full report available at: ${BLUE}$OUTPUT_DIR/performance_analysis_report.txt${NC}"
    fi
fi

echo ""
