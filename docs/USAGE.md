# Usage Guide - PerfAI

This guide covers how to use PerfAI for CUDA performance regression detection in various scenarios.

## Quick Start

### Basic Performance Testing

```bash
# Run a simple benchmark
./scripts/run_benchmark.sh --verbose

# Run benchmark with specific matrix sizes
./scripts/run_benchmark.sh --matrix-sizes 1024,2048,4096 --iterations 5

# Run benchmark only (no analysis)
./scripts/run_benchmark.sh --benchmark-only --output-dir results/
```

### Full Pipeline Execution

```bash
# Run complete pipeline with default configuration
python src/pipeline/perfai_pipeline.py

# Run with custom configuration
python src/pipeline/perfai_pipeline.py --config config.local.json --output-dir results/

# Run with specific baseline
python src/pipeline/perfai_pipeline.py --baseline my_baseline --verbose
```

## Command-Line Interfaces

### CUDA Benchmark Executable

The `cuda_benchmark` executable provides direct access to GPU workloads:

```bash
# Basic usage
./bin/cuda_benchmark --matrix-sizes 512,1024,2048 --iterations 10

# Available options
./bin/cuda_benchmark --help

# Output formats
./bin/cuda_benchmark --output-format json --output-file results.json
./bin/cuda_benchmark --output-format csv --output-file results.csv

# Specific workloads
./bin/cuda_benchmark --workloads gemm,conv,bandwidth

# Test mode (quick validation)
./bin/cuda_benchmark --test-mode --verbose
```

### Data Pipeline

Manage performance data collection and baseline creation:

```bash
# Collect performance data
python src/pipeline/data_pipeline.py collect --executable ./bin/cuda_benchmark

# Create baseline from historical data
python src/pipeline/data_pipeline.py baseline --baseline-name stable_v1.0

# Validate existing data
python src/pipeline/data_pipeline.py validate
```

### Anomaly Detection

Run AI-powered performance analysis:

```bash
# Analyze current vs baseline performance
python src/analysis/detect_anomaly.py \
  --baseline data/baselines/stable_v1.0.json \
  --current data/runs/latest_run.json \
  --output analysis_results/

# Custom confidence threshold
python src/analysis/detect_anomaly.py \
  --baseline data/baselines/stable_v1.0.json \
  --current data/runs/latest_run.json \
  --confidence 0.99 \
  --contamination 0.05
```

## Configuration

### Configuration File Structure

The `config.json` file controls all aspects of PerfAI behavior:

```json
{
  "data_dir": "data",
  "benchmark_executable": "./bin/cuda_benchmark",
  "confidence_threshold": 0.95,
  "contamination_rate": 0.1,
  
  "benchmark_config": {
    "matrix_sizes": [512, 1024, 2048, 4096],
    "iterations": 10,
    "workloads": ["gemm", "conv", "bandwidth"]
  },
  
  "alert_thresholds": {
    "high_severity": 0.3,
    "medium_severity": 0.1,
    "low_severity": 0.05
  }
}
```

### Key Configuration Options

#### Benchmark Settings
- `matrix_sizes`: Array of matrix dimensions for GEMM tests
- `iterations`: Number of repetitions per test
- `workloads`: Types of workloads to run (`gemm`, `conv`, `bandwidth`, `all`)
- `output_format`: Data output format (`json`, `csv`, `text`)

#### Analysis Settings
- `confidence_threshold`: Statistical confidence level (0.90-0.99)
- `contamination_rate`: Expected anomaly rate for Isolation Forest (0.01-0.20)
- `alert_thresholds`: Severity level thresholds

#### Data Management
- `data_dir`: Directory for storing performance data
- `baseline_auto_update`: Automatically update baselines with stable data
- `baseline_min_samples`: Minimum samples required for baseline creation

## Workflow Scenarios

### Scenario 1: CI/CD Integration

Set up automated performance regression detection in your CI/CD pipeline:

```yaml
# .github/workflows/performance.yml
- name: Performance Regression Test
  run: |
    python src/pipeline/perfai_pipeline.py \
      --config ci_config.json \
      --output-dir ${{ github.workspace }}/perf_results
    
    # Exit code indicates severity:
    # 0 = success, 1 = medium severity, 2 = high severity, 3 = error
```

#### CI Configuration Example

```json
{
  "benchmark_config": {
    "matrix_sizes": [1024, 2048],
    "iterations": 5,
    "workloads": ["gemm"]
  },
  "alert_thresholds": {
    "high_severity": 0.20,
    "medium_severity": 0.10
  },
  "ci_cd": {
    "fail_on_high_severity": true,
    "fail_on_medium_severity": false
  }
}
```

### Scenario 2: Development Testing

Test performance impact of code changes during development:

```bash
# Before code changes - establish baseline
git checkout main
./scripts/run_benchmark.sh --output-dir baseline_main/
python src/pipeline/data_pipeline.py baseline --baseline-name main_branch

# After code changes - test for regressions
git checkout feature_branch
python src/pipeline/perfai_pipeline.py \
  --baseline main_branch \
  --output-dir feature_results/
```

### Scenario 3: Production Monitoring

Monitor performance in production environments:

```bash
# Daily performance monitoring (via cron)
0 2 * * * cd /path/to/perfai && python src/pipeline/perfai_pipeline.py \
  --config production_config.json \
  --output-dir /var/log/perfai/$(date +\%Y\%m\%d)
```

#### Production Configuration

```json
{
  "benchmark_config": {
    "matrix_sizes": [2048, 4096, 8192],
    "iterations": 20,
    "workloads": ["all"]
  },
  "database": {
    "enabled": true,
    "path": "/var/lib/perfai/performance.db",
    "retention_days": 90
  },
  "logging": {
    "level": "INFO",
    "file": "/var/log/perfai/perfai.log"
  }
}
```

### Scenario 4: Research and Analysis

Detailed performance analysis for research purposes:

```bash
# Extended benchmark suite
python src/pipeline/perfai_pipeline.py \
  --config research_config.json \
  --output-dir research_results/

# Generate detailed visualizations
python -c "
from src.analysis.detect_anomaly import PerformanceAnalyzer
import pandas as pd

analyzer = PerformanceAnalyzer()
data = pd.read_json('research_results/current_performance_data.json')
analyzer.visualize_results(data, {}, 'research_results/detailed_plots/')
"
```

## Output and Reports

### Performance Analysis Report

The main output is a comprehensive text report:

```
===============================================================================
PERFAI PERFORMANCE REGRESSION ANALYSIS REPORT
===============================================================================
Generated: 2024-01-15T10:30:00Z
Total Samples Analyzed: 50
Confidence Threshold: 0.95

EXECUTIVE SUMMARY
----------------------------------------
Total Anomalies Detected: 5
Severity Level: MEDIUM
Recommendation: Performance monitoring recommended - potential regression detected

ISOLATION FOREST ANALYSIS
----------------------------------------
Anomalies Detected: 3
Detection Rate: 6.00%

STATISTICAL ANALYSIS
----------------------------------------
Anomalies Detected: 2
Anomalies by Workload Type:
  - matrix_multiplication: 1
  - convolution: 1

DETAILED RECOMMENDATIONS
----------------------------------------
• Continue monitoring performance trends
• Review recent optimizations or changes
• Consider running extended benchmarks
```

### JSON Output Format

Machine-readable results for integration:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "total_samples": 50,
  "summary": {
    "total_anomalies": 5,
    "severity": "medium",
    "recommendation": "Performance monitoring recommended"
  },
  "isolation_forest": {
    "anomaly_count": 3,
    "anomaly_indices": [5, 23, 47]
  },
  "statistical": {
    "anomaly_count": 2,
    "anomalies": [...]
  }
}
```

### Visualizations

Generated plots include:
- **Time Series**: Performance metrics over time
- **Distributions**: Histogram of performance values
- **Correlation Heatmap**: Relationships between metrics
- **Anomaly Scatter**: Anomalies highlighted in performance space

## Data Management

### Baseline Management

```bash
# List available baselines
ls data/baselines/

# Create baseline from specific data
python src/pipeline/data_pipeline.py baseline \
  --data-source data/runs/stable_period.json \
  --baseline-name stable_v1.0

# Update existing baseline with new data
python src/pipeline/data_pipeline.py baseline \
  --baseline-name stable_v1.1 \
  --min-samples 100
```

### Data Retention

Configure automatic data cleanup:

```json
{
  "database": {
    "retention_days": 90,
    "cleanup_on_startup": true
  }
}
```

### Backup and Export

```bash
# Export performance database
sqlite3 data/performance.db ".dump" > backup.sql

# Export specific date range
python -c "
from src.pipeline.data_pipeline import DataPipeline
pipeline = DataPipeline()
df = pipeline.load_performance_data(days_back=30)
df.to_csv('last_30_days.csv', index=False)
"
```

## Advanced Usage

### Custom Workloads

Extend PerfAI with custom CUDA kernels:

1. Add kernel implementation to `src/kernel/custom_kernels.cu`
2. Update `workload_engine.h` with new workload method
3. Implement workload in `workload_engine.cu`
4. Update configuration to include new workload type

### Multi-GPU Support

Configure for multiple GPUs:

```json
{
  "gpu_config": {
    "devices": [0, 1, 2, 3],
    "parallel_execution": true,
    "device_selection": "auto"
  }
}
```

### Integration with Monitoring Systems

#### Prometheus Integration

```python
# Custom metrics exporter
from prometheus_client import Gauge, start_http_server

gpu_performance = Gauge('gpu_performance_gflops', 'GPU Performance in GFLOPS')
anomaly_detected = Gauge('performance_anomaly_detected', 'Performance Anomaly Detection')

# Export metrics
start_http_server(8000)
```

#### Grafana Dashboard

Import the provided Grafana dashboard configuration from `docs/grafana_dashboard.json`.

## Troubleshooting

### Common Issues

#### Low Performance Detection
```bash
# Check GPU thermal throttling
nvidia-smi -q -d TEMPERATURE

# Verify GPU isn't shared with other processes
nvidia-smi

# Run with single workload for debugging
./bin/cuda_benchmark --workloads gemm --matrix-sizes 1024 --verbose
```

#### False Positives
```bash
# Increase confidence threshold
python src/pipeline/perfai_pipeline.py --config high_confidence_config.json

# Use larger baseline dataset
python src/pipeline/data_pipeline.py baseline --min-samples 200
```

#### Memory Issues
```bash
# Reduce matrix sizes
./bin/cuda_benchmark --matrix-sizes 512,1024 --verbose

# Check available GPU memory
nvidia-smi --query-gpu=memory.free --format=csv
```

### Debug Mode

Enable debug logging:

```json
{
  "logging": {
    "level": "DEBUG",
    "file": "debug.log"
  }
}
```

## Performance Optimization

### Benchmark Optimization
- Use appropriate matrix sizes for your GPU memory
- Balance iterations vs execution time
- Consider thermal throttling during extended runs

### Analysis Optimization
- Use appropriate baseline size (50-1000 samples)
- Tune contamination rate based on expected anomaly frequency
- Enable database for faster historical data access

---

For more advanced topics, see the [API Documentation](API.md) or [Contributing Guide](CONTRIBUTING.md).
