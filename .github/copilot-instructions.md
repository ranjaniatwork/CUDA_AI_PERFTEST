<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# PerfAI: CUDA-Accelerated Autonomous Performance Regression Detection

This is an enterprise-grade capstone project that combines CUDA GPU programming with AI-powered performance analysis.

## Project Context
- **Course**: CUDA at Scale for the Enterprise - Capstone Project
- **Goal**: Automated performance regression detection for AI workloads
- **Technologies**: CUDA, cuBLAS, Python, scikit-learn, GitHub Actions
- **Target**: Enterprise CI/CD performance monitoring

## Code Guidelines

### CUDA Components
- Use modern CUDA C++ (C++14+) with proper error handling
- Implement cuBLAS for optimized matrix operations
- Add nvToolsExt markers for profiling integration
- Focus on memory bandwidth optimization and kernel efficiency
- Include comprehensive performance telemetry

### Python Components  
- Use scikit-learn for anomaly detection (Isolation Forest, LSTM)
- Implement clean data pipeline with pandas/numpy
- Create modular CLI interface with argparse
- Add comprehensive logging and error handling
- Generate professional CSV/JSON reports

### Performance Monitoring
- Implement CUDA events for precise timing
- Use unified memory counters for bandwidth analysis
- Add GPU utilization and occupancy metrics
- Include memory usage and transfer profiling
- Integrate with NVIDIA Nsight Compute when available

### AI Analysis
- Implement baseline establishment from historical data
- Use statistical models for regression detection
- Add confidence scoring and threshold management
- Create visual analytics with matplotlib/seaborn
- Include automated alerting and reporting

### Enterprise Features
- Docker containerization for reproducible environments
- GitHub Actions CI/CD integration
- Comprehensive documentation and examples
- Production-ready error handling and logging
- Scalable architecture for multiple GPU configurations

## File Structure
- `src/kernel/` - CUDA kernels and cuBLAS implementations
- `src/analysis/` - Python AI analysis and detection algorithms
- `src/pipeline/` - Automation and orchestration logic
- `data/runs/` - Performance metrics and baseline data
- `scripts/` - Build, test, and deployment automation
- `tests/` - Unit tests and integration validation
