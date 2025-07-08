# Installation Guide - PerfAI

This guide covers the installation and setup of PerfAI, a CUDA-accelerated performance regression detection system.

## Prerequisites

### System Requirements
- **GPU**: NVIDIA GPU with compute capability 3.5 or higher
- **CUDA**: CUDA Toolkit 11.0 or later (recommended: 11.8+)
- **OS**: Linux (Ubuntu 18.04+), Windows 10/11, or macOS 10.15+
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: At least 2GB free space

### Software Dependencies
- **C++ Compiler**: GCC 7+ (Linux), MSVC 2019+ (Windows), or Clang 10+ (macOS)
- **Python**: Version 3.8 or later
- **Git**: For source code management
- **CMake**: Version 3.12 or later (optional, for advanced builds)

## Quick Installation

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/perfai.git
cd perfai

# Run automated setup
./scripts/setup_dev.sh
```

The setup script will:
- Check system prerequisites
- Install system dependencies
- Create Python virtual environment
- Install Python packages
- Build CUDA components
- Create initial configuration

### Option 2: Docker Installation

```bash
# Clone repository
git clone https://github.com/your-org/perfai.git
cd perfai

# Build Docker image
docker build -f docker/Dockerfile -t perfai:latest .

# Run with Docker Compose
docker-compose -f docker/docker-compose.yml up perfai
```

### Option 3: Manual Installation

#### Step 1: Install CUDA Toolkit

**Ubuntu/Debian:**
```bash
# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Install CUDA Toolkit
sudo apt-get install cuda-toolkit-11-8
```

**Windows:**
1. Download CUDA Toolkit from [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads)
2. Run the installer and follow instructions
3. Add CUDA to your PATH environment variable

**macOS:**
```bash
# Install using Homebrew (if available)
brew install --cask nvidia-cuda
```

#### Step 2: Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install build-essential cmake git python3 python3-pip python3-dev
```

**CentOS/RHEL:**
```bash
sudo yum groupinstall "Development Tools"
sudo yum install cmake git python3 python3-pip python3-devel
```

**macOS:**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew dependencies
brew install cmake git python3
```

**Windows:**
- Install Visual Studio 2019 or later with C++ development tools
- Install Git for Windows
- Install Python 3.8+ from python.org

#### Step 3: Clone Repository

```bash
git clone https://github.com/your-org/perfai.git
cd perfai
```

#### Step 4: Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt
```

#### Step 5: Build CUDA Components

```bash
# Build using Makefile
make clean
make all

# Test the build
make test
```

#### Step 6: Configuration

```bash
# Copy default configuration
cp config.json config.local.json

# Edit configuration as needed
# vim config.local.json
```

## Verification

### Test CUDA Installation

```bash
# Check CUDA version
nvcc --version

# Check GPU status
nvidia-smi

# Test CUDA benchmark
./bin/cuda_benchmark --test-mode --verbose
```

### Test Python Environment

```bash
# Activate virtual environment (if not already active)
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Run Python tests
python -m pytest tests/ -v

# Test data pipeline
python src/pipeline/data_pipeline.py --help
```

### Run Quick Benchmark

```bash
# Run a quick performance test
./scripts/run_benchmark.sh --benchmark-only --verbose --matrix-sizes 512,1024
```

## Troubleshooting

### Common Issues

#### CUDA Not Found
```bash
# Check CUDA installation
ls /usr/local/cuda/bin/nvcc  # Linux
# or
where nvcc  # Windows

# Add CUDA to PATH (if needed)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

#### Build Errors
```bash
# Check build dependencies
make clean
make all VERBOSE=1  # For detailed output

# For specific architecture
make all CUDA_ARCH=sm_80  # For RTX 30xx series
make all CUDA_ARCH=sm_86  # For RTX 40xx series
```

#### Python Import Errors
```bash
# Ensure virtual environment is active
which python  # Should point to venv/bin/python

# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

#### GPU Access Issues
```bash
# Check GPU permissions
nvidia-smi

# Check Docker GPU access (if using Docker)
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
```

### Performance Issues

#### Low Performance
- Check GPU thermal throttling: `nvidia-smi -q -d TEMPERATURE`
- Verify GPU utilization: `nvidia-smi -l 1`
- Check system RAM usage: `free -h`
- Monitor CPU usage: `top` or `htop`

#### Memory Issues
- Reduce matrix sizes in configuration
- Check available GPU memory: `nvidia-smi --query-gpu=memory.free --format=csv`
- Adjust batch sizes or iterations

### Getting Help

1. **Check Logs**: Look in `logs/perfai.log` for detailed error messages
2. **Run Tests**: Execute `python -m pytest tests/ -v` to identify issues
3. **Verbose Mode**: Use `--verbose` flag with scripts for detailed output
4. **GitHub Issues**: Report bugs at [GitHub Issues](https://github.com/your-org/perfai/issues)

## Next Steps

After successful installation:

1. **Review Configuration**: Edit `config.local.json` for your environment
2. **Create Baseline**: Run `./scripts/run_benchmark.sh` to establish performance baseline
3. **Integration**: Set up CI/CD integration (see `docs/USAGE.md`)
4. **Development**: See `docs/CONTRIBUTING.md` for development guidelines

## Advanced Installation Options

### Custom CUDA Architecture

```bash
# For specific GPU architectures
make CUDA_ARCH=sm_70  # Tesla V100
make CUDA_ARCH=sm_75  # RTX 20xx series
make CUDA_ARCH=sm_80  # A100, RTX 30xx series
make CUDA_ARCH=sm_86  # RTX 40xx series
```

### Multi-GPU Setup

```bash
# Edit config.json for multi-GPU
{
  "gpu_devices": [0, 1, 2, 3],
  "parallel_execution": true
}
```

### Cluster Installation

For distributed setups, see `docs/cluster_setup.md` (advanced users only).

---

For additional help, please refer to the [Usage Guide](USAGE.md) or [Contributing Guide](CONTRIBUTING.md).
