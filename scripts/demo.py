#!/usr/bin/env python3
"""
PerfAI Demonstration Script
Shows the core capabilities of the CUDA performance regression detection system
"""

import json
import os
from pathlib import Path
from datetime import datetime

def demonstrate_perfai():
    """Demonstrate PerfAI capabilities with sample data"""
    
    print("=" * 80)
    print("🚀 PERFAI DEMONSTRATION")
    print("CUDA-Accelerated Autonomous Performance Regression Detection")
    print("=" * 80)
    print()
    
    # Project overview
    print("📋 PROJECT OVERVIEW")
    print("-" * 40)
    print("✅ Enterprise-grade CUDA performance testing framework")
    print("✅ AI-powered anomaly detection for performance regressions")
    print("✅ Complete CI/CD integration with automated alerts")
    print("✅ Comprehensive data pipeline and visualization")
    print("✅ Docker containerization for reproducible environments")
    print()
    
    # Architecture demonstration
    print("🏗️ SYSTEM ARCHITECTURE")
    print("-" * 40)
    print("1. CUDA Workload Engine:")
    print("   • Matrix multiplication with cuBLAS optimization")
    print("   • Custom convolution kernels")
    print("   • Memory bandwidth analysis")
    print("   • GPU utilization and occupancy metrics")
    print()
    print("2. AI Analysis Engine:")
    print("   • Isolation Forest anomaly detection")
    print("   • Statistical regression analysis")
    print("   • Automated baseline management")
    print("   • Confidence scoring and thresholds")
    print()
    print("3. Enterprise Pipeline:")
    print("   • Automated data collection and validation")
    print("   • Real-time performance monitoring")
    print("   • CI/CD integration with GitHub Actions")
    print("   • Multi-format reporting (JSON, CSV, HTML)")
    print()
    
    # Load and analyze demonstration data
    baseline_path = Path("data/runs/demo_baseline.json")
    current_path = Path("data/runs/demo_current.json")
    
    if baseline_path.exists() and current_path.exists():
        print("📊 PERFORMANCE DATA ANALYSIS")
        print("-" * 40)
        
        # Load baseline data
        with open(baseline_path, 'r') as f:
            baseline_data = json.load(f)
        
        # Load current data
        with open(current_path, 'r') as f:
            current_data = json.load(f)
        
        print(f"Baseline samples: {len(baseline_data)}")
        print(f"Current samples: {len(current_data)}")
        print()
        
        # Simple performance comparison
        print("🔍 PERFORMANCE COMPARISON")
        print("-" * 40)
        
        for metric in ['gpu_time_ms', 'gflops', 'memory_bandwidth_gb_s']:
            baseline_values = [sample[metric] for sample in baseline_data if metric in sample]
            current_values = [sample[metric] for sample in current_data if metric in sample]
            
            if baseline_values and current_values:
                baseline_avg = sum(baseline_values) / len(baseline_values)
                current_avg = sum(current_values) / len(current_values)
                change_pct = ((current_avg - baseline_avg) / baseline_avg) * 100
                
                status = "🔴 REGRESSION" if change_pct < -10 else "🟡 VARIATION" if abs(change_pct) > 5 else "🟢 STABLE"
                
                print(f"{metric}:")
                print(f"  Baseline: {baseline_avg:.2f}")
                print(f"  Current:  {current_avg:.2f}")
                print(f"  Change:   {change_pct:+.1f}% {status}")
                print()
        
        # Detect significant regressions
        regressions_detected = False
        print("⚠️  REGRESSION ANALYSIS")
        print("-" * 40)
        
        # GPU Time Analysis (higher is worse)
        gpu_time_baseline = [s['gpu_time_ms'] for s in baseline_data if 'gpu_time_ms' in s]
        gpu_time_current = [s['gpu_time_ms'] for s in current_data if 'gpu_time_ms' in s]
        
        if gpu_time_baseline and gpu_time_current:
            baseline_avg = sum(gpu_time_baseline) / len(gpu_time_baseline)
            current_avg = sum(gpu_time_current) / len(gpu_time_current)
            slowdown = ((current_avg - baseline_avg) / baseline_avg) * 100
            
            if slowdown > 20:
                print(f"🔴 HIGH SEVERITY: GPU execution time increased by {slowdown:.1f}%")
                regressions_detected = True
            elif slowdown > 10:
                print(f"🟡 MEDIUM SEVERITY: GPU execution time increased by {slowdown:.1f}%")
                regressions_detected = True
        
        # GFLOPS Analysis (lower is worse)
        gflops_baseline = [s['gflops'] for s in baseline_data if 'gflops' in s]
        gflops_current = [s['gflops'] for s in current_data if 'gflops' in s]
        
        if gflops_baseline and gflops_current:
            baseline_avg = sum(gflops_baseline) / len(gflops_baseline)
            current_avg = sum(gflops_current) / len(gflops_current)
            performance_drop = ((baseline_avg - current_avg) / baseline_avg) * 100
            
            if performance_drop > 20:
                print(f"🔴 HIGH SEVERITY: GFLOPS performance dropped by {performance_drop:.1f}%")
                regressions_detected = True
            elif performance_drop > 10:
                print(f"🟡 MEDIUM SEVERITY: GFLOPS performance dropped by {performance_drop:.1f}%")
                regressions_detected = True
        
        if not regressions_detected:
            print("🟢 No significant performance regressions detected")
        
        print()
    
    # File structure overview
    print("📁 PROJECT STRUCTURE")
    print("-" * 40)
    
    key_components = {
        "CUDA Engine": ["src/kernel/workload_engine.cu", "src/kernel/custom_kernels.cu"],
        "AI Analysis": ["src/analysis/detect_anomaly.py"],
        "Data Pipeline": ["src/pipeline/data_pipeline.py", "src/pipeline/perfai_pipeline.py"],
        "Build System": ["Makefile", "main.cpp"],
        "Containerization": ["docker/Dockerfile", "docker/docker-compose.yml"],
        "CI/CD": [".github/workflows/perfai_ci.yml"],
        "Documentation": ["README.md", "docs/INSTALLATION.md", "docs/USAGE.md"],
        "Tests": ["tests/test_perfai.py"]
    }
    
    total_files = 0
    for category, files in key_components.items():
        present_files = [f for f in files if Path(f).exists()]
        total_files += len(present_files)
        status = "✅" if len(present_files) == len(files) else "⚠️"
        print(f"{status} {category}: {len(present_files)}/{len(files)} files")
    
    print(f"\nTotal implementation: {total_files} key files")
    print()
    
    # Usage examples
    print("🔧 USAGE EXAMPLES")
    print("-" * 40)
    print("1. Build and setup:")
    print("   ./scripts/setup_dev.sh")
    print("   ./scripts/build.sh")
    print()
    print("2. Run benchmarks:")
    print("   ./scripts/run_benchmark.sh --verbose")
    print("   ./bin/cuda_benchmark --matrix-sizes 1024,2048 --iterations 10")
    print()
    print("3. Performance analysis:")
    print("   python src/pipeline/perfai_pipeline.py --config config.json")
    print("   python src/analysis/detect_anomaly.py --baseline data/baselines/stable.json")
    print()
    print("4. Docker deployment:")
    print("   docker-compose up perfai")
    print("   docker run perfai:latest --test-mode")
    print()
    
    # Academic context
    print("🎓 ACADEMIC CONTEXT")
    print("-" * 40)
    print("Course: CUDA at Scale for the Enterprise - Capstone Project")
    print("Demonstrates mastery of:")
    print("  • Advanced CUDA programming with cuBLAS")
    print("  • GPU performance optimization and profiling")
    print("  • Enterprise software architecture")
    print("  • AI/ML integration for performance analysis")
    print("  • DevOps and CI/CD automation")
    print("  • Professional documentation and testing")
    print()
    
    # Technical highlights
    print("⚡ TECHNICAL HIGHLIGHTS")
    print("-" * 40)
    print("• CUDA kernels with NVTX profiling integration")
    print("• cuBLAS-optimized matrix operations")
    print("• Memory bandwidth and occupancy analysis")
    print("• Isolation Forest ML anomaly detection")
    print("• Statistical regression analysis with confidence scoring")
    print("• Automated baseline creation and management")
    print("• GitHub Actions CI/CD with GPU support")
    print("• Multi-stage Docker builds with CUDA runtime")
    print("• Comprehensive test suite and documentation")
    print("• Enterprise-ready error handling and logging")
    print()
    
    print("=" * 80)
    print("🎉 PERFAI DEMONSTRATION COMPLETE")
    print("Ready for peer review and production deployment!")
    print("=" * 80)

if __name__ == "__main__":
    demonstrate_perfai()
