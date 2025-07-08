#!/usr/bin/env python3
"""
PerfAI Project Summary Generator
Generates a comprehensive overview of the project structure and status
"""

import os
import json
from pathlib import Path
from datetime import datetime

def analyze_file(file_path):
    """Analyze a file and return basic info"""
    try:
        stat = file_path.stat()
        size = stat.st_size
        if size < 1024:
            size_str = f"{size} B"
        elif size < 1024 * 1024:
            size_str = f"{size // 1024} KB"
        else:
            size_str = f"{size // (1024 * 1024)} MB"
        
        # Count lines for text files
        lines = 0
        if file_path.suffix in ['.py', '.cpp', '.cu', '.h', '.sh', '.yml', '.yaml', '.md', '.txt', '.json']:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = sum(1 for _ in f)
            except:
                lines = 0
        
        return {
            'size': size_str,
            'lines': lines if lines > 0 else None,
            'type': file_path.suffix[1:] if file_path.suffix else 'no extension'
        }
    except:
        return {'size': 'unknown', 'lines': None, 'type': 'unknown'}

def scan_directory(base_path, max_depth=3, current_depth=0):
    """Recursively scan directory structure"""
    items = []
    
    if current_depth >= max_depth:
        return items
    
    try:
        for item in sorted(base_path.iterdir()):
            if item.name.startswith('.') and item.name not in ['.github']:
                continue
                
            item_info = {
                'name': item.name,
                'path': str(item.relative_to(base_path.parent)),
                'is_dir': item.is_dir()
            }
            
            if item.is_dir():
                item_info['children'] = scan_directory(item, max_depth, current_depth + 1)
            else:
                item_info.update(analyze_file(item))
            
            items.append(item_info)
    except PermissionError:
        pass
    
    return items

def generate_summary():
    """Generate project summary"""
    base_path = Path(__file__).parent.parent  # Go up one level from scripts/
    project_name = base_path.name
    
    print("=" * 80)
    print(f"PERFAI PROJECT SUMMARY")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project: {project_name}")
    print(f"Path: {base_path}")
    print()
    
    # Scan project structure
    structure = scan_directory(base_path)
    
    # Count files by type
    file_counts = {}
    total_files = 0
    total_lines = 0
    
    def count_files(items):
        nonlocal total_files, total_lines
        for item in items:
            if item['is_dir']:
                count_files(item.get('children', []))
            else:
                total_files += 1
                file_type = item.get('type', 'unknown')
                file_counts[file_type] = file_counts.get(file_type, 0) + 1
                if item.get('lines'):
                    total_lines += item['lines']
    
    count_files(structure)
    
    # Print statistics
    print("PROJECT STATISTICS")
    print("-" * 40)
    print(f"Total files: {total_files}")
    print(f"Total lines of code: {total_lines:,}")
    print()
    
    print("Files by type:")
    for file_type, count in sorted(file_counts.items()):
        print(f"  {file_type}: {count}")
    print()
    
    # Print structure
    print("PROJECT STRUCTURE")
    print("-" * 40)
    
    def print_structure(items, indent=0):
        for item in items:
            prefix = "  " * indent
            if item['is_dir']:
                print(f"{prefix}📁 {item['name']}/")
                print_structure(item.get('children', []), indent + 1)
            else:
                icon = "📄"
                if item.get('type') == 'py':
                    icon = "🐍"
                elif item.get('type') in ['cu', 'cpp', 'h']:
                    icon = "⚡"
                elif item.get('type') == 'md':
                    icon = "📖"
                elif item.get('type') in ['yml', 'yaml']:
                    icon = "⚙️"
                elif item.get('type') == 'json':
                    icon = "📋"
                elif item.get('type') == 'sh':
                    icon = "🔧"
                
                size_info = f" ({item['size']}"
                if item.get('lines'):
                    size_info += f", {item['lines']} lines"
                size_info += ")"
                
                print(f"{prefix}{icon} {item['name']}{size_info}")
    
    print_structure(structure)
    print()
    
    # Key components analysis
    print("KEY COMPONENTS")
    print("-" * 40)
    
    key_files = {
        'CUDA Kernel Engine': 'src/kernel/workload_engine.cu',
        'Custom CUDA Kernels': 'src/kernel/custom_kernels.cu',
        'Kernel Header': 'src/kernel/workload_engine.h',
        'Main Executable': 'main.cpp',
        'AI Anomaly Detection': 'src/analysis/detect_anomaly.py',
        'Data Pipeline': 'src/pipeline/data_pipeline.py',
        'Main Pipeline': 'src/pipeline/perfai_pipeline.py',
        'Build System': 'Makefile',
        'Python Dependencies': 'requirements.txt',
        'Configuration': 'config.json',
        'CI/CD Workflow': '.github/workflows/perfai_ci.yml',
        'Docker Setup': 'docker/Dockerfile',
        'Documentation': 'README.md',
        'Tests': 'tests/test_perfai.py'
    }
    
    for component, file_path in key_files.items():
        full_path = base_path / file_path
        if full_path.exists():
            info = analyze_file(full_path)
            status = "✅"
            details = f"({info['size']}"
            if info['lines']:
                details += f", {info['lines']} lines"
            details += ")"
        else:
            status = "❌"
            details = "(missing)"
        
        print(f"{status} {component}: {details}")
    
    print()
    
    # Check build status
    print("BUILD STATUS")
    print("-" * 40)
    
    build_items = [
        ('CUDA Executable', 'bin/cuda_benchmark'),
        ('Build Directory', 'build/'),
        ('Data Directory', 'data/'),
        ('Logs Directory', 'logs/'),
        ('Virtual Environment', 'venv/'),
    ]
    
    for item_name, item_path in build_items:
        full_path = base_path / item_path
        if full_path.exists():
            print(f"✅ {item_name}: Present")
        else:
            print(f"❌ {item_name}: Missing")
    
    print()
    
    # Configuration analysis
    config_path = base_path / 'config.json'
    if config_path.exists():
        print("CONFIGURATION ANALYSIS")
        print("-" * 40)
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            print(f"Data directory: {config.get('data_dir', 'data')}")
            print(f"Benchmark executable: {config.get('benchmark_executable', './bin/cuda_benchmark')}")
            print(f"Confidence threshold: {config.get('confidence_threshold', 0.95)}")
            
            bench_config = config.get('benchmark_config', {})
            print(f"Matrix sizes: {bench_config.get('matrix_sizes', [])}")
            print(f"Iterations: {bench_config.get('iterations', 10)}")
            print(f"Workloads: {bench_config.get('workloads', [])}")
            
        except Exception as e:
            print(f"❌ Could not parse configuration: {e}")
    else:
        print("❌ Configuration file missing")
    
    print()
    
    # Next steps
    print("NEXT STEPS")
    print("-" * 40)
    print("1. Run setup: ./scripts/setup_dev.sh")
    print("2. Build project: ./scripts/build.sh")
    print("3. Test installation: ./scripts/run_benchmark.sh --test-mode")
    print("4. Run first benchmark: ./scripts/run_benchmark.sh --verbose")
    print("5. Review documentation: docs/INSTALLATION.md")
    print()
    
    print("=" * 80)
    print("PerfAI: CUDA-Accelerated Autonomous Performance Regression Detection")
    print("Enterprise-grade capstone project for CUDA at Scale specialization")
    print("=" * 80)

if __name__ == "__main__":
    generate_summary()
