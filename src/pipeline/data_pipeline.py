#!/usr/bin/env python3
"""
Data pipeline for collecting, processing, and managing CUDA performance metrics

This module handles the data flow from CUDA performance measurements to 
the AI analysis pipeline, including data validation, cleaning, and storage.

@author: PerfAI Project
@date: 2024
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import sqlite3
import subprocess
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPipeline:
    """
    Main data pipeline class for managing CUDA performance data
    """
    
    def __init__(self, data_dir: str = "data", db_path: str = None):
        """
        Initialize the data pipeline
        
        Args:
            data_dir: Directory for storing data files
            db_path: Path to SQLite database (optional)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.runs_dir = self.data_dir / "runs"
        self.baselines_dir = self.data_dir / "baselines"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.baselines_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = db_path
        if self.db_path:
            self._init_database()
        
        logger.info(f"DataPipeline initialized with data directory: {self.data_dir}")
    
    def _init_database(self):
        """Initialize SQLite database for performance data storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create performance_runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                workload_type TEXT NOT NULL,
                matrix_size INTEGER,
                iterations INTEGER,
                gpu_time_ms REAL,
                cpu_time_ms REAL,
                gflops REAL,
                memory_bandwidth_gb_s REAL,
                memory_utilization REAL,
                memory_used_mb INTEGER,
                memory_total_mb INTEGER,
                gpu_name TEXT,
                compute_capability TEXT,
                kernel_occupancy REAL,
                achieved_bandwidth_percent REAL,
                session_id TEXT,
                git_commit TEXT,
                environment_info TEXT
            )
        """)
        
        # Create index for efficient querying
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp_workload 
            ON performance_runs(timestamp, workload_type)
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at: {self.db_path}")
    
    def collect_performance_data(self, executable_path: str, config: Dict = None) -> List[Dict]:
        """
        Execute CUDA performance benchmarks and collect data
        
        Args:
            executable_path: Path to CUDA benchmark executable
            config: Configuration parameters for benchmark
            
        Returns:
            List of performance measurement dictionaries
        """
        logger.info(f"Collecting performance data using: {executable_path}")
        
        if config is None:
            config = self._get_default_config()
        
        # Generate unique session ID
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Build command arguments
        cmd = [executable_path]
        if config.get('matrix_sizes'):
            cmd.extend(['--matrix-sizes', ','.join(map(str, config['matrix_sizes']))])
        if config.get('iterations'):
            cmd.extend(['--iterations', str(config['iterations'])])
        if config.get('output_format'):
            cmd.extend(['--output-format', config['output_format']])
        
        # Add session ID
        cmd.extend(['--session-id', session_id])
        
        try:
            # Execute benchmark
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.error(f"Benchmark execution failed: {result.stderr}")
                return []
            
            # Parse output
            performance_data = self._parse_benchmark_output(result.stdout, session_id)
            
            # Add environment information
            env_info = self._collect_environment_info()
            for record in performance_data:
                record.update(env_info)
            
            logger.info(f"Collected {len(performance_data)} performance records")
            return performance_data
            
        except subprocess.TimeoutExpired:
            logger.error("Benchmark execution timed out")
            return []
        except Exception as e:
            logger.error(f"Error executing benchmark: {e}")
            return []
    
    def _get_default_config(self) -> Dict:
        """Get default benchmark configuration"""
        return {
            'matrix_sizes': [512, 1024, 2048],
            'iterations': 10,
            'output_format': 'json'
        }
    
    def _parse_benchmark_output(self, output: str, session_id: str) -> List[Dict]:
        """Parse benchmark output and extract performance metrics"""
        performance_data = []
        
        try:
            # Try to parse as JSON first
            if output.strip().startswith('[') or output.strip().startswith('{'):
                data = json.loads(output)
                if isinstance(data, list):
                    performance_data = data
                else:
                    performance_data = [data]
            else:
                # Parse line-by-line format
                performance_data = self._parse_text_output(output)
            
            # Add timestamp and session ID
            current_time = datetime.now().isoformat()
            for record in performance_data:
                record['timestamp'] = current_time
                record['session_id'] = session_id
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse benchmark output as JSON: {e}")
            # Try alternative parsing methods
            performance_data = self._parse_text_output(output)
        
        return performance_data
    
    def _parse_text_output(self, output: str) -> List[Dict]:
        """Parse text-based benchmark output"""
        # This is a simplified parser - in practice, you'd implement
        # a more robust parser based on your benchmark output format
        performance_data = []
        
        lines = output.strip().split('\n')
        for line in lines:
            if 'BENCHMARK_RESULT:' in line:
                try:
                    json_part = line.split('BENCHMARK_RESULT:')[1].strip()
                    record = json.loads(json_part)
                    performance_data.append(record)
                except:
                    continue
        
        return performance_data
    
    def _collect_environment_info(self) -> Dict:
        """Collect environment and system information"""
        env_info = {
            'collection_time': datetime.now().isoformat(),
            'python_version': subprocess.check_output(['python', '--version'], text=True).strip(),
        }
        
        # Try to get git commit
        try:
            git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                               text=True, stderr=subprocess.DEVNULL).strip()
            env_info['git_commit'] = git_commit
        except:
            env_info['git_commit'] = 'unknown'
        
        # Try to get CUDA version
        try:
            cuda_version = subprocess.check_output(['nvcc', '--version'], 
                                                 text=True, stderr=subprocess.DEVNULL)
            env_info['cuda_version'] = cuda_version.split('release')[1].split(',')[0].strip()
        except:
            env_info['cuda_version'] = 'unknown'
        
        return env_info
    
    def save_performance_data(self, data: List[Dict], filename: str = None) -> str:
        """
        Save performance data to file and optionally to database
        
        Args:
            data: List of performance records
            filename: Optional filename (auto-generated if not provided)
            
        Returns:
            Path to saved file
        """
        if not data:
            logger.warning("No data to save")
            return ""
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"perf_run_{timestamp}.json"
        
        # Ensure .json extension
        if not filename.endswith('.json'):
            filename += '.json'
        
        # Save to runs directory
        file_path = self.runs_dir / filename
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Performance data saved to: {file_path}")
        
        # Save to database if configured
        if self.db_path:
            self._save_to_database(data)
        
        return str(file_path)
    
    def _save_to_database(self, data: List[Dict]):
        """Save performance data to SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for record in data:
            # Prepare values for insertion
            values = (
                record.get('timestamp'),
                record.get('workload_type'),
                record.get('matrix_size'),
                record.get('iterations'),
                record.get('gpu_time_ms'),
                record.get('cpu_time_ms'),
                record.get('gflops'),
                record.get('memory_bandwidth_gb_s'),
                record.get('memory_utilization'),
                record.get('memory_used_mb'),
                record.get('memory_total_mb'),
                record.get('gpu_name'),
                record.get('compute_capability'),
                record.get('kernel_occupancy'),
                record.get('achieved_bandwidth_percent'),
                record.get('session_id'),
                record.get('git_commit'),
                json.dumps(record.get('environment_info', {}))
            )
            
            cursor.execute("""
                INSERT INTO performance_runs (
                    timestamp, workload_type, matrix_size, iterations,
                    gpu_time_ms, cpu_time_ms, gflops, memory_bandwidth_gb_s,
                    memory_utilization, memory_used_mb, memory_total_mb,
                    gpu_name, compute_capability, kernel_occupancy,
                    achieved_bandwidth_percent, session_id, git_commit, environment_info
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, values)
        
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(data)} records to database")
    
    def load_performance_data(self, file_path: str = None, days_back: int = None) -> pd.DataFrame:
        """
        Load performance data from file or database
        
        Args:
            file_path: Specific file to load (if None, loads from database or latest files)
            days_back: Number of days to look back (for database queries)
            
        Returns:
            DataFrame containing performance data
        """
        if file_path:
            # Load specific file
            logger.info(f"Loading data from file: {file_path}")
            with open(file_path, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        
        elif self.db_path:
            # Load from database
            logger.info("Loading data from database")
            return self._load_from_database(days_back)
        
        else:
            # Load latest files from runs directory
            logger.info("Loading data from latest run files")
            return self._load_latest_files()
    
    def _load_from_database(self, days_back: int = None) -> pd.DataFrame:
        """Load performance data from SQLite database"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM performance_runs"
        params = []
        
        if days_back:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            query += " WHERE timestamp >= ?"
            params.append(cutoff_date.isoformat())
        
        query += " ORDER BY timestamp DESC"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        logger.info(f"Loaded {len(df)} records from database")
        return df
    
    def _load_latest_files(self, max_files: int = 10) -> pd.DataFrame:
        """Load data from the most recent run files"""
        json_files = sorted(self.runs_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        
        if not json_files:
            logger.warning("No performance data files found")
            return pd.DataFrame()
        
        # Load up to max_files most recent files
        all_data = []
        for file_path in json_files[:max_files]:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        df = pd.DataFrame(all_data)
        logger.info(f"Loaded {len(df)} records from {len(json_files[:max_files])} files")
        return df
    
    def create_baseline(self, data_source: Union[str, pd.DataFrame] = None, 
                       baseline_name: str = None, min_samples: int = 50) -> str:
        """
        Create a performance baseline from historical data
        
        Args:
            data_source: Source data (file path, DataFrame, or None for auto-detection)
            baseline_name: Name for the baseline (auto-generated if None)
            min_samples: Minimum number of samples required for baseline
            
        Returns:
            Path to saved baseline file
        """
        logger.info("Creating performance baseline")
        
        # Load data
        if isinstance(data_source, pd.DataFrame):
            df = data_source
        elif isinstance(data_source, str):
            df = self.load_performance_data(data_source)
        else:
            # Auto-detect: load recent stable data
            df = self._load_stable_baseline_data()
        
        if len(df) < min_samples:
            raise ValueError(f"Insufficient data for baseline: {len(df)} samples (minimum: {min_samples})")
        
        # Filter and clean data
        df_clean = self._clean_baseline_data(df)
        
        # Generate baseline name
        if baseline_name is None:
            baseline_name = f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save baseline
        baseline_path = self.baselines_dir / f"{baseline_name}.json"
        df_clean.to_json(baseline_path, orient='records', indent=2)
        
        # Create baseline metadata
        metadata = {
            'name': baseline_name,
            'created': datetime.now().isoformat(),
            'sample_count': len(df_clean),
            'date_range': {
                'start': df_clean['timestamp'].min(),
                'end': df_clean['timestamp'].max()
            },
            'workload_types': df_clean['workload_type'].unique().tolist(),
            'statistics': self._calculate_baseline_stats(df_clean)
        }
        
        metadata_path = self.baselines_dir / f"{baseline_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Baseline created: {baseline_path} ({len(df_clean)} samples)")
        return str(baseline_path)
    
    def _load_stable_baseline_data(self) -> pd.DataFrame:
        """Load data for baseline creation, focusing on stable periods"""
        # Load data from the past 30 days
        df = self.load_performance_data(days_back=30)
        
        if df.empty:
            raise ValueError("No historical data available for baseline creation")
        
        # Filter out potential anomalies using simple statistical methods
        # Remove outliers beyond 2 standard deviations
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in ['gpu_time_ms', 'gflops', 'memory_bandwidth_gb_s']:
                mean = df[col].mean()
                std = df[col].std()
                df = df[abs(df[col] - mean) <= 2 * std]
        
        return df
    
    def _clean_baseline_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data for baseline creation"""
        # Remove rows with missing critical metrics
        critical_columns = ['workload_type', 'gpu_time_ms']
        df_clean = df.dropna(subset=critical_columns)
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        # Sort by timestamp
        if 'timestamp' in df_clean.columns:
            df_clean = df_clean.sort_values('timestamp')
        
        return df_clean
    
    def _calculate_baseline_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate statistical summary for baseline"""
        stats = {}
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in ['gpu_time_ms', 'gflops', 'memory_bandwidth_gb_s', 'memory_utilization']:
                stats[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'median': float(df[col].median()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'q25': float(df[col].quantile(0.25)),
                    'q75': float(df[col].quantile(0.75))
                }
        
        return stats
    
    def validate_data(self, data: List[Dict]) -> List[Dict]:
        """
        Validate performance data for completeness and consistency
        
        Args:
            data: List of performance records
            
        Returns:
            List of validation errors/warnings
        """
        issues = []
        
        required_fields = ['workload_type', 'timestamp']
        recommended_fields = ['gpu_time_ms', 'gflops', 'memory_bandwidth_gb_s']
        
        for i, record in enumerate(data):
            # Check required fields
            for field in required_fields:
                if field not in record or record[field] is None:
                    issues.append({
                        'type': 'error',
                        'record_index': i,
                        'message': f"Missing required field: {field}"
                    })
            
            # Check recommended fields
            for field in recommended_fields:
                if field not in record or record[field] is None:
                    issues.append({
                        'type': 'warning',
                        'record_index': i,
                        'message': f"Missing recommended field: {field}"
                    })
            
            # Validate data ranges
            if 'gpu_time_ms' in record and record['gpu_time_ms'] is not None:
                if record['gpu_time_ms'] <= 0:
                    issues.append({
                        'type': 'error',
                        'record_index': i,
                        'message': "GPU time must be positive"
                    })
            
            if 'memory_utilization' in record and record['memory_utilization'] is not None:
                if not 0 <= record['memory_utilization'] <= 100:
                    issues.append({
                        'type': 'warning',
                        'record_index': i,
                        'message': "Memory utilization should be between 0-100%"
                    })
        
        if issues:
            logger.warning(f"Data validation found {len(issues)} issues")
        else:
            logger.info("Data validation passed")
        
        return issues


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='PerfAI Data Pipeline')
    parser.add_argument('action', choices=['collect', 'baseline', 'validate'], 
                       help='Action to perform')
    parser.add_argument('--executable', help='Path to CUDA benchmark executable')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--db-path', help='SQLite database path')
    parser.add_argument('--config', help='Configuration file for benchmark')
    parser.add_argument('--output', help='Output filename')
    parser.add_argument('--baseline-name', help='Name for baseline')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DataPipeline(args.data_dir, args.db_path)
    
    if args.action == 'collect':
        if not args.executable:
            parser.error("--executable required for collect action")
        
        # Load config if provided
        config = None
        if args.config:
            with open(args.config, 'r') as f:
                config = json.load(f)
        
        # Collect data
        data = pipeline.collect_performance_data(args.executable, config)
        
        if data:
            # Validate data
            issues = pipeline.validate_data(data)
            if any(issue['type'] == 'error' for issue in issues):
                logger.error("Data validation failed with errors")
                for issue in issues:
                    if issue['type'] == 'error':
                        logger.error(f"Record {issue['record_index']}: {issue['message']}")
                return
            
            # Save data
            output_path = pipeline.save_performance_data(data, args.output)
            print(f"Performance data saved to: {output_path}")
        else:
            logger.error("No performance data collected")
    
    elif args.action == 'baseline':
        try:
            baseline_path = pipeline.create_baseline(baseline_name=args.baseline_name)
            print(f"Baseline created: {baseline_path}")
        except Exception as e:
            logger.error(f"Failed to create baseline: {e}")
    
    elif args.action == 'validate':
        # Load and validate recent data
        df = pipeline.load_performance_data()
        if not df.empty:
            data = df.to_dict('records')
            issues = pipeline.validate_data(data)
            
            print(f"Validation completed. Found {len(issues)} issues:")
            for issue in issues:
                print(f"  {issue['type'].upper()}: Record {issue['record_index']} - {issue['message']}")
        else:
            print("No data to validate")


if __name__ == "__main__":
    main()
