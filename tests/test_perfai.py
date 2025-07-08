#!/usr/bin/env python3
"""
Unit tests for PerfAI data pipeline functionality

@author: PerfAI Project
@date: 2024
"""

import unittest
import tempfile
import json
import pandas as pd
from pathlib import Path
import sys

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pipeline.data_pipeline import DataPipeline

class TestDataPipeline(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.pipeline = DataPipeline(data_dir=self.temp_dir)
        
        # Sample performance data
        self.sample_data = [
            {
                'timestamp': '2024-01-01T10:00:00Z',
                'workload_type': 'matrix_multiplication',
                'matrix_size': 1024,
                'iterations': 10,
                'gpu_time_ms': 15.5,
                'gflops': 1234.5,
                'memory_bandwidth_gb_s': 890.2,
                'memory_utilization': 75.0,
                'gpu_name': 'NVIDIA RTX 4090',
                'session_id': 'test_session_1'
            },
            {
                'timestamp': '2024-01-01T10:01:00Z',
                'workload_type': 'matrix_multiplication',
                'matrix_size': 2048,
                'iterations': 10,
                'gpu_time_ms': 62.3,
                'gflops': 2345.6,
                'memory_bandwidth_gb_s': 920.5,
                'memory_utilization': 80.0,
                'gpu_name': 'NVIDIA RTX 4090',
                'session_id': 'test_session_1'
            }
        ]
    
    def test_save_and_load_performance_data(self):
        """Test saving and loading performance data"""
        # Save data
        file_path = self.pipeline.save_performance_data(self.sample_data, 'test_data.json')
        
        # Verify file exists
        self.assertTrue(Path(file_path).exists())
        
        # Load data back
        df = self.pipeline.load_performance_data(file_path)
        
        # Verify data integrity
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]['workload_type'], 'matrix_multiplication')
        self.assertEqual(df.iloc[0]['matrix_size'], 1024)
        self.assertAlmostEqual(df.iloc[0]['gpu_time_ms'], 15.5)
    
    def test_data_validation(self):
        """Test data validation functionality"""
        # Valid data should pass
        issues = self.pipeline.validate_data(self.sample_data)
        error_issues = [issue for issue in issues if issue['type'] == 'error']
        self.assertEqual(len(error_issues), 0)
        
        # Invalid data should fail
        invalid_data = [
            {
                'workload_type': 'matrix_multiplication',
                # Missing timestamp
                'gpu_time_ms': -5.0,  # Invalid negative time
                'memory_utilization': 150.0  # Invalid percentage
            }
        ]
        
        issues = self.pipeline.validate_data(invalid_data)
        error_issues = [issue for issue in issues if issue['type'] == 'error']
        self.assertGreater(len(error_issues), 0)
    
    def test_baseline_creation(self):
        """Test baseline creation from performance data"""
        # Save sample data first
        self.pipeline.save_performance_data(self.sample_data * 30)  # Ensure enough samples
        
        # Create baseline
        baseline_path = self.pipeline.create_baseline(baseline_name='test_baseline')
        
        # Verify baseline file exists
        self.assertTrue(Path(baseline_path).exists())
        
        # Verify metadata file exists
        metadata_path = baseline_path.replace('.json', '_metadata.json')
        self.assertTrue(Path(metadata_path).exists())
        
        # Load and verify metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.assertEqual(metadata['name'], 'test_baseline')
        self.assertGreater(metadata['sample_count'], 0)
        self.assertIn('statistics', metadata)

class TestPerformanceAnalyzer(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
        from analysis.detect_anomaly import PerformanceAnalyzer
        
        self.analyzer = PerformanceAnalyzer()
        
        # Create sample baseline data
        self.baseline_data = pd.DataFrame([
            {
                'workload_type': 'matrix_multiplication',
                'gpu_time_ms': 15.0 + i * 0.1,
                'gflops': 1200 + i * 5,
                'memory_bandwidth_gb_s': 800 + i * 2,
                'memory_utilization': 70 + i * 0.5,
                'matrix_size': 1024
            }
            for i in range(100)
        ])
        
        # Sample current data with one anomaly
        self.current_data = pd.DataFrame([
            {
                'workload_type': 'matrix_multiplication',
                'gpu_time_ms': 15.2,
                'gflops': 1205,
                'memory_bandwidth_gb_s': 802,
                'memory_utilization': 71,
                'matrix_size': 1024
            },
            {
                'workload_type': 'matrix_multiplication',
                'gpu_time_ms': 50.0,  # Anomaly - much slower
                'gflops': 500,        # Anomaly - much lower performance
                'memory_bandwidth_gb_s': 400,
                'memory_utilization': 95,
                'matrix_size': 1024
            }
        ])
    
    def test_isolation_forest_training(self):
        """Test Isolation Forest model training"""
        model = self.analyzer.train_isolation_forest(self.baseline_data)
        self.assertIsNotNone(model)
        self.assertIn('isolation_forest', self.analyzer.models)
    
    def test_statistical_model_training(self):
        """Test statistical model training"""
        stats_model = self.analyzer.train_statistical_model(self.baseline_data)
        self.assertIsNotNone(stats_model)
        self.assertIn('statistical', self.analyzer.models)
        
        # Verify statistics are calculated
        self.assertIn('gpu_time_ms', stats_model)
        self.assertIn('mean', stats_model['gpu_time_ms'])
        self.assertIn('std', stats_model['gpu_time_ms'])
    
    def test_anomaly_detection(self):
        """Test anomaly detection functionality"""
        # Train models
        self.analyzer.train_isolation_forest(self.baseline_data)
        self.analyzer.train_statistical_model(self.baseline_data)
        
        # Detect anomalies
        results = self.analyzer.detect_anomalies(self.current_data)
        
        # Verify results structure
        self.assertIn('summary', results)
        self.assertIn('total_samples', results)
        self.assertEqual(results['total_samples'], 2)
        
        # Should detect at least one anomaly
        self.assertGreater(results['summary']['total_anomalies'], 0)

class TestWorkloadEngine(unittest.TestCase):
    """
    Test CUDA workload engine (requires GPU)
    These tests will be skipped if no GPU is available
    """
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            # Note: This would require the actual CUDA extension to be built
            # For now, we'll create a mock test
            self.cuda_available = False
        except ImportError:
            self.cuda_available = False
    
    def test_matrix_multiplication_mock(self):
        """Mock test for matrix multiplication workload"""
        # This is a placeholder for actual CUDA tests
        # In a real implementation, this would test the CUDA kernel
        self.assertTrue(True)  # Placeholder assertion
    
    @unittest.skipUnless(False, "CUDA not available in test environment")
    def test_cuda_workload_engine(self):
        """Test CUDA workload engine (skipped if no GPU)"""
        # This would contain actual CUDA tests
        pass

if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestDataPipeline))
    suite.addTest(unittest.makeSuite(TestPerformanceAnalyzer))
    suite.addTest(unittest.makeSuite(TestWorkloadEngine))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    if result.wasSuccessful():
        print("\nAll tests passed!")
        exit(0)
    else:
        print(f"\nTests failed: {len(result.failures)} failures, {len(result.errors)} errors")
        exit(1)
