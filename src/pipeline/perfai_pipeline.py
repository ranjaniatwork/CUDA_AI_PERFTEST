#!/usr/bin/env python3
"""
Main orchestration pipeline for PerfAI performance regression detection

This module coordinates the entire workflow from CUDA benchmark execution
to AI-powered anomaly detection and reporting.

@author: PerfAI Project
@date: 2024
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.data_pipeline import DataPipeline
from src.analysis.detect_anomaly import PerformanceAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerfAIPipeline:
    """
    Main orchestration class for the PerfAI performance regression detection pipeline
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the PerfAI pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.data_pipeline = DataPipeline(
            data_dir=self.config.get('data_dir', 'data'),
            db_path=self.config.get('db_path')
        )
        self.analyzer = PerformanceAnalyzer(
            confidence_threshold=self.config.get('confidence_threshold', 0.95)
        )
        
        logger.info("PerfAI Pipeline initialized")
    
    def _load_config(self, config_path: str = None) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            'data_dir': 'data',
            'benchmark_executable': './cuda_benchmark',
            'confidence_threshold': 0.95,
            'contamination_rate': 0.1,
            'benchmark_config': {
                'matrix_sizes': [512, 1024, 2048, 4096],
                'iterations': 10,
                'output_format': 'json'
            },
            'alert_thresholds': {
                'high_severity': 0.3,
                'medium_severity': 0.1,
                'low_severity': 0.05
            },
            'output_formats': ['json', 'csv', 'html'],
            'enable_visualizations': True,
            'baseline_auto_update': True,
            'baseline_min_samples': 50
        }
        
        if config_path and Path(config_path).exists():
            logger.info(f"Loading configuration from: {config_path}")
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)
        else:
            logger.info("Using default configuration")
        
        return default_config
    
    def run_full_pipeline(self, output_dir: str = None, baseline_name: str = None) -> Dict:
        """
        Execute the complete performance regression detection pipeline
        
        Args:
            output_dir: Directory for output files
            baseline_name: Name of baseline to use (auto-detect if None)
            
        Returns:
            Dictionary containing pipeline results and metadata
        """
        logger.info("Starting full PerfAI pipeline execution")
        
        pipeline_start = datetime.now()
        
        # Setup output directory
        if output_dir is None:
            output_dir = f"output_{pipeline_start.strftime('%Y%m%d_%H%M%S')}"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        pipeline_results = {
            'pipeline_id': pipeline_start.strftime('%Y%m%d_%H%M%S'),
            'start_time': pipeline_start.isoformat(),
            'config': self.config,
            'stages': {},
            'output_dir': str(output_path)
        }
        
        try:
            # Stage 1: Collect current performance data
            logger.info("Stage 1: Collecting performance data")
            stage1_start = datetime.now()
            
            current_data = self.data_pipeline.collect_performance_data(
                self.config['benchmark_executable'],
                self.config['benchmark_config']
            )
            
            if not current_data:
                raise RuntimeError("Failed to collect performance data")
            
            # Save current data
            current_data_path = self.data_pipeline.save_performance_data(
                current_data, 
                f"current_run_{pipeline_start.strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            pipeline_results['stages']['data_collection'] = {
                'status': 'success',
                'duration_seconds': (datetime.now() - stage1_start).total_seconds(),
                'samples_collected': len(current_data),
                'data_file': current_data_path
            }
            
            # Stage 2: Load or create baseline
            logger.info("Stage 2: Loading/creating baseline")
            stage2_start = datetime.now()
            
            baseline_path = self._ensure_baseline(baseline_name)
            baseline_data = self.data_pipeline.load_performance_data(baseline_path)
            
            pipeline_results['stages']['baseline_loading'] = {
                'status': 'success',
                'duration_seconds': (datetime.now() - stage2_start).total_seconds(),
                'baseline_path': baseline_path,
                'baseline_samples': len(baseline_data)
            }
            
            # Stage 3: Train AI models
            logger.info("Stage 3: Training AI models")
            stage3_start = datetime.now()
            
            self.analyzer.train_isolation_forest(baseline_data, self.config['contamination_rate'])
            self.analyzer.train_statistical_model(baseline_data)
            
            pipeline_results['stages']['model_training'] = {
                'status': 'success',
                'duration_seconds': (datetime.now() - stage3_start).total_seconds(),
                'models_trained': ['isolation_forest', 'statistical']
            }
            
            # Stage 4: Anomaly detection
            logger.info("Stage 4: Performing anomaly detection")
            stage4_start = datetime.now()
            
            import pandas as pd
            current_df = pd.DataFrame(current_data)
            anomaly_results = self.analyzer.detect_anomalies(current_df)
            
            pipeline_results['stages']['anomaly_detection'] = {
                'status': 'success',
                'duration_seconds': (datetime.now() - stage4_start).total_seconds(),
                'anomalies_detected': anomaly_results['summary']['total_anomalies'],
                'severity': anomaly_results['summary']['severity']
            }
            
            # Stage 5: Generate reports and visualizations
            logger.info("Stage 5: Generating reports and visualizations")
            stage5_start = datetime.now()
            
            # Generate text report
            report_content = self.analyzer.generate_report(
                anomaly_results, 
                output_path / 'performance_analysis_report.txt'
            )
            
            # Save JSON results
            with open(output_path / 'anomaly_results.json', 'w') as f:
                json.dump(anomaly_results, f, indent=2, default=str)
            
            # Generate CSV report if enabled
            if 'csv' in self.config.get('output_formats', []):
                current_df.to_csv(output_path / 'current_performance_data.csv', index=False)
                baseline_data.to_csv(output_path / 'baseline_performance_data.csv', index=False)
            
            # Generate visualizations if enabled
            if self.config.get('enable_visualizations', True):
                self.analyzer.visualize_results(
                    current_df, 
                    anomaly_results, 
                    output_path / 'visualizations'
                )
            
            pipeline_results['stages']['reporting'] = {
                'status': 'success',
                'duration_seconds': (datetime.now() - stage5_start).total_seconds(),
                'reports_generated': [
                    'performance_analysis_report.txt',
                    'anomaly_results.json'
                ]
            }
            
            # Stage 6: Update baseline if configured
            if self.config.get('baseline_auto_update', False):
                logger.info("Stage 6: Updating baseline")
                stage6_start = datetime.now()
                
                # Only update if no high-severity anomalies detected
                if anomaly_results['summary']['severity'] != 'high':
                    try:
                        # Combine current data with baseline for updated baseline
                        combined_data = pd.concat([baseline_data, current_df], ignore_index=True)
                        
                        # Keep only recent data (last 1000 samples)
                        if len(combined_data) > 1000:
                            combined_data = combined_data.tail(1000)
                        
                        new_baseline_path = self.data_pipeline.create_baseline(
                            combined_data,
                            f"auto_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        )
                        
                        pipeline_results['stages']['baseline_update'] = {
                            'status': 'success',
                            'duration_seconds': (datetime.now() - stage6_start).total_seconds(),
                            'new_baseline_path': new_baseline_path
                        }
                        
                    except Exception as e:
                        logger.warning(f"Failed to update baseline: {e}")
                        pipeline_results['stages']['baseline_update'] = {
                            'status': 'skipped',
                            'reason': f"Error: {str(e)}"
                        }
                else:
                    pipeline_results['stages']['baseline_update'] = {
                        'status': 'skipped',
                        'reason': 'High severity anomalies detected'
                    }
            
            # Pipeline completion
            pipeline_end = datetime.now()
            pipeline_results.update({
                'end_time': pipeline_end.isoformat(),
                'total_duration_seconds': (pipeline_end - pipeline_start).total_seconds(),
                'status': 'success',
                'final_results': anomaly_results['summary']
            })
            
            # Save pipeline metadata
            with open(output_path / 'pipeline_metadata.json', 'w') as f:
                json.dump(pipeline_results, f, indent=2, default=str)
            
            logger.info(f"Pipeline completed successfully in {pipeline_results['total_duration_seconds']:.2f} seconds")
            logger.info(f"Results saved to: {output_path}")
            
            # Alert based on severity
            self._handle_alerts(anomaly_results, pipeline_results)
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            pipeline_results.update({
                'end_time': datetime.now().isoformat(),
                'status': 'failed',
                'error': str(e)
            })
            
            # Save partial results
            with open(output_path / 'pipeline_metadata.json', 'w') as f:
                json.dump(pipeline_results, f, indent=2, default=str)
            
            raise
    
    def _ensure_baseline(self, baseline_name: str = None) -> str:
        """Ensure a baseline exists, creating one if necessary"""
        baselines_dir = Path(self.config.get('data_dir', 'data')) / 'baselines'
        
        if baseline_name:
            baseline_path = baselines_dir / f"{baseline_name}.json"
            if baseline_path.exists():
                logger.info(f"Using specified baseline: {baseline_path}")
                return str(baseline_path)
            else:
                logger.warning(f"Specified baseline not found: {baseline_path}")
        
        # Look for existing baselines
        baseline_files = list(baselines_dir.glob("*.json"))
        baseline_files = [f for f in baseline_files if not f.name.endswith('_metadata.json')]
        
        if baseline_files:
            # Use the most recent baseline
            latest_baseline = max(baseline_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Using latest existing baseline: {latest_baseline}")
            return str(latest_baseline)
        
        # Create new baseline from historical data
        logger.info("No baseline found, creating new baseline from historical data")
        try:
            baseline_path = self.data_pipeline.create_baseline(
                baseline_name="auto_generated_baseline"
            )
            return baseline_path
        except Exception as e:
            logger.error(f"Failed to create baseline: {e}")
            raise RuntimeError("No baseline available and failed to create new baseline")
    
    def _handle_alerts(self, anomaly_results: Dict, pipeline_results: Dict):
        """Handle alerting based on anomaly detection results"""
        severity = anomaly_results['summary']['severity']
        
        alert_message = f"""
PerfAI Alert - Performance Analysis Complete

Pipeline ID: {pipeline_results['pipeline_id']}
Severity: {severity.upper()}
Total Anomalies: {anomaly_results['summary']['total_anomalies']}
Samples Analyzed: {anomaly_results['total_samples']}

Recommendation: {anomaly_results['summary']['recommendation']}

Results Location: {pipeline_results['output_dir']}
        """.strip()
        
        if severity == 'high':
            logger.critical("HIGH SEVERITY: Performance regression detected!")
            logger.critical(alert_message)
            # In a real system, this would trigger alerts (email, Slack, PagerDuty, etc.)
            
        elif severity == 'medium':
            logger.warning("MEDIUM SEVERITY: Potential performance issues detected")
            logger.warning(alert_message)
            
        else:
            logger.info("LOW SEVERITY: Performance within normal range")
            logger.info(alert_message)
    
    def run_benchmark_only(self, output_file: str = None) -> str:
        """
        Run only the CUDA benchmark without analysis
        
        Args:
            output_file: Output file for benchmark results
            
        Returns:
            Path to saved benchmark data
        """
        logger.info("Running benchmark only")
        
        current_data = self.data_pipeline.collect_performance_data(
            self.config['benchmark_executable'],
            self.config['benchmark_config']
        )
        
        if not current_data:
            raise RuntimeError("Failed to collect performance data")
        
        output_path = self.data_pipeline.save_performance_data(current_data, output_file)
        logger.info(f"Benchmark data saved to: {output_path}")
        
        return output_path
    
    def run_analysis_only(self, current_data_path: str, baseline_path: str = None, 
                         output_dir: str = None) -> Dict:
        """
        Run only the analysis on existing data
        
        Args:
            current_data_path: Path to current performance data
            baseline_path: Path to baseline data
            output_dir: Output directory for results
            
        Returns:
            Analysis results
        """
        logger.info("Running analysis only")
        
        # Load data
        import pandas as pd
        current_data = pd.read_json(current_data_path)
        
        if baseline_path is None:
            baseline_path = self._ensure_baseline()
        
        baseline_data = self.data_pipeline.load_performance_data(baseline_path)
        
        # Train models
        self.analyzer.train_isolation_forest(baseline_data, self.config['contamination_rate'])
        self.analyzer.train_statistical_model(baseline_data)
        
        # Detect anomalies
        results = self.analyzer.detect_anomalies(current_data)
        
        # Generate outputs
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save results
            with open(output_path / 'anomaly_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Generate report
            self.analyzer.generate_report(results, output_path / 'analysis_report.txt')
            
            # Generate visualizations
            if self.config.get('enable_visualizations', True):
                self.analyzer.visualize_results(current_data, results, output_path / 'plots')
            
            logger.info(f"Analysis results saved to: {output_path}")
        
        return results


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='PerfAI Performance Regression Detection Pipeline')
    
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--output-dir', help='Output directory for results')
    parser.add_argument('--baseline', help='Baseline name to use')
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--full-pipeline', action='store_true', default=True,
                           help='Run full pipeline (default)')
    mode_group.add_argument('--benchmark-only', action='store_true',
                           help='Run benchmark only')
    mode_group.add_argument('--analysis-only', 
                           help='Run analysis only on existing data (provide data file path)')
    
    # Additional options
    parser.add_argument('--benchmark-data', help='Path to current benchmark data (for analysis-only mode)')
    parser.add_argument('--baseline-data', help='Path to baseline data (for analysis-only mode)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize pipeline
        pipeline = PerfAIPipeline(args.config)
        
        if args.benchmark_only:
            # Run benchmark only
            output_path = pipeline.run_benchmark_only()
            print(f"Benchmark completed. Data saved to: {output_path}")
            
        elif args.analysis_only:
            # Run analysis only
            if not args.benchmark_data:
                parser.error("--benchmark-data required for analysis-only mode")
            
            results = pipeline.run_analysis_only(
                args.benchmark_data,
                args.baseline_data,
                args.output_dir
            )
            
            print(f"Analysis completed.")
            print(f"Severity: {results['summary']['severity']}")
            print(f"Anomalies detected: {results['summary']['total_anomalies']}")
            print(f"Recommendation: {results['summary']['recommendation']}")
            
        else:
            # Run full pipeline
            results = pipeline.run_full_pipeline(args.output_dir, args.baseline)
            
            print(f"PerfAI Pipeline completed successfully!")
            print(f"Pipeline ID: {results['pipeline_id']}")
            print(f"Total duration: {results['total_duration_seconds']:.2f} seconds")
            print(f"Final severity: {results['final_results']['severity']}")
            print(f"Results saved to: {results['output_dir']}")
            
            # Exit with appropriate code based on severity
            if results['final_results']['severity'] == 'high':
                sys.exit(2)  # High severity
            elif results['final_results']['severity'] == 'medium':
                sys.exit(1)  # Medium severity
            else:
                sys.exit(0)  # Success/low severity
                
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(3)  # Pipeline failure


if __name__ == "__main__":
    main()
