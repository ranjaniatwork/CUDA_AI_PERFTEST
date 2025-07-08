#!/usr/bin/env python3
"""
Performance Regression Detection using AI/ML techniques

This module implements various anomaly detection algorithms to identify
performance regressions in CUDA workload benchmarks.

@author: PerfAI Project
@date: 2024
"""

import pandas as pd
import numpy as np
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """
    Main class for analyzing CUDA performance data and detecting regressions
    """
    
    def __init__(self, baseline_path: str = None, confidence_threshold: float = 0.95):
        """
        Initialize the performance analyzer
        
        Args:
            baseline_path: Path to baseline performance data
            confidence_threshold: Confidence threshold for anomaly detection
        """
        self.baseline_path = baseline_path
        self.confidence_threshold = confidence_threshold
        self.baseline_data = None
        self.scaler = StandardScaler()
        self.models = {}
        
        logger.info(f"PerformanceAnalyzer initialized with confidence threshold: {confidence_threshold}")
    
    def load_baseline_data(self, path: str = None) -> pd.DataFrame:
        """
        Load baseline performance data from CSV or JSON files
        
        Args:
            path: Path to baseline data file
            
        Returns:
            DataFrame containing baseline performance metrics
        """
        if path is None:
            path = self.baseline_path
            
        if path is None:
            raise ValueError("No baseline path provided")
            
        logger.info(f"Loading baseline data from: {path}")
        
        path = Path(path)
        if path.suffix == '.csv':
            self.baseline_data = pd.read_csv(path)
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
            self.baseline_data = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
            
        logger.info(f"Loaded {len(self.baseline_data)} baseline records")
        return self.baseline_data
    
    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess performance data for ML analysis
        
        Args:
            data: Raw performance data
            
        Returns:
            Preprocessed feature matrix
        """
        # Select numerical performance features
        feature_columns = [
            'gpu_time_ms', 'gflops', 'memory_bandwidth_gb_s', 
            'memory_utilization', 'kernel_occupancy', 'matrix_size'
        ]
        
        # Filter available columns
        available_features = [col for col in feature_columns if col in data.columns]
        logger.info(f"Using features: {available_features}")
        
        # Extract features and handle missing values
        features = data[available_features].fillna(data[available_features].median())
        
        # Apply log transformation to reduce skewness for timing metrics
        if 'gpu_time_ms' in features.columns:
            features['gpu_time_ms'] = np.log1p(features['gpu_time_ms'])
            
        return features.values
    
    def train_isolation_forest(self, data: pd.DataFrame, contamination: float = 0.1) -> IsolationForest:
        """
        Train Isolation Forest model for anomaly detection
        
        Args:
            data: Training data
            contamination: Expected proportion of anomalies
            
        Returns:
            Trained Isolation Forest model
        """
        logger.info("Training Isolation Forest model")
        
        # Preprocess data
        X = self.preprocess_data(data)
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        model.fit(X_scaled)
        
        self.models['isolation_forest'] = model
        logger.info("Isolation Forest training completed")
        
        return model
    
    def train_statistical_model(self, data: pd.DataFrame) -> Dict:
        """
        Train statistical models for performance regression detection
        
        Args:
            data: Training data
            
        Returns:
            Dictionary containing statistical model parameters
        """
        logger.info("Training statistical models")
        
        stats_model = {}
        
        # Calculate statistical baselines for key metrics
        key_metrics = ['gpu_time_ms', 'gflops', 'memory_bandwidth_gb_s']
        
        for metric in key_metrics:
            if metric in data.columns:
                values = data[metric].dropna()
                stats_model[metric] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'median': float(values.median()),
                    'q25': float(values.quantile(0.25)),
                    'q75': float(values.quantile(0.75)),
                    'iqr': float(values.quantile(0.75) - values.quantile(0.25))
                }
        
        self.models['statistical'] = stats_model
        logger.info("Statistical model training completed")
        
        return stats_model
    
    def detect_anomalies(self, current_data: pd.DataFrame) -> Dict:
        """
        Detect performance anomalies in current data
        
        Args:
            current_data: Current performance measurements
            
        Returns:
            Dictionary containing anomaly detection results
        """
        logger.info("Starting anomaly detection")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(current_data),
            'anomalies': [],
            'summary': {}
        }
        
        # Isolation Forest detection
        if 'isolation_forest' in self.models:
            isolation_results = self._detect_with_isolation_forest(current_data)
            results['isolation_forest'] = isolation_results
        
        # Statistical detection
        if 'statistical' in self.models:
            statistical_results = self._detect_with_statistics(current_data)
            results['statistical'] = statistical_results
        
        # Combine results and generate summary
        results['summary'] = self._generate_summary(results)
        
        logger.info(f"Anomaly detection completed. Found {len(results['anomalies'])} potential regressions")
        
        return results
    
    def _detect_with_isolation_forest(self, data: pd.DataFrame) -> Dict:
        """
        Detect anomalies using Isolation Forest
        """
        model = self.models['isolation_forest']
        X = self.preprocess_data(data)
        X_scaled = self.scaler.transform(X)
        
        # Predict anomalies
        predictions = model.predict(X_scaled)
        anomaly_scores = model.decision_function(X_scaled)
        
        # Identify anomalies (prediction = -1)
        anomaly_indices = np.where(predictions == -1)[0]
        
        results = {
            'method': 'isolation_forest',
            'anomaly_count': len(anomaly_indices),
            'anomaly_indices': anomaly_indices.tolist(),
            'anomaly_scores': anomaly_scores.tolist(),
            'threshold': 0.0
        }
        
        return results
    
    def _detect_with_statistics(self, data: pd.DataFrame) -> Dict:
        """
        Detect anomalies using statistical methods (z-score, IQR)
        """
        stats_model = self.models['statistical']
        anomalies = []
        
        for idx, row in data.iterrows():
            anomaly_flags = {}
            
            for metric, baseline in stats_model.items():
                if metric in row and not pd.isna(row[metric]):
                    value = row[metric]
                    
                    # Z-score test
                    z_score = abs((value - baseline['mean']) / baseline['std'])
                    anomaly_flags[f'{metric}_zscore'] = z_score > 2.5
                    
                    # IQR test
                    iqr_threshold = baseline['q75'] + 1.5 * baseline['iqr']
                    anomaly_flags[f'{metric}_iqr'] = value > iqr_threshold
            
            # If any metric shows anomaly, flag the sample
            if any(anomaly_flags.values()):
                anomalies.append({
                    'index': idx,
                    'workload_type': row.get('workload_type', 'unknown'),
                    'flags': anomaly_flags,
                    'metrics': {k: row[k] for k in stats_model.keys() if k in row}
                })
        
        results = {
            'method': 'statistical',
            'anomaly_count': len(anomalies),
            'anomalies': anomalies
        }
        
        return results
    
    def _generate_summary(self, results: Dict) -> Dict:
        """
        Generate summary of anomaly detection results
        """
        summary = {
            'total_anomalies': 0,
            'confidence_level': self.confidence_threshold,
            'recommendation': 'No action required',
            'severity': 'low'
        }
        
        # Count total anomalies from all methods
        if 'isolation_forest' in results:
            summary['total_anomalies'] += results['isolation_forest']['anomaly_count']
        
        if 'statistical' in results:
            summary['total_anomalies'] += results['statistical']['anomaly_count']
        
        # Determine severity and recommendations
        anomaly_rate = summary['total_anomalies'] / results['total_samples']
        
        if anomaly_rate > 0.3:
            summary['severity'] = 'high'
            summary['recommendation'] = 'Immediate investigation required - significant performance regression detected'
        elif anomaly_rate > 0.1:
            summary['severity'] = 'medium'
            summary['recommendation'] = 'Performance monitoring recommended - potential regression detected'
        elif anomaly_rate > 0.05:
            summary['severity'] = 'low'
            summary['recommendation'] = 'Continue monitoring - minor performance variations detected'
        
        return summary
    
    def generate_report(self, results: Dict, output_path: str = None) -> str:
        """
        Generate comprehensive performance analysis report
        
        Args:
            results: Anomaly detection results
            output_path: Path to save the report
            
        Returns:
            Report content as string
        """
        logger.info("Generating performance analysis report")
        
        report = []
        report.append("=" * 80)
        report.append("PERFAI PERFORMANCE REGRESSION ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {results['timestamp']}")
        report.append(f"Total Samples Analyzed: {results['total_samples']}")
        report.append(f"Confidence Threshold: {self.confidence_threshold}")
        report.append("")
        
        # Executive Summary
        summary = results['summary']
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Anomalies Detected: {summary['total_anomalies']}")
        report.append(f"Severity Level: {summary['severity'].upper()}")
        report.append(f"Recommendation: {summary['recommendation']}")
        report.append("")
        
        # Detailed Results
        if 'isolation_forest' in results:
            iso_results = results['isolation_forest']
            report.append("ISOLATION FOREST ANALYSIS")
            report.append("-" * 40)
            report.append(f"Anomalies Detected: {iso_results['anomaly_count']}")
            report.append(f"Detection Rate: {iso_results['anomaly_count']/results['total_samples']*100:.2f}%")
            report.append("")
        
        if 'statistical' in results:
            stat_results = results['statistical']
            report.append("STATISTICAL ANALYSIS")
            report.append("-" * 40)
            report.append(f"Anomalies Detected: {stat_results['anomaly_count']}")
            
            # Breakdown by workload type
            workload_counts = {}
            for anomaly in stat_results['anomalies']:
                workload = anomaly['workload_type']
                workload_counts[workload] = workload_counts.get(workload, 0) + 1
            
            if workload_counts:
                report.append("Anomalies by Workload Type:")
                for workload, count in workload_counts.items():
                    report.append(f"  - {workload}: {count}")
            report.append("")
        
        # Recommendations
        report.append("DETAILED RECOMMENDATIONS")
        report.append("-" * 40)
        if summary['severity'] == 'high':
            report.append("• Immediately halt deployment pipeline")
            report.append("• Investigate recent code changes")
            report.append("• Check GPU hardware and driver status")
            report.append("• Review system resource utilization")
        elif summary['severity'] == 'medium':
            report.append("• Continue monitoring performance trends")
            report.append("• Review recent optimizations or changes")
            report.append("• Consider running extended benchmarks")
        else:
            report.append("• Maintain current monitoring schedule")
            report.append("• Performance variations within acceptable range")
        
        report.append("")
        report.append("=" * 80)
        
        report_content = "\n".join(report)
        
        # Save report if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_content)
            logger.info(f"Report saved to: {output_path}")
        
        return report_content
    
    def visualize_results(self, data: pd.DataFrame, results: Dict, output_dir: str = "plots"):
        """
        Create visualizations for performance analysis
        
        Args:
            data: Performance data
            results: Anomaly detection results
            output_dir: Directory to save plots
        """
        logger.info("Generating performance visualizations")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Time series plot
        if 'timestamp' in data.columns:
            self._plot_time_series(data, results, output_path)
        
        # Performance distribution plots
        self._plot_performance_distributions(data, results, output_path)
        
        # Correlation heatmap
        self._plot_correlation_heatmap(data, output_path)
        
        logger.info(f"Visualizations saved to: {output_path}")
    
    def _plot_time_series(self, data: pd.DataFrame, results: Dict, output_path: Path):
        """Plot time series of key performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Metrics Time Series', fontsize=16)
        
        metrics = ['gpu_time_ms', 'gflops', 'memory_bandwidth_gb_s', 'memory_utilization']
        
        for i, metric in enumerate(metrics):
            if metric in data.columns:
                ax = axes[i//2, i%2]
                ax.plot(data[metric], alpha=0.7, linewidth=1)
                ax.set_title(metric.replace('_', ' ').title())
                ax.grid(True, alpha=0.3)
                
                # Highlight anomalies if available
                if 'isolation_forest' in results:
                    anomaly_indices = results['isolation_forest']['anomaly_indices']
                    if anomaly_indices:
                        ax.scatter(anomaly_indices, data.iloc[anomaly_indices][metric], 
                                 color='red', s=50, alpha=0.8, label='Anomalies')
                        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'time_series.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_distributions(self, data: pd.DataFrame, results: Dict, output_path: Path):
        """Plot performance metric distributions"""
        metrics = ['gpu_time_ms', 'gflops', 'memory_bandwidth_gb_s']
        available_metrics = [m for m in metrics if m in data.columns]
        
        if not available_metrics:
            return
        
        fig, axes = plt.subplots(1, len(available_metrics), figsize=(5*len(available_metrics), 5))
        if len(available_metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(available_metrics):
            ax = axes[i]
            ax.hist(data[metric], bins=30, alpha=0.7, edgecolor='black')
            ax.set_title(f'{metric.replace("_", " ").title()} Distribution')
            ax.set_xlabel(metric)
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_heatmap(self, data: pd.DataFrame, output_path: Path):
        """Plot correlation heatmap of performance metrics"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) < 2:
            return
        
        correlation_matrix = data[numeric_columns].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Performance Metrics Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(output_path / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='PerfAI Performance Regression Detection')
    parser.add_argument('--baseline', required=True, help='Path to baseline performance data')
    parser.add_argument('--current', required=True, help='Path to current performance data')
    parser.add_argument('--output', default='analysis_results', help='Output directory for results')
    parser.add_argument('--confidence', type=float, default=0.95, help='Confidence threshold')
    parser.add_argument('--contamination', type=float, default=0.1, help='Expected contamination rate')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = PerformanceAnalyzer(args.baseline, args.confidence)
    
    # Load and train on baseline data
    baseline_data = analyzer.load_baseline_data()
    analyzer.train_isolation_forest(baseline_data, args.contamination)
    analyzer.train_statistical_model(baseline_data)
    
    # Load current data and detect anomalies
    current_data = pd.read_csv(args.current)
    results = analyzer.detect_anomalies(current_data)
    
    # Generate outputs
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save results as JSON
    with open(output_path / 'anomaly_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Generate report
    report = analyzer.generate_report(results, output_path / 'analysis_report.txt')
    print(report)
    
    # Generate visualizations
    analyzer.visualize_results(current_data, results, output_path / 'plots')
    
    logger.info(f"Analysis complete. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
