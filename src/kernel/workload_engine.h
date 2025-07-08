/**
 * @file workload_engine.h
 * @brief Header file for CUDA workload engine
 * 
 * @author PerfAI Project
 * @date 2024
 */

#ifndef WORKLOAD_ENGINE_H
#define WORKLOAD_ENGINE_H

#include <string>
#include <vector>
#include <cublas_v2.h>

namespace PerfAI {

/**
 * @brief Structure to hold comprehensive performance metrics
 */
struct PerformanceMetrics {
    // Workload identification
    std::string workload_type;
    int matrix_size = 0;
    int iterations = 0;
    size_t data_size_mb = 0;
    
    // Timing metrics
    double gpu_time_ms = 0.0;
    double cpu_time_ms = 0.0;
    double avg_time_per_iteration = 0.0;
    
    // Performance metrics
    double gflops = 0.0;
    double memory_bandwidth_gb_s = 0.0;
    
    // GPU utilization
    std::string gpu_name;
    std::string compute_capability;
    int multiprocessor_count = 0;
    int max_threads_per_sm = 0;
    double memory_utilization = 0.0;
    size_t memory_used_mb = 0;
    size_t memory_total_mb = 0;
    
    // Timestamp
    std::string timestamp;
    
    // Additional metrics
    double kernel_occupancy = 0.0;
    double achieved_bandwidth_percent = 0.0;
    double arithmetic_intensity = 0.0;
};

/**
 * @brief Main workload engine class for GPU performance testing
 */
class WorkloadEngine {
public:
    WorkloadEngine();
    ~WorkloadEngine();
    
    /**
     * @brief Run matrix multiplication workload using cuBLAS
     * @param matrix_size Size of square matrices
     * @param iterations Number of iterations to run
     * @return Performance metrics
     */
    PerformanceMetrics runMatrixMultiplication(int matrix_size, int iterations = 10);
    
    /**
     * @brief Run convolution workload
     * @param input_size Size of input tensor
     * @param filter_size Size of convolution filter
     * @param iterations Number of iterations to run
     * @return Performance metrics
     */
    PerformanceMetrics runConvolution(int input_size, int filter_size = 3, int iterations = 10);
    
    /**
     * @brief Run memory bandwidth test
     * @param data_size Size of data to transfer (bytes)
     * @param iterations Number of iterations to run
     * @return Performance metrics
     */
    PerformanceMetrics runMemoryBandwidth(size_t data_size, int iterations = 10);
    
    /**
     * @brief Run comprehensive benchmark suite
     * @return Vector of performance metrics for all workloads
     */
    std::vector<PerformanceMetrics> runBenchmarkSuite();

private:
    cublasHandle_t cublas_handle_;
    
    void initializeMatrix(float* matrix, int size);
    void getGPUUtilization(PerformanceMetrics& metrics);
    void launchConvolutionKernel(int input_size, int filter_size, int iterations, PerformanceMetrics& metrics);
    void launchMemoryBandwidthKernel(size_t data_size, int iterations, PerformanceMetrics& metrics);
};

} // namespace PerfAI

#endif // WORKLOAD_ENGINE_H
