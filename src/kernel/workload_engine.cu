/**
 * @file workload_engine.cu
 * @brief CUDA-accelerated AI workload simulation engine for performance regression testing
 * 
 * This module implements various GPU workloads that simulate real AI/ML operations:
 * - Matrix multiplication (GEMM) using cuBLAS
 * - Convolution operations
 * - Memory bandwidth tests
 * - Custom CUDA kernels for performance profiling
 * 
 * @author PerfAI Project
 * @date 2024
 */

#include "workload_engine.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvToolsExt.h>
#include <iostream>
#include <chrono>
#include <memory>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// cuBLAS error checking macro
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ \
                      << " - Status: " << status << std::endl; \
            exit(1); \
        } \
    } while(0)

namespace PerfAI {

WorkloadEngine::WorkloadEngine() : cublas_handle_(nullptr) {
    // Initialize CUDA context
    CUDA_CHECK(cudaSetDevice(0));
    
    // Create cuBLAS handle
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    
    // Initialize NVTX for profiling
    nvtxNameOsThread(GetCurrentThreadId(), "WorkloadEngine");
    
    std::cout << "WorkloadEngine initialized successfully" << std::endl;
}

WorkloadEngine::~WorkloadEngine() {
    if (cublas_handle_) {
        cublasDestroy(cublas_handle_);
    }
}

PerformanceMetrics WorkloadEngine::runMatrixMultiplication(int matrix_size, int iterations) {
    nvtxRangePush("MatrixMultiplication");
    
    PerformanceMetrics metrics;
    metrics.workload_type = "matrix_multiplication";
    metrics.matrix_size = matrix_size;
    metrics.iterations = iterations;
    
    // Allocate host memory
    size_t matrix_bytes = matrix_size * matrix_size * sizeof(float);
    std::unique_ptr<float[]> h_A(new float[matrix_size * matrix_size]);
    std::unique_ptr<float[]> h_B(new float[matrix_size * matrix_size]);
    std::unique_ptr<float[]> h_C(new float[matrix_size * matrix_size]);
    
    // Initialize matrices with random data
    initializeMatrix(h_A.get(), matrix_size * matrix_size);
    initializeMatrix(h_B.get(), matrix_size * matrix_size);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, matrix_bytes));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.get(), matrix_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.get(), matrix_bytes, cudaMemcpyHostToDevice));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Warm-up run
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                            matrix_size, matrix_size, matrix_size,
                            &alpha, d_A, matrix_size, d_B, matrix_size,
                            &beta, d_C, matrix_size));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Performance measurement runs
    auto cpu_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaEventRecord(start));
    
    for (int i = 0; i < iterations; ++i) {
        nvtxRangePush("GEMM_Iteration");
        CUBLAS_CHECK(cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                                matrix_size, matrix_size, matrix_size,
                                &alpha, d_A, matrix_size, d_B, matrix_size,
                                &beta, d_C, matrix_size));
        nvtxRangePop();
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    auto cpu_end = std::chrono::high_resolution_clock::now();
    
    // Calculate timing metrics
    float gpu_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, start, stop));
    
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
    
    metrics.gpu_time_ms = gpu_time_ms;
    metrics.cpu_time_ms = cpu_duration.count() / 1000.0;
    metrics.avg_time_per_iteration = gpu_time_ms / iterations;
    
    // Calculate FLOPS
    double flops_per_gemm = 2.0 * matrix_size * matrix_size * matrix_size;
    double total_flops = flops_per_gemm * iterations;
    metrics.gflops = total_flops / (gpu_time_ms * 1e6);
    
    // Memory bandwidth calculation
    double bytes_per_gemm = 3.0 * matrix_bytes; // Read A, B and write C
    double total_bytes = bytes_per_gemm * iterations;
    metrics.memory_bandwidth_gb_s = total_bytes / (gpu_time_ms * 1e6);
    
    // Get GPU utilization and memory info
    getGPUUtilization(metrics);
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    nvtxRangePop();
    return metrics;
}

PerformanceMetrics WorkloadEngine::runConvolution(int input_size, int filter_size, int iterations) {
    nvtxRangePush("Convolution");
    
    PerformanceMetrics metrics;
    metrics.workload_type = "convolution";
    metrics.matrix_size = input_size;
    metrics.iterations = iterations;
    
    // Launch custom convolution kernel
    launchConvolutionKernel(input_size, filter_size, iterations, metrics);
    
    nvtxRangePop();
    return metrics;
}

PerformanceMetrics WorkloadEngine::runMemoryBandwidth(size_t data_size, int iterations) {
    nvtxRangePush("MemoryBandwidth");
    
    PerformanceMetrics metrics;
    metrics.workload_type = "memory_bandwidth";
    metrics.data_size_mb = data_size / (1024 * 1024);
    metrics.iterations = iterations;
    
    // Launch memory bandwidth test kernel
    launchMemoryBandwidthKernel(data_size, iterations, metrics);
    
    nvtxRangePop();
    return metrics;
}

void WorkloadEngine::initializeMatrix(float* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void WorkloadEngine::getGPUUtilization(PerformanceMetrics& metrics) {
    // Get GPU memory info
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    
    metrics.memory_used_mb = (total_mem - free_mem) / (1024 * 1024);
    metrics.memory_total_mb = total_mem / (1024 * 1024);
    metrics.memory_utilization = static_cast<double>(total_mem - free_mem) / total_mem * 100.0;
    
    // Get device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    metrics.gpu_name = prop.name;
    metrics.compute_capability = std::to_string(prop.major) + "." + std::to_string(prop.minor);
    metrics.multiprocessor_count = prop.multiProcessorCount;
    metrics.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
}

} // namespace PerfAI
