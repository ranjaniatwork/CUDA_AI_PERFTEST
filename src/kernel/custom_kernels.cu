/**
 * @file custom_kernels.cu
 * @brief Custom CUDA kernels for specialized workloads and performance testing
 * 
 * @author PerfAI Project
 * @date 2024
 */

#include "workload_engine.h"
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <iostream>
#include <chrono>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

namespace PerfAI {

/**
 * @brief Simple 2D convolution kernel
 */
__global__ void convolution2D(const float* input, const float* filter, float* output,
                             int input_width, int input_height, int filter_size) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (tx >= input_width || ty >= input_height) return;
    
    float sum = 0.0f;
    int half_filter = filter_size / 2;
    
    for (int fy = 0; fy < filter_size; ++fy) {
        for (int fx = 0; fx < filter_size; ++fx) {
            int input_x = tx + fx - half_filter;
            int input_y = ty + fy - half_filter;
            
            if (input_x >= 0 && input_x < input_width && 
                input_y >= 0 && input_y < input_height) {
                sum += input[input_y * input_width + input_x] * 
                       filter[fy * filter_size + fx];
            }
        }
    }
    
    output[ty * input_width + tx] = sum;
}

/**
 * @brief Memory bandwidth test kernel - simple copy operation
 */
__global__ void memoryBandwidthKernel(const float* src, float* dst, size_t num_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < num_elements; i += stride) {
        dst[i] = src[i];
    }
}

/**
 * @brief Compute-intensive kernel for arithmetic intensity testing
 */
__global__ void computeIntensiveKernel(const float* input, float* output, size_t num_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_elements) return;
    
    float value = input[idx];
    
    // Perform many arithmetic operations per memory access
    for (int i = 0; i < 100; ++i) {
        value = sinf(value) * cosf(value) + sqrtf(fabsf(value));
        value = value * 1.001f + 0.001f;
    }
    
    output[idx] = value;
}

void WorkloadEngine::launchConvolutionKernel(int input_size, int filter_size, int iterations, PerformanceMetrics& metrics) {
    nvtxRangePush("ConvolutionKernel");
    
    // Allocate memory
    size_t input_bytes = input_size * input_size * sizeof(float);
    size_t filter_bytes = filter_size * filter_size * sizeof(float);
    
    float *d_input, *d_filter, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_filter, filter_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, input_bytes));
    
    // Initialize data on host and copy to device
    std::vector<float> h_input(input_size * input_size, 1.0f);
    std::vector<float> h_filter(filter_size * filter_size, 0.1f);
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filter, h_filter.data(), filter_bytes, cudaMemcpyHostToDevice));
    
    // Configure kernel launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((input_size + blockSize.x - 1) / blockSize.x,
                  (input_size + blockSize.y - 1) / blockSize.y);
    
    // Create timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Warm-up run
    convolution2D<<<gridSize, blockSize>>>(d_input, d_filter, d_output, 
                                          input_size, input_size, filter_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timed runs
    CUDA_CHECK(cudaEventRecord(start));
    
    for (int i = 0; i < iterations; ++i) {
        convolution2D<<<gridSize, blockSize>>>(d_input, d_filter, d_output,
                                              input_size, input_size, filter_size);
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // Calculate metrics
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    
    metrics.gpu_time_ms = elapsed_ms;
    metrics.avg_time_per_iteration = elapsed_ms / iterations;
    
    // Calculate theoretical performance
    double ops_per_conv = static_cast<double>(input_size) * input_size * filter_size * filter_size;
    double total_ops = ops_per_conv * iterations;
    metrics.gflops = total_ops / (elapsed_ms * 1e6);
    
    // Calculate kernel occupancy
    int max_active_blocks;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                                                           convolution2D,
                                                           blockSize.x * blockSize.y,
                                                           0));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    metrics.kernel_occupancy = static_cast<double>(max_active_blocks) / prop.maxBlocksPerMultiProcessor * 100.0;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_filter));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    nvtxRangePop();
}

void WorkloadEngine::launchMemoryBandwidthKernel(size_t data_size, int iterations, PerformanceMetrics& metrics) {
    nvtxRangePush("MemoryBandwidthKernel");
    
    size_t num_elements = data_size / sizeof(float);
    
    float *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, data_size));
    CUDA_CHECK(cudaMalloc(&d_dst, data_size));
    
    // Initialize source data
    CUDA_CHECK(cudaMemset(d_src, 1, data_size));
    
    // Configure kernel launch parameters
    int blockSize = 256;
    int gridSize = (num_elements + blockSize - 1) / blockSize;
    gridSize = std::min(gridSize, 65535); // Limit grid size
    
    // Create timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Warm-up run
    memoryBandwidthKernel<<<gridSize, blockSize>>>(d_src, d_dst, num_elements);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timed runs
    CUDA_CHECK(cudaEventRecord(start));
    
    for (int i = 0; i < iterations; ++i) {
        memoryBandwidthKernel<<<gridSize, blockSize>>>(d_src, d_dst, num_elements);
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // Calculate metrics
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    
    metrics.gpu_time_ms = elapsed_ms;
    metrics.avg_time_per_iteration = elapsed_ms / iterations;
    
    // Calculate bandwidth (read + write)
    double total_bytes = 2.0 * data_size * iterations; // Read and write
    metrics.memory_bandwidth_gb_s = total_bytes / (elapsed_ms * 1e6);
    
    // Get theoretical peak bandwidth
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    double theoretical_bandwidth = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6; // GB/s
    metrics.achieved_bandwidth_percent = (metrics.memory_bandwidth_gb_s / theoretical_bandwidth) * 100.0;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    nvtxRangePop();
}

std::vector<PerformanceMetrics> WorkloadEngine::runBenchmarkSuite() {
    nvtxRangePush("BenchmarkSuite");
    
    std::vector<PerformanceMetrics> results;
    
    // Matrix multiplication tests with different sizes
    std::vector<int> matrix_sizes = {512, 1024, 2048, 4096};
    for (int size : matrix_sizes) {
        auto metrics = runMatrixMultiplication(size, 5);
        results.push_back(metrics);
    }
    
    // Convolution tests
    std::vector<int> conv_sizes = {256, 512, 1024};
    for (int size : conv_sizes) {
        auto metrics = runConvolution(size, 3, 10);
        results.push_back(metrics);
    }
    
    // Memory bandwidth tests
    std::vector<size_t> data_sizes = {1 << 20, 1 << 24, 1 << 28}; // 1MB, 16MB, 256MB
    for (size_t size : data_sizes) {
        auto metrics = runMemoryBandwidth(size, 10);
        results.push_back(metrics);
    }
    
    nvtxRangePop();
    return results;
}

} // namespace PerfAI
