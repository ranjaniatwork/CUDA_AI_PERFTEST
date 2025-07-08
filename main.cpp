/**
 * @file main.cpp
 * @brief Main entry point for CUDA performance benchmark executable
 * 
 * This is the command-line interface for the PerfAI CUDA benchmark engine.
 * It provides a comprehensive interface for running various GPU workloads
 * and collecting detailed performance metrics.
 * 
 * @author PerfAI Project
 * @date 2024
 */

#include "workload_engine.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <fstream>

void printUsage(const char* program_name) {
    std::cout << "PerfAI CUDA Performance Benchmark\n"
              << "Usage: " << program_name << " [OPTIONS]\n\n"
              << "Options:\n"
              << "  --matrix-sizes SIZE1,SIZE2,...  Matrix sizes for GEMM tests (default: 512,1024,2048)\n"
              << "  --iterations N                  Number of iterations per test (default: 10)\n"
              << "  --workloads TYPE1,TYPE2,...     Workload types to run (gemm,conv,bandwidth,all)\n"
              << "  --output-format FORMAT          Output format (json,csv,text) (default: json)\n"
              << "  --output-file FILE              Output file (default: stdout)\n"
              << "  --session-id ID                 Session identifier for tracking\n"
              << "  --test-mode                     Run in test mode (quick validation)\n"
              << "  --verbose                       Enable verbose output\n"
              << "  --help                          Show this help message\n\n"
              << "Examples:\n"
              << "  " << program_name << " --matrix-sizes 1024,2048 --iterations 5\n"
              << "  " << program_name << " --workloads gemm,bandwidth --output-format csv\n"
              << "  " << program_name << " --test-mode --verbose\n";
}

struct BenchmarkConfig {
    std::vector<int> matrix_sizes = {512, 1024, 2048};
    int iterations = 10;
    std::vector<std::string> workloads = {"all"};
    std::string output_format = "json";
    std::string output_file = "";
    std::string session_id = "";
    bool test_mode = false;
    bool verbose = false;
};

std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    
    return tokens;
}

BenchmarkConfig parseArguments(int argc, char* argv[]) {
    BenchmarkConfig config;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            printUsage(argv[0]);
            exit(0);
        } else if (arg == "--matrix-sizes" && i + 1 < argc) {
            auto size_strings = split(argv[++i], ',');
            config.matrix_sizes.clear();
            for (const auto& size_str : size_strings) {
                config.matrix_sizes.push_back(std::stoi(size_str));
            }
        } else if (arg == "--iterations" && i + 1 < argc) {
            config.iterations = std::stoi(argv[++i]);
        } else if (arg == "--workloads" && i + 1 < argc) {
            config.workloads = split(argv[++i], ',');
        } else if (arg == "--output-format" && i + 1 < argc) {
            config.output_format = argv[++i];
        } else if (arg == "--output-file" && i + 1 < argc) {
            config.output_file = argv[++i];
        } else if (arg == "--session-id" && i + 1 < argc) {
            config.session_id = argv[++i];
        } else if (arg == "--test-mode") {
            config.test_mode = true;
            config.matrix_sizes = {256, 512};
            config.iterations = 3;
        } else if (arg == "--verbose") {
            config.verbose = true;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printUsage(argv[0]);
            exit(1);
        }
    }
    
    return config;
}

std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%dT%H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    ss << std::put_time(std::localtime(&time_t), "%z");
    
    return ss.str();
}

void outputJSON(const std::vector<PerfAI::PerformanceMetrics>& results, 
                std::ostream& out, const BenchmarkConfig& config) {
    out << "[\n";
    
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& metrics = results[i];
        
        out << "  {\n";
        out << "    \"timestamp\": \"" << getCurrentTimestamp() << "\",\n";
        out << "    \"session_id\": \"" << config.session_id << "\",\n";
        out << "    \"workload_type\": \"" << metrics.workload_type << "\",\n";
        out << "    \"matrix_size\": " << metrics.matrix_size << ",\n";
        out << "    \"iterations\": " << metrics.iterations << ",\n";
        out << "    \"gpu_time_ms\": " << std::fixed << std::setprecision(3) << metrics.gpu_time_ms << ",\n";
        out << "    \"cpu_time_ms\": " << std::fixed << std::setprecision(3) << metrics.cpu_time_ms << ",\n";
        out << "    \"avg_time_per_iteration\": " << std::fixed << std::setprecision(3) << metrics.avg_time_per_iteration << ",\n";
        out << "    \"gflops\": " << std::fixed << std::setprecision(2) << metrics.gflops << ",\n";
        out << "    \"memory_bandwidth_gb_s\": " << std::fixed << std::setprecision(2) << metrics.memory_bandwidth_gb_s << ",\n";
        out << "    \"gpu_name\": \"" << metrics.gpu_name << "\",\n";
        out << "    \"compute_capability\": \"" << metrics.compute_capability << "\",\n";
        out << "    \"multiprocessor_count\": " << metrics.multiprocessor_count << ",\n";
        out << "    \"memory_utilization\": " << std::fixed << std::setprecision(1) << metrics.memory_utilization << ",\n";
        out << "    \"memory_used_mb\": " << metrics.memory_used_mb << ",\n";
        out << "    \"memory_total_mb\": " << metrics.memory_total_mb << ",\n";
        out << "    \"kernel_occupancy\": " << std::fixed << std::setprecision(1) << metrics.kernel_occupancy << ",\n";
        out << "    \"achieved_bandwidth_percent\": " << std::fixed << std::setprecision(1) << metrics.achieved_bandwidth_percent << ",\n";
        out << "    \"arithmetic_intensity\": " << std::fixed << std::setprecision(2) << metrics.arithmetic_intensity << "\n";
        out << "  }";
        
        if (i < results.size() - 1) {
            out << ",";
        }
        out << "\n";
    }
    
    out << "]\n";
}

void outputCSV(const std::vector<PerfAI::PerformanceMetrics>& results, 
               std::ostream& out, const BenchmarkConfig& config) {
    // Header
    out << "timestamp,session_id,workload_type,matrix_size,iterations,"
        << "gpu_time_ms,cpu_time_ms,avg_time_per_iteration,gflops,memory_bandwidth_gb_s,"
        << "gpu_name,compute_capability,multiprocessor_count,memory_utilization,"
        << "memory_used_mb,memory_total_mb,kernel_occupancy,achieved_bandwidth_percent,arithmetic_intensity\n";
    
    // Data rows
    for (const auto& metrics : results) {
        out << getCurrentTimestamp() << ","
            << config.session_id << ","
            << metrics.workload_type << ","
            << metrics.matrix_size << ","
            << metrics.iterations << ","
            << std::fixed << std::setprecision(3) << metrics.gpu_time_ms << ","
            << std::fixed << std::setprecision(3) << metrics.cpu_time_ms << ","
            << std::fixed << std::setprecision(3) << metrics.avg_time_per_iteration << ","
            << std::fixed << std::setprecision(2) << metrics.gflops << ","
            << std::fixed << std::setprecision(2) << metrics.memory_bandwidth_gb_s << ","
            << "\"" << metrics.gpu_name << "\","
            << metrics.compute_capability << ","
            << metrics.multiprocessor_count << ","
            << std::fixed << std::setprecision(1) << metrics.memory_utilization << ","
            << metrics.memory_used_mb << ","
            << metrics.memory_total_mb << ","
            << std::fixed << std::setprecision(1) << metrics.kernel_occupancy << ","
            << std::fixed << std::setprecision(1) << metrics.achieved_bandwidth_percent << ","
            << std::fixed << std::setprecision(2) << metrics.arithmetic_intensity << "\n";
    }
}

void outputText(const std::vector<PerfAI::PerformanceMetrics>& results, 
                std::ostream& out, const BenchmarkConfig& config) {
    out << "PerfAI CUDA Performance Benchmark Results\n";
    out << "==========================================\n";
    out << "Session ID: " << config.session_id << "\n";
    out << "Timestamp: " << getCurrentTimestamp() << "\n\n";
    
    for (const auto& metrics : results) {
        out << "Workload: " << metrics.workload_type << "\n";
        out << "Matrix Size: " << metrics.matrix_size << "\n";
        out << "Iterations: " << metrics.iterations << "\n";
        out << "GPU Time (ms): " << std::fixed << std::setprecision(3) << metrics.gpu_time_ms << "\n";
        out << "Performance (GFLOPS): " << std::fixed << std::setprecision(2) << metrics.gflops << "\n";
        out << "Memory Bandwidth (GB/s): " << std::fixed << std::setprecision(2) << metrics.memory_bandwidth_gb_s << "\n";
        out << "GPU: " << metrics.gpu_name << " (Compute " << metrics.compute_capability << ")\n";
        out << "Memory Utilization: " << std::fixed << std::setprecision(1) << metrics.memory_utilization << "%\n";
        out << "Kernel Occupancy: " << std::fixed << std::setprecision(1) << metrics.kernel_occupancy << "%\n";
        out << "---\n";
    }
}

bool runTests() {
    std::cout << "Running PerfAI CUDA benchmark tests...\n";
    
    try {
        PerfAI::WorkloadEngine engine;
        
        // Quick validation test
        auto metrics = engine.runMatrixMultiplication(256, 3);
        
        if (metrics.gflops > 0 && metrics.gpu_time_ms > 0) {
            std::cout << "✓ Basic GEMM test passed\n";
            std::cout << "  Performance: " << std::fixed << std::setprecision(2) 
                      << metrics.gflops << " GFLOPS\n";
            std::cout << "  GPU Time: " << std::fixed << std::setprecision(3) 
                      << metrics.gpu_time_ms << " ms\n";
            return true;
        } else {
            std::cout << "✗ Basic GEMM test failed\n";
            return false;
        }
        
    } catch (const std::exception& e) {
        std::cout << "✗ Test failed with exception: " << e.what() << "\n";
        return false;
    }
}

int main(int argc, char* argv[]) {
    try {
        BenchmarkConfig config = parseArguments(argc, argv);
        
        if (config.verbose) {
            std::cout << "PerfAI CUDA Performance Benchmark\n";
            std::cout << "Configuration:\n";
            std::cout << "  Matrix sizes: ";
            for (size_t i = 0; i < config.matrix_sizes.size(); ++i) {
                std::cout << config.matrix_sizes[i];
                if (i < config.matrix_sizes.size() - 1) std::cout << ",";
            }
            std::cout << "\n";
            std::cout << "  Iterations: " << config.iterations << "\n";
            std::cout << "  Output format: " << config.output_format << "\n";
            std::cout << "  Session ID: " << config.session_id << "\n\n";
        }
        
        // Generate session ID if not provided
        if (config.session_id.empty()) {
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            std::stringstream ss;
            ss << "session_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
            config.session_id = ss.str();
        }
        
        // Run tests if in test mode
        if (config.test_mode) {
            bool test_passed = runTests();
            return test_passed ? 0 : 1;
        }
        
        // Initialize workload engine
        PerfAI::WorkloadEngine engine;
        std::vector<PerfAI::PerformanceMetrics> all_results;
        
        // Determine which workloads to run
        bool run_gemm = false, run_conv = false, run_bandwidth = false;
        
        for (const auto& workload : config.workloads) {
            if (workload == "all") {
                run_gemm = run_conv = run_bandwidth = true;
                break;
            } else if (workload == "gemm") {
                run_gemm = true;
            } else if (workload == "conv") {
                run_conv = true;
            } else if (workload == "bandwidth") {
                run_bandwidth = true;
            }
        }
        
        if (!run_gemm && !run_conv && !run_bandwidth) {
            run_gemm = true; // Default to GEMM if nothing specified
        }
        
        // Run matrix multiplication benchmarks
        if (run_gemm) {
            if (config.verbose) {
                std::cout << "Running matrix multiplication benchmarks...\n";
            }
            
            for (int size : config.matrix_sizes) {
                auto metrics = engine.runMatrixMultiplication(size, config.iterations);
                all_results.push_back(metrics);
                
                if (config.verbose) {
                    std::cout << "  Size " << size << ": " 
                              << std::fixed << std::setprecision(2) << metrics.gflops 
                              << " GFLOPS\n";
                }
            }
        }
        
        // Run convolution benchmarks
        if (run_conv) {
            if (config.verbose) {
                std::cout << "Running convolution benchmarks...\n";
            }
            
            for (int size : config.matrix_sizes) {
                auto metrics = engine.runConvolution(size, 3, config.iterations);
                all_results.push_back(metrics);
                
                if (config.verbose) {
                    std::cout << "  Size " << size << ": " 
                              << std::fixed << std::setprecision(2) << metrics.gflops 
                              << " GFLOPS\n";
                }
            }
        }
        
        // Run memory bandwidth benchmarks
        if (run_bandwidth) {
            if (config.verbose) {
                std::cout << "Running memory bandwidth benchmarks...\n";
            }
            
            std::vector<size_t> data_sizes = {1 << 20, 1 << 24, 1 << 28}; // 1MB, 16MB, 256MB
            for (size_t size : data_sizes) {
                auto metrics = engine.runMemoryBandwidth(size, config.iterations);
                all_results.push_back(metrics);
                
                if (config.verbose) {
                    std::cout << "  Size " << (size >> 20) << "MB: " 
                              << std::fixed << std::setprecision(2) << metrics.memory_bandwidth_gb_s 
                              << " GB/s\n";
                }
            }
        }
        
        // Output results
        std::ostream* out = &std::cout;
        std::ofstream file_out;
        
        if (!config.output_file.empty()) {
            file_out.open(config.output_file);
            if (file_out.is_open()) {
                out = &file_out;
            } else {
                std::cerr << "Warning: Could not open output file " << config.output_file 
                          << ", using stdout\n";
            }
        }
        
        if (config.output_format == "json") {
            outputJSON(all_results, *out, config);
        } else if (config.output_format == "csv") {
            outputCSV(all_results, *out, config);
        } else {
            outputText(all_results, *out, config);
        }
        
        if (config.verbose) {
            std::cout << "\nBenchmark completed successfully!\n";
            std::cout << "Total measurements: " << all_results.size() << "\n";
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
