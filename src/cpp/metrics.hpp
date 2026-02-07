#ifndef METRICS_HPP
#define METRICS_HPP

#include <cmath>
#include <vector>
#include <chrono>
#include <functional>
#include <algorithm>
#include <numeric>

// CPU frequency estimate for Apple Silicon (used for cycle calculations)
constexpr double CPU_FREQ_GHZ = 3.2; // Approximate for M1/M2/M3

// ============================================================================
// BenchmarkMetrics: Comprehensive result container
// ============================================================================
struct BenchmarkMetrics {
    double mean_ms       = 0.0;   // Mean execution time in milliseconds
    double stddev_ms     = 0.0;   // Standard deviation (± σ)
    double min_ms        = 0.0;   // Minimum time
    double max_ms        = 0.0;   // Maximum time
    double cpe           = 0.0;   // Cycles per Element
    double gb_per_sec    = 0.0;   // Throughput in GB/s
    double cycles_per_byte = 0.0; // Cycles per Byte processed
    double speedup       = 0.0;   // Speedup factor (CPP/ASM)
    bool   result_match  = true;  // Verification: ASM matches CPP
    size_t elements      = 0;     // Number of elements processed
    size_t bytes         = 0;     // Bytes processed
};

// ============================================================================
// Multi-iteration measurement with statistics
// ============================================================================
template<typename Func>
BenchmarkMetrics measure_with_stats(Func func, size_t elements, size_t bytes_per_element, int iterations = 10) {
    BenchmarkMetrics result;
    result.elements = elements;
    result.bytes = elements * bytes_per_element;
    
    std::vector<double> times;
    times.reserve(iterations);
    
    // Warmup run
    func();
    
    // Measured runs
    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms > 0.0001 ? ms : 0.0001);
    }
    
    // Calculate mean
    result.mean_ms = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    
    // Calculate standard deviation
    double sq_sum = 0.0;
    for (double t : times) {
        sq_sum += (t - result.mean_ms) * (t - result.mean_ms);
    }
    result.stddev_ms = std::sqrt(sq_sum / times.size());
    
    // Min/Max
    result.min_ms = *std::min_element(times.begin(), times.end());
    result.max_ms = *std::max_element(times.begin(), times.end());
    
    // Calculate derived metrics
    double seconds = result.mean_ms / 1000.0;
    double total_bytes = static_cast<double>(result.bytes);
    double cycles = seconds * CPU_FREQ_GHZ * 1e9;
    
    result.gb_per_sec = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / seconds;
    result.cpe = cycles / static_cast<double>(result.elements);
    result.cycles_per_byte = cycles / total_bytes;
    
    return result;
}

// ============================================================================
// Simple time measurement (for single runs)
// ============================================================================
template<typename Func>
double measure_ms(Func func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    return ms > 0.0001 ? ms : 0.0001;
}

// ============================================================================
// Verification helpers
// ============================================================================
inline bool verify_float_arrays(const float* a, const float* b, size_t n, float tolerance = 1e-5f) {
    for (size_t i = 0; i < n; ++i) {
        if (std::abs(a[i] - b[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

inline bool verify_float_scalar(float a, float b, float tolerance = 1e-3f) {
    return std::abs(a - b) <= tolerance * std::max(1.0f, std::max(std::abs(a), std::abs(b)));
}

inline bool verify_int_arrays(const int* a, const int* b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

inline bool verify_sorted(const int* arr, size_t n) {
    for (size_t i = 1; i < n; ++i) {
        if (arr[i] < arr[i-1]) return false;
    }
    return true;
}

#endif // METRICS_HPP
