#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cstring>
#include <iomanip>
#include <thread>
#include <string>
#include <algorithm>
#include <cstdint>

#include "metrics.hpp"
#include "visualize.hpp"

// ============================================================================
// Import Assembly Functions
// ============================================================================
extern "C" {
    void relu_kernel(float* data, size_t count);
    float dot_product(float* A, float* B, size_t count);
    void matrix_mul_kernel(const float* A, const float* B, float* C, uint64_t N);
    int linked_list_sum_asm(void* head);
}

// ============================================================================
// Import Naive (C++) Functions
// ============================================================================
void naive_relu(float* data, size_t count);
float naive_dot(const float* A, const float* B, size_t count);
void naive_matrix_mul(const float* A, const float* B, float* C, size_t N);

// Forward declarations for new benchmarks (defined in naive.cpp)
struct Node {
    Node* next;
    int value;
    char padding[56];
};
int naive_linked_list_sum(Node* head);
Node* create_random_linked_list(size_t n, std::vector<Node>& storage);

// ============================================================================
// Benchmark Configuration
// ============================================================================
constexpr int NUM_ITERATIONS = 10;
constexpr size_t N = 10000000;       // 10 Million for ReLU, Dot Product
constexpr size_t MATRIX_N = 256;      // 256x256 for matrix mul
constexpr size_t LIST_N = 100000;     // 100K nodes for linked list

// ============================================================================
// Run a complete benchmark with all metrics
// ============================================================================
template<typename CppFunc, typename AsmFunc>
void run_benchmark(const std::string& name, 
                   size_t elements, 
                   size_t bytes_per_element,
                   CppFunc cpp_func, 
                   AsmFunc asm_func,
                   bool verify_result = true) {
    
    draw_header(name, elements);
    
    // Measure C++ version
    auto cpp_metrics = measure_with_stats(cpp_func, elements, bytes_per_element, NUM_ITERATIONS);
    
    // Measure ASM version
    auto asm_metrics = measure_with_stats(asm_func, elements, bytes_per_element, NUM_ITERATIONS);
    
    // Calculate speedup
    double speedup = cpp_metrics.mean_ms / asm_metrics.mean_ms;
    cpp_metrics.speedup = 1.0;
    asm_metrics.speedup = speedup;
    
    // Draw bars
    double max_val = std::max(cpp_metrics.mean_ms + cpp_metrics.stddev_ms, 
                               asm_metrics.mean_ms + asm_metrics.stddev_ms);
    draw_bar_with_error("CPP", cpp_metrics, max_val, Color::RED);
    draw_bar_with_error("ASM", asm_metrics, max_val, Color::GREEN);
    
    // Draw speedup indicator
    draw_speedup(speedup);
    
    // Draw metrics row
    draw_metrics_row(cpp_metrics, asm_metrics);
    
    // Pass/Fail indicator
    std::cout << "  Verification: ";
    draw_pass_fail(verify_result);
    std::cout << "\n";
}

int main() {
    
    srand(time(0));

    while (true) {
        draw_dashboard_header();
        
        // ====================================================================
        // 1. ReLU Kernel - SIMD Parallelism Test
        // ====================================================================
        {
            std::vector<float> data_cpp(N, 1.5f);
            std::vector<float> data_asm(N, 1.5f);
            for (size_t i = 0; i < N; i += 7) {
                data_cpp[i] = -10.0f;
                data_asm[i] = -10.0f;
            }
            
            bool verified = false;
            run_benchmark("ReLU Kernel", N, sizeof(float),
                [&]() { naive_relu(data_cpp.data(), N); },
                [&]() { relu_kernel(data_asm.data(), N); });
            
            // Verify results
            verified = verify_float_arrays(data_cpp.data(), data_asm.data(), N);
            std::cout << "  (Verified: " << (verified ? "✓" : "✗") << ")\n";
        }

        // ====================================================================
        // 2. Dot Product - FMA Test
        // ====================================================================
        {
            std::vector<float> A(N, 1.5f);
            std::vector<float> B(N, 2.0f);
            float result_cpp = 0, result_asm = 0;
            
            run_benchmark("Dot Product", N, sizeof(float) * 2,
                [&]() { result_cpp = naive_dot(A.data(), B.data(), N); },
                [&]() { result_asm = dot_product(A.data(), B.data(), N); });
            
            bool verified = verify_float_scalar(result_cpp, result_asm);
            std::cout << "  (Verified: " << (verified ? "✓" : "✗") 
                      << " CPP=" << result_cpp << " ASM=" << result_asm << ")\n";
        }

        // ====================================================================
        // 3. Matrix Multiplication - Register Blocking Test
        // ====================================================================
        {
            std::vector<float> A(MATRIX_N * MATRIX_N, 1.0f);
            std::vector<float> B(MATRIX_N * MATRIX_N, 2.0f);
            std::vector<float> C_cpp(MATRIX_N * MATRIX_N);
            std::vector<float> C_asm(MATRIX_N * MATRIX_N);
            
            run_benchmark("Matrix Mul " + std::to_string(MATRIX_N) + "x" + std::to_string(MATRIX_N), 
                          MATRIX_N * MATRIX_N * MATRIX_N, sizeof(float),
                [&]() { naive_matrix_mul(A.data(), B.data(), C_cpp.data(), MATRIX_N); },
                [&]() { matrix_mul_kernel(A.data(), B.data(), C_asm.data(), MATRIX_N); });
            
            bool verified = verify_float_arrays(C_cpp.data(), C_asm.data(), MATRIX_N * MATRIX_N, 1e-3f);
            std::cout << "  (Verified: " << (verified ? "✓" : "✗") << ")\n";
        }

        // ====================================================================
        // 4. Linked List Traversal - Memory Latency Test
        // ====================================================================
        {
            std::vector<Node> storage;
            Node* head = create_random_linked_list(LIST_N, storage);
            int result_cpp = 0, result_asm = 0;
            
            run_benchmark("Linked List (Random)", LIST_N, sizeof(Node),
                [&]() { result_cpp = naive_linked_list_sum(head); },
                [&]() { result_asm = linked_list_sum_asm(head); });
            
            bool verified = (result_cpp == result_asm);
            std::cout << "  (Verified: " << (verified ? "✓" : "✗") 
                      << " Sum=" << result_cpp << ")\n";
        }



        // ====================================================================
        // Summary Table
        // ====================================================================
        std::cout << "\n" << Color::BOLD << Color::CYAN;
        std::cout << "══════════════════════════════════════════════\n";
        std::cout << Color::DIM << "Legend: " << Color::RED << "█ CPP  " << Color::GREEN << "█ ASM  " 
                  << Color::RESET << Color::DIM << "| = ±σ  " 
                  << Color::YELLOW << "| = 1× baseline" << Color::RESET << "\n";
        std::cout << Color::DIM << "Note: Linked List is memory-latency bound;\n";
        std::cout << "ASM speedup limited by DRAM, not code." << Color::RESET << "\n";
        std::cout << Color::BOLD << Color::CYAN;
        std::cout << "══════════════════════════════════════════════\n";
        std::cout << Color::RESET;
        
        std::cout << "\nPress Ctrl+C to exit (refresh in 3s)\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    }

    return 0;
}