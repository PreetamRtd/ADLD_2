#ifndef VISUALIZE_HPP
#define VISUALIZE_HPP

#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <cmath>
#include "metrics.hpp"

// ANSI Color codes
namespace Color {
    const std::string RESET  = "\033[0m";
    const std::string RED    = "\033[31m";
    const std::string GREEN  = "\033[32m";
    const std::string YELLOW = "\033[33m";
    const std::string BLUE   = "\033[34m";
    const std::string CYAN   = "\033[36m";
    const std::string BOLD   = "\033[1m";
    const std::string DIM    = "\033[2m";
}

// ============================================================================
// Bar chart with error bars (± σ)
// ============================================================================
inline void draw_bar_with_error(const std::string& label, const BenchmarkMetrics& m, 
                                 double max_val, const std::string& color) {
    const int bar_width = 25;
    int filled = 0;
    
    if (max_val > 0.00001) {
        filled = static_cast<int>((m.mean_ms / max_val) * bar_width);
    }
    filled = std::clamp(filled, 0, bar_width);
    
    // Error bar position
    int error_low = 0, error_high = 0;
    if (max_val > 0.00001) {
        error_low = static_cast<int>(((m.mean_ms - m.stddev_ms) / max_val) * bar_width);
        error_high = static_cast<int>(((m.mean_ms + m.stddev_ms) / max_val) * bar_width);
    }
    error_low = std::clamp(error_low, 0, bar_width);
    error_high = std::clamp(error_high, 0, bar_width);
    
    std::cout << std::left << std::setw(10) << label << " [";
    std::cout << color;
    
    for (int i = 0; i < bar_width; ++i) {
        if (i < filled) std::cout << "█";
        else if (i == error_low || i == error_high) std::cout << "|";
        else if (i > error_low && i < filled) std::cout << "▓";
        else std::cout << " ";
    }
    
    std::cout << Color::RESET << "] ";
    std::cout << std::fixed << std::setprecision(2) << m.mean_ms << " ms";
    std::cout << Color::DIM << " ±" << std::setprecision(2) << m.stddev_ms << Color::RESET;
    std::cout << "\n";
}

// ============================================================================
// Pass/Fail indicator
// ============================================================================
inline void draw_pass_fail(bool pass) {
    if (pass) {
        std::cout << Color::GREEN << "✓ PASS" << Color::RESET;
    } else {
        std::cout << Color::RED << "✗ FAIL" << Color::RESET;
    }
}

// ============================================================================
// Speedup indicator with visual bar
// ============================================================================
inline void draw_speedup(double speedup) {
    std::cout << "  Speedup: " << Color::BOLD;
    if (speedup >= 2.0) {
        std::cout << Color::GREEN;
    } else if (speedup >= 1.0) {
        std::cout << Color::YELLOW;
    } else {
        std::cout << Color::RED;
    }
    std::cout << std::fixed << std::setprecision(2) << speedup << "x" << Color::RESET;
    
    // Draw speedup bar (1x is the baseline)
    std::cout << "  [";
    const int bar_width = 20;
    int baseline = bar_width / 4; // 1x baseline at 25%
    int filled = static_cast<int>((speedup / 4.0) * bar_width);
    filled = std::clamp(filled, 0, bar_width);
    
    for (int i = 0; i < bar_width; ++i) {
        if (i == baseline) std::cout << Color::YELLOW << "|" << Color::RESET;
        else if (i < filled) std::cout << Color::GREEN << "▪" << Color::RESET;
        else std::cout << Color::DIM << "·" << Color::RESET;
    }
    std::cout << "] 1x";
    if (speedup > 1.0) std::cout << "→" << std::fixed << std::setprecision(1) << speedup << "x";
    std::cout << "\n";
}

// ============================================================================
// Metrics row (GB/s, CPE, Cycles/Byte)
// ============================================================================
inline void draw_metrics_row(const BenchmarkMetrics& cpp, const BenchmarkMetrics& asm_m) {
    std::cout << Color::DIM;
    std::cout << "  │ CPP: " << std::fixed << std::setprecision(2) << cpp.gb_per_sec << " GB/s";
    std::cout << " | " << std::setprecision(1) << cpp.cpe << " CPE";
    std::cout << " | " << std::setprecision(2) << cpp.cycles_per_byte << " cyc/B\n";
    std::cout << "  │ ASM: " << std::setprecision(2) << asm_m.gb_per_sec << " GB/s";
    std::cout << " | " << std::setprecision(1) << asm_m.cpe << " CPE";
    std::cout << " | " << std::setprecision(2) << asm_m.cycles_per_byte << " cyc/B";
    std::cout << Color::RESET << "\n";
}

// ============================================================================
// Section header
// ============================================================================
inline void draw_header(const std::string& title, size_t n) {
    std::cout << "\n" << Color::BOLD << Color::CYAN;
    std::cout << "╔════════════════════════════════════════════╗\n";
    std::cout << "║ " << std::left << std::setw(42) << (title + " (N=" + std::to_string(n) + ")") << " ║\n";
    std::cout << "╚════════════════════════════════════════════╝" << Color::RESET << "\n";
}

// ============================================================================
// Comparison Table (Small / Medium / Large)
// ============================================================================
inline void draw_comparison_table(const std::string& name,
                                   const BenchmarkMetrics& small_cpp, const BenchmarkMetrics& small_asm,
                                   const BenchmarkMetrics& med_cpp, const BenchmarkMetrics& med_asm,
                                   const BenchmarkMetrics& large_cpp, const BenchmarkMetrics& large_asm) {
    std::cout << "\n" << Color::BOLD << "┌─────────────┬──────────┬──────────┬──────────┐\n";
    std::cout << "│ " << std::left << std::setw(11) << name << " │  Small   │  Medium  │  Large   │\n";
    std::cout << "├─────────────┼──────────┼──────────┼──────────┤" << Color::RESET << "\n";
    
    std::cout << "│ CPP (ms)    │ " 
              << std::right << std::setw(8) << std::fixed << std::setprecision(2) << small_cpp.mean_ms << " │ "
              << std::setw(8) << med_cpp.mean_ms << " │ "
              << std::setw(8) << large_cpp.mean_ms << " │\n";
    
    std::cout << "│ ASM (ms)    │ " 
              << std::setw(8) << small_asm.mean_ms << " │ "
              << std::setw(8) << med_asm.mean_ms << " │ "
              << std::setw(8) << large_asm.mean_ms << " │\n";
    
    std::cout << "│ Speedup     │ "
              << Color::GREEN << std::setw(7) << (small_cpp.mean_ms / small_asm.mean_ms) << "x" << Color::RESET << " │ "
              << Color::GREEN << std::setw(7) << (med_cpp.mean_ms / med_asm.mean_ms) << "x" << Color::RESET << " │ "
              << Color::GREEN << std::setw(7) << (large_cpp.mean_ms / large_asm.mean_ms) << "x" << Color::RESET << " │\n";
    
    std::cout << Color::BOLD << "└─────────────┴──────────┴──────────┴──────────┘" << Color::RESET << "\n";
}

// ============================================================================
// Dashboard header
// ============================================================================
inline void draw_dashboard_header() {
    std::cout << "\033[2J\033[1;1H"; // Clear screen
    std::cout << Color::BOLD << Color::BLUE;
    std::cout << "╔══════════════════════════════════════════════╗\n";
    std::cout << "║    " << Color::CYAN << "EDUCATIONAL BENCHMARK SUITE" << Color::BLUE << "        ║\n";
    std::cout << "║  " << Color::DIM << "ARM64 Assembly vs C++ Performance" << Color::RESET << Color::BOLD << Color::BLUE << "  ║\n";
    std::cout << "╚══════════════════════════════════════════════╝" << Color::RESET << "\n";
}

#endif // VISUALIZE_HPP
