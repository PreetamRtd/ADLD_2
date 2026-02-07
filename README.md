# Assembly Benchmark Suite

An ARM64 (Apple Silicon) assembly optimization project demonstrating the power of SIMD (NEON) instructions and low-level optimizations compared to high-level C++ (-O2 optimized).

## 🚀 Overview

This suite benchmarks common algorithms implemented in:
1.  **Naive C++**: Clean, readable code optimized with `g++ -O2`.
2.  **Hand-Optimized ARM64 Assembly**: Utilizing SIMD (NEON), manual loop unrolling, and architecture-specific optimizations.

## 📊 Benchmarks Included

| Kernel | What It Tests | ASM Strategy | Expected Speedup |
|--------|---------------|--------------|------------------|
| **ReLU** | SIMD Parallelism | 4-wide NEON fmax | ~1.8× |
| **Dot Product** | Fused Multiply-Add | NEON fmla instruction | ~3.0× |
| **Matrix Mul (256×256)** | Register Blocking | SIMD unrolled inner loop | ~3.6× |
| **Linked List (Random)** | Memory Latency | Pointer chasing with prefetch | ~1.2× (limited by DRAM) |

## 📈 Enhanced Metrics

The suite now reports:
- **Mean ± σ**: Execution time with standard deviation
- **GB/s**: Throughput (shows memory bandwidth limits)
- **CPE**: Cycles Per Element (hardware-agnostic efficiency)
- **Cycles/Byte**: How tight the code is
- **Verification**: ✓/✗ Pass/Fail indicator

## 📁 Project Structure

```bash
.
├── Makefile                    # Multi-stage build system
├── README.md                   # Project documentation
├── report.md                   # Detailed performance analysis
├── src/
│   ├── cpp/
│   │   ├── main.cpp           # Visual benchmark dashboard
│   │   ├── naive.cpp          # Reference C++ implementations
│   │   ├── metrics.hpp        # Measurement infrastructure
│   │   └── visualize.hpp      # Terminal visualization utilities
│   └── asm/
│       ├── relu_kernel.s      # ARM64 SIMD ReLU
│       ├── dot_product.s      # ARM64 SIMD Dot Product
│       ├── matrix_mul_kernel.s # SIMD Matrix Mul
│       └── linked_list.s      # Pointer-chasing traversal
└── build/                      # Compiled object files
```

## 🛠️ Build & Run

### Prerequisites
-   Apple Silicon Mac (M1/M2/M3)
-   `g++` and `as` (XCode Command Line Tools)

### Build
```bash
make clean && make
```

### Run
```bash
./benchmark
```

## 📉 Understanding the Output

```
╔════════════════════════════════════════════╗
║ ReLU Kernel (N=10000000)                   ║
╚════════════════════════════════════════════╝
CPP        [█████████████████████    ] 3.56 ms ±0.52
ASM        [███████████||            ] 1.93 ms ±0.04
  Speedup: 1.84x  [▪▪▪▪▪|▪▪▪···········] 1x→1.8x
  │ CPP: 10.46 GB/s | 1.1 CPE | 0.29 cyc/B
  │ ASM: 19.26 GB/s | 0.6 CPE | 0.15 cyc/B
  Verification: ✓ PASS
```

- **Error bars** `|` show measurement variance (±σ)
- **Speedup bar** shows ASM improvement over CPP baseline
- **GB/s** indicates memory throughput
- **CPE** shows cycles per element (lower = better)
- **Verification** confirms ASM matches CPP output

---
*Created for Educational Purposes - ADLD Project 2*
