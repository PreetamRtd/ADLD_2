# Technical Report: Assembly Optimization Suite

## 1. Introduction
The objective of this project is to explore the performance limitations of high-level compiled languages (C++) compared to hand-optimized assembly on the Apple Silicon (ARM64) architecture. This suite focuses on SIMD-friendly workloads where assembly can provide substantial speedups, plus one memory-latency benchmark to demonstrate hardware limitations.

## 2. Methodology
The benchmarks are executed on an ARM64 processor. Each test case compares a C++ implementation compiled with `-O2` (Standard optimization) against a manual assembly implementation. Workloads are designed to showcase different optimization opportunities.

### 2.1 Measurement Infrastructure
- **Multi-iteration sampling**: Each benchmark runs 10+ iterations
- **Standard deviation**: Reported as ±σ for stability analysis
- **Derived metrics**: CPE, GB/s, Cycles/Byte calculated automatically
- **Verification**: ASM output compared against CPP reference

## 3. Implementation Details

### 3.1 ReLU Kernel (SIMD Parallelism)
**Concept**: $f(x) = \max(0, x)$
- **CPP**: Sequential loop with conditional branch
- **ASM**: NEON `fmax v0.4s, v0.4s, v1.4s` (4 floats/iteration, no branches)
- **Speedup**: ~1.8× (SIMD + branch elimination)

### 3.2 Dot Product (Fused Multiply-Add)
**Concept**: $\sum A_i \cdot B_i$
- **CPP**: Sequential multiply-accumulate
- **ASM**: NEON `fmla v0.4s, v1.4s, v2.4s` processes 4 elements in one cycle
- **Speedup**: ~3.0× (SIMD + FMA fusion)

### 3.3 Matrix Multiplication (Register Blocking)
**Concept**: $C[i][j] = \sum_k A[i][k] \cdot B[k][j]$
- **Optimization**: Inner loop processes 4 elements with register blocking
- **Benefit**: Minimizes load/store traffic by keeping accumulators in SIMD registers
- **Speedup**: ~3.6× (SIMD + register blocking)

### 3.4 Linked List Traversal (Memory Latency)
**Concept**: Sum values by chasing random pointers
- **Purpose**: Demonstrates hardware limits - breaks CPU prefetcher to expose true memory latency
- **Implementation**: 64-byte padded nodes shuffled via Fisher-Yates
- **Speedup**: ~1.2× (limited by DRAM latency ~100ns, not instruction throughput)

## 4. Experimental Results

Based on real-time benchmarks on Apple Silicon:

| Algorithm | Mean CPP Time | Mean ASM Time | Speedup | GB/s (ASM) | Notes |
|:----------|:--------------|:--------------|:--------|:-----------|:------|
| **ReLU** (N=10M) | ~3.5 ms | ~1.9 ms | **1.8×** | ~19 GB/s | SIMD wins |
| **Dot Product** (N=10M) | ~15.5 ms | ~5.2 ms | **3.0×** | ~14 GB/s | FMA wins |
| **Matrix Mul** (256×256) | ~25 ms | ~7 ms | **3.6×** | ~9 GB/s | Register blocking wins |
| **Linked List** (100K) | ~1.9 ms | ~1.6 ms | **1.2×** | ~4 GB/s | Memory-bound |

### Key Observations
1. **SIMD benchmarks** (ReLU, Dot, Matrix) show 1.8-3.6× speedup
2. **Memory-latency** (Linked List) limited by DRAM access patterns (~100ns latency)
3. **Hand-tuning helps most** when CPU can execute many operations in parallel

## 5. Metrics Interpretation

| Metric | Meaning | Ideal Value |
|:-------|:--------|:------------|
| **CPE** | Cycles/Element | Lower = better |
| **GB/s** | Throughput | Closer to hardware max |
| **±σ** | Stability | Lower = consistent |

## 6. Conclusion
Hand-optimizing assembly provides maximum benefit for:
- **Data-parallel tasks**: SIMD provides 2-4× speedup
- **Fused operations**: Single-cycle FMA vs multi-cycle multiply+add
- **Register blocking**: Keeping hot data in SIMD registers

However, for:
- **Memory-latency bound code**: DRAM access (~100ns) dominates
- **Branch-heavy code**: Compiler typically optimizes well

Assembly remains a powerful tool when the bottleneck is computation, not memory access.

---
*End of Report*
