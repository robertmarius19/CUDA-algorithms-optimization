# High-Performance CUDA Optimization on NVIDIA Jetson Orin

## Project Overview
This repository hosts a comparative study and implementation of high-performance computing algorithms on embedded edge hardware. The project focuses on optimizing three computationally intensive algorithms: **2D Convolution**, **General Matrix Multiplication (GEMM)**, and **Discrete Fourier Transform (DFT)**.

The implementations target the **NVIDIA Jetson Orin Nano** platform, demonstrating how specific CUDA memory hierarchy optimizations (Shared Memory Tiling, Constant Memory, and Tensor Core utilization) can overcome memory bandwidth bottlenecks inherent in embedded systems.

## Hardware Specifications
Benchmarks were conducted on the following hardware:
* **Device:** NVIDIA Jetson Orin Nano Developer Kit
* **GPU Architecture:** NVIDIA Ampere (1024 CUDA Cores, 32 Tensor Cores)
* **CPU:** 6-core Arm Cortex-A78AE v8.2 64-bit
* **Memory:** 8GB LPDDR5 (Unified Memory Architecture)
* **Storage:** NVMe SSD

## Performance Results

The following table summarizes the peak performance achieved for the maximum dataset sizes tested ($N=16384$ for Conv2D/GEMM, $N=32768$ for DFT).

| Algorithm | Input Size (N) | CPU Sequential | CPU Parallel (OpenMP/BLAS) | GPU CUDA (Optimized) | Speedup (vs Seq) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Conv2D** | $16,384 \times 16,384$ | 1448.45 ms | 794.09 ms | **2.33 ms** | **621x** |
| **GEMM** | $16,384 \times 16,384$ | N/A (Timeout) | 17,983 ms (17.9s) | **1,679 ms (1.67s)** | **10.7x** (vs Parallel) |
| **DFT** | $32,768$ | 21,632 ms (21.6s)| 586.46 ms | **29.33 ms** | **737x** |

> **Note:** For GEMM, the comparison is made against the optimized OpenBLAS implementation on CPU, as sequential execution times out for large $N$.

## Technical Implementation Details

### 1. 2D Convolution (Conv2D)
* **Bottleneck:** High memory bandwidth usage due to repetitive filter access.
* **Optimization:**
    * **Constant Memory:** The convolution kernel is stored in `__constant__` memory to leverage the hardware broadcast cache.
    * **Halo Loading:** A collaborative thread loading strategy moves image tiles plus their required "halo" (borders) into **Shared Memory**, eliminating redundant global memory reads.

### 2. Matrix Multiplication (GEMM)
* **Bottleneck:** Arithmetic intensity and cache coherence.
* **Optimization (Hybrid Approach):**
    * **Small N (< 2048):** Custom CUDA kernel utilizing 2D Shared Memory Tiling ($32 \times 32$ blocks) to minimize global memory traffic.
    * **Large N (>= 2048):** Offloads computation to **cuBLAS** to utilize hardware **Tensor Cores** for mixed-precision acceleration.

### 3. Discrete Fourier Transform (DFT)
* **Bottleneck:** $O(N^2)$ complexity with non-coalesced memory access patterns.
* **Optimization:**
    * **Shared Memory Tiling:** Input data is loaded into shared memory in chunks, allowing threads to compute partial sums without accessing global memory repeatedly.
    * **Vectorization:** Utilizes `__sincosf()` intrinsic functions to compute sine and cosine in a single hardware instruction.
