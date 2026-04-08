# SGEMM_comparisonSGEMM CUDA Kernel Optimization & Benchmarking
## Overview
This repository implements and benchmarks multiple SGEMM (Single-Precision Matrix Multiplication) kernels in CUDA, comparing their performance against:
* CPU naive implementation
* GPU naive implementation
* Optimized CUDA kernels (tiling, shared memory, vectorization, etc.)

The goal is to study how different optimization strategies affect performance and to build intuition about GPU execution behavior.

## Features
* Multiple CUDA SGEMM kernel implementations
* CPU baseline for correctness and comparison
* Performance benchmarking across kernels
* Automatic timing and CSV logging
* Ready-to-use Python script for visualization
* Designed for profiling with Nsight Compute (NCU)
