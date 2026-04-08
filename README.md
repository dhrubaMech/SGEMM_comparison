# SGEMM CUDA Kernel Optimization & Benchmarking
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

## Repository Structure
.
├── matMul.cu                # CUDA kernels (naive + optimized)
├── helperFunction.cpp      # Utility functions (timing, initialization, etc.)
├── Makefile                # Build script
├── KernelTimings/          # Stores timing results (CSV files)
├── plotTILESIZE.py         # Example plotting script
└── README.md

## Build Instructions
Make sure you have:
* CUDA Toolkit installed
* nvcc available in your PATH
Then simply run:
```
make
```
This generates the executable:
```
res
```

## Running the Code
Run the executable:
```
./res
```

What happens during execution:
* Each kernel is executed multiple times (to reduce noise)
* Execution time is recorded for each run
* Results are saved as CSV files

## Output Format
Timing results are stored in:
```
KernelTimings/
```
Each file follows the naming convention:
```
CUDA{kernel_type}_N{repetitions}.csv
```

## Visualization
You can visualize performance trends using Python and Matplotlib.
An example script is provided:
```
python plotTILESIZE.py
```
This script demonstrates how to:
* Load CSV timing data
* Plot performance vs tile size / configuration
* Compare different kernels

## Profiling (Recommended)
For deeper analysis, use Nsight Compute (NCU):
```
ncu ./res
```
Important metrics to analyze:
* Achieved Occupancy
* Warp Efficiency
* Memory Throughput
* Register Usage
* Shared Memory Utilization

## Requirements
* CUDA-capable GPU (tested on RTX 2080 Ti)
* CUDA Toolkit
* Python (for plotting)
* Matplotlib, NumPy

