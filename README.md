# SGEMM_comparison
This repository presents a comprehensive study of SGEMM (Single-Precision General Matrix Multiplication) implementations in CUDA, with a focus on performance optimization and comparative analysis.

The project includes multiple GPU kernel variants, ranging from a naive implementation to progressively optimized tiled and vectorized kernels, alongside a baseline CPU implementation for reference. Each kernel is evaluated using NVIDIA profiling tools to analyze performance characteristics such as occupancy, memory throughput, and compute efficiency.

Key highlights of the repository:

* Implementation of CPU naive, GPU naive, and optimized CUDA SGEMM kernels
* Exploration of tiling strategies, shared memory usage, and vectorized memory access
* Performance comparison across different configurations (tile sizes, thread block shapes, etc.)
* Profiling using Nsight Compute (NCU) to extract metrics like:
* Achieved occupancy
* Memory bandwidth utilization
* Warp execution efficiency
* Analysis of how architectural factors such as register usage, memory coalescing, and bank conflicts impact performance

This repository is intended as a learning and benchmarking resource for understanding GPU performance optimization, as well as a practical guide for designing high-performance CUDA kernels.
