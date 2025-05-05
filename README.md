# Algebraic Multigrid (AMG) Solver Optimization Project

This repository contains the code and resources for the CS 5914 Final Project, focusing on the optimization and performance analysis of Algebraic Multigrid (AMG) solvers. The project explores both manual and Large Language Model (LLM)-assisted optimization techniques to enhance the performance of AMG solvers for solving large linear systems.

## Project Overview

The project investigates the optimization of AMG solvers, which are widely used in fields such as structural mechanics, electromagnetics, computational fluid dynamics, image processing, machine learning, and neural network training. The goal is to improve the computational efficiency of AMG solvers through manual tuning and LLM-driven code generation, comparing their performance in terms of execution time, memory usage, FLOPs, and convergence behavior.

### Key Objectives
- **Manual Optimization**: Adjust parameters of the PyAMG library (e.g., max levels, max coarse, strength-of-connection, smoothers) and cache solver hierarchies for improved performance.
- **LLM Optimization**: Iteratively develop and optimize AMG solver implementations using LLMs (e.g., OpenAI's O1 Pro, Grok 3) across five versions, incorporating techniques like C++ offloading and parallel processing.
- **Performance Evaluation**: Benchmark the optimized solvers for various matrix sizes (N = 16, 128, 256, 512, 1024, 4096) and analyze execution time, memory usage, and convergence.

## Repository Structure

The repository is organized into three main directories: `baseline`, `manual`, and `llmcode`. Below is a description of each directory and its contents:

### `baseline/`
Contains the baseline implementation and benchmarking scripts for the unoptimized AMG solver.
- `benchmark.py`: Script to benchmark the baseline AMG solver.
- `program.prof`: Profiling data for the baseline implementation.

### `manual/`
Contains the manually optimized AMG solver implementation and benchmarking scripts.
- `benchmark.py`: Script to benchmark the manually optimized AMG solver.
- `program.prof`: Profiling data for the manually optimized implementation.

### `llmcode/`
Contains the LLM-optimized AMG solver implementations, divided into two subdirectories for different versions.

#### `llmcode/Versions 1-3/`
Contains Python-based LLM-optimized implementations for Versions 1 to 3.
- `LLM_Initial.py`: Initial LLM-generated code using SciPy's conjugate gradient solver.
- `LLM_Version2.py`: Version 2 with proper smoothed aggregation, avoiding SciPy's CG solver.
- `LLM_Version3.py`: Version 3 with optimizations like avoiding redundant sparse matrix conversions, precomputing diagonal inverses, and adjusting aggregation strategies.
- `Tester_Program.py`: Script to test and benchmark Versions 1-3.

#### `llmcode/Versions 4-5/`
Contains LLM-optimized implementations that incorporate C++ for performance-critical tasks.

##### `llmcode/Versions 4-5/version4/`
Version 4 offloads computations to C++ using PyBind11 and the Eigen library.
- `amg_solver.cpp`: C++ implementation of the AMG solver.
- `amg_solver.cpython-312-x86_64-linux-gnu.so`: Compiled shared object for Python integration.
- `benchmark_amg.py`: Benchmarking script for Version 4.
- `my_amg.py`: Python wrapper for the C++ AMG solver.
- `setup.py`: Setup script for building the C++ module.
- `build/`: Build directory containing compiled objects and shared libraries.
- `__pycache__/`: Cached Python bytecode.

##### `llmcode/Versions 4-5/version5/`
Version 5 optimizes for larger matrix sizes (N = 1024, 4096) using OpenMP for parallel matrix construction and size-based solver selection.
- `amg_solver_opt.cpp`: Optimized C++ implementation with OpenMP and geometric coarsening.
- `amg_solver_opt.cpython-312-x86_64-linux-gnu.so`: Compiled shared object for Python integration.
- `benchmark_amg_opt.py`: Benchmarking script for Version 5.
- `my_amg_opt.py`: Python wrapper for the optimized C++ AMG solver.
- `setup_opt.py`: Setup script for building the optimized C++ module.
- `build/`: Build directory containing compiled objects and shared libraries.
- `__pycache__/`: Cached Python bytecode.

## Installation

To run the code in this repository, you need Python 3.12, PyAMG, SciPy, NumPy, PyBind11, and a C++ compiler with Eigen and OpenMP support (for Versions 4 and 5).

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/amg-solver-optimization.git
   cd amg-solver-optimization
   ```

2. **Set up a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install pyamg scipy numpy pybind11
   ```

4. **Install C++ dependencies** (for Versions 4 and 5):
   - Install a C++ compiler (e.g., `g++`) with OpenMP support.
   - Install the Eigen library (e.g., via `sudo apt-get install libeigen3-dev` on Ubuntu or equivalent).
   - Ensure PyBind11 is installed globally for C++ bindings:
     ```bash
     pip install pybind11[global]
     ```

5. **Build C++ modules** (for Versions 4 and 5):
   ```bash
   cd llmcode/Versions 4-5/version4
   python setup.py build_ext --inplace
   cd ../version5
   python setup_opt.py build_ext --inplace
   ```

## Usage

### Running Benchmarks
- **Baseline**: Run the baseline benchmarking script:
  ```bash
  python baseline/benchmark.py
  ```
- **Manual**: Run the manually optimized benchmarking script:
  ```bash
  python manual/benchmark.py
  ```
- **LLM Versions 1-3**: Use the tester program to benchmark Versions 1-3:
  ```bash
  python llmcode/Versions 1-3/Tester_Program.py
  ```
- **LLM Version 4**: Run the Version 4 benchmarking script:
  ```bash
  python llmcode/Versions 4-5/version4/benchmark_amg.py
  ```
- **LLM Version 5**: Run the Version 5 benchmarking script:
  ```bash
  python llmcode/Versions 4-5/version5/benchmark_amg_opt.py
  ```

### Expected Output
Each benchmarking script generates performance metrics (execution time, memory usage, FLOPs, and convergence behavior) for matrix sizes N = 16, 128, 256, 512, 1024, and 4096. Results are saved or printed as specified in the scripts.

## Results

- **Manual Optimization**: Achieved significant memory reduction with performance comparable to the baseline, within the margin of error.
- **LLM Optimization**:
  - **Versions 1-3**: Progressed from a basic SciPy-based solver to a proper smoothed aggregation solver with optimizations like precomputing diagonal inverses and adjusting aggregation strategies.
  - **Version 4**: Offloaded computations to C++ using Eigen, improving performance for memory-intensive tasks.
  - **Version 5**: Optimized for large matrices (N = 1024, 4096) with parallel matrix construction and size-based solver selection, achieving significant efficiency gains at higher N values, albeit with slightly increased memory usage.

## Dependencies
- Python 3.12
- PyAMG
- SciPy
- NumPy
- PyBind11
- C++ compiler (e.g., g++)
- Eigen library
- OpenMP

## References
- Bell, Nathan, et al. "PyAMG: Algebraic Multigrid Solvers in Python." *https://likeces.illinois.edu/files/2022_BeOlSc_pyamg.pdf*
- PyAMG Documentation: *https://pyamg.readthedocs.io/*

## Acknowledgments
- This project was developed as part of the CS 5914 course.
- Special thanks to the PyAMG developers and the authors of the referenced paper for their foundational work.
- LLM optimizations were performed using OpenAI's O1 Pro model and Grok 3 by xAI.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For questions or issues, please open an issue on this GitHub repository or contact the project maintainers.