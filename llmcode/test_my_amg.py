import time
import numpy as np

import my_amg
from my_amg import poisson

try:
    from memory_profiler import memory_usage
    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False
try:
    from pypapi import events, high_level
    HAS_PAPI = True
except ImportError:
    HAS_PAPI = False

def benchmark_solver(n, method='smoothed_aggregation'):
    # 1) Define the matrix (2D Poisson)
    A = poisson((n, n), format='csr')
    b = np.ones(A.shape[0], dtype=float)

    # 2) Construct the AMG hierarchy (via our custom methods)
    if method == 'smoothed_aggregation':
        ml = my_amg.SmoothedAggregationSolver.smoothed_aggregation_solver(A)
    elif method == 'ruge_stuben':
        ml = my_amg.ruge_stuben_solver(A)
    else:
        raise ValueError(f"Unknown method: {method}")

    # 3) Solve and measure time
    start_time = time.perf_counter()
    _ = ml.solve(b, tol=1e-8)
    end_time = time.perf_counter()
    runtime = end_time - start_time

    return runtime

def benchmark_solver_memory(n, method='smoothed_aggregation'):
    if not HAS_MEMORY_PROFILER:
        raise ImportError("memory_profiler module not found. Install via pip install memory_profiler.")

    def run_solver():
        # Build the matrix and solver
        A = poisson((n, n), format='csr')
        b = np.ones(A.shape[0], dtype=float)
        if method == 'smoothed_aggregation':
            ml = my_amg.SmoothedAggregationSolver.smoothed_aggregation_solver(A)
        elif method == 'ruge_stuben':
            ml = my_amg.ruge_stuben_solver(A)
        else:
            raise ValueError(f"Unknown method: {method}")
        # Solve
        ml.solve(b, tol=1e-8)

    # Measure memory usage during run_solver
    mem_usage = memory_usage((run_solver,), interval=0.1)
    # Return the difference between max and min usage in MB
    return max(mem_usage) - min(mem_usage)

def benchmark_solver_flops(n, method='smoothed_aggregation'):
    if not HAS_PAPI:
        raise ImportError("pypapi not installed or system does not support PAPI events.")
    
    # Build the matrix and solver
    A = poisson((n, n), format='csr')
    b = np.ones(A.shape[0], dtype=float)
    
    if method == 'smoothed_aggregation':
        ml = my_amg.SmoothedAggregationSolver.smoothed_aggregation_solver(A)
    elif method == 'ruge_stuben':
        ml = my_amg.ruge_stuben_solver(A)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Start counters using high_level
    high_level.start_counters([events.PAPI_FP_OPS])
    
    _ = ml.solve(b, tol=1e-8)
    
    # Stop counters and read FLOPs
    counters = high_level.stop_counters()
    flops = counters[0]  # Since we started only one counter: [PAPI_FP_OPS]
    return flops

def benchmark_solver_convergence(n, method='smoothed_aggregation'):
    A = poisson((n, n), format='csr')
    b = np.ones(A.shape[0], dtype=float)
    
    if method == 'smoothed_aggregation':
        ml = my_amg.SmoothedAggregationSolver.smoothed_aggregation_solver(A)
    elif method == 'ruge_stuben':
        ml = my_amg.ruge_stuben_solver(A)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    residuals = []
    x0 = np.zeros_like(b)
    
    def callback(x):
        # Compute the residual norm at each iteration
        r = np.linalg.norm(b - A @ x)
        residuals.append(r)

    _ = ml.solve(b, x0=x0, tol=1e-8, callback=callback)
    return residuals

def run_all_benchmarks():
    # Problem sizes for demonstration
    sizes = [64, 128, 256, 512, 1024, 4096]
    method = 'smoothed_aggregation'
    
    print(f"Benchmarking with method: {method}\n")
    
    ## 1) Time benchmark
    print("--------------- Execution Time ---------------")
    for n in sizes:
        runtime = benchmark_solver(n, method)
        print(f"n={n}, runtime={runtime:.4f} s")
    
    # 2) Memory usage
    if HAS_MEMORY_PROFILER:
        print("\n--------------- Memory Usage ---------------")
        for n in sizes:
            mem_used = benchmark_solver_memory(n, method)
            print(f"n={n}, memory usage ≈ {mem_used:.2f} MB")
    else:
        print("\nMemory usage not tested (memory_profiler not installed).")
    
    # 3) Convergence check
    print("\n--------------- Convergence ---------------")
    n_test = 128
    residuals = benchmark_solver_convergence(n_test, method)
    print(f"n={n_test}, residual norms per iteration:")
    for i, r in enumerate(residuals, 1):
        print(f"  Iter {i}: residual = {r:.2e}")

run_all_benchmarks()