import time
import numpy as np
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

from scipy.sparse import spdiags, csr_matrix
from smoothed_aggregation_solver import SmoothedAggregationSolver

def poisson_2d(n, format='csr'):
    N = n * n
    e = np.ones(N)
    # For 2D Poisson, we have: T (main diagonal), and neighbors offsets
    # We can build via spdiags:
    diagonals = [
    -4.0 * e,
    1.0 * np.roll(e, 1),
    1.0 * np.roll(e, -1),
    1.0 * np.roll(e, n),
    1.0 * np.roll(e, -n)
    ]
    offsets = [0, -1, 1, -n, n]
    # Zero out the 'wrap-around' connections
    for i in range(n):
        # Each row in the grid, remove horizontal wrap
        diagonals[1][i*n] = 0.0
        diagonals[2][i*n + (n-1)] = 0.0

    A = spdiags(diagonals, offsets, N, N, format=format)
    return A

def benchmark_solver(n):
    A = poisson_2d(n, format='csr')
    b = np.ones(A.shape[0], dtype=float)

    # Construct the solver
    solver = SmoothedAggregationSolver.smoothed_aggregation_solver(A)

    # Solve and measure performance
    start_time = time.perf_counter()
    x = solver.solve(b)
    end_time = time.perf_counter()

    runtime = end_time - start_time

    # Compute final residual norm
    r = b - A.dot(x)
    res_norm = np.linalg.norm(r)

    return runtime, res_norm

def benchmark_solver_memory(n):
    if not HAS_MEMORY_PROFILER:
        raise ImportError("memory_profiler module not found.")
    def run_solver():
        A = poisson_2d(n, format='csr')
        b = np.ones(A.shape[0], dtype=float)
        solver = SmoothedAggregationSolver.smoothed_aggregation_solver(A)
        solver.solve(b, tol=1e-8)
        mem_usage = memory_usage((run_solver,), interval=0.1)
    return max(mem_usage) - min(mem_usage)
    
    def benchmark_solver_flops(n):
        if not HAS_PAPI:
            raise ImportError("pypapi not installed or system does not support PAPI events.")
        A = poisson_2d(n, format='csr')
        b = np.ones(A.shape[0], dtype=float)
        solver = SmoothedAggregationSolver.smoothed_aggregation_solver(A)
        high_level.start_counters([events.PAPI_FP_OPS])
        solver.solve(b, tol=1e-8)
        counters = high_level.stop_counters()
    return counters[0]

def benchmark_solver_convergence(n):
    A = poisson_2d(n, format='csr')
    b = np.ones(A.shape[0], dtype=float)
    solver = SmoothedAggregationSolver.smoothed_aggregation_solver(A)

    residuals = []
    x0 = np.zeros_like(b)

    def callback(x):
        r = b - A.dot(x)
        residuals.append(np.linalg.norm(r))

    # Modify the solver to accept a callback (if you want it integrated).
    # This SmoothedAggregationSolver doesn't have a callback parameter by default,
    # so for demonstration, we'll simply capture residuals in each cycle below.

    # Example: If you instrument your solver's solve method to call a callback each cycle
    # solver.solve(b, x0=x0, tol=1e-8, callback=callback)
    #
    # For the current implementation, we can manually replicate the cycle structure:
    tol = 1e-8
    normb = np.linalg.norm(b)
    x = x0
    max_cycles = 50
    for i in range(max_cycles):
        x = solver.v_cycle(solver.hierarchy, b, level=0, x=x)
        r = b - solver.A.dot(x)
        nr = np.linalg.norm(r)
        residuals.append(nr)
        if nr < tol * normb:
            break
    return residuals

def run_all_tests():
    sizes = [64, 128, 256, 512, 1024, 4096]

    print("========== SmoothedAggregationSolver Tests ==========\n")

    # 1) Timing and final residual
    print("-------------- Timing and Residual --------------")
    for n in sizes:
        runtime, res_norm = benchmark_solver(n)
        print(f"n={n:4d}, runtime={runtime:.4e} s, final residual={res_norm:.4e}")

    # 2) Convergence test
    n_test = 16
    print("\n-------------- Convergence Test --------------")
    residuals = benchmark_solver_convergence(n_test)
    for i, r in enumerate(residuals, start=1):
        print(f"   Cycle {i}: residual norm = {r:.4e}")

    # 3) (Optional) Memory usage
    # if HAS_MEMORY_PROFILER:
    #     print("\n-------------- Memory Usage --------------")
    #     for n in sizes:
    #         mem_used = benchmark_solver_memory(n)
    #         print(f"n={n}, memory usage ~ {mem_used:.2f} MB")
    # else:
    #     print("\nMemory usage not tested (install memory_profiler to enable).")

    # 4) (Optional) FLOP count
    # if HAS_PAPI:
    #     print("\n-------------- FLOP Count --------------")
    #     for n in sizes:
    #         flops = benchmark_solver_flops(n)
    #         print(f"n={n}, FLOPs={flops}")
    # else:
    #     print("\nFLOP count not tested (install pypapi or ensure PAPI support).")

run_all_tests()
