import numpy as np
from scipy import __version__ as scipy_version
from scipy.sparse import csr_matrix, diags, isspmatrix_csr
from scipy.sparse.linalg import spsolve, cg
from numba import njit
import pkg_resources

# Check SciPy version for tolerance parameter compatibility
SCIPY_VERSION = pkg_resources.parse_version(scipy_version)
USE_TOL = SCIPY_VERSION >= pkg_resources.parse_version('1.1.0')

# Strength-of-connection aggregation
def strength_of_connection_aggregation(A, theta=0.25):
    if not isspmatrix_csr(A):
        A = A.tocsr()
    n = A.shape[0]
    agg = np.full(n, -1, dtype=np.int32)
    agg_id = 0

    A_abs = np.abs(A)
    row_max = np.maximum(A_abs.max(axis=1).toarray().ravel(), 1e-10)
    threshold = theta * row_max

    for i in range(n):
        if agg[i] != -1:
            continue
        agg[i] = agg_id
        row_start, row_end = A.indptr[i], A.indptr[i + 1]
        for idx in range(row_start, row_end):
            j = A.indices[idx]
            if j != i and agg[j] == -1 and abs(A.data[idx]) > threshold[i]:
                agg[j] = agg_id
        agg_id += 1

    agg[agg == -1] = np.arange(agg_id, agg_id + np.sum(agg == -1))
    return agg

def build_tentative_prolongation(agg):
    n_fine = len(agg)
    n_coarse = agg.max() + 1
    return csr_matrix((np.ones(n_fine, dtype=np.float64), (np.arange(n_fine, dtype=np.int32), agg)), 
                      shape=(n_fine, n_coarse))

def smooth_prolongation(A, P, omega=4.0/3.0, iterations=1):
    D_inv = 1.0 / A.diagonal()
    DA = diags(D_inv) @ A
    P_smoothed = P
    for _ in range(iterations):
        P_smoothed -= omega * (DA @ P_smoothed)
    return P_smoothed.tocsr()

def restrict_operator(P):
    return P.T.tocsr()

def build_coarse_operator(A, P, R):
    return R @ (A @ P)

def build_hierarchy(A, max_levels=10, min_coarse_size=10):
    hierarchy = []
    A_current = A.copy()

    for _ in range(max_levels):
        if A_current.shape[0] <= min_coarse_size:
            hierarchy.append({'A': A_current})
            break

        agg = strength_of_connection_aggregation(A_current)
        P_tent = build_tentative_prolongation(agg)
        P_smoothed = smooth_prolongation(A_current, P_tent)
        R = restrict_operator(P_smoothed)
        A_coarse = build_coarse_operator(A_current, P_smoothed, R)

        hierarchy.append({'A': A_current, 'P': P_smoothed, 'R': R})
        A_current = A_coarse

    return hierarchy

@njit
def gauss_seidel_smoother(A_data, A_indices, A_indptr, x, b, iterations=2, omega=0.8):
    n = len(b)
    for _ in range(iterations):
        for i in range(n):
            row_start = A_indptr[i]
            row_end = A_indptr[i + 1]
            Ax_i = 0.0
            diag = 1.0
            for j in range(row_start, row_end):
                col = A_indices[j]
                val = A_data[j]
                if col == i:
                    diag = val
                else:
                    Ax_i += val * x[col]
            if abs(diag) > 1e-10:
                x[i] += omega * (b[i] - Ax_i) / diag
    return x

def v_cycle(hierarchy, b, level=0, x=None):
    A = hierarchy[level]['A']
    n = A.shape[0]
    if x is None:
        x = np.zeros(n, dtype=A.dtype)

    if level == len(hierarchy) - 1 or 'P' not in hierarchy[level]:
        return spsolve(A, b)

    x = gauss_seidel_smoother(A.data, A.indices, A.indptr, x, b)
    r = b - A @ x
    R = hierarchy[level]['R']
    r_coarse = R @ r
    x_coarse = v_cycle(hierarchy, r_coarse, level + 1)
    P = hierarchy[level]['P']
    x += P @ x_coarse
    x = gauss_seidel_smoother(A.data, A.indices, A.indptr, x, b)

    return x

class SmoothedAggregationSolver:
    def __init__(self, A, max_levels=10, min_coarse_size=10):
        self.A = A.tocsr()
        self.hierarchy = build_hierarchy(self.A, max_levels, min_coarse_size)

    def v_cycle(self, b, x):
        return v_cycle(self.hierarchy, b, level=0, x=x)

    def solve(self, b, x0=None, tol=1e-8, max_cycles=50, verbose=False):
        normb = np.linalg.norm(b)
        x = x0 if x0 is not None else np.zeros_like(b)

        def preconditioner(b):
            return v_cycle(self.hierarchy, b, level=0, x=np.zeros_like(b))


        x, info = cg(self.A, b, x0=x0, maxiter=max_cycles, callback=None)
        
        if verbose and info == 0:
            print(f"Converged in {max_cycles} iterations or less")
        elif verbose:
            print(f"Did not converge, info = {info}")
        
        return x

    @staticmethod
    def smoothed_aggregation_solver(A):
        return SmoothedAggregationSolver(A)

# Example usage:
if __name__ == "__main__":
    n = 1000
    A = diags([np.ones(n-1), -2*np.ones(n), np.ones(n-1)], [-1, 0, 1], shape=(n, n)).tocsr()
    b = np.random.rand(n)
    solver = SmoothedAggregationSolver(A)
    x = solver.solve(b, verbose=True)
    print("Residual norm:", np.linalg.norm(b - A @ x))