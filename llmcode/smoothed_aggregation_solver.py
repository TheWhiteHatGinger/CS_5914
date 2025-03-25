import numpy as np
from scipy.sparse import csr_matrix, spdiags, diags
from scipy.sparse.linalg import spsolve

def naive_aggregation(A):
    n_rows = A.shape[0]
    agg = np.zeros(n_rows, dtype=np.int32)
    agg_id = np.arange(0, n_rows // 2, dtype=np.int32)
    agg[0::2] = agg_id  # Even indices
    agg[1::2] = agg_id  # Odd indices (paired with previous)
    if n_rows % 2:  # Handle odd-sized input
        agg[-1] = agg_id[-1] + 1
    return agg

def build_tentative_prolongation(agg):
    n_fine = len(agg)
    n_coarse = agg.max() + 1
    return csr_matrix((np.ones(n_fine, dtype=np.float64), (np.arange(n_fine, dtype=np.int32), agg)), 
                      shape=(n_fine, n_coarse))

def smooth_prolongation(A, P, omega=4.0/3.0, iterations=1):
    D_inv = 1.0 / A.diagonal()  # Precompute inverse diagonal
    P_smoothed = P  # Avoid copy, modify in-place where possible
    DA = diags(D_inv) @ A  # Precompute D_inv * A as sparse matrix
    
    for _ in range(iterations):
        P_smoothed -= omega * (DA @ P_smoothed)
    
    return P_smoothed.tocsr()

def restrict_operator(P):
    return P.T.tocsr()  # Transpose directly in CSR format

def build_coarse_operator(A, P, R):
    return R @ (A @ P)  # Use @ operator for clarity and efficiency

def build_hierarchy(A, max_levels=10, min_coarse_size=10):
    hierarchy = []
    A_current = A.copy()  # Ensure A isn't modified

    for _ in range(max_levels):
        if A_current.shape[0] <= min_coarse_size:
            hierarchy.append({'A': A_current})
            break

        agg = naive_aggregation(A_current)
        P_tent = build_tentative_prolongation(agg)
        P_smoothed = smooth_prolongation(A_current, P_tent)
        R = restrict_operator(P_smoothed)
        A_coarse = build_coarse_operator(A_current, P_smoothed, R)

        hierarchy.append({'A': A_current, 'P': P_smoothed, 'R': R})
        A_current = A_coarse

    return hierarchy

def jacobi_smoother(A, x, b, D_inv=None, iterations=2, omega=0.8):
    if D_inv is None:
        D_inv = 1.0 / A.diagonal()
    D_inv = diags(D_inv)  # Convert to sparse diagonal matrix
    for _ in range(iterations):
        x += omega * D_inv @ (b - A @ x)
    return x

def v_cycle(hierarchy, b, level=0, x=None):
    A = hierarchy[level]['A']
    n = A.shape[0]
    if x is None:
        x = np.zeros(n, dtype=A.dtype)

    if level == len(hierarchy) - 1 or 'P' not in hierarchy[level]:
        return spsolve(A, b)  # Use sparse solver instead of dense

    # Precompute D_inv once per level
    D_inv = 1.0 / A.diagonal()
    x = jacobi_smoother(A, x, b, D_inv=D_inv)
    r = b - A @ x
    R = hierarchy[level]['R']
    r_coarse = R @ r
    x_coarse = v_cycle(hierarchy, r_coarse, level + 1)
    P = hierarchy[level]['P']
    x += P @ x_coarse
    x = jacobi_smoother(A, x, b, D_inv=D_inv)

    return x

class SmoothedAggregationSolver:
    def __init__(self, A, max_levels=10, min_coarse_size=10):
        self.A = A
        self.hierarchy = build_hierarchy(A, max_levels, min_coarse_size)

    def solve(self, b, x0=None, tol=1e-8, max_cycles=50, verbose=False):
        normb = np.linalg.norm(b)
        x = x0 if x0 is not None else np.zeros_like(b)

        for i in range(max_cycles):
            x = v_cycle(self.hierarchy, b, level=0, x=x)
            r = b - self.A @ x
            nr = np.linalg.norm(r)
            if verbose:
                print(f"Cycle {i+1:2d}: residual = {nr:.6e}")
            if nr < tol * normb:
                break
        return x

    @staticmethod
    def smoothed_aggregation_solver(A):
        return SmoothedAggregationSolver(A)

# Example usage:
if __name__ == "__main__":
    n = 100
    A = spdiags([np.ones(n), -2*np.ones(n), np.ones(n)], [-1, 0, 1], n, n).tocsr()
    b = np.random.rand(n)
    solver = SmoothedAggregationSolver(A)
    x = solver.solve(b, verbose=True)