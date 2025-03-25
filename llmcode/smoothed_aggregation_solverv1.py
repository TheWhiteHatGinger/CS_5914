import numpy as np
from scipy.sparse import csr_matrix, spdiags

def naive_aggregation(A):
    n_rows = A.shape[0]
    agg = np.zeros(n_rows, dtype=np.int32)
    agg_id = 0

    i = 0
    while i < n_rows:
        agg[i] = agg_id
        if i + 1 < n_rows:
            agg[i + 1] = agg_id
        agg_id += 1
        i += 2
    return agg

def build_tentative_prolongation(agg):
    n_fine = len(agg)
    n_coarse = agg.max() + 1

    row_inds = np.arange(n_fine, dtype=np.int32)
    col_inds = agg
    data = np.ones(n_fine, dtype=np.float64)

    # Build P in CSR format
    P = csr_matrix( (data, (row_inds, col_inds)), shape=(n_fine, n_coarse) )
    return P

def smooth_prolongation(A, P, omega=4.0/3.0, iterations=1):
    D_inv = 1.0 / A.diagonal()
    P_smoothed = P.copy().tolil()  # LIL is easier for row-by-row updates

    # One approach is to multiply: M = D_inv * A (elementwise for D_inv).
    # Then P <- P - omega * M * P. We'll do it iteratively a few times.
    for _ in range(iterations):
        # M * P:
        MP = A.dot(P_smoothed)
        # Multiply each row i of MP by D_inv[i]
        for i in range(P_smoothed.shape[0]):
            MP[i, :] *= D_inv[i]
        P_smoothed = P_smoothed - omega * MP

    return P_smoothed.tocsr()

def restrict_operator(P):
    return P.transpose().tocsr()

def build_coarse_operator(A, P, R):
    return R.dot(A.dot(P))

def build_hierarchy(A, max_levels=10, min_coarse_size=10):
    hierarchy = []
    A_current = A

    for _ in range(max_levels):
        level = {}
        level['A'] = A_current

        # If already small enough, stop coarsening
        if A_current.shape[0] <= min_coarse_size:
            hierarchy.append(level)
            break

        # 1) Build naive aggregates
        agg = naive_aggregation(A_current)

        # 2) Build tentative prolongation
        P_tent = build_tentative_prolongation(agg)

        # 3) Smooth the tentative prolongation
        P_smoothed = smooth_prolongation(A_current, P_tent, omega=4.0/3.0, iterations=1)

        # 4) Build R = P^T
        R = restrict_operator(P_smoothed)

        # 5) Build coarse operator
        A_coarse = build_coarse_operator(A_current, P_smoothed, R)

        # Store in the hierarchy
        level['P'] = P_smoothed
        level['R'] = R
        
        hierarchy.append(level)

        # Move on to next coarser level
        A_current = A_coarse

    return hierarchy


def jacobi_smoother(A, x, b, iterations=2, omega=0.8):
    D_inv = 1.0 / A.diagonal()
    for _ in range(iterations):
        r = b - A.dot(x)
        x += omega * (D_inv * r)
    return x

def v_cycle(hierarchy, b, level=0, x=None):
    A = hierarchy[level]['A']
    n = A.shape[0]

    if x is None:
        x = np.zeros(n, dtype=A.dtype)

    # If we're at the coarsest level, solve directly (e.g. via dense solve):
    if (level == len(hierarchy) - 1) or ('P' not in hierarchy[level]):
        return np.linalg.solve(A.toarray(), b)

    # 1) Pre-smooth
    x = jacobi_smoother(A, x, b, iterations=2, omega=0.8)
    # 2) Compute residual
    r = b - A.dot(x)
    # 3) Restrict residual to coarser space
    R = hierarchy[level]['R']
    r_coarse = R.dot(r)
    # 4) Recursively solve on coarse level
    x_coarse = v_cycle(hierarchy, r_coarse, level+1)
    # 5) Prolong correction
    P = hierarchy[level]['P']
    x += P.dot(x_coarse)
    # 6) Post-smooth
    x = jacobi_smoother(A, x, b, iterations=2, omega=0.8)

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
            r = b - self.A.dot(x)
            nr = np.linalg.norm(r)
            if verbose:
                print(f"Cycle {i+1:2d}: residual = {nr:.6e}")
            if nr < tol * normb:
                break
        return x

    @staticmethod
    def smoothed_aggregation_solver(A):
        return SmoothedAggregationSolver(A)
