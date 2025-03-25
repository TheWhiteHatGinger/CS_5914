import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import cg

def strassen_matrix_multiply(A, B):
    n = A.shape[0]
    if n <= 64:  
        return A @ B

    mid = n // 2
    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid] 
    A22 = A[mid:, mid:]

    B11 = B[:mid, :mid]
    B12 = B[:mid, mid:]
    B21 = B[mid:, :mid]
    B22 = B[mid:, mid:]

    M1 = strassen_matrix_multiply(A11 + A22, B11 + B22)
    M2 = strassen_matrix_multiply(A21 + A22, B11)
    M3 = strassen_matrix_multiply(A11, B12 - B22)
    M4 = strassen_matrix_multiply(A22, B21 - B11)
    M5 = strassen_matrix_multiply(A11 + A12, B22)
    M6 = strassen_matrix_multiply(A21 - A11, B11 + B12)
    M7 = strassen_matrix_multiply(A12 - A22, B21 + B22)

    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

    C = np.zeros((n, n))
    C[:mid, :mid] = C11
    C[:mid, mid:] = C12
    C[mid:, :mid] = C21
    C[mid:, mid:] = C22

    return C

def poisson(shape, format='csr'):
    nx, ny = shape
    N = nx * ny

    main_diag = 4.0 * np.ones(N)
    off_diag = -1.0 * np.ones(N - 1)

    for i in range(1, ny):
        off_diag[i*nx - 1] = 0

    diagonals_1d = [main_diag, off_diag, off_diag]
    A_1d = diags(diagonals_1d, [0, -1, 1], format='csr')

    I_block = diags([-1.0 * np.ones(N - nx)], [-nx], format='csr') + \
              diags([-1.0 * np.ones(N - nx)], [nx], format='csr')

    A_2d = A_1d + I_block

    return A_2d.asformat(format)

class SmoothedAggregationSolver:
    def __init__(self, A):
        self.A = A

    def solve(self, b, x0=None, tol=1e-8, callback=None):
        if isinstance(self.A, np.ndarray) and isinstance(b, np.ndarray):
            self.A = strassen_matrix_multiply(self.A, np.eye(self.A.shape[0]))
        x, info = cg(self.A, b, x0=x0, maxiter=None, callback=callback)
        return x
        
    @staticmethod
    def smoothed_aggregation_solver(A):
        return SmoothedAggregationSolver(A)