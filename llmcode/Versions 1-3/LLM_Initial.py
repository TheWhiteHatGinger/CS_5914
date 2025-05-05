import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import cg

def poisson(shape, format='csr'):
    nx, ny = shape
    N = nx * ny
    
    # Main diagonal and neighboring diagonals
    main_diag = 4.0 * np.ones(N)
    off_diag  = -1.0 * np.ones(N - 1)
    # Zeros at boundaries between rows in 1D indexing
    for i in range(1, ny):
        off_diag[i*nx - 1] = 0
    
    # 1D discretization in x-direction
    diagonals_1d = [main_diag, off_diag, off_diag]
    A_1d = diags(diagonals_1d, [0, -1, 1], format='csr')
    
    # Block structure in y-direction
    I_block = diags([-1.0 * np.ones(N - nx)], [-nx], format='csr') + \
              diags([-1.0 * np.ones(N - nx)], [nx], format='csr')
    
    A_2d = A_1d + I_block
    
    return A_2d.asformat(format)

class SmoothedAggregationSolver:

    def __init__(self, A):
        self.A = A
    
    def solve(self, b, x0=None, tol=1e-8, callback=None):
        x, info = cg(self.A, b, x0=x0, maxiter=None, callback=callback)
        return x
    
    def smoothed_aggregation_solver(A):
        return SmoothedAggregationSolver(A)
    
    class RugeStubenSolver:
        def init(self, A):
            self.A = A
        
        def solve(self, b, x0=None, tol=1e-8, callback=None):
            x, info = cg(self.A, b, x0=x0, maxiter=None, callback=callback)
            return x
    
    def ruge_stuben_solver(A):
        return RugeStubenSolver(A)
