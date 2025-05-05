import numpy as np
from scipy.sparse import csr_matrix
import amg_solver

def poisson(shape, format='csr'):
    nx, ny = shape
    values, colind, rowptr, rows, cols = amg_solver.poisson(nx, ny)
    A = csr_matrix((values, colind, rowptr), shape=(rows, cols))
    return A.asformat(format)

class SmoothedAggregationSolver:
    def __init__(self, A):
        self.A = A
        self.values = A.data
        self.colind = A.indices
        self.rowptr = A.indptr
        self.rows, self.cols = A.shape

    def solve(self, b, x0=None, tol=1e-8, callback=None):
        # Convert inputs to lists for pybind11
        b = b.tolist()
        x = amg_solver.cg_solve(self.values, self.colind, self.rowptr,
                               self.rows, self.cols, b, tol, 0)
        x = np.array(x)
        # Apply callback if provided
        if callback:
            callback(x)
        return x

    @staticmethod
    def smoothed_aggregation_solver(A):
        return SmoothedAggregationSolver(A)

class RugeStubenSolver:
    def __init__(self, A):
        self.A = A
        self.values = A.data
        self.colind = A.indices
        self.rowptr = A.indptr
        self.rows, self.cols = A.shape

    def solve(self, b, x0=None, tol=1e-8, callback=None):
        # Convert inputs to lists for pybind11
        b = b.tolist()
        x = amg_solver.cg_solve(self.values, self.colind, self.rowptr,
                               self.rows, self.cols, b, tol, 0)
        x = np.array(x)
        # Apply callback if provided
        if callback:
            callback(x)
        return x

    @staticmethod
    def ruge_stuben_solver(A):
        return RugeStubenSolver(A)
