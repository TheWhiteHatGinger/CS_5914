import numpy as np
#import cupy as cp
import pymp
import multiprocessing as mp
from numba import jit, prange
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
import threading
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import cg

@jit(nopython=True, parallel=True)
def numba_off_diag(ny, nx, N):
	off_diag = -np.ones(N - 1)
	for i in prange(1, ny):
		off_diag[i * nx - 1] = 0
	return off_diag

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
	num_threads = mp.cpu_count()
	main_diag = pymp.shared.array((N,), dtype='float64')
	off_diag = numba_off_diag(ny, nx, N)
	with pymp.Parallel(num_threads) as p:
		for i in p.range(N):
			main_diag[i] = 4.0
	
	def build_block():
		return diags([-1.0 * np.ones(N - nx)], [-nx], format='csr') + diags([-1.0 * np.ones(N - nx)], [nx], format='csr')
	with ThreadPoolExecutor(max_workers=2) as executor:
		future_block = executor.submit(build_block)
	I_block = future_block.result()
	diagonals_1d = [main_diag, off_diag, off_diag]
	A_1d = diags(diagonals_1d, [0, -1, 1], format='csr')
	A_2d = A_1d + I_block
	return A_2d.asformat(format)

class SmoothedAggregationSolver:
	def __init__(self, A):
		self.A = A
		self._lock = threading.Lock()
		#if hasattr(A, 'shape') and cp.cuda.is_available():
		#	self.A_gpu = cp.sparse.csr_matrix(A)
		#else:
		self.A_gpu = None

	def solve(self, b, x0=None, tol=1e-8, callback=None):
		with self._lock:
			if isinstance(self.A, np.ndarray) and isinstance(b, np.ndarray):
				self.A = strassen_matrix_multiply(self.A, np.eye(self.A.shape[0]))
			#if self.A_gpu is not None:
			#	b_gpu = cp.asarray(b)
			#	x0_gpu = cp.asarray(x0) if x0 is not None else None
			#	x_gpu, info = cp.sparse.linalg.cg(self.A_gpu, b_gpu, x0=x0_gpu, tol=tol)
			#	return cp.asnumpy(x_gpu)
			#else:
			with Parallel(n_jobs=-1):
				x, info = cg(self.A, b, x0=x0, maxiter=None, callback=callback)
				return x
	@staticmethod
	def smoothed_aggregation_solver(A):
		return SmoothedAggregationSolver(A)

