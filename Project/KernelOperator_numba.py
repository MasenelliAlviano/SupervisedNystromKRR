
import numpy as np
from scipy.sparse.linalg import LinearOperator
import numba
from numba import jit, prange
from kernels import gaussian_kernel_numba

#TO DO make the class accept the kernel as a choice instead of a fixed gaussian


@jit
def gaussian_matvec_numba(X, v, r):
    N, D = X.shape
    result = np.zeros(N)

    # Parallelizza sul ciclo esterno (righe)
    for i in prange(N):
        val = 0.0
        for j in range(N):
            # Calcolo elemento kernel e moltiplicazione
            k_ij = gaussian_kernel_numba(X[i],X[j],r)
            val += k_ij * v[j]
        
        result[i] = val
    return result

class NumbaKernelOperator(LinearOperator):
    def __init__(self, X, r=1.0):
        self.X = X
        self.r = r
        self.shape = (X.shape[0], X.shape[0])
        self.dtype = np.float64
        
    def _matvec(self, v):
        # Chiama la funzione compilata JIT
        return gaussian_matvec_numba(self.X, v, self.r)