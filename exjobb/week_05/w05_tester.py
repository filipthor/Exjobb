import numpy as np
from scipy import sparse
import scipy.sparse.linalg
import timeit

ni = 6
nj = 4
A = np.zeros((ni,nj))
u = np.zeros((nj,1))
counter = 0
for i in range(ni):
    for j in range(nj):
        counter += 1
        A[i,j] = counter
A = np.asarray(A).reshape(-1)

