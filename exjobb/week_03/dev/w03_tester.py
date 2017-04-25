import numpy as np
from scipy import sparse
import scipy.sparse.linalg


ni = 9
nj = 4
A = np.zeros((ni,nj))
u = np.zeros((nj,1))
counter = 0
for i in range(ni):
    for j in range(nj):
        counter += 1
        A[i,j] = counter
#print(A)
#print(u)

print(A)
B = np.zeros((10,4))
print(A[1:,:])



