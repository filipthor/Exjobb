import numpy as np
from scipy import sparse
import scipy.sparse.linalg


ni = 3
nj = 4
A = np.zeros((ni,nj))
u = np.zeros((nj,1))
counter = 0
for i in range(ni):
    u[i] = counter
    for j in range(nj):
        counter += 1
        A[i,j] = counter
#print(A)
#print(u)
print(np.dot(A,u))

B = np.zeros((2*ni,2*nj))
v = np.zeros((2*nj,1))
v[0:len(u)] = u
B[0:A.shape[0],0:A.shape[1]] = A
print(B)
print(v)
print(np.dot(B,v))



