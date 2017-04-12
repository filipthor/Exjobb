import numpy as np
from scipy import sparse
import scipy.sparse.linalg

n = 6
Ni = 2*n-3
Nj = n-2 # -1
A = np.zeros((Ni,Nj))
u = np.zeros((Nj,1))
counter = 0
for i in range(Ni):
    for j in range(Nj):
        counter += 1
        A[i,j] = counter


ni = n-2
nj = n-1
print(A)
print("______access things from matrix_____")
print(A[0:ni,:]) # = u1
print(A[ni,:]) # = u_r
print(A[ni+1:,:])
print("______access things from vector_____")
V = np.asarray(A).reshape(-1)
#print(V)
print(V[0:ni**2]) # accessa u1
print(V[ni**2:ni**2+ni]) # accessa u_r
print(V[0:ni**2+ni]) # accessa u1+u_r
print(V[ni**2+ni:]) # accessa u2