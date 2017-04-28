import numpy as np
from scipy import sparse
import scipy.sparse.linalg


ni = 6
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

# B = np.array([[1,1],[1,2],[2,1]],np.float64)
# b = np.array([[2],[3],[3.01]],np.float64)
# sol = np.linalg.lstsq(B,b)
# print("LSQ sol",sol[0])
# print("Res",sol[1])
# print("Rank",sol[2])
# print("s",sol[3])
a = scipy.sparse.csr_matrix(A)

n = 6
B = np.ones(((n-2),(n-2)**2))
b = np.ones(((n-2)**2,1))

print(np.dot(B,b))


