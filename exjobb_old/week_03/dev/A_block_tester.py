import numpy as np
from scipy import sparse
import scipy.sparse.linalg
from scipy.sparse import hstack
from scipy.sparse import vstack




def create_column(Ni):
    M = np.zeros((Ni ** 2,Ni))
    for i in range(Ni):
        for j in range(Ni):
            if i == j:
                M[Ni+i*Ni-1,j] = 1
    return M


def create_row(Ni):
    return np.transpose(create_column(Ni))


def create_bottom(Ni):
    return np.diag(-3 * np.ones(Ni),0)+np.diag(1*np.ones(Ni-1),-1)+np.diag(1*np.ones(Ni-1),1)


def create_boundary(Ni):
    b = np.zeros((Ni, Ni))
    b[:,0] += 30 # north
    b[0,:] += 30 # west
    b[-1,:] += 15 # south
    return np.asarray(b).reshape(-1)



n = 5
Ni = n - 2
Nj = n - 1

sup = np.ones(Ni * Ni - 1)

for i in range(1, len(sup) + 1):
    if i % Ni == 0: sup[i - 1] = 0
A1 = np.diag(np.ones(Ni * Ni - Ni), -Ni) \
    + np.diag(sup, -1) \
    + np.diag(-4. * np.ones(Ni * Ni), 0) \
    + np.diag(sup, 1) \
    + np.diag(np.ones(Ni * Ni - Ni), Ni)

A1 = sparse.csr_matrix(A1)
A2 = A1.copy()
c = sparse.csr_matrix(create_column(Ni))
r = sparse.csr_matrix(create_row(Ni))
b = sparse.csr_matrix(create_bottom(Ni))

A1 = sparse.csr_matrix(vstack((hstack((A1,c)),hstack((r,b)))))


zeros1 = sparse.csr_matrix(np.zeros((Ni*Nj,Ni**2)))
e1 = hstack((A1,zeros1))
zeros2 = np.transpose(zeros1)
eye = sparse.csr_matrix(np.eye(Ni**2))
e2 = hstack((zeros2,eye))
E1 = sparse.csr_matrix(vstack((e1,e2)))



boundary = create_boundary(Ni)
edge = np.zeros((Ni,1)) # set new aj here
edge[0] += 30 #north
edge[-1] += 15 #south
boundary = np.concatenate((boundary,np.asarray(edge).reshape(-1)))
largerboundary = np.concatenate((boundary,np.asarray(np.zeros((Ni**2,1)).reshape(-1))))


#solution1 = scipy.sparse.linalg.spsolve(A1, -1*boundary)
solution2 = scipy.sparse.linalg.spsolve(E1, -1*largerboundary)

#print(solution1)
#print(solution2)




# Creating the A2-matrix core
eye = sparse.csr_matrix((np.eye(Ni*Nj)))
e1 = hstack((eye,zeros1))
e2 = hstack((zeros2,A2))
E2 = sparse.csr_matrix(vstack((e1,e2)))

b2 = np.zeros((Ni,Ni))
b2[0,:] += 30
b2[-1,:] += 15
b2[:,-1] += 15
b2[:,0] += solution2[Ni**2:Ni**2+Ni] # access u_r
b2_zeros = np.zeros((Nj,Ni))
b2 = np.asarray(np.vstack((b2_zeros,b2))).reshape(-1)

solution_r2 = scipy.sparse.linalg.spsolve(E2,-1*b2)
print(solution_r2)








