import timeit
from  pylab import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.sparse.linalg

class simple_tworoom:

    # initial values
    n = 40
    iterations = 10
    relaxation = 0.8


    ht = 30
    lt = 15
    N = n-2     # internal grid points
    dx = 1 / (N+1)

    u1 = zeros((n,n))
    u2 = zeros((n,n))

    # boundary values
        # room 1
    u1n = lt
    u1w = ht
    u1s = lt
    u1e = np.ones(N)*lt# np.zeros(N) # neumann (initial value = 0)
        # room 2
    u2n = lt
    u2w = np.zeros(N)#np.ones(N)*lt # dirichlet (initial value = default room temp) # n-1?
    u2s = lt
    u2e = lt

    # setup of A matrices
    sup = ones(N**2 - 1)
    for i in range(1,len(sup)+1):
        if i % N == 0: sup[i-1] = 0
    A1 = np.diag(ones(N ** 2 - N), -N) \
         + np.diag(sup, -1) \
         + np.diag(-4. * ones(N ** 2), 0) \
         + np.diag(sup, 1) \
         + np.diag(ones(N ** 2 - N), N)

    A2 = A1.copy()
    for i in range(0, N):
        #A1[i * N + N - 1, i * N + N - 1] = -3
        A2[i * N, i * N] = -3

    

    A1 = sparse.csr_matrix(1 / (dx ** 2) * A1)
    A2 = sparse.csr_matrix(1 / (dx ** 2) * A2)



    # setting up boundary
    bc1 = np.zeros((N,N))
    bc1[0, :] += u1n / dx ** 2
    bc1[-1, :] += u1s / dx ** 2
    bc1[:, 0] += u1w / dx ** 2
    bc1[:, -1] += u1e / dx
    bc1 = np.asarray(bc1).reshape(-1)

    bc2 = np.zeros((N,N))
    bc2[0, :] += u2n / dx ** 2
    bc2[-1, :] += u2s / dx ** 2
    bc2[:, 0] += u2w / dx
    bc2[:, -1] += u2e / dx ** 2
    bc2 = np.asarray(bc2).reshape(-1)


    solvetime = timeit.default_timer()

    # solve
    for i in range(0,iterations):
        print("iteration:", (i+1))


    # room 2
        # updating boundary from room 1
        if i > 0:
            bc2 = np.reshape(bc2, (N, N))
            bc2[:, 0] = u2w / dx
            bc2 = np.asarray(bc2).reshape(-1)

        u_inner_2 = np.reshape(scipy.sparse.linalg.spsolve(A2,-bc2),(N,N))

        u2[1:-1,1:-1] = u_inner_2
        u2[0, :] = u2n
        u2[-1, :] = u2s
        u2[1:-1, 0] = u2[1:-1, 1] + dx * u2w * u2[1:-1, 1]
        u2[1:-1, -1] = u2e

        if i > 0: u2_prev = relaxation * u2 + (1-relaxation) * u2_prev
        else: u2_prev = u2

        u1e = u2_prev[1:-1,0]

    # room 1
        # updating boundary from room 2
        bc1 = np.reshape(bc1,(N,N))
        bc1[:, -1] = u1e / dx
        bc1 = np.asarray(bc1).reshape(-1)

        u_inner_1 = np.reshape(scipy.sparse.linalg.spsolve(A1, -bc1), (N, N))

        u1[1:-1, 1:-1] = u_inner_1
        u1[0, :] = u1n
        u1[-1, :] = u1s
        u1[:, 0] = u1w

        u1[1:-1, -1] = u1[1:-1, -2] + dx * u1e * u1[1:-1, -2]
        
        if i > 0: u1_prev = relaxation * u1 + (1-relaxation) * u1_prev
        else: u1_prev = u1

        u2w = (u1_prev[1:-1,-2]-u1_prev[1:-1,-1]) / dx

    solvetime = timeit.default_timer() - solvetime
    print("It took {} seconds to iterate a solution.".format(solvetime))

    u_sol = zeros((n,2*n))
    u_sol[:,0:n] = u1_prev
    u_sol[:,n:2*n] = u2_prev

    plt.pcolor(flipud(u_sol))
    plt.colorbar()
    plt.show()












