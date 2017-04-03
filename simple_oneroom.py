import os
import timeit
from  pylab import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.sparse.linalg


class simple_oneroom:

    # initial conditions
    highTemp = 50
    lowTemp = 5
    n = 50
    N = n - 2
    dx = 1 / (n + 1)
    u = np.zeros((n,n))

    sup = np.ones(N**2 - 1)

    for i in range(1, len(sup)+1):
        rem = i % N
        if (rem == 0): sup[i - 1] = 0
    A = np.diag(ones(N**2 - N), -N) \
            + np.diag(sup, -1) \
            + np.diag(-4.*ones(N**2), 0) \
            + np.diag(sup, 1) \
            + np.diag(ones(N**2 - N),N)
    D = sparse.csr_matrix(1 / (dx ** 2) * A)

    # boundary conditions
    bc = np.zeros((N, N))
    bc[0, :] = lowTemp / (dx ** 2)       # y = 0 (v)
    bc[-1, :] = lowTemp / (dx ** 2)      # y = N (h)
    bc[:, -1] = lowTemp / (dx ** 2)      # x = N (t)
    bc[:, 0] = highTemp / (dx ** 2)      # x = 0 (b)

    bc[0, 0] = bc[0, 0] + highTemp / (dx ** 2)      # bv
    bc[0, -1] = bc[0, -1] + lowTemp / (dx ** 2)     # bh
    bc[-1, 0] = bc[-1, 0] + highTemp / (dx ** 2)    # tv
    bc[-1, -1] = bc[-1, -1] + lowTemp / (dx ** 2)   # th

    bc = np.asarray(bc).reshape(-1)  # bc as vector

    # solve
    solvetime = timeit.default_timer()
    utemp = scipy.sparse.linalg.spsolve(D, -bc)
    solvetime = timeit.default_timer() - solvetime
    print("Solver took {} seconds to run".format(solvetime))

    # reshape for plot
    umatrix = np.reshape(utemp, (N, N))
    u[1:-1, 1:-1] = umatrix
    u[0, :] = lowTemp  # y = 0
    u[-1, :] = lowTemp  # y = N
    u[:, 0] = highTemp  # x = 0
    u[:, -1] = lowTemp  # x = N

    #plt.pcolor(np.flipud(u))
    plt.pcolor(u)
    plt.show()

