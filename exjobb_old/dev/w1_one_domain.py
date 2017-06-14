
import timeit
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.sparse.linalg


class one_domain:

    def __init__(self, n=40, solution=[]):
        self.n = n
        self.solution = []

    def solve(self):
        # initial conditions
        ht = 30
        lt = 15

        n = self.n
        N = n - 2
        Ny = 2*n - 3
        Nx= N
        dx = 1 / (N + 1)

        # wall temperatures
        un = ht
        uw = ht
        us = lt
        ue = lt

        u = np.zeros((n,2*n-1))

        sup = np.ones(Nx*Ny - 1)

        # creating A matrix
        for i in range(1, len(sup)+1):
            rem = i % Ny
            if (rem == 0): sup[i - 1] = 0
        A = np.diag(ones(Nx*Ny - Ny), -Ny) \
                + np.diag(sup, -1) \
                + np.diag(-4.*ones(Nx*Ny), 0) \
                + np.diag(sup, 1) \
                + np.diag(ones(Nx*Ny - Ny),Ny)

        D = sparse.csr_matrix((1 /dx ** 2) * A)

        # boundary conditions
        bc = np.zeros((Nx, Ny))
        bc[:, -1] += ue
        bc[0, :] += us
        bc[:, 0] += uw
        bc[-1, :] += un
        bc = bc / (dx ** 2)

        bc = np.asarray(bc).reshape(-1)  # bc as vector

        # solve

        u_sol_inner = np.reshape(scipy.sparse.linalg.spsolve(D, -bc), (Nx, Ny))

        # reshape for plot
        u[1:-1, 1:-1] = u_sol_inner

        u[0, :] = us
        u[:, 0] = uw
        u[:, -1] = ue
        u[-1, :] = un
        self.set_solution(u)

    def set_solution(self,solution):
        self.solution = solution

    def get_solution(self):
        return self.solution

    def visualize(self):
        plt.pcolor(self.solution)
        plt.colorbar()
        plt.show()



#if __name__ == "__main__": Simple_One_Domain.main()