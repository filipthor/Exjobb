import timeit
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.sparse.linalg


class Simple_Two_Domain:

    def __init__(self,n=40,iterations=10,relaxation=0.8,solution=[]):
        self.n = n
        self.iterations = iterations
        self.relaxation = relaxation
        self.solution = []
        self.residual_1 = []
        self.residual_2 = []

    def main(self):
        # initial values
        n = self.n
        iterations = self.iterations
        relaxation = self.relaxation


        ht = 30
        lt = 15
        N = n-2
        dx = 1 / (N+1)
        self.residual_1 = np.zeros((iterations,N**2))
        self.residual_2 = np.zeros((iterations, N ** 2))

        u1 = zeros((n,n))
        u2 = zeros((n,n))

        # boundary values
        # room 1
        u1n = ht
        u1w = ht
        u1s = lt
        u1e = np.ones(N)# np.zeros(N) # neumann (initial value = 0)

        # room 2
        u2n = ht
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
            A2[i * N, i * N] = -3

        A1 = sparse.csr_matrix(1 / (dx ** 2) * A1)
        A2 = sparse.csr_matrix(1 / (dx ** 2) * A2)

        solvetime = timeit.default_timer()


        # solve
        for i in range(0,iterations):
            #print("iteration:", (i+1))
        # room 1
            bc1 = np.zeros((N, N))
            bc1[0, :] += u1n / dx ** 2
            bc1[-1, :] += u1s / dx ** 2
            bc1[:, 0] += u1w / dx ** 2
            bc1[:, -1] += u1e / dx ** 2
            bc1 = np.asarray(bc1).reshape(-1)

            u_inner_1 = np.reshape(scipy.sparse.linalg.spsolve(A1, -bc1), (N, N))

            u1[1:-1, 1:-1] = u_inner_1
            u1[0, :] = u1n
            u1[-1, :] = u1s
            u1[:, 0] = u1w
            u1[1:-1, -1] = u1e

            if i > 0:
                u1_prev = relaxation * u1 + (1 - relaxation) * u1_prev
            else:
                u1_prev = u1

            x = A1*np.asarray(u1[1:-1,1:-1]).reshape(-1)*(dx**2)
            self.residual_1[i] = x

            u2w = (u1_prev[1:-1, -2] - u1_prev[1:-1, -1])

        # room 2
            bc2 = np.zeros((N, N))
            bc2[0, :] += u2n / dx ** 2
            bc2[-1, :] += u2s / dx ** 2
            bc2[:, 0] += u2w / dx  # med ^2 fel på 0.04, utan, 0.8
            bc2[:, -1] += u2e / dx ** 2
            bc2 = np.asarray(bc2).reshape(-1)

            u_inner_2 = np.reshape(scipy.sparse.linalg.spsolve(A2,-bc2),(N,N))

            u2[1:-1,1:-1] = u_inner_2
            u2[0, :] = u2n
            u2[-1, :] = u2s
            u2[1:-1, 0] = u2[1:-1, 1] + dx * u2w * u2[1:-1, 1]
            u2[1:-1, -1] = u2e

            if i > 0:
                u2_prev = relaxation * u2 + (1-relaxation) * u2_prev
            else:
                u2_prev = u2

            x = A2 * np.asarray(u2[1:-1,1:-1]).reshape(-1)*(dx**2)
            self.residual_2[i] = x

            u1e = u2_prev[1:-1,0]



        solvetime = timeit.default_timer() - solvetime
        print("It took {} seconds to iterate a solution.".format(solvetime))

        u_sol = zeros((n,2*n))
        u_sol[:,0:n] = u1_prev
        u_sol[:,n:2*n] = u2_prev

        self.set_solution(flipud(u_sol))

    def set_solution(self, solution):
        self.solution = solution

    def get_solution(self):
        return self.solution

    def visualize(self):
        plt.pcolor(self.solution)
        plt.colorbar()
        plt.show()

    def get_residual(self, room):
        if room == 1: return self.residual_1
        if room == 2: return self.residual_2

#if __name__ == '__main__': Simple_Two_Domain.main()
