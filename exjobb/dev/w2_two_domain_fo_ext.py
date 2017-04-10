import timeit
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.sparse.linalg


class two_domain:

    def __init__(self,n=40, iterations=10, relaxation=0.8):
        self.n = n
        self.iterations = iterations
        self.relaxation = relaxation
        self.solution = []
        self.ht = 30
        self.lt = 15
        self.dx = 1 / (self.n+1)
        self.neumann_east = 0
        self.dirichlet_west = 0
        self.u1_prev = []
        self.u2_prev = []
        self.diff_vector = np.zeros((iterations,1))

    def get_a(self,roomtype):
        if roomtype == "Dirichlet":
            N = self.n-2
            sup = ones(N ** 2 - 1)
            for i in range(1, len(sup) + 1):
                if i % N == 0: sup[i - 1] = 0
            A = np.diag(ones(N ** 2 - N), -N) \
                 + np.diag(sup, -1) \
                 + np.diag(-4. * ones(N ** 2), 0) \
                 + np.diag(sup, 1) \
                 + np.diag(ones(N ** 2 - N), N)
            return sparse.csr_matrix(1 / (self.dx ** 2) * A)
        if roomtype == "Neumann":
            Ni = self.n - 1
            Nj = self.n - 2
            sup = ones(Ni * Nj - 1)
            for i in range(1, len(sup) + 1):
                if i % Ni == 0: sup[i - 1] = 0
            A = np.diag(ones(Ni * Nj - Ni), -Ni) \
                 + np.diag(sup, -1) \
                 + np.diag(-4. * ones(Ni * Nj), 0) \
                 + np.diag(sup, 1) \
                 + np.diag(ones(Ni * Nj - Ni), Ni)
            for i in range(0, Nj):
                A[i * Ni+Ni-1 , i * Ni +Ni-1] = -3
            return sparse.csr_matrix(1 / (self.dx ** 2) * A)


    def get_wall(self,roomtype,wall):
        if roomtype == "Neumann":
            if wall == "North": return self.ht
            if wall == "West": return self.ht
            if wall == "South": return self.lt
            if wall == "East": return self.neumann_east
        if roomtype == "Dirichlet":
            if wall == "North": return self.ht
            if wall == "West": return self.dirichlet_west
            if wall == "South": return self.lt
            if wall == "East": return self.lt

    def update_boundary(self,roomtype,boundary):
        if roomtype == "Neumann":
            self.neumann_east = boundary
        if roomtype == "Dirichlet":
            self.dirichlet_west = boundary

    def visualize(self):
        plt.pcolor(self.solution)
        plt.colorbar()
        plt.show()

    def get_solution(self):
        return self.solution

    def set_solution(self,room1,room2):
        u1 = np.zeros((self.n,self.n))
        u2 = u1.copy()
        u = np.zeros((self.n,self.n*2-1))
        u1[0,:] = self.get_wall("Neumann","North")
        u1[-1,:] = self.get_wall("Neumann","South")
        u1[:,0] = self.get_wall("Neumann","West")
        u1[1:-1,1:] = room1

        u2[-1,:] = self.get_wall("Dirichlet","South")
        u2[:,-1] = self.get_wall("Dirichlet","East")
        u2[0,:] = self.get_wall("Dirichlet","North")
        u2[1:-1,1:-1] = room2

        u[:, self.n - 1:2 * self.n - 1] = u2
        u[:, 0:self.n] = u1
        self.solution = flipud(u)

    def get_diff_vector(self):
        return self.diff_vector

    def relax(self,i,room,itr):
        if room == 1:
            if i == 0:
                self.u1_prev = itr
            else:
                self.u1_prev = self.relaxation * itr + (1 - self.relaxation) * self.u1_prev
            return self.u1_prev
        if room == 2:
            if i == 0:
                self.u2_prev = itr
            else:
                self.u2_prev = self.relaxation * itr + (1 - self.relaxation) * self.u2_prev
            return self.u2_prev

    def solve(self):
        A1 = self.get_a("Neumann")
        A2 = self.get_a("Dirichlet")

        solvetime = timeit.default_timer()
        for i in range(self.iterations):

            b1 = np.zeros((self.n-2,self.n-1))
            b1[0,:] += -1*self.get_wall("Neumann","North")
            b1[:, 0] += -1*self.get_wall("Neumann", "West")
            b1[-1,:] += -1*self.get_wall("Neumann","South")
            b1[:,-1] += -1*self.get_wall("Neumann","East")*(self.dx)
            b1 = np.asarray(b1).reshape(-1)/(self.dx ** 2)

            u1_itr = self.relax(i,1,np.reshape(scipy.sparse.linalg.spsolve(A1, b1), (self.n-2, self.n-1)))

            self.update_boundary("Dirichlet", u1_itr[:,-1])


            #self.solution = flipud(u1_in)

            b2 = np.zeros((self.n-2,self.n-2))
            b2[0,:] += -1*self.get_wall("Dirichlet","North")
            b2[:, 0] += -1*self.get_wall("Dirichlet", "West")
            b2[-1,:] += -1*self.get_wall("Dirichlet","South")
            b2[:,-1] += -1*self.get_wall("Dirichlet","East")
            b2 = np.asarray(b2).reshape(-1)/ (self.dx ** 2)

            u2_itr = self.relax(i,2,np.reshape(scipy.sparse.linalg.spsolve(A2,b2), (self.n-2, self.n-2)))

            aj = (u2_itr[:,1] - u2_itr[:,0])/(self.dx)
            self.update_boundary("Neumann",aj)
            self.diff_vector[i] = np.linalg.norm(u1_itr[:,-1]-u2_itr[:,0],ord=inf)




        solvetime = timeit.default_timer() - solvetime
        print("It took {} seconds to iterate a solution.".format(solvetime))

        self.set_solution(u1_itr,u2_itr)
