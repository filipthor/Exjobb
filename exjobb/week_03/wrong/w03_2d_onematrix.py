import timeit
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.sparse.linalg
from scipy.sparse import hstack
from scipy.sparse import vstack


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
        self.u_gamma = np.zeros((n-2,1))
        self.error = np.zeros((iterations,1))
        self.u1_prev = []
        self.u2_prev = []
        self.diff_vector = np.zeros((iterations,1))
        self.residual = np.zeros((iterations,2))
        self.aj = 0#np.zeros((n-2,1))

    def set_ugamma(self,ugamma):
        self.u_gamma = ugamma

    def get_error(self):
        return self.error

    def create_column(self,Ni):
        M = np.zeros((Ni ** 2, Ni))
        for i in range(Ni):
            for j in range(Ni):
                if i == j:
                    M[Ni + i * Ni - 1, j] = 1
        return M

    def create_row(self,Ni):
        return np.transpose(self.create_column(Ni))

    def create_bottom(self,Ni):
        return np.diag(-3 * np.ones(Ni), 0) + np.diag(1 * np.ones(Ni - 1), -1) + np.diag(1 * np.ones(Ni - 1), 1)

    def get_a(self,roomtype):
        if roomtype == "Neumann":
            Ni = self.n - 2
            Nj = self.n - 1

            sup = np.ones(Ni * Ni - 1)

            for i in range(1, len(sup) + 1):
                if i % Ni == 0: sup[i - 1] = 0
            A1 = np.diag(np.ones(Ni * Ni - Ni), -Ni) \
                 + np.diag(sup, -1) \
                 + np.diag(-4. * np.ones(Ni * Ni), 0) \
                 + np.diag(sup, 1) \
                 + np.diag(np.ones(Ni * Ni - Ni), Ni)

            A1 = sparse.csr_matrix(A1)
            c = sparse.csr_matrix(self.create_column(Ni))
            r = sparse.csr_matrix(self.create_row(Ni))
            b = sparse.csr_matrix(self.create_bottom(Ni))

            A1 = sparse.csr_matrix(vstack((hstack((A1, c)), hstack((r, b)))))

            zeros1 = sparse.csr_matrix(np.zeros((Ni * Nj, Ni ** 2)))
            e1 = hstack((A1, zeros1))
            zeros2 = transpose(zeros1)
            eye = sparse.csr_matrix(np.eye(Ni ** 2))
            e2 = hstack((zeros2, eye))
            return sparse.csr_matrix(vstack((e1, e2)))

        if roomtype == "Dirichlet":
            Ni = self.n - 2
            Nj = self.n - 1

            sup = np.ones(Ni * Ni - 1)

            for i in range(1, len(sup) + 1):
                if i % Ni == 0: sup[i - 1] = 0
            A = np.diag(np.ones(Ni * Ni - Ni), -Ni) \
                 + np.diag(sup, -1) \
                 + np.diag(-4. * np.ones(Ni * Ni), 0) \
                 + np.diag(sup, 1) \
                 + np.diag(np.ones(Ni * Ni - Ni), Ni)

            I = sparse.csr_matrix((np.eye(Ni * Nj)))
            zeros1 = sparse.csr_matrix(np.zeros((Ni * Nj, Ni ** 2)))
            zeros2 = np.transpose(zeros1)
            e1 = hstack((I, zeros1))
            e2 = hstack((zeros2, A))
            return sparse.csr_matrix(vstack((e1, e2)))

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

    def visualize(self):
        plt.pcolor(self.solution)
        plt.colorbar()
        plt.show()

    def get_solution(self):
        return self.solution

    def set_solution(self,domain1,domain2):
        u1 = np.zeros((self.n,self.n))
        u2 = u1.copy()
        u = np.zeros((self.n,self.n*2-1))
        u1[0,:] = self.get_wall("Neumann","North")
        u1[-1,:] = self.get_wall("Neumann","South")
        u1[:,0] = self.get_wall("Neumann","West")
        u1[1:-1,-1] = self.u_gamma
        u1[1:-1,1:-1] = np.reshape(domain1[0:(self.n-2)**2],(self.n-2,self.n-2))


        u2[-1,:] = self.get_wall("Dirichlet","South")
        u2[:,-1] = self.get_wall("Dirichlet","East")
        u2[0,:] = self.get_wall("Dirichlet","North")
        u2[1:-1,1:-1] = np.reshape(domain2[(self.n-2)**2+(self.n-2):],(self.n-2,self.n-2))

        u[:, self.n - 1:2 * self.n - 1] = u2
        u[:, 0:self.n] = u1
        self.solution = flipud(u)

    def get_diff_vector(self):
        return self.diff_vector

    def get_residuals(self):
        return self.residual

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

    def get_b(self,domain):
        if domain == "Neumann":
            b1 = np.zeros((self.n - 2, self.n - 1))
            b1[0, :] += -1*self.get_wall("Neumann", "North")
            b1[:, 0] += -1*self.get_wall("Neumann", "West")
            b1[-1, :] += -1*self.get_wall("Neumann", "South")
            b1[:, -1] += -1*self.aj
            b = np.zeros((self.n-1, self.n-2))
            b[:-1,:] = b1[:,:-1]
            b[-1,:] = b1[:,-1]
            return np.asarray(np.vstack((b,np.zeros((self.n - 2, self.n - 2))))).reshape(-1)
        if domain == "Dirichlet":
            b2 = np.zeros((self.n - 2, self.n - 2))
            b2[0, :] += -1 * self.get_wall("Dirichlet", "North")
            b2[:, 0] += -1 * self.u_gamma
            b2[-1, :] += -1 * self.get_wall("Dirichlet", "South")
            b2[:, -1] += -1 * self.get_wall("Dirichlet", "East")
            return np.asarray(np.vstack((np.zeros((self.n - 1, self.n - 2)), b2))).reshape(-1)

    def solve(self):

        A1 = self.get_a("Neumann")
        A2 = self.get_a("Dirichlet")

        solvetime = timeit.default_timer()
        for i in range(self.iterations):

            b1 = self.get_b("Neumann")
            u1_itr = np.reshape(scipy.sparse.linalg.spsolve(A1, b1),(2*self.n-3,self.n-2))
            #self.u_gamma = u1_itr[(self.n-2)**2:(self.n-2)**2+(self.n-2)] #vector form
            self.u_gamma = u1_itr[self.n-2,:] # matrix form
            #print(self.u_gamma)
            #print("===")
            b2 = self.get_b("Dirichlet")
            u2_itr = np.reshape(scipy.sparse.linalg.spsolve(A2, b2),(2*self.n-3,self.n-2))

            #self.aj = (u2_itr[(self.n-2)**2+(self.n-2):(self.n-2)**2+2*(self.n-2)] - self.u_gamma) # vector
            self.aj = (u2_itr[self.n-1:,0]-self.u_gamma)







        solvetime = timeit.default_timer() - solvetime
        print("It took {} seconds to iterate a solution.".format(solvetime))

        self.set_solution(np.asarray(u1_itr).reshape(-1),np.asarray(u2_itr).reshape(-1))
