import timeit
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.sparse.linalg
from scipy.sparse import hstack
from scipy.sparse import vstack


class two_domain:

    def __init__(self,n=40, iterations=10, relaxation=1):


        self.n = n
        self.iterations = iterations
        self.relaxation = relaxation
        self.solution = []
        self.ht = 30
        self.lt = 15
        self.dx = 1 / (self.n+1)
        self.neumann_east = 0
        self.dirichlet_west = 0
        self.u_gamma_d1 = np.zeros((n-2))
        self.u_gamma_d2 = np.zeros((n-2))
        self.error = np.zeros((iterations,1))
        self.u1_prev = []
        self.u2_prev = []
        self.diff_vector = np.zeros((iterations,1))
        self.residual = np.zeros((iterations,2))
        self.aj_d1 = 0#np.zeros((n-2,1)) #Spännande saker händer vid -10
        self.aj_d2 = 0
        self.utrue = []

    def set_utrue(self,utrue):
        self.utrue = utrue

    def get_error(self):
        return self.error

    def create_column_east(self,Ni):
        M = np.zeros((Ni ** 2, Ni))
        for i in range(Ni):
            for j in range(Ni):
                if i == j:
                    M[Ni + i * Ni - 1, j] = 1
        return M

    def create_column_west(self,Ni):
        M = np.zeros((Ni ** 2, Ni))
        for i in range(Ni):
            for j in range(Ni):
                if i == j:
                    M[i * Ni, j] = 1
        return M

    def create_row(self,Ni):
        return np.transpose(self.create_column_east(Ni))

    def create_AGG_1(self,Ni):
        return np.diag(-4 * np.ones(Ni), 0) + np.diag(1 * np.ones(Ni - 1), -1) + np.diag(1 * np.ones(Ni - 1), 1)

    def create_AGG_2(self,Ni):
        return np.diag(-3 * np.ones(Ni), 0) + np.diag(1 * np.ones(Ni - 1), -1) + np.diag(1 * np.ones(Ni - 1), 1)


    def get_a(self):
        Ni = self.n - 2
        Nj = self.n - 1
        sup = np.ones(Ni * Ni - 1)
        for i in range(1, len(sup) + 1):
            if i % Ni == 0: sup[i - 1] = 0
        self.A11 = sparse.csr_matrix(np.diag(np.ones(Ni * Ni - Ni), -Ni) \
             + np.diag(sup, -1) \
             + np.diag(-4. * np.ones(Ni * Ni), 0) \
             + np.diag(sup, 1) \
             + np.diag(np.ones(Ni * Ni - Ni), Ni))

        #if d1.shape == d2.shape:
        self.A22 = self.A11.copy()

        #if domaintype == "Neumann":
        self.A1G = sparse.csr_matrix(self.create_column_east(Ni))
        self.AG1 = sparse.csr_matrix(self.create_row(Ni))
        self.AGG = sparse.csr_matrix(self.create_AGG_1(Ni))
        zeros = sparse.csr_matrix(np.zeros((Ni ** 2, Ni**2)))
        self.A2G = sparse.csr_matrix(self.create_column_west(Ni))
        self.AG2 = sparse.csr_matrix(np.transpose(self.A2G))

        r1 = sparse.csr_matrix(hstack((self.A11,self.A1G,zeros)))
        r2 = sparse.csr_matrix(hstack((self.AG1,self.AGG,self.AG2)))
        r3 = sparse.csr_matrix(hstack((zeros,self.A2G,self.A22)))

        return sparse.csr_matrix(vstack((r1,r2,r3)))


    def get_wall(self,domaintype,wall):
        if domaintype == "Neumann":
            if wall == "North": return self.ht
            if wall == "West": return self.ht
            if wall == "South": return self.lt
            if wall == "East": return self.neumann_east
        if domaintype == "Dirichlet":
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
        u1[1:-1,-1] = self.u_gamma_d1
        u1[1:-1,1:-1] = domain1


        u2[-1,:] = self.get_wall("Dirichlet","South")
        u2[:,-1] = self.get_wall("Dirichlet","East")
        u2[0,:] = self.get_wall("Dirichlet","North")
        u2[1:-1,1:-1] = domain2

        u[:, self.n - 1:2 * self.n - 1] = u2
        u[:, 0:self.n] = u1
        self.solution = flipud(u)

    def get_diff_vector(self):
        return self.diff_vector

    def get_residuals(self):
        return self.residual

    def relax(self,itr,domain,current):
        if domain == "Neumann":
            if itr == 0:
                self.u1_prev = current
            else:
                self.u1_prev = self.relaxation * current + (1 - self.relaxation) * self.u1_prev
            return self.u1_prev
        if domain == "Dirichlet":
            if itr == 0:
                self.u2_prev = current
            else:
                self.u2_prev = self.relaxation * current + (1 - self.relaxation) * self.u2_prev
            return self.u2_prev

    def get_b(self,):
        #if domain == "Neumann":
        b1 = np.zeros((self.n - 2, self.n - 2))
        b1[0, :] += -1 * self.get_wall("Neumann", "North")
        b1[:, 0] += -1 * self.get_wall("Neumann", "West")
        b1[-1, :] += -1 * self.get_wall("Neumann", "South")
        self.b1 = np.asarray(b1).reshape(-1)
        bG = np.zeros((1,self.n-2))
        bG[0,0] = -1*self.get_wall("Neumann", "North")
        bG[0,-1] = -1 * self.get_wall("Neumann", "South")
        self.bG = np.asarray(bG).reshape(-1)


        b2 = np.zeros((self.n - 2, self.n - 2))
        b2[0, :] += -1 * self.get_wall("Dirichlet", "North")
        b2[-1, :] += -1 * self.get_wall("Dirichlet", "South")
        b2[:, -1] += -1 * self.get_wall("Dirichlet", "East")
        self.b2 = np.asarray(b2).reshape(-1)

        b = np.vstack((b1,bG,b2))
        return np.asarray(b).reshape(-1)


    def solve(self):
        A = self.get_a()
        print(A.toarray())
        print("A:",A.shape)

        b = self.get_b()
        print(b)
        print("b:",b.shape)
        #self.set_solution(u1_itr[:-1,:],u2_itr[1:,:])
        u = scipy.sparse.linalg.spsolve(A,b)
        u = np.reshape(u,(self.n-2+self.n-1,self.n-2))
        u1 = u[:self.n-2,:]
        self.u_gamma_d1 = u[self.n-2:self.n-1,:]
        u2 = u[self.n-1:,:]
        self.set_solution(u1,u2)
