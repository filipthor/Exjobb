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
        self.u_gamma = np.zeros((n-2,1))
        self.error = np.zeros((iterations,1))
        self.u1_prev = []
        self.u2_prev = []
        self.diff_vector = np.zeros((iterations,1))
        self.residual = np.zeros((iterations,2))
        self.aj = 0#np.zeros((n-2,1)) #Spännande saker händer vid -10
        self.utrue = []

    def set_utrue(self,utrue):
        self.utrue = utrue

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

    def get_a(self,domaintype):
        Ni = self.n - 2
        Nj = self.n - 1
        sup = np.ones(Ni * Ni - 1)
        for i in range(1, len(sup) + 1):
            if i % Ni == 0: sup[i - 1] = 0
        A = sparse.csr_matrix(np.diag(np.ones(Ni * Ni - Ni), -Ni) \
             + np.diag(sup, -1) \
             + np.diag(-4. * np.ones(Ni * Ni), 0) \
             + np.diag(sup, 1) \
             + np.diag(np.ones(Ni * Ni - Ni), Ni))

        if domaintype == "Neumann":
            A12 = sparse.csr_matrix(self.create_column(Ni))
            A21 = sparse.csr_matrix(self.create_row(Ni))
            AG = sparse.csr_matrix(self.create_bottom(Ni))
            return sparse.csr_matrix(vstack((hstack((A, A12)), hstack((A21, AG)))))

        if domaintype == "Dirichlet":
            A23 = sparse.csr_matrix(np.zeros((Ni,Ni**2)))
            A32 = sparse.csr_matrix(np.zeros((Ni ** 2, Ni)))
            AG = sparse.csr_matrix(np.eye(Ni))
            return sparse.csr_matrix(vstack((hstack((AG,A23)), hstack((A32, A)))))

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
        u1[1:-1,-1] = self.u_gamma
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
            return np.asarray(b).reshape(-1)
        if domain == "Dirichlet":
            b2 = np.zeros((self.n - 2, self.n - 2))
            b2[0, :] += -1 * self.get_wall("Dirichlet", "North")
            b2[:, 0] += -1 * self.u_gamma
            b2[-1, :] += -1 * self.get_wall("Dirichlet", "South")
            b2[:, -1] += -1 * self.get_wall("Dirichlet", "East")
            b = np.zeros((self.n-1,self.n-2))
            b[1:,:] = b2
            return np.asarray(b).reshape(-1)

    def solve(self):
        A1 = self.get_a("Neumann")
        A2 = self.get_a("Dirichlet")

        solvetime = timeit.default_timer()
        for i in range(self.iterations):

            b1 = self.get_b("Neumann")
            u1_itr = np.reshape(scipy.sparse.linalg.spsolve(A1, b1),(self.n-1,self.n-2))
            u1_itr = self.relax(i,"Neumann",u1_itr)
            self.u_gamma = u1_itr[-1,:]

            b2 = self.get_b("Dirichlet")
            u2_itr = np.reshape(scipy.sparse.linalg.spsolve(A2, b2),(self.n-1,self.n-2))
            u2_itr = self.relax(i, "Dirichlet", u2_itr)
            self.aj = (u2_itr[1:,0]-self.u_gamma)



            self.error[i] = np.linalg.norm(self.utrue - self.u_gamma,ord=inf)

            #res1 = np.reshape(A1 * np.asarray(u1_itr).reshape(-1)-b1,(self.n-2,self.n-1))
            #res2 = np.reshape(A2 * np.asarray(u2_itr).reshape(-1)-b2, (self.n - 2, self.n - 2))
            #self.residual[i,0] = np.linalg.norm(res1[:,-1],ord=inf)
            #self.residual[i, 1] = np.linalg.norm(res2[:, 0], ord=inf)







        solvetime = timeit.default_timer() - solvetime
        print("It took {} seconds to iterate a solution.".format(solvetime))

        self.set_solution(u1_itr[:-1,:],u2_itr[1:,:])
