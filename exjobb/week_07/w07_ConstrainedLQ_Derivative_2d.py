import timeit
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.sparse.linalg
from scipy.sparse import hstack
from scipy.sparse import vstack
from IPython import embed
from week_two import w1_one_domain

'''
Written 2017-05-16
Updated 2017-06-13

This class holds the overdetermined system solving using a constrained least squares method
putting a constraint on the derivatives

'''


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
        self.difference = np.zeros((iterations,1))
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
        self.AGG1 = sparse.csr_matrix(self.create_AGG_1(Ni))
        self.AGG2 = sparse.csr_matrix(self.create_AGG_2(Ni))
        zeros = sparse.csr_matrix(np.zeros((Ni ** 2, Ni**2)))
        self.A2G = sparse.csr_matrix(self.create_column_west(Ni))
        self.AG2 = sparse.csr_matrix(np.transpose(self.A2G))

        r1 = sparse.csr_matrix(hstack((self.A11,self.A1G,zeros)))
        r2 = sparse.csr_matrix(hstack((self.AG1,self.AGG1,self.AG2)))
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

    def set_solution(self,domain1,domainG,domain2):
        u1 = np.zeros((self.n,self.n))
        u2 = u1.copy()
        domain1 = np.reshape(domain1, (self.n - 2, self.n - 2))
        domain2 = np.reshape(domain2, (self.n - 2, self.n - 2))

        u1[0,:] = self.get_wall("Neumann","North")
        u1[-1,:] = self.get_wall("Neumann","South")
        u1[:,0] = self.get_wall("Neumann","West")
        u1[1:-1,-1] = domainG
        u1[1:-1,1:-1] = domain1


        u2[-1,:] = self.get_wall("Dirichlet","South")
        u2[:,-1] = self.get_wall("Dirichlet","East")
        u2[0,:] = self.get_wall("Dirichlet","North")
        u2[1:-1,1:-1] = domain2

        u = np.zeros((self.n, self.n * 2 - 1))
        u[:, self.n - 1:2 * self.n - 1] = u2
        u[:, 0:self.n] = u1
        self.solution = flipud(u)

    def get_difference(self):
        return self.difference

    def get_residual(self):
        return self.residual

    def get_A_over(self):
        zblock = np.zeros(((self.n-2) ** 2, (self.n-2)**2))
        zrow = np.zeros((self.n-2, (self.n-2)**2))
        zcolumn = np.transpose(zrow)
        r1 = scipy.sparse.csr_matrix(hstack((self.A11,zcolumn,zblock)))
        r2 = scipy.sparse.csr_matrix(hstack((self.AG1,self.AGG2,zrow)))
        r3 = scipy.sparse.csr_matrix(hstack((zrow,self.AGG2,self.AG2)))
        r4 = scipy.sparse.csr_matrix(hstack((zblock,zcolumn,self.A22)))
        return scipy.sparse.csr_matrix(vstack((r1,r2,r3,r4)))

    def get_A1(self):
        zrow = np.zeros((self.n-2, (self.n-2)**2))
        I = np.eye((self.n-2))
        zcolumn = np.transpose(zrow)
        r1 = scipy.sparse.csr_matrix(hstack((self.A11,zcolumn)))
        r2 = scipy.sparse.csr_matrix(np.hstack((zrow,I)))
        r3 = scipy.sparse.csr_matrix(hstack((self.AG1,self.AGG2)))
        return scipy.sparse.csr_matrix(vstack((r1,r2,r3)))

    def get_A2(self):
        zrow = np.zeros((self.n-2, (self.n-2)**2))
        zcolumn = np.transpose(zrow)
        r3 = scipy.sparse.csr_matrix(hstack((self.AGG2,self.AG2)))
        r4 = scipy.sparse.csr_matrix(np.hstack((np.eye((self.n-2)),zrow)))
        r5 = scipy.sparse.csr_matrix(hstack((zcolumn,self.A22)))
        return scipy.sparse.csr_matrix(vstack((r3,r4,r5)))


    def get_A_over_2(self):
        zblock = np.zeros(((self.n-2) ** 2, (self.n-2)**2))
        zrow = np.zeros((self.n-2, (self.n-2)**2))
        zcolumn = np.transpose(zrow)
        r1 = scipy.sparse.csr_matrix(hstack((self.A11,self.A1G,zblock)))
        r2 = scipy.sparse.csr_matrix(hstack((self.AG1,self.AGG2,self.AG2)))
        r3 = scipy.sparse.csr_matrix(hstack((self.AG1,self.AGG2,self.AG2)))
        r4 = scipy.sparse.csr_matrix(hstack((zblock,self.A2G,self.A22)))
        return scipy.sparse.csr_matrix(vstack((r1,r2,r3,r4)))


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
        b_over = np.vstack((b1,bG,bG,b2))
        b11 = np.vstack((b1,np.zeros((self.n-2)),bG))
        b22 = np.vstack((bG,np.zeros((self.n-2)),b2))
        self.b11 = np.asarray(b11).reshape(-1)
        self.b22 = np.asarray(b22).reshape(-1)
        self.b_over = np.asarray(b_over).reshape(-1)
        return np.asarray(b).reshape(-1)

    def get_b_over(self):
        return self.b_over

    def get_bl1(self,uG,u2): # Here the extra condition is imposed
        r1 = -1*np.dot(self.A1G.toarray(),np.transpose(uG))
        self.d1 = -1*(np.dot(self.AG2.toarray(),np.transpose(u2))-uG)
        return np.concatenate((r1,uG,self.d1))

    def get_bl2(self,u1,uG): # Here the extra condition is imposed
        self.d2 = (uG-np.dot(self.AG1.toarray(),np.transpose(u1)))
        r4 = -1*np.dot(self.A2G.toarray(),np.transpose(uG))
        return np.concatenate((self.d2,uG,r4))

    def vis(self,solution):
        u = np.reshape(solution,(self.n-2+self.n-1,self.n-2))
        u1 = u[:self.n-2,:]
        uG = u[self.n-2:self.n-1,:]
        u2 = u[self.n-1:,:]
        self.set_solution(u1,uG,u2)
        self.visualize()


    def initiate_constrained_matrices(self):
        self.A1 = self.get_A1()  # D and N block matrix 2 by 2 for Omega 1
        self.A2 = self.get_A2()  # D and N block matrix 2 by 2 for Omega 2

        # SETTING CONSTRAINT OVER DERIVATIVE CONDITION!
        C1 = scipy.sparse.csr_matrix(hstack((self.AG1,self.AGG2)))
        C2 = scipy.sparse.csr_matrix(hstack((self.AGG2, self.AG2)))
        # ================

        r11 = scipy.sparse.csr_matrix(hstack((np.transpose(self.A1)*self.A1,np.transpose(C1))))
        r12 = scipy.sparse.csr_matrix(hstack((C1,np.zeros((self.n-2,self.n-2)))))

        r21 = scipy.sparse.csr_matrix(hstack((np.transpose(self.A2)*self.A2,np.transpose(C2))))
        r22 = scipy.sparse.csr_matrix(hstack((C2,np.zeros((self.n-2,self.n-2)))))

        self.A1_constr = scipy.sparse.csr_matrix(vstack((r11,r12)))
        self.A2_constr = scipy.sparse.csr_matrix(vstack((r21, r22)))


    def get_constrained_b1(self,bl1):
        r1 = np.transpose(self.A1)*(self.b11+bl1)
        return np.concatenate((r1, self.d1+self.bG))

    def get_constrained_b1_0(self):
        r1 = np.transpose(self.A1) * (self.b11)
        return np.concatenate((r1, self.bG))

    def get_constrained_b2(self, bl2):
        r1 = np.transpose(self.A2) * (self.b22 + bl2)
        return np.concatenate((r1, self.d2+self.bG))


    def solve(self):
        one_domain = w1_one_domain.one_domain(self.n)
        one_domain.solve()
        u_simple = one_domain.get_solution()

        A = self.get_a() # A Global
        b = self.get_b() # b global

        self.initiate_constrained_matrices()


        solvetime = timeit.default_timer()

        for i in range(self.iterations):
            if i == 0:
                u1_itr = scipy.sparse.linalg.spsolve(self.A1_constr, self.get_constrained_b1_0())
                #u1_residual = u1_itr[1]
                #u1_itr = u1_itr[0]
            else:
                u1_itr = scipy.sparse.linalg.spsolve(self.A1_constr, self.get_constrained_b1(bl1))
                #u1_residual = u1_itr[1]
                #u1_itr = u1_itr[0]
            u1_itr = u1_itr[:(self.n-2)**2+(self.n-2)]

            # Temporär residual
            if i > 0:
                temp_res_top = self.A1 * u1_itr - (self.b11 + bl1)


            u1 = u1_itr[:(self.n-2)**2]
            uG1 = u1_itr[(self.n-2)**2:(self.n-2)**2+(self.n-2)]

            # Relaxation
            if i > 0:
                uG1 = self.relaxation*uG1+(1-self.relaxation)*uG1_old
            uG1_old = uG1

            bl2 = self.get_bl2(u1,uG1)

            u2_itr = scipy.sparse.linalg.spsolve(self.A2_constr, self.get_constrained_b2(bl2))
            u2_itr = u2_itr[:(self.n - 2) ** 2 + (self.n - 2)]

            #Temporär residual
            temp_res_bot = self.A2 * u2_itr - (self.b22 + bl2)

            uG2 = u2_itr[:(self.n-2)]
            u2 = u2_itr[(self.n-2):(self.n-2)**2+(self.n-2)]

            # Relaxation
            if i > 0:
                uG2 = self.relaxation * uG2 + (1 - self.relaxation) * uG2_old
            uG2_old = uG2

            bl1 = self.get_bl1(uG2,u2)

            #self.residual[i, 0] = u1_residual
            #self.residual[i, 1] = u2_residual


            self.set_solution(u1,uG2,u2)
            self.difference[i] = np.linalg.norm(self.solution-u_simple,ord='fro')



            '''
            if i == 20 or i == 50 or i == 100:
                plt.subplot(311)
                plt.plot(temp_res_top,'.')
                plt.title("Iteration %d" % i)

                plt.subplot(312)
                plt.plot(temp_res_bot, '.')
                plt.title("Residual $\Omega_2$")

                plt.subplot(325)
                plt.title("Current iterated solution")
                plt.pcolor(self.solution)

                plt.subplot(326)
                plt.title("Difference current iterated exact discrete")
                plt.pcolor(self.solution-u_simple)
                plt.colorbar()
                plt.show()
            '''
            '''
            if i == 20:
                plt.figure(figsize=(10,4))
                plt.subplot(211)
                plt.plot(temp_res_top, '.')
                plt.title("Residual over $\Omega_1$")

                plt.subplot(212)
                plt.plot(temp_res_bot, '.')
                plt.title("Residual over $\Omega_2$")
                savefig('residual.pdf')
                plt.show()
            '''






        solvetime = timeit.default_timer() - solvetime
        print("iteration took {} seconds.".format(solvetime))

        self.set_solution(u1, uG2, u2)
