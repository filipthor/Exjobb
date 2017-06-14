from week_two import w1_one_domain
from week_05 import w05_2d_third_constraint
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed

'''
Written 2017-05-16

This script runs the overconstraint system for both neumann
and dirichlet conditions on both u1 and u2

'''

n = 20
itr = 400
relaxation = 1


# iterated domain initialization and solving
itr_domain = w05_2d_third_constraint.two_domain(n,itr,relaxation)
itr_domain.solve()
u_itr = itr_domain.get_solution()

# single domain comparison
one_domain = w1_one_domain.one_domain(n)
one_domain.solve()
u_simple = one_domain.get_solution()

# calculating difference iterated and single domain
diff = u_itr-u_simple

# get residuals from iterated domain
residuals = itr_domain.get_residual()


# get difference ||u_itr(i)-u_simple||fro
difference = itr_domain.get_difference()







# plotting
plt.subplot(231)
plt.title("$u_{itr}^{(%d)}$, n = %d, itr = %d, relax =  %g" % (itr,n,itr,relaxation))
plt.pcolor(u_itr)
plt.colorbar()

plt.subplot(232)
plt.title("$u_{simple}$")
plt.pcolor(u_simple)
plt.colorbar()

plt.subplot(233)
plt.title("$u_{itr}-u_{simple}$")
plt.pcolor(diff)
plt.colorbar()

plt.subplot(234)
plt.title("Residual from lstsq $\Omega_1$")
plt.semilogy(residuals[:,0])

plt.subplot(235)
plt.title("Residual from lstsq $\Omega_2$")
plt.semilogy(residuals[:,1])

plt.subplot(236)
plt.title("$||u_{itr}(i)-u_{simple}||_{F}$")
plt.semilogy(difference)

plt.show()


