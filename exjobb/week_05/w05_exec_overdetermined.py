from week_two import w1_one_domain
from week_05 import w05_2d
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed

'''
Written 2017-05-16

This script tests running an non-overdetermined system 
imposing both neumann and dirichlet conditions on boundary
for both subdomains u1 and u2

'''

n = 30
itr = 500
relaxation = 1


# iterated domain initialization and solving
itr_domain = w05_2d.two_domain(n,itr,relaxation)
itr_domain.solve()
u_itr = itr_domain.get_solution()

# single domain comparison
one_domain = w1_one_domain.one_domain(n)
one_domain.solve()
u_simple = one_domain.get_solution()

# calculating difference iterated and single domain
diff = u_simple-u_itr

# get difference ||u_itr(i)-u_simple||fro
difference = itr_domain.get_difference()







# plotting
plt.subplot(221)
plt.title("Iterated Domain solution, n = %d, itr = %d, relax =  %g" % (n,itr,relaxation))
plt.pcolor(u_itr)
plt.colorbar()

plt.subplot(222)
plt.title("Simple Domain solution")
plt.pcolor(u_simple)
plt.colorbar()

plt.subplot(223)
plt.title("Difference itr - simple")
plt.pcolor(diff)
plt.colorbar()

plt.subplot(224)
plt.title("$||u_{itr}(i)-u_{simple}||_{F}$")
plt.semilogy(difference)

plt.show()


