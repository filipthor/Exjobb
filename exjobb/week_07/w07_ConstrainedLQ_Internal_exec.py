from week_two import w1_one_domain
from week_07 import w07_ConstrainedLQ_Internal_2d
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed

'''
Written 2017-05-16
Updated 2017-06-13

This script runs the overdetermined system solving using constrained least squares
constraining internal points

'''

n = 40
itr = 300
relaxation = 1


# iterated domain initialization and solving
itr_domain = w07_ConstrainedLQ_Internal_2d.two_domain(n,itr,relaxation)
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
plt.figure(figsize=(16,8))

plt.subplot(221)
plt.title("$u_{itr}^{(%d)}$, n = %d, itr = %d, relax =  %g" % (itr,n,itr,relaxation))
plt.pcolor(u_itr)#,cmap="YlOrRd")
plt.colorbar()

plt.subplot(222)
plt.title("$u_{exact\quad discrete}$")
plt.pcolor(u_simple)

plt.subplot(223)
plt.title("$u_{itr}-u_{exact\quad discrete}$")
plt.pcolor(diff)
plt.colorbar()
'''
plt.subplot(234)
plt.title("Residual from lstsq $\Omega_1$")
plt.semilogy(residuals[:,0])

plt.subplot(235)
plt.title("Residual from lstsq $\Omega_2$")
plt.semilogy(residuals[:,1])
'''
plt.subplot(224)
plt.title("$||u_{itr}(i)-u_{simple}||_{F}$")
plt.semilogy(difference)


plt.show()


