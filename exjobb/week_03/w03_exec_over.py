import matplotlib.pyplot as plt

from week_03 import w03_2d_over
from week_03 import w03_2d_over_nonsparse
from week_04.AdaptLsq import w04_2d_over
from week_two import w1_one_domain
import numpy as np
from IPython import embed

N = 8
itr = 40
rel = 1


'''
Initially, this exec was the first attempt at looking at what happens when
simply enforcing both N and D on each subdomain, iterating forward
using a Least Squares method. It worked surprisingly well.

Now the code has been used to look at the residual between the 
"perfect" solution and the iterated LSQ residual. 
'''

onedomain = w1_one_domain.one_domain(n = N)
onedomain.solve()
od = onedomain.get_solution()


twodomain = w03_2d_over_nonsparse.two_domain(n = N, iterations=itr, relaxation=rel)
twodomain.set_utrue(np.flipud(od[1:-1,N-1]))
twodomain.solve()
td = twodomain.get_solution()
res = twodomain.get_residuals()


adaptLQ = w04_2d_over.two_domain(n=N,iterations=itr,relaxation=rel)
adaptLQ.solve()
perf_resi = adaptLQ.get_residual()




A_over = adaptLQ.get_A_over()
b_over = adaptLQ.get_b_over()

u_over = twodomain.get_u_over() # solution from overdetermined iteration
bl = adaptLQ.get_bl()
plt.figure(figsize=(9,3))
plt.plot(bl,'.')
plt.savefig('test.pdf')
plt.show()

itr_bl = A_over*u_over-b_over
bl_diff = -bl+itr_bl

plt.plot(bl_diff,'.b')
plt.show()
residual = A_over*u_over-(b_over+bl)
plt.plot(residual,'g.')
#plt.plot(perf_resi,'.b')
plt.show()
#embed()


err = twodomain.get_error()
diff = td-od

plt.subplot(231)
plt.title("Iterated Domain solution, %d itr %g relax" % (itr,rel))
plt.pcolor(td)
plt.colorbar()
plt.subplot(232)
plt.title("Simple Domain solution")
plt.pcolor(od)
plt.colorbar()

plt.subplot(233)
plt.title("Difference itr - simple")
plt.pcolor(diff)
plt.colorbar()

plt.subplot(234)
plt.title("LeastSquare residual domain 1")
plt.plot(res[:,0])
plt.yscale('log')

plt.subplot(235)
plt.title("LeastSquare residual domain 2")
plt.plot(res[:,1])
plt.yscale('log')

plt.subplot(236)
plt.title("$||u_{true}-u_{\gamma}||_{\infty}$")
plt.plot(err)
plt.yscale('log')
plt.show()

#embed()