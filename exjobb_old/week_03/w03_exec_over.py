import matplotlib.pyplot as plt

#from week_03 import w03_2d_over
from week_03 import w03_2d_over_nonsparse
from week_two import w1_one_domain
import numpy as np
from IPython import embed

N = 30
itr = 80
rel = 1

onedomain = w1_one_domain.one_domain(n = N)
onedomain.solve()
od = onedomain.get_solution()


twodomain = w03_2d_over_nonsparse.two_domain(n = N, iterations=itr, relaxation=rel)
twodomain.set_utrue(np.flipud(od[1:-1,N-1]))
twodomain.solve()
td = twodomain.get_solution()
res = twodomain.get_residuals()

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

embed()