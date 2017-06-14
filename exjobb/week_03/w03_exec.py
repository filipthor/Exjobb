import matplotlib.pyplot as plt

from week_03 import w03_2d_block
from week_two import w1_one_domain
import numpy as np

N = 20
itr = 200
rel = 1

onedomain = w1_one_domain.one_domain(n = N)
onedomain.solve()
od = onedomain.get_solution()


twodomain = w03_2d_block.two_domain(n = N, iterations=itr, relaxation=rel)
twodomain.set_utrue(np.flipud(od[1:-1,N-1]))
twodomain.solve()
td = twodomain.get_solution()
difference = twodomain.get_difference()

err = twodomain.get_error()
diff = td-od


# plotting
plt.figure(figsize=(16,8))

plt.subplot(221)
#plt.title("Iterated Domain solution, %d itr %g relax" % (itr,rel))
plt.title("$u_{itr}^{(%d)}$, n = %d, itr = %d, relax =  %g" % (itr,N,itr,rel))
plt.pcolor(td)#,cmap="YlOrBr")
plt.colorbar()
plt.subplot(222)
#plt.title("Simple Domain solution")
plt.title("$u_{exact\quad discrete}$")
plt.pcolor(od)
plt.colorbar()

plt.subplot(223)
#plt.title("Difference itr - simple")
plt.title("$u_{itr}-u_{exact\quad discrete}$")
plt.pcolor(diff)
plt.colorbar()

plt.subplot(224)
#plt.title("$||u_{true}-u_{\gamma}||_{\infty}$")
plt.title("$||u_{itr}(i)-u_{simple}||_{F}$")
plt.plot(difference)
plt.yscale('log')
plt.show()