from week_two import w2_two_domain
from week_two import w1_one_domain
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed


N = 40
itr = 20
rel = 1
onedomain = w1_one_domain.one_domain(n = N)
onedomain.solve()
od = onedomain.get_solution()

twodomain = w2_two_domain.two_domain(n = N,iterations=itr,relaxation=rel)
twodomain.set_ugamma(np.flipud(od[1:-1,N-1]))
twodomain.solve()
td = twodomain.get_solution()
#twodomain.visualize()


#embed()

dv = twodomain.get_diff_vector()
err = twodomain.get_error()


diff = td-od

plt.subplot(221)
plt.title("Iterated Domain solution, %d itr %g relax" % (itr,rel))
#plt.figtext(0.1,0.95,"Solution after 10 iterations, relaxation factor 0.8")
plt.pcolor(td)
plt.colorbar()
plt.subplot(222)
plt.title("Simple Domain solution")
plt.pcolor(od)
plt.colorbar()

plt.subplot(223)
plt.title("Difference itr - simple")
plt.pcolor(diff)
plt.colorbar()
#plt.show()
#
# plt.subplot(224)
# plt.title("inf-Norm(boundaries u1e-u2w)")
# plt.plot(dv)
# plt.yscale('log')
plt.subplot(224)
plt.title("inf-Norm(boundaries u_gamma-u_true)")
plt.plot(err)
plt.yscale('log')
plt.show()


