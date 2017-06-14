import matplotlib.pyplot as plt

from week_03.wrong import w03_2d_onematrix
from week_two import w1_one_domain

N = 30
itr = 300
rel = 1

twodomain = w03_2d_onematrix.two_domain(n = N, iterations=itr, relaxation=rel)
twodomain.solve()
td = twodomain.get_solution()

onedomain = w1_one_domain.one_domain(n = N)
onedomain.solve()
od = onedomain.get_solution()


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


plt.show()