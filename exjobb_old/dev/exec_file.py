from dev import w2_two_domain
from dev import w1_one_domain
import matplotlib.pyplot as plt


twodomain = w2_two_domain.two_domain(n = 40,iterations=40)
twodomain.solve()
td = twodomain.get_solution()
#twodomain.visualize()

onedomain = w1_one_domain.one_domain(n = 40)
onedomain.solve()
od = onedomain.get_solution()

dv = twodomain.get_diff_vector()

diff = td - od

plt.subplot(221)
plt.title("Iterated Domain solution")
#plt.figtext(0.1,0.95,"Solution after 10 iterations, relaxation factor 0.8")
plt.pcolor(td)
plt.colorbar()
plt.subplot(222)
plt.title("Simple Domain solution")
plt.pcolor(od)
plt.colorbar()

plt.subplot(223)
plt.title("Difference")
plt.pcolor(diff)
plt.colorbar()
#plt.show()

plt.subplot(224)
plt.plot(dv)
plt.show()


