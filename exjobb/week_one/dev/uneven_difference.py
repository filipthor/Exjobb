from week_one.dev import simple_oneroom_resize
from week_one.dev import simple_tworoom_test_resize
import matplotlib.pyplot as plt

simple_domain = simple_oneroom_resize.Simple_One_Domain()
simple_domain.main()
simpledomain_solution = simple_domain.get_solution()
#simple_domain.visualize()

itr_domain = simple_tworoom_test_resize.Simple_Two_Domain(iterations=100)
itr_domain.main()
twodomain_solution = itr_domain.get_solution()

difference = simpledomain_solution - twodomain_solution



plt.subplot(221)
plt.title("Iterated Domain solution")
plt.figtext(0.1,0.95,"Solution after 10 iterations, relaxation factor 0.8")
plt.pcolor(twodomain_solution)
plt.colorbar()
plt.subplot(222)
plt.title("Simple Domain solution")
plt.pcolor(simpledomain_solution)
plt.colorbar()

plt.subplot(223)
plt.title("Difference")
plt.pcolor(difference)
plt.colorbar()
plt.show()