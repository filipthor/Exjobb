import matplotlib.pyplot as plt
from dev import simple_tworoom_test
from dev import simple_oneroom


itr_domain = simple_tworoom_test.Simple_Two_Domain(n=70)
itr_domain.main()
twodomain_solution = itr_domain.get_solution()
#itr_room.visualize()


simple_domain = simple_oneroom.Simple_One_Domain(n=70)
simple_domain.main()
simpledomain_solution = simple_domain.get_solution()
#simple_room.visualize()

difference = simpledomain_solution - twodomain_solution

#plt.pcolor(difference)
#plt.colorbar()
#plt.show()

# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111)
# plt.title("Iterated solution")
# plt.pcolor(tworoom_solution)
# plt.colorbar()
#
#
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111)
# plt.title("Simple Solution")
# plt.pcolor(oneroom_solution)
# plt.colorbar()
#
# plt.show()



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



