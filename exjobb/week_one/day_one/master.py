import matplotlib.pyplot as plt
from dev import simple_tworoom_test
from dev import simple_oneroom


itr_room = simple_tworoom_test.Simple_Tworoom(iterations=10)
itr_room.main()
tworoom_solution = itr_room.get_solution()
#itr_room.visualize()


simple_room = simple_oneroom.simple_oneroom()
simple_room.main()
oneroom_solution = simple_room.get_solution()
#simple_room.visualize()

difference = oneroom_solution - tworoom_solution

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
plt.title("Iterated solution")
plt.figtext(0,0,"Solution after 10 iterations, relaxation factor 0.8")
plt.pcolor(tworoom_solution)
plt.colorbar()
plt.subplot(222)
plt.title("Simple Solution")
plt.pcolor(oneroom_solution)
plt.colorbar()

plt.subplot(223)
plt.title("Difference")
plt.pcolor(difference)
plt.colorbar()
plt.show()



