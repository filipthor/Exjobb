from dev import simple_oneroom
from dev import simple_tworoom_test

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

#simple_domain = simple_oneroom.Simple_One_Domain()
#simple_domain.main()
#simpledomain_solution = simple_domain.get_solution()
#simple_domain.visualize()


itr_domain = simple_tworoom_test.Simple_Two_Domain(iterations=10)
itr_domain.main()
twodomain_solution = itr_domain.get_solution()

residuals = itr_domain.get_residual(2)

resval = np.zeros((10,36))



print(twodomain_solution[:,38]-twodomain_solution[:,39])
"""
for i in range(10):
    res = np.reshape(residuals[i,:],(38,38))
    resval[i,:] = res[1:-1,0]
    print(res[:,0])
    plt.pcolor(res)
    plt.colorbar()
    plt.show()

plt.pcolor(resval)
plt.colorbar()
plt.show()


# Set up grid and test data
nx, ny = 80, 40
x = range(nx)
y = range(ny)

#data = simpledomain_solution
data = twodomain_solution

hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')

X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
ha.plot_surface(X, Y, data,cmap='coolwarm')

plt.show()
"""