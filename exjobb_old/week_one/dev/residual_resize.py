from dev import simple_oneroom
from dev import simple_tworoom_test_resize

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

itr_domain = simple_tworoom_test_resize.Simple_Two_Domain(iterations=10000)
itr_domain.main()
twodomain_solution = itr_domain.get_solution()

residuals = itr_domain.get_residual(2)

resval = np.zeros((10,36))


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