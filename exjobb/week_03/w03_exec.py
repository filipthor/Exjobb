from week_03 import w03_2d
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed


N = 20
itr = 20
rel = 1

twodomain = w03_2d.two_domain(n = N,iterations=itr,relaxation=rel)
twodomain.solve()
td = twodomain.get_solution()

plt.pcolor(td)
plt.colorbar()
plt.show()