import scipy as sp
import numpy as np


A = np.ones((2,2))
B = np.zeros((5,5))
B[0:A.shape[0],0:A.shape[1]] = A
u = np.array([1,2,0,0,0])
print(np.eye(3,3))