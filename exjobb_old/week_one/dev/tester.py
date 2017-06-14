import numpy as np
from week_one.dev import simple_twodomain

#itr_domain = simple_twodomain.Simple_Two_Domain(iterations=10)
#itr_domain.main()
#twodomain_solution = itr_domain.get_solution()
#itr_domain.visualize()


n = 4
A = np.zeros((n,n))
count = 1
for i in range(n):
    for j in range(n):
        A[i,j] = count
        count += 1
print(A)
print(A[:,-2])



#for i in range(4):
 #   C = np.reshape(A[:,i],(2,2))
  #  print(C)