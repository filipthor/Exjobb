# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from  scipy import *
from  pylab import *
import numpy as np
import time

class simpleSolution:
############################################################################## 
######################## #create discrete laplace operator ###################
    totaltime = time.clock()

    n=20
    N=n-2
    dx=1/(N+1)  
    u=np.zeros((n,n))
   
    sup=np.ones(N**2-1)
  
    for i in range(1,len(sup)+1):
        rem=i%N
        if (rem==0):
            sup[i-1]=0
    A = np.diag(ones(N**2-N),-N) \
            + np.diag(sup, -1) \
            + np.diag(-4.*ones(N**2), 0) \
            + np.diag(sup, 1) \
            + np.diag(ones(N**2-N),N)
    #D=linalg.block_diag(A,A,A)
  
    D=1/(dx**2)*A
 
   
###############################################################################
######################### SET Boundry conditions ##############################
   
    bc=np.zeros((N,N))
    bc[0,:]=40/(dx**2)                    # y = 0
    bc[-1,:]=40/(dx**2)                  # y = N
    bc[1:,0]=20/(dx**2)                    # x = 0
    bc[:,-1]=20/(dx**2)
    
    bc[0,0]=bc[0,0]+40/(dx**2)
    bc[0,-1]=bc[0,-1]+40/(dx**2)
    bc[-1,0]=bc[-1,0]+40/(dx**2)
    bc[-1,-1]=bc[-1,-1]+40/(dx**2)
                     # x = N
    bc=np.asarray(bc).reshape(-1)              #bc as vector

###############################################################################
################################  SOLVE   #####################################
    
#   print("D",D.round(1))
    solvetime = time.clock()
    utemp =np.linalg.solve(D,-bc)
    solvetime = time.clock() - solvetime
    print("solve took {} seconds".format(solvetime))
#   test=np.linalg.norm(D.dot(utemp)+bc)
#   print("test",test)
    umatrix=np.reshape(utemp,(N,N))
#   print("utemp",umatrix.round())   
   
    u[1:-1,1:-1]=umatrix
    u[0,:]=40                # y = 0
    u[-1,:]=40                  # y = N
    u[:,0]=20                  # x = 0
    u[:,-1]=20                   # x = N
    print(u.round())
   
###############################################################################
    totaltime = time.clock() - totaltime
    print("entire program took {} seconds".format(totaltime))

    plt.pcolor(np.flipud(u))
    plt.show()
             
    # print(b)
    # print(matrixb)








def extractBoundary(u,n,room):
    if room == 1:
        #dim på rum är = np.sqrt(len(u))
        m = np.reshape(u,(n,n))
        return np.array(m[0:n,-1])
    if room == 2:
        #dim på rum är = np.sqrt(len(u)/2)
        m = np.reshape(u,(2*n,n))
        return np.column_stack((np.array(m[n:2*n,0]),np.array(m[0:n,-1])))
    if room == 3:
        #dim på rum är = np.sqrt(len(u))
        m = np.reshape(u,(n,n))
        return np.array(m[0:n,0])
    else:
        return 0