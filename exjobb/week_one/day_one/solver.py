#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 17:10:54 2017

@author: filipthor
"""

from scipy import *
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm

nbrOfIterations = 10
normalWall = 15
heaterWall = 40
windowWall = 5
omega = 0.8
dx = 0.05

# Define the problem here

print("Skapa tre (2) grids")
print("skapa statiska randvillkor")
print("skapa delta-matris")
print("skapa dynamiska randvillkor")

for i in range(nbrOfIterations):
    print("Lös stora rummet")
    print("uppdatera dynamiska randvillkor")
    print("Lös första lilla rummet")
    print("Lös andra lilla rummet")
    print("Lös stora rummet")
    print("uppdatera dynamiska randvillkor")
    print("Relaxera")

#=== Klar med beräkningar, dags för redovisning: ===#

#=== Testvektorer att plotta ===#
m = 20 # storlek på testvektor (n^2)

u1 = np.arange(1,m**2+1) 
u2 = np.arange(1,2*m**2+1)
u3 = np.arange(1,m**2+1)
#u1 = np.random.rand(1,m**2)
#u2 = np.random.rand(1,2*m**2)
#u3 = np.random.rand(1,m**2)

def plot( u1, u2, u3):
    n = np.sqrt(len(u1))
    if n.is_integer():
        n = int(n)
    else:
        raise Exception # fyll på med mer info
    
    m1 = np.reshape(u1,(n,n))
    m2 = np.reshape(u2,(2*n,n))
    m3 = np.reshape(u3,(n,n))
    empty = np.zeros(shape=(n,n))
    
    block1 = np.concatenate((empty,m1))
    block2 = m2
    block3 = np.concatenate((m3,empty))
    
    #block1 = np.delete(block1,-1,1) #uncomment to remove double grid points along Γ1
    #block3 = np.delete(block3,0,1) #uncomment to remove double grid points along Γ2
    
    solution = np.concatenate((block1,block2,block3),axis=1)
    
    plt.pcolor(np.flipud(solution))
    plt.show()

plot(u1,u2,u3)