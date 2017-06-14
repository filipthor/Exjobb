from week_two import w1_one_domain
from week_07 import w07_ConstrainedLQ_Internal_2d
from week_07 import w07_ConstrainedLQ_Derivative_2d
from week_07 import w07_2d_third_constraint
from week_03 import w03_2d_block
from week_two import w1_one_domain
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed


'''
Written 2017-06-14

Script running a comparison between Constrained Internal Least Squares,
Standard Least Squares and Neumann-Dirichlet iteration


'''

# Data initalization
n = 20
itr = 400
relaxation = 1

# Constrained Internal Least Squares
constrained_internal= w07_ConstrainedLQ_Internal_2d.two_domain(n,itr,relaxation)

# Standard Least Squares
least_square = w07_2d_third_constraint.two_domain(n,itr,relaxation)

# Standard Neumann-Dirichlet iteration
ND_method = w03_2d_block.two_domain(n, itr, relaxation)

'''
# Constrained Derivative Least Squares
constrained_derivative = w07_ConstrainedLQ_Derivative_2d.two_domain(n,4000,relaxation)
'''


# Running scripts
print("Starting constrained Internal LQ")
constrained_internal.solve()
print("Starting Standard LQ")
least_square.solve()
print("Starting Neumann-Dirichlet iteration")
ND_method.solve()
'''
print("Starting constrained Derivative LQ")
constrained_derivative.solve()
'''


# Getting convergence info
# CILQ
c_i_converge = constrained_internal.get_difference()

# SLQ
s_lq_converge = least_square.get_difference()

# ND-itr
nd_converge = ND_method.get_difference()

'''
# CDLQ
c_d_converge = constrained_derivative.get_difference()
'''



plt.figure(figsize=(11,4.5))
plt.semilogy(c_i_converge,'b-',label="Constrained Internal points Least Squares")
plt.semilogy(s_lq_converge,'r-',label="Standard Least Squares")
plt.semilogy(nd_converge,'m-',label="Neumann-Dirichlet iteration")

#plt.semilogy(c_d_converge,'k-',label="Constrained Derivative Least Squares")

plt.title("A Comparison of errors between different methods")
plt.ylabel("Error")
plt.xlabel("Iteration")
plt.legend()
plt.savefig("out.pdf")
plt.show()

