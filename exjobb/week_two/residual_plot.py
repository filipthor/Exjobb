from week_two import w2_two_domain
from week_two import w1_one_domain
import matplotlib.pyplot as plt


twodomain = w2_two_domain.two_domain(n = 40,iterations=100,relaxation=0.3)
twodomain.solve()
td = twodomain.get_solution()
#twodomain.visualize()

onedomain = w1_one_domain.one_domain(n = 40)
onedomain.solve()
od = onedomain.get_solution()

#dv = twodomain.get_diff_vector()
res = twodomain.get_residuals()
print(res)



plt.subplot(121)
plt.title("Neumann domain residual")
#plt.figtext(0.1,0.95,"Solution after 10 iterations, relaxation factor 0.8")
plt.semilogy(res[:,0])
plt.yscale('log')
plt.grid(b=True,which="minor",linestyle="-")
plt.grid(b=True,which="major",linestyle="--")
plt.xlabel("Iterations")

plt.subplot(122)
plt.title("Dirichlet domain residual")
plt.plot(res[:,1])
plt.yscale('log')
plt.xlabel("Iterations")
plt.grid(True,which="major")
plt.grid(True,which="minor",linestyle="--")
plt.show()