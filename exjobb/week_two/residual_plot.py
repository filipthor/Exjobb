from week_two import w2_two_domain
from week_two import w1_one_domain
import matplotlib.pyplot as plt



twodomain = w2_two_domain.two_domain(n = 40,iterations=30,relaxation=0.8)
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
plt.title("Neumann domain residual $||A_1 u_1-b_1||_{\infty}$")

#plt.figtext(0.1,0.95,"Solution after 10 iterations, relaxation factor 0.8")
plt.semilogy(res[:,0])

plt.yscale('log')
plt.grid(b=True,which="major",linestyle="--")
plt.grid(b=True,which="minor",linestyle="--")
plt.xlabel("Iterations")

plt.subplot(122)
plt.title("Dirichlet domain residual $||A_2 u_2-b_2||_{\infty}$")
plt.plot(res[:,1])
plt.yscale('log')
plt.xlabel("Iterations")
plt.grid(True,which="major", linestyle="--")
plt.grid(True,which="minor",linestyle="--")
plt.figtext(0.1,0.95,"$n=40$, $\gamma = 0.8$, 30 iterations.")
#plt.minorticks_on()
#plt.axes().tick_params(axis="x",which="minor",bottom="off")
plt.show()