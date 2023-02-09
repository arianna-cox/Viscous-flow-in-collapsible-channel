import numpy as np
import matplotlib.pyplot as plt
from Shooting import shooting
plt.rcParams.update({'font.size': 16})
from direct_finite_difference import continuation_technique
from tabulate import tabulate

# No flux Q=0
Q = 0
# number of grid points
n = 101

# plot static wall shapes for shooting method
X_start = 1
X_end = 0
H_initial = 1
relative_tol = 1e-11
absolute_tol = 1e-11
H_shooting = []
for H_XX_initial in [-1, -0.5, -0.1, 0.1, 0.5, 1]:
    X, H, H_XX = shooting(X_start, X_end, Q, H_initial, H_XX_initial, relative_tol, absolute_tol, n)
    H_shooting.append(H)
    plt.plot(X, H, label = rf"$P_d =$ {-H_XX_initial}")
plt.xlabel('X')
plt.ylabel('H')
plt.grid()
plt.legend(fontsize=11)
plt.show()

# plot static wall shapes for newton-raphson method
X = np.linspace(0,1,n)
H_initial = np.ones((n,1))
upstream = 0
tol_X = 1e-12
tol_F = 1e-12
max_number_of_iterations = int(1e3)
Q_first = 1e-4
P_first = 0
continuation_steps = 100
H_newton = []
for H_XX_initial in [-1, -0.5, -0.1, 0.1, 0.5, 1]:
    H = continuation_technique(H_initial, Q, -H_XX_initial, upstream, tol_F, tol_X, max_number_of_iterations, Q_first, P_first, continuation_steps)
    H = H.flatten()
    H_newton.append(H)
    plt.plot(X, H, label = rf"$P_d =$ {-H_XX_initial}")
plt.xlabel('X')
plt.ylabel('H')
plt.grid()
plt.legend(fontsize=11)
plt.show()

# plot static wall shapes for analytic solution
def analytic(X, H_XX_initial):
    return (H_XX_initial/2)*(X**2-X)+1

H_analytic = []
for H_XX_initial in [-1, -0.5, -0.1, 0.1, 0.5, 1]:
    H = analytic(X, H_XX_initial)
    H_analytic.append(H)
    plt.plot(X, H, label = rf"$P_d =$ {-H_XX_initial}")
plt.xlabel('X')
plt.ylabel('H')
plt.grid()
plt.legend(fontsize=11)
plt.show()

# comparing error
error_shooting_max = []
error_newton_max = []
for i in range(6):
    error_shooting_max.append(np.max(abs(H_shooting[i]-H_analytic[i])))
    error_newton_max.append(np.max(abs(H_newton[i]-H_analytic[i])))

# tabulate
head = ["P_d", "Maximum error in shooting method","Maximum error in finite differences method"]
print(tabulate(np.column_stack(([-1, -0.5, -0.1, 0.1, 0.5, 1],error_shooting_max, error_newton_max)), head))
