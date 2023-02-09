import numpy as np
import matplotlib.pyplot as plt
from Shooting import shooting
plt.rcParams.update({'font.size': 15})
from direct_finite_difference import continuation_technique, delta_P_finite_difference
from tabulate import tabulate

# function that computes delta P given Q and P_d using shooting method
def delta_P_shooting_downstream(Q, P_d, relative_tol, absolute_tol, n):
    X_start = 1
    X_end = 0
    H_initial = 1
    P = -shooting(X_start, X_end, Q, H_initial, -P_d, relative_tol, absolute_tol, n)[2]
    if P[0] != P_d:
        print("P[0] != P_d")
    # deltaP = P_u - P_d
    return P[-1] - P[0]


# using shooting method
Q_length = 61
Q_vector = np.linspace(0,6,Q_length)
relative_tol = 1e-5
absolute_tol = 1e-5
n = 101
plt.figure('Delta P against Q via shooting')
for P_d in [-3,-1,0,1,3]:
    delta_P_shooting_vector = []
    for Q in Q_vector:
        deltaP = delta_P_shooting_downstream(Q, P_d, relative_tol, absolute_tol, n)
        delta_P_shooting_vector.append(deltaP)
    print('plot')
    plt.plot(Q_vector, delta_P_shooting_vector, label = rf"$P_d =$ {P_d}")
plt.xlabel('Q')
plt.ylabel(r'$\Delta P$')
plt.grid()
plt.legend(fontsize=11)
plt.show()


# using direct finite differences method
Q_length = 61
Q_vector = np.linspace(0,6,Q_length)
H_initial = np.ones((51,1))
tol_F = 1e-7
tol_X = 1e-7
max_number_of_iterations = 1000
upstream = 0
Q_first = 0.01
P_first = 0
continuation_steps = 10
plt.figure('Delta P against Q via direct_finite_difference')
for P_d in [-3,-1,0,1,3]:
    delta_P_newton_vector = []
    for Q in Q_vector:
        H = continuation_technique(H_initial, Q, P_d, upstream, tol_F, tol_X, max_number_of_iterations, Q_first, P_first, continuation_steps)
        deltaP = delta_P_finite_difference(H, P_d, upstream)
        delta_P_newton_vector.append(deltaP)
    print('plot')
    plt.plot(Q_vector, delta_P_newton_vector, label = rf"$P_d =$ {P_d}")
plt.xlabel('Q')
plt.ylabel(r'$\Delta P$')
plt.grid()
plt.legend(fontsize=11)
plt.show()


# plot channel shapes for shooting method
# number of grid points
n = 101
X_start = 1
X_end = 0
H_initial = 1
relative_tol = 1e-11
absolute_tol = 1e-11
H_shooting = []
for P_d in [-3, -1, 0, 1, 3]:
    plt.figure(f"Shooting P_d = {P_d}")
    for Q in [0,1,2,3,4,5,6]:
        X, H, H_XX = shooting(X_start, X_end, Q, H_initial, -P_d, relative_tol, absolute_tol, n)
        H_shooting.append(H)
        plt.plot(X, H, label = rf"$Q =$ {Q}")
    plt.xlabel('X')
    plt.ylabel('H')
    plt.grid()
    plt.legend(fontsize=11)
    plt.show()

# plot channel shapes for newton-raphson method
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
for P_d in [-1, 1]:
    plt.figure(f"Finite difference P_d = {P_d}")
    for Q in [0,1,2,3,4,5,6]:
        H = continuation_technique(H_initial, Q, P_d, upstream, tol_F, tol_X, max_number_of_iterations, Q_first, P_first, continuation_steps)
        H = H.flatten()
        H_newton.append(H)
        plt.plot(X, H, label = rf"$Q =${Q}")
    plt.xlabel('X')
    plt.ylabel('H')
    plt.grid()
    plt.legend(fontsize=11)
    plt.show()
