import numpy as np
import matplotlib.pyplot as plt
from Shooting import shooting
plt.rcParams.update({'font.size': 15})
from direct_finite_difference import continuation_technique, pressure_distribution
from tabulate import tabulate

# number of grid points
n = 101
# downstream
for Q in [0.03]:
    # plot H for shooting method with downstream pressure P_d
    X_start = 1
    X_end = 0
    H_initial = 1
    relative_tol = 1e-11
    absolute_tol = 1e-11
    H_shooting_Pd = []
    H_XX_shooting_Pd = []
    plt.figure(f"H for shooting method with downstream pressures P_d and Q = {Q}")
    for P_d in [-Q, -2*Q/3, -Q/2, -Q/3, -Q/6]:
        X, H, H_XX = shooting(X_start, X_end, Q, H_initial, -P_d, relative_tol, absolute_tol, n)
        H_shooting_Pd.append(H)
        H_XX_shooting_Pd.append(H_XX)
        plt.plot(X, H, label = rf"$P_d =$ {P_d}")
    plt.xlabel('X')
    plt.ylabel('H')
    plt.grid()
    plt.legend(fontsize=11)
    plt.show()

    # plot H for newton-raphson method with downstream pressure P_d
    X = np.linspace(0,1,n)
    H_initial = np.ones((n,1))
    upstream = 0
    tol_X = 1e-11
    tol_F = 1e-11
    max_number_of_iterations = int(1e4)
    Q_first = 1e-6
    P_first = 0
    continuation_steps = 100
    H_newton = []
    P_newton = []
    plt.figure(f"H for newton-raphson method with downstream pressures P_d and Q = {Q}")
    for P_d in [-Q, -2*Q/3, -Q/2, -Q/3, -Q/6]:
        H = continuation_technique(H_initial, Q, P_d, upstream, tol_F, tol_X, max_number_of_iterations, Q_first, P_first, continuation_steps)
        P_newton.append(pressure_distribution(H,P_d,upstream))
        H = H.flatten()
        H_newton.append(H)
        plt.plot(X, H, label = rf"$P_d =$ {P_d}")
    plt.xlabel('X')
    plt.ylabel('H')
    plt.grid()
    plt.legend(fontsize=11)
    plt.show()

    #plot the pressure distribution for shooting method with downstream pressure P_d
    plt.figure(f"Pressure distribution for shooting method with downstream pressures P_d and Q = {Q}")
    i = 0
    for H_XX in H_XX_shooting_Pd:
        X = np.linspace(1, 0, n)
        plt.plot(X, -H_XX, label = rf"$P_d =$ {[-Q, -2*Q/3, -Q/2, -Q/3, -Q/6][i]}")
        i += 1
    plt.xlabel('X')
    plt.ylabel('P')
    plt.grid()
    plt.legend(fontsize=11)
    plt.show()

    #plot the pressure distribution for newton-raphson method with downstream pressure P_d
    plt.figure(f"pressure distribution for newton-raphson method with downstream pressures P_d and Q = {Q}")
    i = 0
    X = np.linspace(0, 1, n)
    for P in P_newton:
        plt.plot(X, P, label = rf"$P_d =$ {[-Q, -2*Q/3, -Q/2, -Q/3, -Q/6][i]}")
        i += 1
    plt.xlabel('X')
    plt.ylabel('P')
    plt.grid()
    plt.legend(fontsize=11)
    plt.show()

    # comparing the methods
    shooting_minus_newton_max_difference = []
    for i in range(5):
        # maximal difference in the methods
        difference = np.flip(H_shooting_Pd[i]) - H_newton[i]
        maximal_difference = np.max(abs(difference))
        shooting_minus_newton_max_difference.append(maximal_difference)
    # tabulate
    print(f"Q={Q}")
    head = ["P_d", "Maximum difference in the methods"]
    print(tabulate(np.column_stack(([-Q, -2*Q/3, -Q/2, -Q/3, -Q/6],shooting_minus_newton_max_difference)), head))

# upstream
for Q in [0.003]:
    # plot H for shooting method with upstream pressure P_u
    X_start = 0
    X_end = 1
    H_initial = 1
    relative_tol = 1e-11
    absolute_tol = 1e-11
    H_shooting_Pu = []
    H_XX_shooting_Pu = []
    plt.figure(f"H for shooting method with upstream pressures P_u and Q = {Q}")
    for P_u in [Q/6,Q/3,Q/2,2*Q/3,Q]:
        X, H, H_XX = shooting(X_start, X_end, Q, H_initial, -P_u, relative_tol, absolute_tol, n)
        H_shooting_Pu.append(H)
        H_XX_shooting_Pu.append(H_XX)
        plt.plot(X, H, label = rf"$P_u =$ {P_u}")
    plt.xlabel('X')
    plt.ylabel('H')
    plt.grid()
    plt.legend(fontsize=11)
    plt.show()

    # plot H for newton-raphson method with downstream pressure P_d
    X = np.linspace(0,1,n)
    H_initial = np.ones((n,1))
    upstream = 1
    tol_X = 1e-11
    tol_F = 1e-11
    max_number_of_iterations = int(1e4)
    Q_first = 1e-6
    P_first = 0
    continuation_steps = 100
    H_newton = []
    P_newton = []
    plt.figure(f"H for newton-raphson method with upstream pressures P_u and Q = {Q}")
    for P_u in [Q/6,Q/3,Q/2,2*Q/3,Q]:
        H = continuation_technique(H_initial, Q, P_u, upstream, tol_F, tol_X, max_number_of_iterations, Q_first, P_first, continuation_steps)
        P_newton.append(pressure_distribution(H,P_u,upstream))
        H = H.flatten()
        H_newton.append(H)
        plt.plot(X, H, label = rf"$P_u =$ {P_u}")
    plt.xlabel('X')
    plt.ylabel('H')
    plt.grid()
    plt.legend(fontsize=11)
    plt.show()

    #plot the pressure distribution for shooting method with upstream pressure P_u
    plt.figure(f"Pressure distribution for shooting method with upstream pressures P_u and Q = {Q}")
    i = 0
    for H_XX in H_XX_shooting_Pu:
        X = np.linspace(0, 1, n)
        plt.plot(X, -H_XX, label = rf"$P_u =$ {[Q/6,Q/3,Q/2,2*Q/3,Q][i]}")
        i += 1
    plt.xlabel('X')
    plt.ylabel('P')
    plt.grid()
    plt.legend(fontsize=11)
    plt.show()

    # comparing the methods
    shooting_minus_newton_max_difference = []
    for i in range(5):
        # maximal difference in the methods
        difference = H_shooting_Pu[i]- H_newton[i]
        maximal_difference = np.max(abs(difference))
        shooting_minus_newton_max_difference.append(maximal_difference)
    # tabulate
    print(f"Q={Q}")
    head = ["P_u", "Maximum difference in the methods"]
    print(tabulate(np.column_stack(([Q/6,Q/3,Q/2,2*Q/3,Q],shooting_minus_newton_max_difference)), head))
