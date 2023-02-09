import numpy as np
import matplotlib.pyplot as plt
from Shooting import shooting, SolutionExistenceException
plt.rcParams.update({'font.size': 15})
from direct_finite_difference import continuation_technique, delta_P_finite_difference
from tabulate import tabulate

# function that computes delta P given Q and P_u using shooting method
def delta_P_shooting_upstream(Q, P_u, relative_tol, absolute_tol, n):
    X_start = 0
    X_end = 1
    H_initial = 1
    solution2 = shooting(X_start, X_end, Q, H_initial, -P_u, relative_tol, absolute_tol, n)
    P2 = -solution2[2]
    # P1 = -solution1[2]
    # deltaP = P_u - P_d
    # print(f"P_d1 = {P1[-1]} and P_d2 = {P2[-1]}")
    return P2[0] - P2[-1], solution2[1]

# using shooting method
Q_length_shooting = 601
Q_vector_original = np.linspace(0,6,Q_length_shooting)
relative_tol = 1e-5
absolute_tol = 1e-5
n = 51
plt.figure('Delta P against Q via shooting')
maxQ = np.zeros((5,2))
H_shooting_vector = [[],[],[],[],[]]
Q_vector = [[],[],[],[],[]]
i = 0
for P_u in [-3,-1,0,1,3]:
    delta_P_shooting_vector = []
    Q_no_solution = []
    for Q in Q_vector_original:
        try:
            deltaP, H_shooting = delta_P_shooting_upstream(Q, P_u, relative_tol, absolute_tol, n)
            delta_P_shooting_vector.append(deltaP)
            Q_vector[i].append(Q)
            H_shooting_vector[i].append(H_shooting)
        except SolutionExistenceException:
            Q_no_solution.append(Q)
    plt.plot(Q_vector[i], delta_P_shooting_vector, label=rf"$P_u =$ {P_u}")
    try:
        if Q_vector[i][-1] < Q_no_solution[0]:
            print(f"For P_u = {P_u} the largest value of Q is {Q_vector[i][-1]}")
        else:
            print("There are values of Q with no solution below the maximum for Q")
    except:
        pass
    maxQ[i, :] = P_u, Q_vector[i][-1]
    i += 1
plt.xlabel('Q')
plt.ylabel(r'$\Delta P$')
plt.grid()
plt.legend(fontsize=11)
plt.show()
np.save('Q3_Q_vector.npy', Q_vector)
np.save('Q3_max_values_of_Q.npy', maxQ)
np.save('Q3_H_shooting_vector.npy', H_shooting_vector)

Q_vector = np.load('Q3_Q_vector.npy', allow_pickle=True)
H_shooting_vector = np.load('Q3_H_shooting_vector.npy', allow_pickle=True)

# plot the channel shapes for the shooting method
i = -1
for P_u in [-3,-1,0,1,3]:
    i +=1
    Q_current_vector = Q_vector[i]
    number_of_samples = 5
    def index(j):
        step_size = np.floor((len(Q_current_vector)-1)/(number_of_samples-1))
        remainder = (len(Q_current_vector)-1)%(number_of_samples-1)
        index = np.linspace(0,step_size*(number_of_samples-1), number_of_samples)
        if remainder != 0:
            index[-1] = (len(Q_current_vector)-1)
        return int(index[j])
    plt.figure(f"H using finite difference method with P_u = {P_u}")
    for j in range(number_of_samples):
        Q = Q_current_vector[index(j)]
        X = np.linspace(0, 1, 51)
        plt.plot(X, H_shooting_vector[i][index(j)], label = rf"$Q =$ {Q}")
    plt.xlabel('X')
    plt.ylabel('H')
    plt.grid()
    plt.legend(fontsize=11)
    plt.show()

maxQ = np.load('Q3_max_values_of_Q.npy')[:,1]

# using direct finite differences method
Q_length_newton = 51
H_initial = np.ones((51,1))
tol_F = 1e-5
tol_X = 1e-5
max_number_of_iterations = 1000
upstream = 1
Q_first = 0.01
P_first = 0
continuation_steps = 100
plt.figure("Finite difference")
i = -1
H_newton_vector = [[],[],[],[],[]]
for P_u in [-3,-1,0,1,3]:
    i += 1
    Q_vector = np.linspace(0, maxQ[i], Q_length_newton)
    delta_P_newton_vector = []
    for Q in Q_vector:
        H = continuation_technique(H_initial, Q, P_u, upstream, tol_F, tol_X, max_number_of_iterations, Q_first, P_first, continuation_steps)
        H_newton_vector[i].append(H.flatten())
        deltaP = delta_P_finite_difference(H, P_u, upstream)
        delta_P_newton_vector.append(deltaP)
    plt.plot(Q_vector, delta_P_newton_vector, label = rf"$P_u =$ {P_u}")
plt.xlabel('Q')
plt.ylabel(r'$\Delta P$')
plt.grid()
plt.legend(fontsize=11)
plt.show()

np.save('Q3_H_newton_vector.npy', H_newton_vector)

# plot the channel shapes using the finite difference method
H_newton_vector = np.load('Q3_H_newton_vector.npy')
i = -1
for P_u in [-3,-1,0,1,3]:
    i +=1
    Q_vector = np.linspace(0, maxQ[i], Q_length_newton)
    plt.figure(f"H using finite difference method with P_u = {P_u}")
    for j in [0,1,2,3,4,5]:
        Q = Q_vector[10*j]
        X = np.linspace(0, 1, 51)
        plt.plot(X, H_newton_vector[i][10*j], label = rf"$Q =$ {Q}")
    plt.xlabel('X')
    plt.ylabel('H')
    plt.grid()
    plt.legend(fontsize=11)
    plt.show()

head = ["P_u", "Maximum Q"]
print(tabulate(np.column_stack([[-3,-1,0,1,3],maxQ]), headers = head))