import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

def backward_substitution(U, b):
    #This function solves the linear system Ux = b when U is upper triangular
    n = len(b)
    x = np.zeros((n,1))
    for j in range(n):
        i = n-1-j
        x[i] = (b[i] - np.dot(U[i, i:n], x[i:n])) / U[i, i]
    return x

def forward_substitution(L, b):
    #This function solves the linear system Lx = b when L is lower triangular
    x = np.zeros((len(b),1))
    for i in range(len(b)):
        x[i] = (b[i] - np.dot(L[i, 0:i], x[0:i])) / L[i, i]
    return x

def linear_equation_solver(A,b):
    # This function solves the linear system Ax = b and outputs x
    P, L, U = scipy.linalg.lu(A)
    # L b_new = P^T b
    b_new = forward_substitution(L, np.matmul(P.T,b))
    x = backward_substitution(U, b_new)
    return x

def Jacobian(H, upstream):
    n = len(H)
    J = np.zeros((n, n))
    J[0,0] = 1
    J[1,n-1] = 1
    for i in range(2,n-2):
        J[i, i] = 3*H[i]**2*(H[i+2]-2*H[i+1]+2*H[i-1]-H[i-2])
        H_cubed = H[i]**3
        J[i,i+2] = H_cubed
        J[i,i+1] = -2*H_cubed
        J[i,i-1] = 2*H_cubed
        J[i,i-2] = -H_cubed

    H_cubed = H[n - 2] ** 3
    J[n-2, n-2] = 3*H[n-2]**2 *(3*H[n-1] + 12*H[n-3] - 6*H[n-4] + H[n-5]) - 40*H_cubed
    J[n-2, n-1] = 3*H_cubed
    J[n-2, n-3] = 12* H_cubed
    J[n-2, n-4] = -6* H_cubed
    J[n-2, n-5] = H_cubed

    if upstream == 1:
        J[n-1,0:4] = [2,-5,4,-1]
    if upstream == 0:
        J[n-1, n-4:n] = [-1, 4, -5, 2]
    return J

def linear_system(H, Q, P, upstream):
    n = len(H)
    delta = 1/(n-1)
    F = np.zeros((n,1))
    F[0] = H[0]-1
    F[1] = H[n-1]-1
    F[2:n-2] = H[2:n-2]**3*(H[4:n]-2*H[3:n-1]+2*H[1:n-3]-H[0:n-4]) -2*delta**3*Q
    F[n-2] = H[n-2]**3*(3*H[n-1]-10*H[n-2]+12*H[n-3]-6*H[n-4]+H[n-5]) -2*delta**3*Q
    if upstream == 1:
        F[n-1] = 2*H[0] -5*H[1]+4*H[2]-H[3]+delta**2*P
    if upstream == 0:
        F[n-1] = 2*H[n-1] -5*H[n-2]+4*H[n-3]-H[n-4]+delta**2*P
    return F

def Newton_Raphson(H_initial, Q, P, upstream, tol_F, tol_X, max_number_of_iterations):
    H = H_initial
    F = np.ones((len(H),1))
    H_delta = np.ones((len(H), 1))
    iterations = 0
    while np.sum(np.abs(F)) > tol_F and np.sum(np.abs(H_delta)) > tol_X:
        iterations += 1
        F = linear_system(H, Q, P, upstream)
        J = Jacobian(H, upstream)
        H_delta = linear_equation_solver(J, -F)
        H += H_delta
        if iterations >= max_number_of_iterations:
            print(f"Ran for more than {max_number_of_iterations} iterations")
            break
    return H

def continuation_technique(H_initial, Q, P, upstream, tol_F, tol_X, max_number_of_iterations, Q_first, P_first, continuation_steps):
    # this function finds H of given values of Q and P_d or P_u = P, using a continuation technique
    Q_vector = np.linspace(Q_first,Q,continuation_steps)
    P_vector = np.linspace(P_first,P,continuation_steps)
    H = H_initial
    for i in range(continuation_steps):
        H = Newton_Raphson(H, Q_vector[i], P_vector[i], upstream, tol_F, tol_X, max_number_of_iterations)
    return H

def pressure_distribution(H, P_endpoint, upstream):
    n = len(H)
    delta = 1/(n-1)
    P = np.zeros((n,1))
    if upstream == 1:
        P[0] = P_endpoint
        P[n-1] = -(2*H[n-1]-5*H[n-2]+4*H[n-3]-H[n-4])/delta**2
    if upstream == 0:
        P[-1] = P_endpoint
        P[0] = -(2*H[0]-5*H[1]+4*H[2]-H[3])/delta**2
    P[1:n-1] = -(H[0:n-2]-2*H[1:n-1]+H[2:n])/delta**2
    return P

def delta_P_finite_difference(H, P_endpoint, upstream):
    n = len(H)
    delta = 1/(n-1)
    if upstream == 1:
        P_u = P_endpoint
        P_d = -(2*H[n-1]-5*H[n-2]+4*H[n-3]-H[n-4])/delta**2
    if upstream == 0:
        P_d = P_endpoint
        P_u = -(2*H[0]-5*H[1]+4*H[2]-H[3])/delta**2
    return P_u - P_d
