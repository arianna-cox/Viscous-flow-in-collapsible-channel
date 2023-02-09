import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import optimize

# error for if the integration cannot be carried out
class IntegrationException(Exception):
    pass

# error for if the solution does not exist
class SolutionExistenceException(Exception):
    pass

def Runge_Kutta(X_start, X_end, Q, H_initial, H_X_initial, H_XX_initial, relative_tol, absolute_tol, n):
    # This function returns X, H and H_XX
    G_initial = np.array([H_initial, H_X_initial, H_XX_initial])
    def ODE(t, G):
        return np.array([G[1], G[2], Q*(G[0]**(-3))])
    solution = solve_ivp(ODE, (X_start, X_end), G_initial, method='RK45', t_eval=np.linspace(X_start, X_end, n), rtol=relative_tol, atol=absolute_tol)
    if solution.t[-1] != X_end:
        # The error is that for solve_ivp the 'Required step size is less than spacing between numbers' so the solution doesn't reach X_end
        raise IntegrationException()
    return solution.t, solution.y[0], solution.y[2]

def shooting(X_start, X_end, Q, H_initial, H_XX_initial, relative_tol, absolute_tol, n):
    # H_root_function returns the value of H(X_end) - 1
    def H_root_function(beta_test):
        solution = Runge_Kutta(X_start, X_end, Q, H_initial, beta_test, H_XX_initial, relative_tol, absolute_tol, n)
        return solution[1][-1] - 1

    # the following code finds an interval for beta over which there is a sign change of H-1

    if X_start == 1:
        M = 11
        # initial try for the lower bound of the interval
        initial_lower_bound = -1
        interval_lower = 'not assigned'
        # initial try for the upper bound of the interval
        interval_upper = 1
        # find the lower bound of the interval
        while interval_lower == 'not assigned':
            try:
                if H_root_function(initial_lower_bound) > 0:
                    interval_lower = initial_lower_bound
                else:
                    initial_lower_bound -= 1
            except IntegrationException:
                interval_upper = initial_lower_bound
                initial_lower_bound -= 1
        # find a sign change in the interval
        found = 0
        while found == 0:
            beta_test = np.linspace(interval_lower, interval_upper, M + 1)
            for i in range(M):
                try:
                    result_1 = H_root_function(beta_test[i])
                except IntegrationException:
                    # when the integration fails H - 1 < 0
                    print(f'Error for {beta_test[i]}')
                    # up to this point there has been no sign change, so the sign change must lie between beta_test i and i+1
                    interval_upper = beta_test[i]
                    interval_lower = beta_test[i-1]
                    break
                try:
                    result_2 = H_root_function(beta_test[i+1])
                except IntegrationException:
                    print(f'Error for {beta_test[i+1]}')
                    interval_upper = beta_test[i+1]
                    interval_lower = beta_test[i]
                    break
                if result_1 * result_2 < 0:
                    found = 1
                    a = beta_test[i]
                    b = beta_test[i+1]
                    break
                # if there is no error and no sign change the for loop will not have stopped, so we increase the upper bound
                if i == M-1:
                    interval_upper += 1

        # once we have found the interval over which there's a sign change, use a bisection method to narrow down the root
        beta = optimize.bisect(H_root_function, a, b)
        return Runge_Kutta(X_start, X_end, Q, H_initial, beta, H_XX_initial, relative_tol, absolute_tol, n)


    # the algorithm assumes H_root_function tends to infinity for |beta| large
    # Step 0. choose an initial grid of values of beta
    # Step 1. find the minimum of H_root_function over the grid:
    #     CASE 1: if the minimum is negative, call the value of beta at which H_root_function is negative beta_negative. Go to Step 2.
    #     CASE 2: if the minimum is positive and on an endpoint of the grid, extend the grid and repeat Step 1
    #     CASE 3: if the minimum is positive and not an endpoint let the new grid have start and endpoints being the grid points before and after the value of beta at the minimum. Then repeat Step 1.
    #     If CASE 3 happens more than a certain number of times, assume there is no root
    # Step 2. Find values of beta either side of beta_negative for which H_root_function is positive. Run the bisection method on both intervals over which there is a sign change. This gives two solutions for H_X_initial.
    if X_start == 0:
        #number of points in grid
        M = 61
        # Step 0
        interval_lower = -2
        interval_upper = 2

        # Step 1
        beta_negative_found = 0
        # check that you want this as the max iterations
        max_iterations = 10
        iterations = 0
        while beta_negative_found == 0:
            beta_test = np.linspace(interval_lower, interval_upper, M)
            H_root_values = np.zeros(M)
            for i in range(M):
                H_root_values[i] = H_root_function(beta_test[i])
            index_of_minimum = np.argmin(H_root_values)
            # Case 1
            if H_root_values[index_of_minimum] < 0:
                beta_negative = beta_test[index_of_minimum]
                beta_negative_found = 1
            # Case 2
            if index_of_minimum == 0:
                interval_lower += -1
                interval_upper = interval_lower
            elif index_of_minimum == M-1:
                interval_lower = interval_upper
                interval_upper += 1
            # Case 3
            else:
                interval_lower = beta_test[index_of_minimum - 1]
                interval_upper = beta_test[index_of_minimum + 1]
            # If no negative value can be found within the maximum number of iterations, the function returns 'There may be no solution for beta'
            iterations += 1
            if iterations > max_iterations:
                raise SolutionExistenceException()

        # Step 2
        # a = beta_negative - 1
        b = beta_negative + 1
        # while H_root_function(a) <= 0:
        #     a -= 1
        while H_root_function(b) <= 0:
            b += 1

        # beta_lower = optimize.bisect(H_root_function, a, beta_negative)
        beta_upper = optimize.bisect(H_root_function, beta_negative, b)

        # Runge_Kutta(X_start, X_end, Q, H_initial, beta_lower, H_XX_initial, relative_tol, absolute_tol, n)
        return Runge_Kutta(X_start, X_end, Q, H_initial, beta_upper, H_XX_initial, relative_tol, absolute_tol, n)