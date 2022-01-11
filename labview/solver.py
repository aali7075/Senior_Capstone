import cvxpy as cp
import numpy as np


def solve_currents(b, x_t, I_max, r):
    n = len(x_t)  # number of coils to solve

    # setup minimization problem
    x = cp.Variable(n)
    objective = cp.Minimize(cp.norm_inf(b @ x - r))

    # re-center constraints using the previous solution
    constraints = [-I_max - x_t <= x, x <= I_max - x_t]
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # return solution
    return np.array(x.value)
