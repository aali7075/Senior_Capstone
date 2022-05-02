import numpy as np
import cvxpy as cp

from .simulation_helper import get_full_b


class FieldNuller:
    def __init__(self, shape, coil_size, coil_spacing, wall_spacing, turns_per_coil, max_current, point=None):
        self.shape = shape
        self.coil_size = coil_size
        self.coil_spacing = coil_spacing
        self.wall_spacing = wall_spacing
        self.max_current = max_current
        self.turns_per_coil = turns_per_coil

        self.n_coils = np.prod(shape)
        self.currents = np.zeros(self.n_coils)

        if point is not None:
            self.set_point(point)
        else:
            self.point = point
            self.b_mat = None

    def set_point(self, point):
        if len(point) != 3:
            print("FieldNuller ERROR: point must have 3 dimensions! point not set")
            return

        self.point = point
        self.b_mat = get_full_b(self.shape, self.coil_size, self.coil_spacing,
                                self.wall_spacing, self.turns_per_coil, point)

        print(f"FieldNuller: Point is now {point}.")
        print(self.b_mat)

    def reset_currents(self):
        self.currents = np.zeros(self.n_coils)

    def solve(self, readings, lambda_value=0):
        if self.point is None:
            print("FieldNuller ERROR: Must first set measurement point!")
            return None

        # Setup minimization problem
        x = cp.Variable(int(self.n_coils))
        lambd = cp.Parameter(nonneg=True)
        lambd.value = lambda_value
        objective = cp.Minimize(cp.norm_inf(self.b_mat @ x + readings) + lambd * cp.norm2(x))

        # Re-center constraints using the previous solution
        constraints = [-self.max_current - self.currents <= x, x <= self.max_current - self.currents]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        delta_currents = np.array(x.value)
        self.currents += delta_currents

        # return solution
        return self.currents

    def simulate_fields(self):
        return self.b_mat @ self.currents