import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Units : A * H / (m * cm), therefore need to multiply by 100x.
# Units are then T / A
#
# Amperes * H (what is H?) / (meters * centimeters)
# Then Tesla / Ampere

# H / m (what is H => ? / meter...)
MU_0 = 4 * np.pi * 1.0e-7

# rectangular in the X, Y plane @ z0 distance from the origin
# has side lengths 2*a1 and 2*b1

# QUESTION: what are r, d and c ???

r1 = lambda x, y, z, a1, b1, z0: np.sqrt((a1 + x) ** 2 + (y + b1) ** 2 + (z - z0) ** 2)
r2 = lambda x, y, z, a1, b1, z0: np.sqrt((a1 - x) ** 2 + (y + b1) ** 2 + (z - z0) ** 2)
r3 = lambda x, y, z, a1, b1, z0: np.sqrt((a1 - x) ** 2 + (y - b1) ** 2 + (z - z0) ** 2)
r4 = lambda x, y, z, a1, b1, z0: np.sqrt((a1 + x) ** 2 + (y - b1) ** 2 + (z - z0) ** 2)

d1 = lambda x, y, z, a1, b1, z0: y + b1
d2 = lambda x, y, z, a1, b1, z0: y + b1
d3 = lambda x, y, z, a1, b1, z0: y - b1
d4 = lambda x, y, z, a1, b1, z0: y - b1

c1 = lambda x, y, z, a1, b1, z0: a1 + x
c2 = lambda x, y, z, a1, b1, z0: a1 - x
c3 = lambda x, y, z, a1, b1, z0: -a1 + x
c4 = lambda x, y, z, a1, b1, z0: -a1 - x


def _eval_r(x, y, z, a1, b1, z0):
    return r1(x, y, z, a1, b1, z0), r2(x, y, z, a1, b1, z0), r3(x, y, z, a1, b1, z0), r4(x, y, z, a1, b1, z0)


def _eval_d(x, y, z, a1, b1, z0):
    return d1(x, y, z, a1, b1, z0), d2(x, y, z, a1, b1, z0), d3(x, y, z, a1, b1, z0), d4(x, y, z, a1, b1, z0)


def _eval_c(x, y, z, a1, b1, z0):
    return c1(x, y, z, a1, b1, z0), c2(x, y, z, a1, b1, z0), c3(x, y, z, a1, b1, z0), c4(x, y, z, a1, b1, z0)


def Bz(x, y, z, a1, b1, z0, N):
    """
    Magnetic flux density in Z direction at point (x, y, z)

    :param x: x location
    :param y: y location
    :param z: z location
    :param a1: (half of) coil side length A
    :param b1: (half of) coil side length B
    :param z0: Distance of coil from origin
    :param N: Current (Milliamps?)

    :return: Magnetic flux density
    """
    _k = 100.0 * 1e6 * MU_0 * N / (4 * np.pi)

    # compute each sub-function once and store result
    _r = _eval_r(x, y, z, a1, b1, z0)
    _d = _eval_d(x, y, z, a1, b1, z0)
    _c = _eval_c(x, y, z, a1, b1, z0)

    # return constant * inner sum of various function combinations
    # tried to keep things clean, or at least cleaner than the original mathematica code

    return _k * np.sum([((-1**(i+1))*_d[i] / (_r[i] * (_r[i] + (-1**i)*_c[i]))) -
                        (_c[i] / (_r[i] * (_r[i] - _d[i]))) for i in range(4)], axis=0)


def Bx(x, y, z, a1, b1, z0, N):
    """
    Magnetic flux density in X direction at point (x, y, z)

    :param x: x location
    :param y: y location
    :param z: z location
    :param a1: (half of) coil side length A
    :param b1: (half of) coil side length B
    :param z0: Distance of coil from origin
    :param N: Current (Milliamps?)

    :return: Magnetic flux density
    """

    _k = 100.0 * MU_0 * N / (4 * np.pi)
    _z = z - z0
    _r = _eval_r(x, y, z, a1, b1, z0)
    _d = _eval_d(x, y, z, a1, b1, z0)

    return _k * np.sum([(-1**i) * _z / (_r[i] * (_r[i] + _d[i])) for i in range(4)], axis=0)

def By(x, y, z, a1, b1, z0, N):
    """
    Magnetic flux density in X direction at point (x, y, z)

    :param x: x location
    :param y: y location
    :param z: z location
    :param a1: (half of) coil side length A
    :param b1: (half of) coil side length B
    :param z0: Distance of coil from origin
    :param N: Current (Milliamps?)

    :return: Magnetic flux density
    """

    _k = 100.0 * MU_0 * N / (4 * np.pi)
    _z = z - z0

    _r = _eval_r(x, y, z, a1, b1, z0)
    _c = _eval_c(x, y, z, a1, b1, z0)

    return _k * np.sum([(-1**i) * _z / (_r[i] * (_r[i] + (-1**i) * _c[i])) for i in range(4)], axis=0)

def get_coordinates(a1, b1, s, N, x, y):
    # get (x, y) coordinates given spacing between coils, coil sizes,
    # and number of coils s.t. the center of the coils is the origin

    f = lambda x: (s * (N-1) + 2 * x * (N-1)) / 2.0

    f_a1, f_b1 = f(a1), f(b1)
    xx = np.linspace(-1.0 * f_a1 + x, f_a1 + x, num=N)
    yy = np.linspace(-1.0 * f_b1 + y, f_b1 + y, num=N)

    return np.meshgrid(xx, yy)

def get_gradient(x, y, z, a1, b1, z0, current, num_coils, coil_spacing):
    xx, yy = get_coordinates(a1, b1, coil_spacing, num_coils, x, y)

    bx = Bx(x, y, z, a1, b1, z0, current)
    by = By(x, y, z, a1, b1, z0, current)
    bz = Bz(x, y, z, a1, b1, z0, current)

    return np.array([bx, by, bz])

if __name__ == '__main__':
    # define some random constants to plot
    x_axis = np.linspace(-10, 10, num=100)
    y_axis = np.linspace(-10, 10, num=100)
    Z, A1, B1, Z0, I = 1.0, 2, 2, 0.0, 12

    grid = np.zeros((len(x_axis), len(y_axis)))
    for i, X in enumerate(x_axis):
        B = get_gradient(X, y_axis, Z, A1, B1, Z0, I, num_coils=1, coil_spacing=1.0)
        grid[i] = np.linalg.norm(B, axis=0)

    ax = sns.heatmap(grid)
    plt.show()

    # density = np.zeros(len(Z))
    # spacing = 1.0
    # GRID_SIZE = 3
    #
    # gx, gy = get_coordinates(A1, B1, spacing, GRID_SIZE)
    # for i in range(GRID_SIZE):
    #     for j in range(GRID_SIZE):
    #         density += Bz1(gx[i, j], gy[i, j], Z, A1, B1, Z0, N)
    #
    # plt.plot(Z, density, label=f'multi-coils ({GRID_SIZE}x{GRID_SIZE})')
    # plt.show()
