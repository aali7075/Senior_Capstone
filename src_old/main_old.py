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

    # c = 100.0 * 1e6 * MU_0 * N / (4 * np.pi)
    #
    # # compute each sub-function once and store result
    # _r1, _r2, _r3, _r4 = _eval_r(x, y, z, a1, b1, z0)
    # _d1, _d2, _d3, _d4 = _eval_d(x, y, z, a1, b1, z0)
    # _c1, _c2, _c3, _c4 = _eval_c(x, y, z, a1, b1, z0)
    #
    # # return constant * inner sum of various function combinations
    # # tried to keep things clean, or at least cleaner than the original mathematica code
    # return c * np.sum([
    #     -1.0 *  _d1 / (_r1 * (_r1 + _c1)),
    #     -1.0 *  _c1 / (_r1 * (_r1 + _d1)),
    #             _d2 / (_r2 * (_r2 - _c2)),
    #     -1.0 *  _c2 / (_r2 * (_r2 + _d2)),
    #     -1.0 *  _d3 / (_r3 * (_r3 + _c3)),
    #     -1.0 *  _c3 / (_r3 * (_r3 + _d3)),
    #             _d4 / (_r4 * (_r4 - _c4)),
    #     -1.0 *  _c4 / (_r4 * (_r4 + _c4))
    # ], axis=0)

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

    bx, by, bz = None, None, None
    for i in range(num_coils):
        for j in range(num_coils):
            if bx is None:
                bx = Bx(xx[i, j], yy[i, j], z, a1, b1, z0, current)
            else:
                bx += Bx(xx[i, j], yy[i, j], z, a1, b1, z0, current)

            if by is None:
                by = By(xx[i, j], yy[i, j], z, a1, b1, z0, current)
            else:
                by += By(xx[i, j], yy[i, j], z, a1, b1, z0, current)

            if bz is None:
                bz = Bz(xx[i, j], yy[i, j], z, a1, b1, z0, current)
            else:
                bz += Bz(xx[i, j], yy[i, j], z, a1, b1, z0, current)

    return np.array([bx, by, bz])

def overlay_coils(a1, b1, num_coils, coil_spacing, unit, ax):
    # convert coil sizes to matplotlib units
    a1 *= unit
    b1 *= unit
    coil_spacing *= unit

    f = lambda x: (coil_spacing * (num_coils - 1) + 2 * x * (num_coils - 1)) / 2.0

    f_a1, f_b1 = f(a1), f(b1)

    # shift up and over s.t. we have the top left corner of each coil
    xp = np.linspace(-1.0 * f_a1 + 50, f_a1 + 50, num=num_coils) - a1
    yp = np.linspace(-1.0 * f_b1 + 50, f_b1 + 50, num=num_coils) - b1
    xx, yy = np.meshgrid(xp, yp)
    for i in range(num_coils):
        for j in range(num_coils):
            px, py = xx[i, j], yy[i, j]
            print(px, py, 2*a1, 2*b1)
            rect = plt.Rectangle((px, py), 2*a1, 2*b1, fc='none', ec='green', lw=2.0, alpha=0.75)
            ax.add_patch(rect)

if __name__ == '__main__':
    # define some random constants to plot
    x_axis = np.linspace(-10, 10, num=100)
    y_axis = np.linspace(-10, 10, num=100)

    # A1 and B1 are (half of) the rectangle length and width
    A1, B1, I, spacing = 2, 2, 1, 1.0
    Z = 1.0

    z_wall_1 = -100.0 # 100 cm back
    z_wall_2 = 100.0 # 100 cm forward

    GRID = 3
    xx, yy = get_coordinates(A1, B1, spacing, GRID, 0, 0)
    size = max((GRID-1), 1) * A1 * spacing * 2

    grid = np.zeros((100, 100, 100))

    x_axis = np.linspace(-size, size, 100)
    y_axis = np.linspace(-size, size, 100)

    unit = 100 / (2*size) # convert cm grid (2*size x 2*size) to matplotlib unit grid (100x100 plot)
    for i, x in enumerate(x_axis):
        for j, y in enumerate(y_axis):
            # for k, z in ...
            # visualize coil placement by plotting distance to closest coil for each (x,y)

            # v = [np.sqrt((x - xx[ii, jj])**2 + (y - yy[ii, jj])**2)
            #     for ii in range(GRID) for jj in range(GRID)]
            # grid[i, j] = min(v)

            # get the magnitude of the field at each x, y

            B_wall_1 = get_gradient(x, y, Z, A1, B1, z_wall_1, I, GRID, spacing)
            B_wall_2 = get_gradient(x, y, Z, A1, B1, z_wall_2, I, GRID, spacing)

            B = B_wall_1 + B_wall_2

            u, v, w = B[0], B[1], B[2]

            grid[i, j] = np.linalg.norm(B, axis=0)

    # plot the grid
    xticks = ['' if i % 10 else f'{x_axis[i]:.2f}' for i in range(len(x_axis))][:-1] + [x_axis[-1]]
    yticks = ['' if i % 10 else f'{y_axis[i]:.2f}' for i in range(len(y_axis))][:-1] + [y_axis[-1]]
    ax = sns.heatmap(grid, xticklabels=xticks, yticklabels=yticks)

    overlay_coils(A1, B1, GRID, spacing, unit, ax)

    plt.title(f'Magnetic Field Magnitude (nT) of a {GRID}x{GRID} of Coils at Z={Z}cm.')
    plt.xlabel('X position (cm)')
    plt.ylabel('Y position (cm)')

    plt.show()
