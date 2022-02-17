import numpy as np

MU_0 = 4 * np.pi * 1.0e-7

_r = (
    lambda _x, _y, _z, _a1, _b1, _z0: np.sqrt((_a1 + _x) ** 2 + (_y + _b1) ** 2 + (_z - _z0) ** 2),
    lambda _x, _y, _z, _a1, _b1, _z0: np.sqrt((_a1 - _x) ** 2 + (_y + _b1) ** 2 + (_z - _z0) ** 2),
    lambda _x, _y, _z, _a1, _b1, _z0: np.sqrt((_a1 - _x) ** 2 + (_y - _b1) ** 2 + (_z - _z0) ** 2),
    lambda _x, _y, _z, _a1, _b1, _z0: np.sqrt((_a1 + _x) ** 2 + (_y - _b1) ** 2 + (_z - _z0) ** 2)
)


def _eval_r(_x, _y, _z, _a1, _b1, _z0):
    return [r(_x, _y, _z, _a1, _b1, _z0) for r in _r]


_d = (
    lambda _x, _y, _z, _a1, _b1, _z0: _y + _b1,
    lambda _x, _y, _z, _a1, _b1, _z0: _y + _b1,
    lambda _x, _y, _z, _a1, _b1, _z0: _y - _b1,
    lambda _x, _y, _z, _a1, _b1, _z0: _y - _b1
)


def _eval_d(_x, _y, _z, _a1, _b1, _z0):
    return [d(_x, _y, _z, _a1, _b1, _z0) for d in _d]


_c = (
    lambda _x, _y, _z, _a1, _b1, _z0: _a1 + _x,
    lambda _x, _y, _z, _a1, _b1, _z0: _a1 - _x,
    lambda _x, _y, _z, _a1, _b1, _z0: -_a1 + _x,
    lambda _x, _y, _z, _a1, _b1, _z0: -_a1 - _x
)


def _eval_c(_x, _y, _z, _a1, _b1, _z0):
    return [c(_x, _y, _z, _a1, _b1, _z0) for c in _c]


def field_x(x, y, z, a1, b1, z0, N):
    """
    Magnetic flux density in X direction at point (x, y, z) for a
    coil of size 2*a1, 2*b1 residing on the x, y plane defined by z0

    :param x: x location
    :param y: y location
    :param z: z location
    :param a1: (half of) coil side length A
    :param b1: (half of) coil side length B
    :param z0: Distance of coil from origin
    :param N: Current

    :return: Magnetic flux density
    """

    k = 100.0 * MU_0 * N / (4 * np.pi)
    z = z - z0
    r_vals = _eval_r(x, y, z, a1, b1, z0)
    d_vals = _eval_d(x, y, z, a1, b1, z0)

    return k * np.sum([(-1 ** i) * z / (r_vals[i] * (r_vals[i] + d_vals[i]))
                       for i in range(4)], axis=0)


def field_y(x, y, z, a1, b1, z0, N):
    """
    Magnetic flux density in Y direction at point (x, y, z) for a
    coil of size 2*a1, 2*b1 residing on the x, y plane defined by z0

    :param x: x location
    :param y: y location
    :param z: z location
    :param a1: (half of) coil side length A
    :param b1: (half of) coil side length B
    :param z0: Distance of coil from origin
    :param N: Current

    :return: Magnetic flux density
    """

    k = 100.0 * MU_0 * N / (4 * np.pi)
    z = z - z0

    r_vals = _eval_r(x, y, z, a1, b1, z0)
    c_vals = _eval_c(x, y, z, a1, b1, z0)

    return k * np.sum([(-1 ** i) * z / (r_vals[i] * (r_vals[i] + (-1 ** i) * c_vals[i]))
                       for i in range(4)], axis=0)


def field_z(x, y, z, a1, b1, z0, N):
    """
    Magnetic flux density in Z direction at point (x, y, z) for a
    coil of size 2*a1, 2*b1 residing on the x, y plane defined by z0

    :param x: x location
    :param y: y location
    :param z: z location
    :param a1: (half of) coil side length A
    :param b1: (half of) coil side length B
    :param z0: Distance of coil from origin
    :param N: Current

    :return: Magnetic flux density
    """

    k = 100.0 * 1e6 * MU_0 * N / (4 * np.pi)

    # compute each sub-function once and store result
    r_vals = _eval_r(x, y, z, a1, b1, z0)
    d_vals = _eval_d(x, y, z, a1, b1, z0)
    c_vals = _eval_c(x, y, z, a1, b1, z0)

    # return constant * inner sum of various function combinations
    # tried to keep things clean, or at least cleaner than the original mathematica code

    return k * np.sum([((-1 ** (i + 1)) * d_vals[i] / (r_vals[i] * (r_vals[i] + (-1 ** i) * c_vals[i]))) -
                       (c_vals[i] / (r_vals[i] * (r_vals[i] - d_vals[i]))) for i in range(4)], axis=0)
