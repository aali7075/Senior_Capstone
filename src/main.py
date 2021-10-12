import numpy as np
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

def Bz1(x, y, z, a1, b1, z0, N):
    """
    Magnetic flux density in Z direction at point (x, y, z)

    :param x: x location
    :param y: y location
    :param z: z location
    :param a1: (half of) coil side length A
    :param b1: (half of) coil side length B
    :param z0: Distance of coil from origin
    :param N: ???

    :return: Magnetic flux density
    """
    c = 100.0 * 1e6 * MU_0 * N / (4 * np.pi)

    # compute each sub-function once and store result
    _r1, _r2, _r3, _r4 = _eval_r(x, y, z, a1, b1, z0)
    _d1, _d2, _d3, _d4 = _eval_d(x, y, z, a1, b1, z0)
    _c1, _c2, _c3, _c4 = _eval_c(x, y, z, a1, b1, z0)

    # return constant * inner sum of various function combinations
    # tried to keep things clean, or at least cleaner than the original mathematica code
    return c * np.sum([
        -1.0 *  _d1 / (_r1 * (_r1 + _c1)),
        -1.0 *  _c1 / (_r1 * (_r1 + _d1)),
                _d2 / (_r2 * (_r2 - _c2)),
        -1.0 *  _c2 / (_r2 * (_r2 + _d2)),
        -1.0 *  _d3 / (_r3 * (_r3 + _c3)),
        -1.0 *  _c3 / (_r3 * (_r3 + _d3)),
                _d4 / (_r4 * (_r4 - _c4)),
        -1.0 *  _c4 / (_r4 * (_r4 + _c4))
    ], axis=0)

def Bx1(x, y, z, a1, b1, z0, N):
    """
    Magnetic flux density in X direction at point (x, y, z)

    :param x: x location
    :param y: y location
    :param z: z location
    :param a1: (half of) coil side length A
    :param b1: (half of) coil side length B
    :param z0: Distance of coil from origin
    :param N: ???

    :return: Magnetic flux density
    """

    c = 100.0 * MU_0 * N / (4 * np.pi)

    _z = z - z0
    _r1, _r2, _r3, _r4 = _eval_r(x, y, z, a1, b1, z0)
    _d1, _d2, _d3, _d4 = _eval_d(x, y, z, a1, b1, z0)

    return c * np.sum([
                _z / (_r1 * (_r1 + _d1)),
        -1.0 *  _z / (_r2 * (_r2 + _d2)),
                _z / (_r3 * (_r3 + _d3)),
        -1.0 *  _z / (_r4 * (_r4 + _d4))
    ], axis=0)

if __name__ == '__main__':
    # define some random constants to plot
    Z = np.linspace(-5, 5, num=100)
    X, Y, A1, B1, Z0, N = 0, 0, 3.5, 3.5, 1.75, 12
    B = Bz1(X, Y, Z, A1, B1, Z0, N)

    plt.plot(Z, B)
    plt.title('Single coil')
    plt.xlabel('Z (cm)')
    plt.ylabel('Bz field strength (nT/mA)')
    plt.show()
