from types import Union

from field_constants import *

def _get_coil_coordinates(a1, b1, s, shape, x, y):
    """
    Get the (x, y) coordinates for the center of the NxN grid of rectangles
    so they satisfy the distances set by the rectangle width, height, and spacing.
    The coils are positioned such that their combined center is at x, y

    :param a1: Rectangle width (x-direction) (cm)
    :param b1: Rectangle height (y-direction) (cm)
    :param s: Spacing between rectangles (cm)
    :param shape: Dimensions of coil matrix (rows, cols)
    :param x: x position of rectangle's center
    :param y: y position of rectangle's center

    :return: np.meshgrid in ij format of each coil's center for NxN grid
    """

    M, N = shape
    f_a = lambda _x: (s * (N - 1) + 2 * x * (N - 1)) / 2.0
    f_b = lambda _y: (s * (M - 1) + 2 * y * (M - 1)) / 2.0

    f_a1, f_b1 = f_a(a1), f_b(b1)
    xx = np.linspace(-1.0 * f_a1, f_a1, num=N) + x
    yy = np.linspace(-1.0 * f_b1, f_b1, num=M) + y

    return (xx, yy)

def _query_wall(self,
                x: Union[int, float],
                y: Union[int, float],
                z: Union[int, float],
                currents: Union[list, tuple],
                wall_args: dict
                ):
    """
    Compute the magnetic field at (x, y, z) created by the specified wall

    :param x: x coordinate to query (cm)
    :param y: y coordinate to query (cm)
    :param z: z coordinate to query (cm)
    :param currents: Currents (Amps) going through each coil
    :param wall_args: Specifics of the wall generating the field

    :return: np.array containing the field x, y, z components in that order
    """
    a1 = wall_args['a1']
    b1 = wall_args['b1']
    z0 = wall_args['z0']
    coil_spacing = wall_args['coil_spacing']
    shape = wall_args['shape']
    xx, yy = _get_coil_coordinates(a1, b1, coil_spacing, shape, x, y)

    bx, by, bz = None, None, None
    for i in range(shape[0]):
        for j in range(shape[1]):
            if bx is None:
                bx = field_x(xx[i, j], yy[i, j], z, a1, b1, z0, currents[i + j * num_coils])
            else:
                bx += field_x(xx[i, j], yy[i, j], z, a1, b1, z0, currents[i + j * num_coils])

            if by is None:
                by = field_y(xx[i, j], yy[i, j], z, a1, b1, z0, currents[i + j * num_coils])
            else:
                by += field_y(xx[i, j], yy[i, j], z, a1, b1, z0, currents[i + j * num_coils])

            if bz is None:
                bz = field_z(xx[i, j], yy[i, j], z, a1, b1, z0, currents[i + j * num_coils])
            else:
                bz += field_z(xx[i, j], yy[i, j], z, a1, b1, z0, currents[i + j * num_coils])

    return np.array([bx, by, bz])

class Simulation():
    def __init__(self):
        self._walls = []



    def add_wall(self,
                 a1: Union[int, float],
                 b1: Union[int, float],
                 z0: Union[int, float],
                 coil_spacing: Union[int, float],
                 shape: Union[list[int, int], tuple[int, int]] = (1, 1)
                 ):
        """
        Add a wall to the simulation such the the coils lay in the x,y plane and the positive z-axis is the
        front of the plane.

        :param a1: Half of the width (x-axis) of each coil
        :param b1: Half of the height (y-axis) of each coil
        :param z0: z value the wall goes through
        :param coil_spacing: spacing between the edges of each coil
        :param shape: dimensions of coil grid: (rows, columns)
        """

        if type(a1) not in (int, float) or a1 <= 0:
            raise ValueError('a1 must be a positive number')

        if type(b1) not in (int, float) or b1 <= 0:
            raise ValueError('b1 must be a positive number')

        if type(z0) not in (int, float):
            raise ValueError('z0 must be a number')

        if type(coil_spacing) not in (int, float) or coil_spacing < 0:
            raise ValueError('coil_spacing must be a non-negative number')

        if type(shape) not in (tuple, list) or len(shape) != 2 or not all([isinstance(x, int) for x in shape]):
            raise ValueError('shape must be a 2 integer tuple or list to specify the number of rows x columns of coils')

        self._walls.append({'a1': a1, 'b1': b1, 'z0': z0, 'coil_spacing': coil_spacing, 'shape': shape})



