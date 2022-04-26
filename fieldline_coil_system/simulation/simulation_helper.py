import numpy as np

from .simulation import field_x, field_y, field_z


def _rotation_matrix(theta, u):
    return np.array([[np.cos(theta) + u[0] ** 2 * (1 - np.cos(theta)),
                      u[0] * u[1] * (1 - np.cos(theta)) - u[2] * np.sin(theta),
                      u[0] * u[2] * (1 - np.cos(theta)) + u[1] * np.sin(theta)],
                     [u[0] * u[1] * (1 - np.cos(theta)) + u[2] * np.sin(theta),
                      np.cos(theta) + u[1] ** 2 * (1 - np.cos(theta)),
                      u[1] * u[2] * (1 - np.cos(theta)) - u[0] * np.sin(theta)],
                     [u[0] * u[2] * (1 - np.cos(theta)) - u[1] * np.sin(theta),
                      u[1] * u[2] * (1 - np.cos(theta)) + u[0] * np.sin(theta),
                      np.cos(theta) + u[2] ** 2 * (1 - np.cos(theta))]])


def _get_coil_coordinates(a1, b1, s, shape, center):
    """
    Get the (x, y) coordinates for the center of the NxN grid of rectangles
    so they satisfy the distances set by the rectangle width, height, and spacing.
    The coils are positioned such that their combined center is at x, y

    :param a1: Rectangle width (x-direction) (meters)
    :param b1: Rectangle height (y-direction) (meters)
    :param s: Spacing between rectangles (meters)
    :param shape: Dimensions of coil matrix (x, y)
    :param center: center of panel (x, y), (meters)

    :return: Tuple containing x and y
    """

    m, n = shape
    f_a = lambda x: (s * (m-1) + 2 * x * (m-1)) / 2.0
    f_b = lambda y: (s * (n-1) + 2 * y * (n-1)) / 2.0

    f_a1, f_b1 = f_a(a1), f_b(b1)
    xx = np.linspace(-1.0 * f_a1, f_a1, num=m) + center[0]
    yy = np.linspace(-1.0 * f_b1, f_b1, num=n) + center[1]

    return xx, yy


def _panel_b(x_c, y_c, z_c, shape, a1, b1, coil_spacing, x_p, y_p, z_p, turns_per_coil, rot_axis=None, rot_angle=0):
    """

    :param x_c: Panel center x
    :param y_c: Panel center y
    :param z_c: Panel center z
    :param shape: rows, columns of coils in panel
    :param a1: half of the coil width
    :param b1: half of the coil height
    :param coil_spacing: Spacing between the edges of the coils
    :param x_p: Query point x
    :param y_p: Query point y
    :param z_p: Query point z
    :param turns_per_coil: Number of turns on each individual coil
    :param rot_axis: Axis to rotate the panel around
    :param rot_angle: Angle to rotate the panel, radians

    :return:
    """

    # 1)    move the point s.t. the panel center is at 0, 0
    #       this is done so we can do the rotation relative to the panel center
    x_p_t = x_p - x_c
    y_p_t = y_p - y_c
    z_p_t = z_p - z_c

    p = np.array([x_p_t, y_p_t, z_p_t])
    #print("p is ", p)

    # 2) apply the (opposite) rotation to the point, simulation the panel rotation
    u = [0, 0, 0]
    if rot_axis == 0:
        u[0] = 1
    elif rot_axis == 1:
        u[1] = 1
    elif rot_axis == 2:
        u[2] = 1

    r = _rotation_matrix(rot_angle, u)

    p = p @ r

    # 3) solve for each coil's unit field
    #print(a1, b1, coil_spacing, shape)
    xx, yy = _get_coil_coordinates(a1, b1, coil_spacing, shape, (0, 0))
    #print("xx is ", xx, "yy is ", yy)

    x = []
    y = []
    z = []
    for coil_y in yy:
        for coil_x in xx:

            # move each measurement s.t. the coil is at 0, 0, 0 and the measurement is relative to that
            x_q = p[0] - coil_x
            y_q = p[1] - coil_y
            z_q = p[2]

            x.append(field_x(x_q, y_q, z_q, a1, b1, 0, turns_per_coil))
            y.append(field_y(x_q, y_q, z_q, a1, b1, 0, turns_per_coil))
            z.append(field_z(x_q, y_q, z_q, a1, b1, 0, turns_per_coil))
            # print(f"b for coil at ({x_q}, {y_q}, {z_q}): [{x[-1]}, {y[-1]}, {z[-1]}]")

    return np.array([x, y, z])


def get_full_b_from_walls(wall1, wall2, turns_per_coil, p):
    # All units in meters
    w1_center = wall1['center'] # wall center
    w1_shape = wall1['shape'] # (rows, columns)
    w1_a1 = wall1['a1'] # half width (metric)
    w1_b1 = wall1['b1'] # half height (metric)
    w1_coil_spacing = wall1['coil_spacing'] # spacing between coils (metric)
    w1_rx = wall1['rotation_axis'] # None or 'x', 'y', 'z'
    w1_theta = wall1['theta'] # angle to rotate around axis (ccw respective to positive axis), radians
    b1 = _panel_b(*w1_center, w1_shape, w1_a1, w1_b1, w1_coil_spacing, *p, turns_per_coil, w1_rx, w1_theta)

    w2_center = wall2['center']
    w2_shape = wall2['shape']
    w2_a1 = wall2['a1']
    w2_b1 = wall2['b1']
    w2_coil_spacing = wall2['coil_spacing']
    w2_rx = wall2['rotation_axis']
    w2_theta = wall2['theta']
    b2 = _panel_b(*w2_center, w2_shape, w2_a1, w2_b1, w2_coil_spacing, *p, turns_per_coil, w2_rx, w2_theta)

    b = np.concatenate([b1, b2], axis=1)
    return b


def get_full_b(shape, coil_size, coil_spacing, wall_spacing, turns_per_coil, point):
    """
    Get the b matrix for a given measurement point

    :param shape: shape of the panels [panels, x, y]. Panels is assumed to be 2
    :param coil_size: Tuple of size in meters (x, y)
    :param coil_spacing: Space between coils in meters
    :param wall_spacing: Space between panels in meters
    :param turns_per_coil: Number of turns on each coil
    :param point: measurement point in meters (x, y, z)
    :return: ndarray of shape ()
    """

    # Convert units to cm as get_full_b_from_walls
    half_wall_spacing = wall_spacing / 2
    a1 = coil_size[0] / 2
    b1 = coil_size[1] / 2

    wall1 = {
        'center': (0, 0, -half_wall_spacing),
        'shape': shape[1:3],
        'a1': a1,
        'b1': b1,
        'coil_spacing': coil_spacing,
        'rotation_axis': None,
        'theta': 0
    }
    wall2 = {
        'center': (0, 0, half_wall_spacing),
        'shape': shape[1:3],
        'a1': a1,
        'b1': b1,
        'coil_spacing': coil_spacing,
        'rotation_axis': None,
        'theta': 0
    }

    return get_full_b_from_walls(wall1, wall2, turns_per_coil, point)
