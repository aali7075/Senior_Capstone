import numpy as np

from .field_constants import field_x, field_y, field_z


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

    :return: Tuple containing x and y
    """

    m, n = shape
    m = int(m)
    n = int(n)
    f_a = lambda _x: (s * (n - 1) + 2 * x * (n - 1)) / 2.0
    f_b = lambda _y: (s * (m - 1) + 2 * y * (m - 1)) / 2.0

    f_a1, f_b1 = f_a(a1), f_b(b1)
    #print("f_a1", f_a1, "f_b1", f_b1)
    xx = np.linspace(-1.0 * f_a1, f_a1, num=n) + x
    yy = np.linspace(-1.0 * f_b1, f_b1, num=m) + y

    return xx, yy


def _panel_b(x_c, y_c, z_c, shape, a1, b1, coil_spacing, x_p, y_p, z_p, rot_axis=None, rot_angle=0):
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
    xx, yy = _get_coil_coordinates(a1, b1, coil_spacing, shape, 0, 0)
    #print("xx is ", xx, "yy is ", yy)

    x = []
    y = []
    z = []
    for coil_x in xx:
        for coil_y in yy:
            # move each measurement s.t. the coil is at 0, 0, 0 and the measurement is relative to that
            x_q = p[0] - coil_x
            y_q = p[1] - coil_y
            z_q = p[2]

            x.append(field_x(x_q, y_q, z_q, a1, b1, 0, 1))
            y.append(field_y(x_q, y_q, z_q, a1, b1, 0, 1))
            z.append(field_z(x_q, y_q, z_q, a1, b1, 0, 1))

    return np.array([x, y, z])


def get_full_b(wall1, wall2, p):
    w1_center = wall1['center'] # wall center (metric)
    w1_shape = wall1['shape'] # (rows, columns)
    w1_a1 = wall1['a1'] # half width (metric)
    w1_b1 = wall1['b1'] # half height (metric)
    w1_coil_spacing = wall1['coil_spacing'] # spacing between coils (metric)
    w1_rx = wall1['rotation_axis'] # None or 'x', 'y', 'z'
    w1_theta = wall1['theta'] # angle to rotate around axis (ccw respective to positive axis), radians

    b1 = _panel_b(*w1_center, w1_shape, w1_a1, w1_b1, w1_coil_spacing, *p, w1_rx, w1_theta)
    #print("b1 in orginal is ", b1)
    w2_center = wall2['center']
    w2_shape = wall2['shape']
    w2_a1 = wall2['a1']
    w2_b1 = wall2['b1']
    w2_coil_spacing = wall2['coil_spacing']
    w2_rx = wall2['rotation_axis']
    w2_theta = wall2['theta']

    b2 = _panel_b(*w2_center, w2_shape, w2_a1, w2_b1, w2_coil_spacing, *p, w2_rx, w2_theta)

    b = np.concatenate([b1, b2], axis=1)
    print("b original is", b)
    return b

def pls(lst):
    #lst= np.array(lst)
    w1_center = (lst[0],lst[1],lst[2]) # wall center (metric)
    w1_shape = (lst[3],lst[4]) # (rows, columns)
    w1_a1 = lst[5] # half width (metric)
    w1_b1 = lst[6] # half height (metric)
    w1_coil_spacing = lst[7] # spacing between coils (metric)
    w1_rx = lst[8] # None or 'x', 'y', 'z'
    w1_theta = lst[9] # angle to rotate around axis (ccw respective to positive axis), radians
    p =(lst[20],lst[21],lst[22])
    b1 = _panel_b(*w1_center, w1_shape, w1_a1, w1_b1, w1_coil_spacing, *p, w1_rx, w1_theta)
    w2_center = (lst[10],lst[11],lst[12])
    w2_shape = (lst[13],lst[14])
    w2_a1 = lst[15]
    w2_b1 = lst[16]
    w2_coil_spacing = lst[17]
    w2_rx = lst[18]
    w2_theta = lst[19]

    b2 = _panel_b(*w2_center, w2_shape, w2_a1, w2_b1, w2_coil_spacing, *p, w2_rx, w2_theta)

    b = np.concatenate([b1, b2], axis=1)
    b=b.flatten()
    check = np.zeros((4,3))
    check = check+3
    check = check.flatten()
    #print(check)
    #lst.append(180)
    lst[:]=check # pass by reference for labview
    print(lst)
    return check

def getFullB1(lst):
    #print(len(lst))
    w1_center = (lst[0],lst[1],lst[2]) # wall center (metric)
    w1_shape = (lst[3],lst[4]) # (rows, columns)
    w1_a1 = lst[5] # half width (metric)
    w1_b1 = lst[6] # half height (metric)
    w1_coil_spacing = lst[7] # spacing between coils (metric)
    w1_rx = lst[8] # None or 'x', 'y', 'z'
    w1_theta = lst[9] # angle to rotate around axis (ccw respective to positive axis), radians
    p =(lst[20],lst[21],lst[22])

    b1 = _panel_b(*w1_center, w1_shape, w1_a1, w1_b1, w1_coil_spacing, *p, w1_rx, w1_theta)
    #print("b1 is ", b1)
    w2_center = (lst[10],lst[11],lst[12])
    w2_shape = (lst[13],lst[14])
    w2_a1 = lst[15]
    w2_b1 = lst[16]
    w2_coil_spacing = lst[17]
    w2_rx = lst[18]
    w2_theta = lst[19]

    b2 = _panel_b(*w2_center, w2_shape, w2_a1, w2_b1, w2_coil_spacing, *p, w2_rx, w2_theta)

    b = np.concatenate([b1, b2], axis=1)
    b=b.flatten()
    #print("b new is ", b)
    lst[:]=[0,0,0,0]
    return 0

w1 = {
    'center': (0, 0, 0), # 3 points, x, y, z (float)
    'shape': (2, 2), # 2 points, rows columns (int)
    'a1': .5, # float
    'b1': .5, # float
    'coil_spacing': .5, # float
    'rotation_axis': None, # string, can be one of None, 'x', 'y', 'z' (could replace with int)
    'theta': 0 # float, radians
}

w2 = {
    'center': (0, 0, 0), # 3 points, x, y, z (float)
    'shape': (2, 2), # 2 points, rows columns (int)
    'a1': .5, # float
    'b1': .5, # float
    'coil_spacing': .5, # float
    'rotation_axis': None, # string, can be one of None, 'x', 'y', 'z' (could replace with int)
    'theta': 0 # float, radians
}
lst = [0,0,0,2,2,.5,.5,.5,-1,0,0,0,0,2,2,.5,.5,.5,-1,0,1,0,0]
# get_full_b(w1,w2,(1,0,0))
#get_full_b1(lst)
pls(lst)

