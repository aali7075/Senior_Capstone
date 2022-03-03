import numpy as np
import itertools
from simulation import field_x, field_y, field_z

def get_coordinates(shape, coil_size, coil_spacing, wall_spacing):
    x_limit = (coil_spacing * (shape[1] - 1) + coil_size[0] * (shape[1] - 1)) / 2.0
    y_limit = (coil_spacing * (shape[2] - 1) + coil_size[1] * (shape[2] - 1)) / 2.0
    z_limit = wall_spacing / 2.0 if shape[0] == 2 else 0

    x_coords = np.linspace(-x_limit, x_limit, num=shape[1])
    y_coords = np.linspace(-y_limit, y_limit, num=shape[2])
    z_coords = np.array([-z_limit, z_limit] if shape[0] == 2 else [0])

    coords_shape = np.append(shape, 3)
    coords_list = [z_coords, y_coords, x_coords]
    coords = np.array(list(itertools.product(*coords_list)))
    coords = np.flip(coords, axis=1).reshape(coords_shape)

    return coords


def get_full_b(coil_coords, coil_size, p):
    half_x, half_y = coil_size
    half_x /= 2.0
    half_y /= 2.0

    b =

    return b


# def get_panel_b(panel_center, shape, half_x, half_y, coil_spacing, point, rot_axis, rot_angle):
#     """
#
#         :param panel_center: Panel center (x, y, z)
#         :param shape: rows, columns of coils in panel
#         :param half_x: half of the coil width
#         :param half_y: half of the coil height
#         :param coil_spacing: Spacing between the edges of the coils
#         :param point: Query point (x, y, z)
#         :param rot_axis: Axis to rotate the panel around
#         :param rot_angle: Angle to rotate the panel, radians
#
#         :return:
#         """
#
#     # 1)    move the point s.t. the panel center is at 0, 0
#     #       this is done so we can do the rotation relative to the panel center
#     point -= panel_center
#
#     # 2) apply the (opposite) rotation to the point, simulation the panel rotation
#     u = [0, 0, 0]
#     if rot_axis == 0:
#         u[0] = 1
#     elif rot_axis == 1:
#         u[1] = 1
#     elif rot_axis == 2:
#         u[2] = 1
#
#     r = _rotation_matrix(rot_angle, u)
#
#     point = point @ r
#
#     # 3) solve for each coil's unit field
#     # print(a1, b1, coil_spacing, shape)
#     xx, yy = _get_coil_coordinates(a1, b1, coil_spacing, shape, 0, 0)
#     # print("xx is ", xx, "yy is ", yy)
#
#     x = []
#     y = []
#     z = []
#     for coil_x in xx:
#         for coil_y in yy:
#             # move each measurement s.t. the coil is at 0, 0, 0 and the measurement is relative to that
#             x_q = p[0] - coil_x
#             y_q = p[1] - coil_y
#             z_q = p[2]
#
#             x.append(field_x(x_q, y_q, z_q, a1, b1, 0, 1))
#             y.append(field_y(x_q, y_q, z_q, a1, b1, 0, 1))
#             z.append(field_z(x_q, y_q, z_q, a1, b1, 0, 1))
#
#     return np.array([x, y, z])

def get_coil_b(coil_coords, half_x, half_y, point):
    point -= coil_coords

    b_vector = np.zeros(3)
    b_vector[0] = field_x(point, half_x, half_y, 0, 1)
    b_vector[1] = field_y(point, half_x, half_y, 0, 1)
    b_vector[2] = field_z(point, half_x, half_y, 0, 1)

    return b_vector
