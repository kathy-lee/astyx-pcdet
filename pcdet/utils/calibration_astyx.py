import numpy as np
import json
import math


def inv_trans(T):
    rotation = np.linalg.inv(T[0:3, 0:3])  # rotation matrix

    translation = T[0:3, 3]
    translation = -1 * np.dot(rotation, translation.T)
    translation = np.reshape(translation, (3, 1))
    Q = np.hstack((rotation, translation))
    return Q


def get_calib_from_file(calib_file):
    with open(calib_file, 'r') as f:
        data = json.load(f)
    T_fromLidar = np.array(data['sensors'][1]['calib_data']['T_to_ref_COS'])
    T_fromCamera = np.array(data['sensors'][2]['calib_data']['T_to_ref_COS'])
    K = np.array(data['sensors'][2]['calib_data']['K'])

    T_toLidar = inv_trans(T_fromLidar)
    T_toCamera = inv_trans(T_fromCamera)
    return {'T_toLidar': T_toLidar,
            'T_toCamera': T_toCamera,
            'K': K}


def get_objects_lidar(objects, T_toLidar):
    objects_lidar = []
    for obj in objects:
        obj_lidar = np.dot(T_toLidar[0:3, 0:3], np.transpose(obj))
        T = T_toLidar[0:3, 3]
        obj_lidar = obj_lidar + T
        obj_lidar = np.transpose(obj_lidar)
        objects_lidar.append(obj_lidar)
    return np.array(objects_lidar)


def get_rot_lidar(orient, T_toLidar):
    rot_lidar = []
    for k in orient:
        T = quat_to_rotation(k)
        T = np.dot(T_toLidar[:,0:3], T)
        rot = math.atan2(T[1,0], T[0,0])
        # rot = math.atan2(-T[2,0], np.sqrt(T[2,0]*T[2,0], T[2,2]*T[2,2]))
        # rot = math.atan2(T[2,1], T[2,2])
        rot_lidar.append(rot)
    rot_lidar = np.array(rot_lidar)
    return rot_lidar[:, np.newaxis]


def quat_to_rotation(quat):
    m = np.sum(np.multiply(quat, quat))
    q = quat.copy()
    q = np.array(q)
    n = np.dot(q, q)
    if n < np.finfo(q.dtype).eps:
        rot_matrix = np.identity(4)
        return rot_matrix
    q = q * np.sqrt(2.0 / n)
    q = np.outer(q, q)
    rot_matrix = np.array(
        [[1.0 - q[2, 2] - q[3, 3], q[1, 2] + q[3, 0], q[1, 3] - q[2, 0]],
         [q[1, 2] - q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] + q[1, 0]],
         [q[1, 3] + q[2, 0], q[2, 3] - q[1, 0], 1.0 - q[1, 1] - q[2, 2]]],
        dtype=q.dtype)
    rot_matrix = np.transpose(rot_matrix)
    # # test if it is truly a rotation matrix
    # d = np.linalg.det(rotation)
    # t = np.transpose(rotation)
    # o = np.dot(rotation, t)
    return rot_matrix


def boxes_lidar_to_astyx_camera(boxes, calib):

    return boxes_camera


def boxes3d_camera_to_astyx_imageboxes(boxes, calib, image_shape):

    return boxes_img