import numpy as np
import math


class Object3dAstyx(object):
    def __init__(self, dimension3d, score):
        self.h = dimension3d[2]
        self.w = dimension3d[0]
        self.l = dimension3d[1]
        self.loc = [None]*3
        self.orient = [None]*4
        self.score = score
        self.src = {}
        self.cls_type = ''
        self.cls_id = -1
        self.occlusion = -1
        self.level = -1
        self.level_str = ''
        self.loc_lidar = [None]*3
        self.rot_lidar = 0.0

    @classmethod
    def from_label(cls, labelinfo):
        obj = cls(labelinfo['dimension3d'], labelinfo['score'])
        obj.src = dict
        obj.cls_type = labelinfo['classname'] if labelinfo['classname'] != 'Person' else 'Pedestrian'
        cls_type_to_id = {
            'Bus': 0, 'Car': 1, 'Cyclist': 2, 'Motorcyclist': 3, 'Pedestrian': 4, 'Trailer': 5, 'Truck': 6,
            'Towed Object': 5, 'Other Vehicle': 5
        }
        obj.cls_id = cls_type_to_id[obj.cls_type]
        # self.truncation = float(label[1])
        obj.occlusion = float(
            labelinfo['occlusion'])  # 0:fully visible 1:partly occluded 2:largely occluded 3:fully occluded
        # self.alpha = float(label[3])
        # self.box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)
        # self.loc = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
        obj.loc = np.array(labelinfo['center3d'])
        # self.dis_to_cam = np.linalg.norm(self.loc)
        # self.ry = float(label[14])
        obj.orient = labelinfo['orientation_quat']
        # self.score = float(label[15]) if label.__len__() == 16 else -1.0
        obj.level_str = None
        obj.level = obj.get_astyx_obj_level()
        return obj

    @classmethod
    def from_prediction(cls, pred_boxes, pred_labels, pred_scores):
        obj = cls(pred_boxes[3:6], pred_scores)
        obj.cls_id = pred_labels
        obj.loc_lidar = pred_boxes[:3]
        obj.rot_lidar = pred_boxes[-1]
        return obj

    def get_astyx_obj_level(self):
        # height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if self.occlusion == 0:
            self.level_str = 'Easy'
            return 0  # Easy
        elif self.occlusion == 1:
            self.level_str = 'Moderate'
            return 1  # Moderate
        elif self.occlusion >= 2:
            self.level_str = 'Hard'
            return 2  # Hard
        else:
            self.level_str = 'UnKnown'
            return -1

    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        # x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        # y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        # z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        #
        # R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
        #               [0, 1, 0],
        #               [-np.sin(self.ry), 0, np.cos(self.ry)]])
        # corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        # corners3d = np.dot(R, corners3d).T
        # corners3d = corners3d + self.loc

        x_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        y_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
        # rotate and translate 3d bounding box
        R = quat_to_rot(self.orient)
        bbox = np.vstack([x_corners, y_corners, z_corners])
        bbox = np.dot(R, bbox)
        bbox = bbox + self.loc[:, np.newaxis]
        bbox = np.transpose(bbox)

        return bbox

    def to_str(self):
        print_str = '%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                    % (self.cls_type, self.truncation, self.occlusion, self.alpha, self.box2d, self.h, self.w, self.l,
                       self.loc, self.ry)
        return print_str

    def to_kitti_format(self):
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.cls_type, self.truncation, int(self.occlusion), self.alpha, self.box2d[0], self.box2d[1],
                       self.box2d[2], self.box2d[3], self.h, self.w, self.l, self.loc[0], self.loc[1], self.loc[2],
                       self.ry)
        return kitti_str

    def from_radar_to_camera(self, calib):
        loc_camera = np.dot(calib['T_from_radar_to_camera'][0:3, 0:3], np.transpose(self.loc))
        loc_camera += calib['T_from_radar_to_camera'][0:3, 3]
        self.loc_camera = np.transpose(loc_camera)

        T = quat_to_rot(self.orient)
        T = np.dot(calib['T_from_radar_to_camera'][:, 0:3], T)
        self.rot_camera = math.atan2(T[1, 0], T[0, 0])

    def from_radar_to_lidar(self, calib):
        loc_lidar = np.dot(calib['T_from_radar_to_lidar'][0:3, 0:3], np.transpose(self.loc))
        loc_lidar += calib['T_from_radar_to_lidar'][0:3, 3]
        self.loc_lidar = np.transpose(loc_lidar)

        T = quat_to_rot(self.orient)
        T = np.dot(calib['T_from_radar_to_lidar'][:, 0:3], T)
        self.rot_lidar = math.atan2(T[1, 0], T[0, 0])

    def from_lidar_to_camera(self, calib):
        loc_radar = np.dot(calib['T_from_lidar_to_radar'][0:3, 0:3], np.transpose(self.loc_lidar))
        loc_radar += calib['T_from_lidar_to_radar'][0:3, 3]
        self.loc = np.transpose(loc_radar)
        self.orient = rot_to_quat(self.rot_lidar, 0, 0)
        self.from_radar_to_camera(calib)


def rot_to_quat(yaw, pitch, roll):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [w, x, y, z]


def inv_trans(T):
    rotation = np.linalg.inv(T[0:3, 0:3])  # rotation matrix

    translation = T[0:3, 3]
    translation = -1 * np.dot(rotation, translation.T)
    translation = np.reshape(translation, (3, 1))
    Q = np.hstack((rotation, translation))
    return Q


def quat_to_rot(quat):
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
