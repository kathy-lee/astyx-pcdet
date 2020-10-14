import numpy as np
import json
import math
from . calibration_astyx import quat_to_rotation


def get_objects_from_label(label_file):
    with open(label_file, 'r') as f:
        data = json.load(f)
    objects = [Object3dAstyx(obj) for obj in data['objects']]
    return objects


class Object3dAstyx(object):
    def __init__(self, dict):
        self.src = dict
        self.cls_type = dict['classname'] if dict['classname']!='Person' else 'Pedestrian'
        cls_type_to_id = {
            'Bus': 0, 'Car': 1, 'Cyclist': 2, 'Motorcyclist': 3, 'Pedestrian': 4, 'Trailer': 5, 'Truck': 6,
            'Towed Object': 5, 'Other Vehicle': 5
        }
        self.cls_id = cls_type_to_id[self.cls_type]
        # self.truncation = float(label[1])
        self.occlusion = float(dict['occlusion'])# 0:fully visible 1:partly occluded 2:largely occluded 3:fully occluded
        # self.alpha = float(label[3])
        # self.box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)
        self.h = float(dict['dimension3d'][2])
        self.w = float(dict['dimension3d'][0])
        self.l = float(dict['dimension3d'][1])
        # self.loc = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
        self.loc = np.array(dict['center3d'])
        # self.dis_to_cam = np.linalg.norm(self.loc)
        # self.ry = float(label[14])
        self.orient = dict['orientation_quat']
        # self.score = float(label[15]) if label.__len__() == 16 else -1.0
        self.score = float(dict['score'])
        self.level_str = None
        self.level = self.get_astyx_obj_level()


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
        R = quat_to_rotation(self.orient)
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


    def convert_to_camera3d_obj(self, calib):
        loc_camera = np.dot(calib['T_from_radar_to_camera'][0:3, 0:3], np.transpose(self.loc))
        loc_camera += calib['T_from_radar_to_camera'][0:3, 3]
        self.loc_camera = np.transpose(loc_camera)

        T = quat_to_rotation(self.orient)
        T = np.dot(calib['T_from_radar_to_camera'][:, 0:3], T)
        self.rot_camera = math.atan2(T[1, 0], T[0, 0])


    def convert_to_lidar_obj(self, calib):
        loc_lidar = np.dot(calib['T_from_radar_to_lidar'][0:3, 0:3], np.transpose(self.loc))
        loc_lidar += calib['T_from_radar_to_lidar'][0:3, 3]
        self.loc_lidar = np.transpose(loc_lidar)

        T = quat_to_rotation(self.orient)
        T = np.dot(calib['T_from_radar_to_lidar'][:, 0:3], T)
        self.rot_lidar = math.atan2(T[1, 0], T[0, 0])