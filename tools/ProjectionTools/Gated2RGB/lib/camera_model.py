import numpy as np
import json
import os
import cv2

class CameraInfo:
    pass

class CameraModel:

    def __init__(self, file_path):
        if os.path.splitext(file_path)[1] == '.ini':
            self.from_ini_file(file_path)
        elif os.path.splitext(file_path)[1] == '.json':
            self.from_cam_file(file_path)


    def from_ini_file(self, ini_file_path):
        data = {}
        data['roi'] = {
            "width": 0,
            "do_rectify": False,
            "y_offset": 0,
            "x_offset": 0,
            "height": 0
        }
        data['header'] = {}
        data['binning_y'] = 0
        data['binning_x'] = 0
        data['distortion_model'] = 'plumb_bob'

        with open(ini_file_path) as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            if 'width' in line:
                data['width'] = int(lines[i + 1])

            if 'height' in line:
                data['height'] = int(lines[i + 1])

            if 'camera matrix' in line:
                string = lines[i + 1][:-1] + lines[i + 2][:-1] + lines[i + 3][:-1]
                nums = np.fromstring(string, dtype=float, sep=' ')
                data['K'] = []
                for num in nums:
                    data['K'].append(num)

            if 'distortion' in line:
                string = lines[i + 1][:-1] + lines[i + 2][:-1] + lines[i + 3][:-1]
                nums = np.fromstring(string, dtype=float, sep=' ')
                data['D'] = []
                for num in nums:
                    data['D'].append(num)

            if 'rectification' in line:
                string = lines[i + 1][:-1] + lines[i + 2][:-1] + lines[i + 3][:-1]
                nums = np.fromstring(string, dtype=float, sep=' ')
                data['R'] = []
                for num in nums:
                    data['R'].append(num)

            if 'projection' in line:
                string = lines[i + 1][:-1] + lines[i + 2][:-1] + lines[i + 3][:-1]
                nums = np.fromstring(string, dtype=float, sep=' ')
                data['P'] = []
                for num in nums:
                    data['P'].append(num)
        self.from_dict(data)

    def from_cam_file(self, cam_file_path):
        with open(cam_file_path) as f:
            data = json.load(f)
        self.from_dict(data)

    def from_dict(self, dict):
        self.K = np.array(dict['K']).reshape((3, 3))
        self.P = np.array(dict['P']).reshape((3, 4))
        self.R = np.array(dict['R']).reshape((3, 3))
        self.D = np.array(dict['D'])
        self.height = int(dict['height'])
        self.width = int(dict['width'])
        self.rect_mat44 = np.identity(4)
        self.rect_mat44[0:3, 0:3] = self.R

    def rectifyImage(self, image):
        # taken from image_geometry.PinholeCameraModel (ROS)
        mapx = np.ndarray(shape=(self.height, self.width, 1), dtype='float32')
        mapy = np.ndarray(shape=(self.height, self.width, 1), dtype='float32')
        cv2.initUndistortRectifyMap(self.K, self.D, self.R, self.P, (self.width, self.height), cv2.CV_32FC1, mapx, mapy)
        image_rect = cv2.remap(image, mapx, mapy, cv2.INTER_CUBIC)

        return image_rect

    def image2pointcloud(self, image_rect, depth):

        # Depth to 3D points
        idx = np.indices((self.height, self.width))
        index_matrix = np.vstack((idx[1].flatten(), idx[0].flatten(),np.ones((self.height * self.width,))))
        inv_proj_mat = np.linalg.inv(np.array(self.P[:, 0:3]))
        # print(self.height, self.width)
        pc = np.multiply(np.dot(inv_proj_mat, index_matrix), depth.flatten())

        # pc = np.r_[pc, np.ones((1, pc.shape[1]))]
        # pc = np.dot(np.transpose(self.rect_mat44), pc)
        # pc = pc[0:3, :]

        # points_stereo = np.multiply(np.matmul(inv_proj_mat, index_matrix), depth.flatten())


        # x = (uv[0] - self.cx()) / self.fx()
        # y = (uv[1] - self.cy()) / self.fy()
        # norm = math.sqrt(x * x + y * y + 1)
        # x /= norm
        # y /= norm
        # z = 1.0 / norm
        # return (x, y, z)

        return pc, idx