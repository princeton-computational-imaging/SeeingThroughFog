import glob
import os
import json
from pyquaternion import Quaternion
from sklearn.impute import SimpleImputer

import cv2
import numpy as np
from matplotlib import pyplot as plt

from tools.ProjectionTools.Gated2RGB.lib.camera_model import CameraModel
from tools.ProjectionTools.Gated2RGB.lib.data_loader import load_image
from scipy.interpolate import NearestNDInterpolator

def load_sweden_calib_data(tf_tree, target='cam_stereo_left_optical', source='bwv_cam_optical'):
    """
    :param path_total_sweden_dataset: Path to dataset root dir
    :param name_camera_calib: Camera calib file containing image Intrinsic
    :param tf_tree: TF (Tranformation) tree containing Translations from Velodyne to Cameras
    :return:
    """

    with open(os.path.join(tf_tree), 'r') as f:
        data_extrinsics = json.load(f)
    translations = []
    T_c = {}
    T_v = {}

    # Scan data extrinsics for transformation from lidar to camera
    Important_translations = ['bwv_cam_optical', 'cam_stereo_left_optical']
    for item in data_extrinsics:
        if item['child_frame_id'] in Important_translations:
            translations.append(item)
            if item['child_frame_id'] == target:
                T_c = item['transform']
            elif item['child_frame_id'] == source:
                T_v = item['transform']

    # print(T_c)
    # print(T_v)
    # Use pyquaternion to setup rotation matrix properly
    R_c = Quaternion(w=T_c['rotation']['w']*360/2/np.pi, x=T_c['rotation']['x']*360/2/np.pi, y=T_c['rotation']['y']*360/2/np.pi, z=T_c['rotation']['z']*360/2/np.pi)
    R_v = Quaternion(w=T_v['rotation']['w']*360/2/np.pi, x=T_v['rotation']['x']*360/2/np.pi, y=T_v['rotation']['y']*360/2/np.pi, z=T_v['rotation']['z']*360/2/np.pi)

    # Setup rotation Matrixes
    R_c_m = R_c.rotation_matrix
    R_v_m = R_v.rotation_matrix

    # Setup translation Vectors
    Tr_c = np.asarray([T_c['translation']['x'], T_c['translation']['y'], T_c['translation']['z']])
    Tr_v = np.asarray([T_v['translation']['x'], T_v['translation']['y'], T_v['translation']['z']])

    # Setup Translation Matrix camera to lidar -> ROS spans transformation from it children to its parents
    # Therefore 1 inversion step is needed for zero_to_camera. -> <parent_child>
    zero_to_camera = np.zeros((3,4))
    zero_to_velo = np.zeros((3,4))
    zero_to_camera[0:3, 0:3] = R_c_m
    zero_to_camera[0:3, 3] = Tr_c
    zero_to_camera = np.vstack((zero_to_camera, np.array([0, 0, 0, 1])))
    zero_to_velo[0:3, 0:3] = R_v_m
    zero_to_velo[0:3, 3] = Tr_v
    zero_to_velo = np.vstack((zero_to_velo, np.array([0, 0, 0, 1])))

    # calculate total exstrinsic trafo to camera
    mat44 = np.matmul(np.linalg.inv(zero_to_camera),zero_to_velo)


    return mat44

def disparity2depth_psm(disparity):
    baseline = 0.202993
    focal = 2355.722801
    depth = 250*np.ones(disparity.shape)
    # In SGM there are NAN values, as areas could not be calculated. The SimpleImputer interpolates those wholes.
    # Larger wholes in some cases can not be closed. Check this publication for better results: In Defense of Classical Image Processing: Fast Depth Completion on the CPU
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    disparity = imp.fit_transform(disparity)
    depth[disparity == 0] = 250
    depth[disparity != 0] = focal * baseline / disparity[disparity != 0]

    return np.clip(depth,0,250)

class ImageTransformer:

    def __init__(self, source_frame, target_frame, source_cam_file, target_cam_file, tf_file, source_cam_file_stereo=None):
        self.source_frame = source_frame
        self.target_frame = target_frame
        self.mat44 = load_sweden_calib_data(tf_file)
        self.mat44_inv = load_sweden_calib_data(tf_file, target='bwv_cam_optical', source='cam_stereo_left_optical')
        self.source_cam_model = CameraModel(source_cam_file)
        self.target_cam_model = CameraModel(target_cam_file)
        if source_cam_file_stereo:
            self.source_cam_model_stereo = CameraModel(source_cam_file_stereo)

    def transform_with_disparity(self, source_image, disparity, baseline):

        # only for SGM native
        # # interpolate
        # valid_disparity = np.logical_and(disparity != float('nan'), disparity != 0)
        # current_coordinates = np.where(valid_disparity)
        # missing_coordinates = np.where(~valid_disparity)
        # current_disparity = disparity[valid_disparity]
        # disparity_interpolator = NearestNDInterpolator(current_coordinates, current_disparity)
        # missing_disparity = disparity_interpolator(missing_coordinates[0], missing_coordinates[1])
        # disparity[~valid_disparity] = missing_disparity

        # position the disparity
        #disparity_full = np.zeros((1024, 1920))
        #disparity_full[70:70 + 824, :] = cv2.resize(disparity, (1920, 824))

        # disparity to depth
        depth = self.disparity2depth(disparity, baseline)

        return self.transform_with_depth(source_image, depth)

    def disparity2depth(self, disparity, baseline):
        depth = np.zeros(disparity.shape)
        depth[disparity == 0] = float('inf')
        depth[disparity != 0] = -baseline / disparity[disparity != 0]
        return depth


    def transform(self, source_image):
        depth = 100*np.ones((source_image.shape[0], source_image.shape[1]))
        return self.transform_with_depth(source_image, depth)

    def transform_with_stereo(self, source_image_left, source_image_right, baseline):
        imgL = source_image_left.copy()
        imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        imgL = self.source_cam_model.rectifyImage(imgL)

        imgR = source_image_right.copy()
        imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        imgR = self.source_cam_model_stereo.rectifyImage(imgR)

        stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
        disparity = stereo.compute(imgL, imgR)

        return self.transform_with_disparity(source_image_left, disparity, baseline)

    def transform_with_depth(self, source_image, depth):

        source_image = source_image.copy()
        # source image must be rectified. This is already handeled in tools/ProjectionTools/Gated2RGB/lib/warp_gatedimage.py
        # inside the WarpingClass. If you want to use this class somewhere else make sure to use:
        # source_image_rect = self.source_cam_model.rectifyImage(source_image)
        source_image_rect = source_image

        # Depth to 3D points
        pc, idx = self.source_cam_model.image2pointcloud(source_image_rect, depth)

        # change coordinate system
        points_stereo = np.r_[pc, np.ones((1, pc.shape[1]))]
        points_gated = np.matmul(self.mat44, points_stereo)

        # project 3D points into gated frame
        coordinates_hom = np.matmul(np.array(self.target_cam_model.P), points_gated)
        coordinates = coordinates_hom[:-1, :] / coordinates_hom[-1, :]
        coordinates = np.round(coordinates).astype(int)

        valid = np.logical_and(np.logical_and(coordinates[0] >= 0, coordinates[0] < self.target_cam_model.width),
                               np.logical_and(coordinates[1] >= 0, coordinates[1] < self.target_cam_model.height))

        out = np.zeros((self.target_cam_model.height*self.target_cam_model.width, 3), dtype=np.uint8)
        image = source_image_rect[idx[0].flatten(), idx[1].flatten(), :]

        coordinates = coordinates[:,valid]
        image = image[valid]

        out[coordinates[1]*self.target_cam_model.width + coordinates[0],:] = image

        out = out.reshape((self.target_cam_model.height, self.target_cam_model.width, 3))

        # #


        # # interpolate all pixels without color information
        # # outcomment if used for projecting with gated depth to camera coordinate system
        # valid_color = out[:,:] != [0,0,0]
        # current_coordinates = np.where(valid_color)
        # missing_coordinates = np.where(~valid_color)
        # current_color = out[valid_color]
        # #griddata()
        # #
        # color_interpolator = NearestNDInterpolator(current_coordinates, current_color)
        # missing_color = color_interpolator(missing_coordinates[0], missing_coordinates[1])
        # out[~valid_color] = missing_color

        return out

    def transform_with_target_depth(self, source_image, target_image, depth, vehicle_speed=0, delay=0, angle=0):

        source_image = source_image.copy()
        source_image_rect = source_image

        # Depth to 3D points
        pc, idx = self.target_cam_model.image2pointcloud(source_image_rect, depth)
        points_target = np.vstack((pc, np.ones((1,pc.shape[1]))))
        points_source = np.matmul(self.mat44_inv, points_target)
        points_source[2,:] = points_source[2,:] - np.cos(angle*np.pi/180)*vehicle_speed*delay
        points_source[1,:] = points_source[1,:] + np.sin(angle*np.pi/180)*vehicle_speed*delay


        # project 3D points into gated frame
        coordinates_hom_target = np.matmul(np.array(self.target_cam_model.P), points_target) # should be equal to idx
        coordinates_hom_source = np.matmul(np.array(self.source_cam_model.P), points_source)
        coordinates_t = coordinates_hom_target[:-1, :] / coordinates_hom_target[-1, :]
        coordinates_t = np.round(coordinates_t).astype(int)
        coordinates_s = coordinates_hom_source[:-1, :] / coordinates_hom_source[-1, :]
        coordinates_s = np.round(coordinates_s).astype(int)

        valid_t = np.logical_and(np.logical_and(coordinates_t[0] >= 0, coordinates_t[0] < self.target_cam_model.width),
                               np.logical_and(coordinates_t[1] >= 0, coordinates_t[1] < self.target_cam_model.height))
        valid_s = np.logical_and(np.logical_and(coordinates_s[0] >= 0, coordinates_s[0] < self.source_cam_model.width),
                               np.logical_and(coordinates_s[1] >= 0, coordinates_s[1] < self.source_cam_model.height))

        out = np.zeros((self.target_cam_model.height, self.target_cam_model.width, 3), dtype=source_image.dtype)

        coordinates_s = coordinates_s[:, valid_s]
        coordinates_t = coordinates_t[:, valid_s]
        out[coordinates_t[1], coordinates_t[0], :] = source_image_rect[coordinates_s[1].flatten(), coordinates_s[0].flatten(), :]

        return out



