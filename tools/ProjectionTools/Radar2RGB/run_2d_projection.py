from tools.DatasetViewer.lib.read import load_radar_points
from tools.DatasetViewer.lib.read import load_calib_data
from tools.ProjectionTools.Lidar2RGB.lib.visi import plot_image_projection

# import cv2
import numpy as np

import os
import argparse


def parsArgs():
    parser = argparse.ArgumentParser(description='Radar 2d projection tool')
    parser.add_argument('--root', '-r', help='Enter the root folder')

    args = parser.parse_args()

    return args


interesting_samples = [
    '2018-02-06_14-25-51_00400',
    '2019-09-11_16-39-41_01770',
    '2018-02-12_07-16-32_00100',
    '2018-10-29_16-42-03_00560',
]

echos = [
    ['last', 'strongest'],
]

if __name__ == '__main__':

    args = parsArgs()

    velodyne_to_camera, camera_to_velodyne, P, R, vtc, radar_to_camera, zero_to_camera = load_calib_data(
        args.root, name_camera_calib='calib_cam_stereo_left.json', tf_tree='calib_tf_tree_full.json')

    for interesting_sample in interesting_samples:

        radar_file = os.path.join(args.root, 'radar_targets',
                                      interesting_sample + '.json')

        radar_data = load_radar_points(radar_file)

        plot_image_projection(radar_data, np.matmul(np.matmul(P, R), radar_to_camera), radar_to_camera, title='Camera Projection Radar')
