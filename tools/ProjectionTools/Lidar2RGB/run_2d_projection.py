from tools.DatasetViewer.lib.read import load_velodyne_scan
from tools.DatasetViewer.lib.read import load_calib_data
from tools.ProjectionTools.Lidar2RGB.lib.utils import filter, \
    find_closest_neighbors, find_missing_points, transform_coordinates
from tools.ProjectionTools.Lidar2RGB.lib.visi import plot_spherical_scatter_plot, plot_image_projection
# import cv2
import numpy as np

import os
import argparse


def parsArgs():
    parser = argparse.ArgumentParser(description='Lidar 2d projection tool')
    parser.add_argument('--root', '-r', help='Enter the root folder')
    parser.add_argument('--lidar_type', '-t', help='Enter the root folder', default='lidar_hdl64',
                        choices=['lidar_hdl64', 'lidar_vlp32'])
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
        args.root, name_camera_calib='calib_cam_stereo_left.json', tf_tree='calib_tf_tree_full.json',
        velodyne_name='lidar_hdl64_s3_roof' if args.lidar_type == 'lidar_hdl64' else 'lidar_vlp32_roof')

    for interesting_sample in interesting_samples:
        velo_file_last = os.path.join(args.root, args.lidar_type + '_' + echos[0][0],
                                      interesting_sample + '.bin')
        velo_file_strongest = os.path.join(args.root, args.lidar_type + '_' + echos[0][1],
                                           interesting_sample + '.bin')
        lidar_data_last = load_velodyne_scan(velo_file_last)
        lidar_data_strongest = load_velodyne_scan(velo_file_strongest)

        print('last shape:', lidar_data_last.shape)
        lidar_data_last = filter(lidar_data_last, 1.5)
        print('strongest shape:', lidar_data_strongest.shape)
        lidar_data_strongest = filter(lidar_data_strongest, 1.5)

        remaining_last, remaining_strong = find_missing_points(lidar_data_last, lidar_data_strongest)
        valid = find_closest_neighbors(transform_coordinates(remaining_strong), transform_coordinates(remaining_last))

        plot_spherical_scatter_plot(lidar_data_last, pattern='hot', title='Spherical Plot Last Echo')
        plot_spherical_scatter_plot(lidar_data_strongest, pattern='cool', title='Spherical Plot Strongest Echo')

        print(len(remaining_strong), '/', len(lidar_data_strongest), len(remaining_last), '/', len(lidar_data_last))
        print("intensity_mean", np.mean(lidar_data_strongest[:, 3]), np.mean(lidar_data_last[:, 3]))
        print("intensity_mean", np.mean(remaining_strong[:, 3]), np.mean(remaining_last[:, 3]))

        plot_spherical_scatter_plot(remaining_last, pattern='hot', plot_show=False)
        plot_spherical_scatter_plot(remaining_strong, pattern='cool', title='Not matching echos')

        plot_image_projection(lidar_data_last, vtc, velodyne_to_camera, title='Camera Projection Last Echo')
        plot_image_projection(lidar_data_strongest, vtc, velodyne_to_camera, title='Camera Projection Strongest Echo')
