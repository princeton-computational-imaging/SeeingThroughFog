from tools.DatasetViewer.lib.read import load_velodyne_scan
from tools.CreateTFRecords.generic_tf_tools.resize import resize
from tools.DatasetViewer.lib.read import load_calib_data
from tools.ProjectionTools.Lidar2RGB.lib.utils import transform_coordinates, filter_below_groundplane
# import cv2
import numpy as np
import os
import open3d as o3d

import matplotlib as mpl
import matplotlib.cm as cm

import argparse

def parsArgs():
    parser = argparse.ArgumentParser(description='Lidar 3d illustration tool')
    parser.add_argument('--root', '-r', help='Enter the root folder')
    parser.add_argument('--lidar_type', '-t', help='Enter the root folder', default='lidar_hdl64',
                        choices=['lidar_hdl64', 'lidar_vlp32'])
    parser.add_argument('--cmap', '-c', help='Illustration color map', default='jet')
    args = parser.parse_args()

    return args



interesting_samples = [
    '2018-02-06_14-25-51_00400',
    '2019-09-11_16-39-41_01770',
    '2018-02-12_07-16-32_00100',
    '2018-10-29_16-42-03_00560',
]

if __name__ == '__main__':
    """
    Illustrate Pointclouds in a 3D dimensional using open3D - Make sure install open3d: e.g. pip install open3d
    """
    args = parsArgs()

    velodyne_to_camera, camera_to_velodyne, P, R, vtc, radar_to_camera, zero_to_camera = load_calib_data(
        args.root, name_camera_calib='calib_cam_stereo_left.json', tf_tree='calib_tf_tree_full.json',
        velodyne_name='lidar_hdl64_s3_roof' if args.lidar_type == 'lidar_hdl64' else 'lidar_vlp32_roof')


    for interesting_sample in interesting_samples[:100]:
        velo_file_last = os.path.join(args.root, args.lidar_type + '_' + 'last',
                                      interesting_sample + '.bin')

        lidar_data_last = load_velodyne_scan(velo_file_last)
        lidar_data_last = filter_below_groundplane(lidar_data_last)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(lidar_data_last[:,:3])

        # Colorize Ring Value
        norm = mpl.colors.Normalize(vmin=0, vmax=64)

        cmap = None
        if args.cmap == 'hot':
            cmap = cm.hot
        elif args.cmap == 'cool':
            cmap = cm.cool
        elif args.cmap == 'jet':
            cmap = cm.jet
        else:
            print('Wrong color map specified')
            exit()
        m = cm.ScalarMappable(norm, cmap)
        transformed_pointcloud = transform_coordinates(lidar_data_last)

        depth_map_color = m.to_rgba(lidar_data_last[:,4])
        print(depth_map_color.shape)

        pcd.colors = o3d.utility.Vector3dVector(depth_map_color[:,:3])

        o3d.visualization.draw_geometries([pcd])

