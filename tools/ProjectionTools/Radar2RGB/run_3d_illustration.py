from tools.DatasetViewer.lib.read import load_radar_points
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
    parser = argparse.ArgumentParser(description='Radar 3d illustration tool')
    parser.add_argument('--root', '-r', help='Enter the root folder')
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
        args.root, name_camera_calib='calib_cam_stereo_left.json', tf_tree='calib_tf_tree_full.json')


    for interesting_sample in interesting_samples[:100]:
        radar_file = os.path.join(args.root, 'radar_targets',
                                      interesting_sample + '.json')

        radar_data = load_radar_points(radar_file)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(radar_data[:,:3])

        # Colorize Doppler Speed
        norm = mpl.colors.Normalize(vmin=-50, vmax=+50)

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

        depth_map_color = m.to_rgba(radar_data[:,4])
        print(depth_map_color.shape)

        pcd.colors = o3d.utility.Vector3dVector(depth_map_color[:,:3])

        o3d.visualization.draw_geometries([pcd])

