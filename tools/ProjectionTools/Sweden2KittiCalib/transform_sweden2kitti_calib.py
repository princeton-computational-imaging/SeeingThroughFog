from tools.ProjectionTools.Sweden2KittiCalib.lib.utils import  export_as_kitti_calib
from tools.DatasetViewer.lib.read import load_calib_data
import argparse
import numpy as np

def parsArgs():
    parser = argparse.ArgumentParser(description='Calib projection tool')
    parser.add_argument('--root', '-r', help='Enter the root folder', default='../../DatasetViewer/calibs')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    """
    This tool enables you to transform the dense calibration files to match the kitti format
    """

    args = parsArgs()

    # Export Kitti Calibs for Velodyne HDL

    velodyne_to_cameraP2, cameraP2_to_velodyne, P2, R2, vtcP2, radar_to_cameraP2, zero_to_cameraP2 = load_calib_data(args.root, 'calib_cam_stereo_left.json', 'calib_tf_tree_full.json')
    velodyne_to_cameraP3, cameraP3_to_velodyne, P3, R3, vtcP3, radar_to_cameraP3, zero_to_cameraP3 = load_calib_data(args.root, 'calib_cam_stereo_right.json', 'calib_tf_tree_full.json')
    velodyne_to_cameraP0, cameraP0_to_velodyne, P0, R0, vtcP0, radar_to_cameraP0, zero_to_cameraP0 = load_calib_data(args.root, 'calib_gated_bwv.json', 'calib_tf_tree_full.json')


    #Export calib for stereo left camera as main camera
    export_as_kitti_calib(P0, P0, P2, P3, np.identity(3), velodyne_to_cameraP2, radar_to_cameraP2, 'kitti_stereo_velodynehdl_calib.txt')

    #Export calib for gated camera as main camera -> P0 and P2 are changed
    export_as_kitti_calib(P2, P3, P0, P0, np.identity(3), velodyne_to_cameraP0, radar_to_cameraP0, 'kitti_gated_velodynehdl_calib.txt')


    #Export Kitti Calibs for Velodyne VLP23

    vlp_to_cameraP2, cameraP2_to_vlp, P2, R2, vlptcP2, radar_to_cameraP2, zero_to_cameraP2 = load_calib_data(args.root, 'calib_cam_stereo_left.json', 'calib_tf_tree_full.json', velodyne_name='lidar_vlp32_roof')
    vlp_to_cameraP3, cameraP3_to_vlp, P3, R3, vlptcP3, radar_to_cameraP3, zero_to_cameraP3 = load_calib_data(args.root, 'calib_cam_stereo_right.json', 'calib_tf_tree_full.json', velodyne_name='lidar_vlp32_roof')
    vlp_to_cameraP0, cameraP0_to_vlp, P0, R0, vlptcP0, radar_to_cameraP0, zero_to_cameraP0 = load_calib_data(args.root, 'calib_gated_bwv.json', 'calib_tf_tree_full.json', velodyne_name='lidar_vlp32_roof')

    #Export calib for stereo left camera as main camera
    export_as_kitti_calib(P0, P0, P2, P3, np.identity(3), vlp_to_cameraP2, radar_to_cameraP2, 'kitti_stereo_velodynevlp_calib.txt')

    #Export calib for gated camera as main camera -> P0 and P2 are changed
    export_as_kitti_calib(P2, P3, P0, P0, np.identity(3), vlp_to_cameraP0, radar_to_cameraP0, 'kitti_gated_velodynevlp_calib.txt')


