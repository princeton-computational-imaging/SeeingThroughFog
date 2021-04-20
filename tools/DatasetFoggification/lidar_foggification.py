import os
import sys
import math
import time
import socket
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from lib.SeeingThroughFog.tools.DatasetFoggification.beta_modification import BetaRadomization

if os.name == 'posix' and "DISPLAY" not in os.environ:
    headless_server = True
else:
    headless_server = False
    import pyqtgraph.opengl as gl
    from PyQt5.QtCore import Qt
    from pyqtgraph.Qt import QtGui
    from PyQt5.QtWidgets import QApplication, QDesktopWidget, QWidget, QGridLayout, QLabel

SUPPORTED_SENSORS = ['Velodyne HDL-64E S2',
                     'Velodyne HDL-64E S3D']



def load_lidar_scan(file, n_features=5):
    """Load and parse a lidar binary file. According to Kitti Dataset"""

    assert n_features >= 4, 'points must have at least 4 features (e.g. x, y, z, reflectance)'
    scan = np.fromfile(file, dtype=np.float32)

    return scan.reshape((-1, n_features))[:,0:4]


def parsArgs():

    parser = argparse.ArgumentParser(description='LiDAR foggification')

    parser.add_argument('-r', '--root', help='root folder of dataset', default=str(Path.home() / 'datasets/DENSE/SeeingThroughFog'))
    parser.add_argument('-l', '--lidar_folder', help='relative path to LiDAR data', default='lidar_hdl64_strongest')
    parser.add_argument('-f', '--n_features', help='number of point features', type=int, default=5)
    parser.add_argument('-b', '--beta', type=float, help='fogdensity parameter', default=0.05)
    parser.add_argument('-v', '--visualize', type=bool, help='visualization mode', default=True)
    parser.add_argument('-m', '--monitor', help='display number for visualization', type=int, default=1)
    parser.add_argument('-n', '--normalize', type=bool, help='if reflectance in [0, 255] => set to True', default=True)
    parser.add_argument('-p', '--param_set', help='specify which parameter set', default='DENSE')
    parser.add_argument('--fraction_random', type=float, default=0.05,  # fraction of 0.05 was found empirically
                        help ='fraction of random scattered points')
    parser.add_argument('-s', '--sensor_type', type=str, default='Velodyne HDL-64E S3D',
                        help='sensor type [see SUPPORTED_SENSORS]')
    parser.add_argument('--seed', help='random seed', default=None)
    parser.add_argument('--scale', type=float, help='scale for image size', default=2.0)

    arguments = parser.parse_args()

    assert arguments.sensor_type in SUPPORTED_SENSORS, 'LiDAR sensor not supported yet'

    return arguments


def haze_point_cloud(pts_3D, beta_radomization, arguments):
    # foggyfication should be applied to sequences to ensure time correlation inbetween frames

    n, g, dmin = None, None, None

    # print('minmax_values\n',
    #       min(pts_3D[:, 0]), max(pts_3D[:, 0]), '\n',
    #       min(pts_3D[:, 1]), max(pts_3D[:, 1]), '\n',
    #       min(pts_3D[:, 2]), max(pts_3D[:, 2]))

    '''
    the noise-level n specifies the minimum reflectance value that can be detected by the sensor
    the gain g specifies the (adaptive) amount of laser power that gets emitted by the sensor
    '''

    if arguments.sensor_type== 'Velodyne HDL-64E S3D':
        n = 0.04    # noise-level
        g = 0.45    # gain
        dmin = 2    # minimal detectable distance

    elif arguments.sensor_type== 'Velodyne HDL-64E S2':
        n = 0.05    # noise-level
        g = 0.35    # gain
        dmin = 2    # minimal detectable distance

    d = np.sqrt(pts_3D[:,0] * pts_3D[:,0] + pts_3D[:,1] * pts_3D[:,1] + pts_3D[:,2] * pts_3D[:,2])
    detectable_points = np.where(d>dmin)
    d = d[detectable_points]
    pts_3D = pts_3D[detectable_points]

    beta = beta_radomization.get_beta(pts_3D[:, 0], pts_3D[:, 1], pts_3D[:, 2])

    d_max = -np.divide(np.log(np.divide(n,(pts_3D[:,3] + g))),(2 * beta))
    d_new = -np.log(1 - 0.5) / beta

    probability_lost = 1 - np.exp(-beta*d_max)
    lost = np.random.uniform(0, 1, size=probability_lost.shape) < probability_lost

    if beta_radomization.beta == 0.0:

        pts_3d_augmented = np.zeros((pts_3D.shape[0], pts_3D.shape[1]+1))
        pts_3d_augmented[:, 0:4] = pts_3D
        pts_3d_augmented[:, -1] = np.zeros(np.shape(pts_3D[:, 3]))

        return pts_3d_augmented,  []

    cloud_scatter = np.logical_and(d_new < d, np.logical_not(lost))
    random_scatter = np.logical_and(np.logical_not(cloud_scatter), np.logical_not(lost))

    idx_stable = np.where(d<d_max)[0]

    old_points = np.zeros((len(idx_stable), pts_3D.shape[1]+1))
    old_points[:,0:-1] = pts_3D[idx_stable, :]
    old_points[:,3] = old_points[:,3]*np.exp(-beta[idx_stable]*d[idx_stable])
    old_points[:, -1] = np.zeros(np.shape(old_points[:,3]))

    cloud_scatter_idx = np.where(np.logical_and(d_max<d, cloud_scatter))[0]

    cloud_scatter = np.zeros((len(cloud_scatter_idx), pts_3D.shape[1]+1))
    cloud_scatter[:,0:-1] =  pts_3D[cloud_scatter_idx, :]
    cloud_scatter[:,0:3] = (cloud_scatter[:,0:3].T * d_new[cloud_scatter_idx] / d[cloud_scatter_idx]).T
    cloud_scatter[:,3] = cloud_scatter[:,3]*np.exp(-beta[cloud_scatter_idx]*d_new[cloud_scatter_idx])
    cloud_scatter[:, -1] = np.ones(np.shape(cloud_scatter[:, 3]))

    # Subsample random scatter abhaengig vom noise im Lidar
    random_scatter_idx = np.where(random_scatter)[0]

    # scatter outside min detection range
    scatter_max = np.min(np.stack((d_max, d)).transpose(), axis=1)
    d_rand = np.random.uniform(high=scatter_max[random_scatter_idx])
    drand_idx = np.where(d_rand>dmin)
    d_rand = d_rand[drand_idx]
    random_scatter_idx = random_scatter_idx[drand_idx]

    # not all points are getting randomly scattered, subsample fraction given in arguments
    subsampled_idx = np.random.choice(len(random_scatter_idx), int(arguments.fraction_random * len(random_scatter_idx)),
                                      replace=False)
    d_rand = d_rand[subsampled_idx]
    random_scatter_idx = random_scatter_idx[subsampled_idx]

    random_scatter = np.zeros((len(random_scatter_idx), pts_3D.shape[1]+1))
    random_scatter[:,0:-1] = pts_3D[random_scatter_idx, :]
    random_scatter[:,0:3] = (random_scatter[:,0:3].T * d_rand / d[random_scatter_idx]).T
    random_scatter[:,3] = random_scatter[:,3]*np.exp(-beta[random_scatter_idx]*d_rand)
    random_scatter[:, -1] = 2*np.ones(np.shape(random_scatter[:, 3]))

    pts_3d_augmented = np.concatenate((old_points, cloud_scatter, random_scatter), axis=0)

    return pts_3d_augmented


def add_random_noise(lidar_scan):
    random_noise = np.random.normal(0.0, 5, np.shape(lidar_scan))
    lidar_scan = lidar_scan + random_noise
    return  lidar_scan


def set_color(dist_pts_3d):
    color = []
    for pts in dist_pts_3d:
        if pts[4] == 0:
            color.append([0, 255, 255, 1])  # cyan
        elif pts[4] == 1:
            color.append([255, 0, 0, 1])    # red
        else: # pts[4] == 2:
            color.append([255, 255, 0, 1])  # yellow

    return np.asarray(color)



def main(walk_path, dest_path, arguments):

    visualization_mode = arguments.visualize

    files_all = []
    layout = None

    for root, dirs, files in os.walk(walk_path, followlinks=True):
        assert(root==walk_path)
        files_all = sorted(files)

    if visualization_mode:

        app = QApplication(sys.argv)
        monitor = QDesktopWidget().screenGeometry(arguments.monitor)

        main_widget = QWidget()
        main_widget.move(monitor.left(), monitor.top())

        widget_01 = gl.GLViewWidget()
        widget_02 = gl.GLViewWidget()
        widget_03 = gl.GLViewWidget()
        widget_04 = gl.GLViewWidget()

        widgets = [widget_01, widget_02, widget_03, widget_04]

        for screen in app.screens():

            print('')
            print('Screen: %s' % screen.name())
            size = screen.size()
            print('Size: %d x %d' % (size.width(), size.height()))
            rect = screen.availableGeometry()
            print('Available: %d x %d' % (rect.width(), rect.height()))

        layout = QGridLayout()

        layout.addWidget(widget_01, 1, 0)
        layout.addWidget(widget_02, 1, 1)
        layout.addWidget(widget_03, 2, 0)
        layout.addWidget(widget_04, 2, 1)

        pos = QtGui.QVector3D(0, 0, 0)

        bev_distance = 200
        bev_azimuth = 180
        bev_elevation = 90

        ego_distance = 2
        ego_azimuth = 180
        ego_elevation = 10

        # near_distance = 35
        # near_azimuth = 180
        # near_elevation = 40
        #
        # far_distance = 86.8051380062
        # far_azimuth = 180
        # far_elevation = 40

        # distance = 86.8051380062
        # azimuth = 273
        # elevation = -59

        widget_01.setCameraPosition(pos=pos, distance=ego_distance, azimuth=ego_azimuth, elevation=ego_elevation)
        widget_02.setCameraPosition(pos=pos, distance=ego_distance, azimuth=ego_azimuth, elevation=ego_elevation)

        widget_03.setCameraPosition(pos=pos, distance=bev_distance, azimuth=bev_azimuth, elevation=bev_elevation)
        widget_04.setCameraPosition(pos=pos, distance=bev_distance, azimuth=bev_azimuth, elevation=bev_elevation)

    iterator = range(len(files_all))

    for i in (iterator if visualization_mode else tqdm(iterator)):

        B = BetaRadomization(arguments.beta, arguments.seed, param_set=arguments.param_set)
        B.propagate_in_time(10)

        current_file = files_all[i]

        current_img_path = os.path.join(walk_path.replace('velodyne', 'image_2'),
                                        current_file.replace('bin', 'png'))

        current_foggy_img_path = current_img_path.replace('image_2',
                                                          f'image_2_DENSE_TRI_beta_{arguments.beta:.3f}')

        if visualization_mode:

            scale = arguments.scale

            pixmap = QtGui.QPixmap(current_img_path)
            pixmap = pixmap.scaled(int(pixmap.width() / scale), int(pixmap.height() / scale))

            pixmap_foggy = QtGui.QPixmap(current_foggy_img_path)
            pixmap_foggy = pixmap_foggy.scaled(int(pixmap_foggy.width() / scale), int(pixmap_foggy.height() / scale))

            image_widget = QLabel()
            image_widget.setPixmap(pixmap)

            foggy_image_widget = QLabel()
            foggy_image_widget.setPixmap(pixmap_foggy)

            layout.addWidget(image_widget, 0, 0, Qt.AlignHCenter)
            layout.addWidget(foggy_image_widget, 0, 1, Qt.AlignHCenter)

            main_widget.setLayout(layout)

        start_loading = time.time()
        lidar_scan = load_lidar_scan(os.path.join(walk_path, current_file), n_features=arguments.n_features)
        end_loading = time.time()

        if arguments.normalize:
            lidar_scan[:,3] = lidar_scan[:,3]/255

        start = time.time()
        pts_3d_augmented = haze_point_cloud(lidar_scan, B, arguments)
        end = time.time()

        pts_3d = np.zeros((lidar_scan.shape[0], 5))
        pts_3d[:, 0:4] = lidar_scan

        pts_3d_display = [pts_3d, pts_3d_augmented, pts_3d, pts_3d_augmented]

        if visualization_mode:

            print('')
            print(current_file)
            print(f'loading time: {math.ceil(end_loading - start_loading)}s')

            plots = []

            for j in range(len(widgets)):
                color = set_color(pts_3d_display[j])
                plot = gl.GLScatterPlotItem(pos=pts_3d_display[j][:, 0:3], size=3, color=color)
                widgets[j].addItem(plot)

                plots.append(plot)

                # g = gl.GLGridItem()
                # widgets[j].addItem(g)

            main_widget.setWindowTitle(f'{current_file} - beta {arguments.beta} - {arguments.param_set}')
            main_widget.update()
            main_widget.show()
            main_widget.showMaximized()

            app.exec_()

            # save_processed_image
            # widget.grabFrameBuffer().save(f'{current_file.split('.')[0]}.png')

            for k in range(len(widgets)):
                widgets[k].removeItem(plots[k])

            print(f'augmentation time:  {(end - start) * 1000:.0f}ms')
            print(f'distance:           {widget_01.opts["distance"]:.0f}')
            print(f'azimuth:            {widget_01.opts["azimuth"]:.0f}')
            print(f'elevation:          {widget_01.opts["elevation"]:.0f}')

        else:

            lidar_save_path = os.path.join(dest_path, current_file)
            pts_3d_augmented.astype(np.float32).tofile(lidar_save_path)

        #Update position in time
        B.propagate_in_time(5)



if __name__ == '__main__':

    args = parsArgs()
    args.seed = 0
    args.visualize = False
    # args.param_set = 'CVL'

    hostname = socket.gethostname()

    if 'MacBook' in hostname or '.ee.ethz.ch' in hostname:
        args.scale = 1.8
    else:                                                                       # assume CVL host
        args.scale = 1.325

    betas = [0.10, 0.12, 0.15, 0.2]
    betas.reverse()

    lidar_folders = ['lidar_hdl64_strongest', 'lidar_hdl64_last']

    for lidar_folder in lidar_folders:

        args.lidar_folder = lidar_folder

        for b in betas:

            args.beta = b

            src_folder = os.path.join(args.root, args.lidar_folder)
            dst_folder = f'{src_folder}_DENSE_beta_{args.beta:.3f}'

            print('')
            print(f'beta {b}')
            print('')
            print(f'{args.param_set} parameter set')
            print('')
            print(f'searching for point clouds in    {src_folder}')

            if not args.visualize:
                print(f'saving augmented point clouds to {dst_folder}')

            if not os.path.exists(dst_folder):
                os.makedirs(dst_folder)

            main(src_folder, dst_folder, arguments=args)