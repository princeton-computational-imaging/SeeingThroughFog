import os
import sys
import time
import socket
import argparse
import numpy as np
import pyqtgraph.opengl as gl

from PyQt5.QtCore import Qt
from pyqtgraph.Qt import QtGui
from PyQt5.QtWidgets import QApplication, QDesktopWidget, QWidget, QGridLayout, QLabel
from tools.DatasetFoggification.beta_modification import BetaRadomization

SUPPORTED_SENSORS = ['Velodyne HDL-64E S2',
                     'Velodyne HDL-64E S3D']



def load_lidar_scan(file, n_features=5):
    """Load and parse a lidar binary file. According to Kitti Dataset"""

    assert n_features >= 4, 'points must have at least 4 features (e.g. x, y, z, reflectance)'
    scan = np.fromfile(file, dtype=np.float32)

    # if n_features == 5
    #     scan = scan.reshape((-1, n_features))
    #     scan = scan[scan[:, 1].argsort()]       # sort by channel

    return scan.reshape((-1, n_features))[:,0:4]


def parsArgs():

    parser = argparse.ArgumentParser(description='LiDAR foggification')

    parser.add_argument('-r', '--root', help='root folder of dataset', default=None)
    parser.add_argument('-l', '--lidar_folder', help='relative path to LiDAR data', default=None)
    parser.add_argument('-f', '--n_features', help='number of point features', type=int, default=4)
    parser.add_argument('-n', '--normalize', type=bool, help='if reflectance in [0, 255] => set to True', default=False)
    parser.add_argument('--fraction_random', type=float, default=0.05,  # fraction of 0.05 found empirically
                        help ='fraction of random scattered points')
    parser.add_argument('-s', '--sensor_type', type=str, default='Velodyne HDL-64E S2',
                        help='sensor type [see SUPPORTED_SENSORS]')
    parser.add_argument('--seed', help='random seed', default=None)
    parser.add_argument('--noise', help='whether to add noise or not', default=True)
    parser.add_argument('--scale', type=float, help='scale for image size', default=2.0)
    parser.add_argument('--distance', type=float, help='viewpoint parameter', default=35.0)
    parser.add_argument('--azimuth', type=float, help='viewpoint parameter', default=180.0)
    parser.add_argument('--elevation', type=float, help='viewpoint parameter', default=40.0)
    parser.add_argument('--first_lidar_monitor', help='display number for LiDAR visualization', type=int, default=0)
    parser.add_argument('--second_lidar_monitor', help='display number for LiDAR visualization', type=int, default=1)
    parser.add_argument('--image_monitor', help='display number for image visualization', type=int, default=2)
    parser.add_argument('--widescreen', help='specify if image_monitor is widescreen or not', type=bool, default=True)

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

    if arguments.sensor_type== 'Velodyne HDL-64E S3D':
        n = 0.04
        g = 0.45
        dmin = 2 # Minimal detectable distance

    elif arguments.sensor_type== 'Velodyne HDL-64E S2':
        n = 0.05
        g = 0.35
        dmin = 2

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

        pts_3d_augmented = np.zeros((pts_3D.shape[0], 5))
        pts_3d_augmented[:, 0:4] = pts_3D
        pts_3d_augmented[:, 4] = np.zeros(np.shape(pts_3D[:, 3]))

        return pts_3d_augmented,  []

    cloud_scatter = np.logical_and(d_new < d, np.logical_not(lost))
    random_scatter = np.logical_and(np.logical_not(cloud_scatter), np.logical_not(lost))

    idx_stable = np.where(d<d_max)[0]

    old_points = np.zeros((len(idx_stable), 5))
    old_points[:,0:4] = pts_3D[idx_stable,:]
    old_points[:,3] = old_points[:,3]*np.exp(-beta[idx_stable]*d[idx_stable])
    old_points[:, 4] = np.zeros(np.shape(old_points[:,3]))

    cloud_scatter_idx = np.where(np.logical_and(d_max<d, cloud_scatter))[0]

    cloud_scatter = np.zeros((len(cloud_scatter_idx), 5))
    cloud_scatter[:,0:4] =  pts_3D[cloud_scatter_idx,:]
    cloud_scatter[:,0:3] = (cloud_scatter[:,0:3].T * d_new[cloud_scatter_idx] / d[cloud_scatter_idx]).T
    cloud_scatter[:,3] = cloud_scatter[:,3]*np.exp(-beta[cloud_scatter_idx]*d_new[cloud_scatter_idx])
    cloud_scatter[:, 4] = np.ones(np.shape(cloud_scatter[:, 3]))

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

    random_scatter = np.zeros((len(random_scatter_idx), 5))
    random_scatter[:,0:4] = pts_3D[random_scatter_idx,:]
    random_scatter[:,0:3] = (random_scatter[:,0:3].T * d_rand / d[random_scatter_idx]).T
    random_scatter[:,3] = random_scatter[:,3]*np.exp(-beta[random_scatter_idx]*d_rand)
    random_scatter[:, 4] = 2*np.ones(np.shape(random_scatter[:, 3]))

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


def main(walk_path, arguments):

    all_files = []

    app = QApplication(sys.argv)

    image_widget = QWidget()
    first_lidar_widget = QWidget()
    second_lidar_widget = QWidget()

    first_lidar_widget_01 = gl.GLViewWidget()
    first_lidar_widget_02 = gl.GLViewWidget()
    first_lidar_widget_03 = gl.GLViewWidget()
    first_lidar_widget_04 = gl.GLViewWidget()
    first_lidar_widget_05 = gl.GLViewWidget()
    first_lidar_widget_06 = gl.GLViewWidget()

    second_lidar_widget_01 = gl.GLViewWidget()
    second_lidar_widget_02 = gl.GLViewWidget()
    second_lidar_widget_03 = gl.GLViewWidget()
    second_lidar_widget_04 = gl.GLViewWidget()
    second_lidar_widget_05 = gl.GLViewWidget()
    second_lidar_widget_06 = gl.GLViewWidget()

    first_lidar_widgets = [first_lidar_widget_01,
                           first_lidar_widget_02,
                           first_lidar_widget_03,
                           first_lidar_widget_04,
                           first_lidar_widget_05,
                           first_lidar_widget_06]

    second_lidar_widgets = [second_lidar_widget_01,
                            second_lidar_widget_02,
                            second_lidar_widget_03,
                            second_lidar_widget_04,
                            second_lidar_widget_05,
                            second_lidar_widget_06]

    for root, dirs, files in os.walk(walk_path, followlinks=True):

        assert(root==walk_path)
        all_files = sorted(files)

    for screen in app.screens():

        print('')
        print('Screen: %s' % screen.name())
        size = screen.size()
        print('Size: %d x %d' % (size.width(), size.height()))
        rect = screen.availableGeometry()
        print('Available: %d x %d' % (rect.width(), rect.height()))

    image_layout = QGridLayout()
    first_lidar_layout = QGridLayout()
    second_lidar_layout = QGridLayout()

    first_lidar_layout.addWidget(first_lidar_widget_01, 0, 0)
    first_lidar_layout.addWidget(first_lidar_widget_02, 0, 1)
    first_lidar_layout.addWidget(first_lidar_widget_03, 0, 2)
    first_lidar_layout.addWidget(first_lidar_widget_04, 1, 0)
    first_lidar_layout.addWidget(first_lidar_widget_05, 1, 1)
    first_lidar_layout.addWidget(first_lidar_widget_06, 1, 2)

    second_lidar_layout.addWidget(second_lidar_widget_01, 0, 0)
    second_lidar_layout.addWidget(second_lidar_widget_02, 0, 1)
    second_lidar_layout.addWidget(second_lidar_widget_03, 0, 2)
    second_lidar_layout.addWidget(second_lidar_widget_04, 1, 0)
    second_lidar_layout.addWidget(second_lidar_widget_05, 1, 1)
    second_lidar_layout.addWidget(second_lidar_widget_06, 1, 2)

    first_lidar_widget.setLayout(first_lidar_layout)
    second_lidar_widget.setLayout(second_lidar_layout)

    iterator = range(len(all_files))

    betas = [0.005, 0.01, 0.02, 0.03, 0.06]

    for i in iterator:

        if i % 2:   # only show uneven samples

            B = BetaRadomization(betas[0], arguments.seed, param_set='DENSE')
            C = BetaRadomization(betas[1], arguments.seed, param_set='DENSE')
            D = BetaRadomization(betas[2], arguments.seed, param_set='DENSE')
            E = BetaRadomization(betas[3], arguments.seed, param_set='DENSE')
            F = BetaRadomization(betas[4], arguments.seed, param_set='DENSE')

            G = BetaRadomization(betas[0], arguments.seed, param_set='DENSE_no_noise')
            H = BetaRadomization(betas[1], arguments.seed, param_set='DENSE_no_noise')
            I = BetaRadomization(betas[2], arguments.seed, param_set='DENSE_no_noise')
            J = BetaRadomization(betas[3], arguments.seed, param_set='DENSE_no_noise')
            K = BetaRadomization(betas[4], arguments.seed, param_set='DENSE_no_noise')

            dense_randomizations = [B, C, D, E, F]
            cvl_randomizations = [G, H, I, J, K]

            for A in dense_randomizations:
                A.propagate_in_time(10)

            for A in cvl_randomizations:
                A.propagate_in_time(10)

            current_file = all_files[i]

            img_path_01 = os.path.join(walk_path.replace('velodyne', 'image_2'), current_file.replace('bin', 'png'))
            img_path_02 = img_path_01.replace('image_2', f'image_2_DENSE_TRI_beta_{betas[0]:.3f}')
            img_path_03 = img_path_01.replace('image_2', f'image_2_DENSE_TRI_beta_{betas[1]:.3f}')
            img_path_04 = img_path_01.replace('image_2', f'image_2_DENSE_TRI_beta_{betas[2]:.3f}')
            img_path_05 = img_path_01.replace('image_2', f'image_2_DENSE_TRI_beta_{betas[3]:.3f}')
            img_path_06 = img_path_01.replace('image_2', f'image_2_DENSE_TRI_beta_{betas[4]:.3f}')

            img_paths = [img_path_01, img_path_02, img_path_03, img_path_04, img_path_05, img_path_06]
            img_widgets = []

            scale = arguments.scale

            for img_path in img_paths:

                pix_map = QtGui.QPixmap(img_path)
                pix_map = pix_map.scaled(int(pix_map.width() / scale), int(pix_map.height() / scale))

                img_widget = QLabel()
                img_widget.setPixmap(pix_map)

                img_widgets.append(img_widget)

            if arguments.widescreen:

                image_layout.addWidget(img_widgets[0], 0, 0, Qt.AlignHCenter)
                image_layout.addWidget(img_widgets[1], 0, 1, Qt.AlignHCenter)
                image_layout.addWidget(img_widgets[2], 1, 0, Qt.AlignHCenter)
                image_layout.addWidget(img_widgets[3], 1, 1, Qt.AlignHCenter)
                image_layout.addWidget(img_widgets[4], 2, 0, Qt.AlignHCenter)
                image_layout.addWidget(img_widgets[5], 2, 1, Qt.AlignHCenter)

            else:

                image_layout.addWidget(img_widgets[0], 0, 0, Qt.AlignHCenter)
                image_layout.addWidget(img_widgets[1], 1, 0, Qt.AlignHCenter)
                image_layout.addWidget(img_widgets[2], 2, 0, Qt.AlignHCenter)
                image_layout.addWidget(img_widgets[3], 3, 0, Qt.AlignHCenter)
                image_layout.addWidget(img_widgets[4], 4, 0, Qt.AlignHCenter)
                image_layout.addWidget(img_widgets[5], 5, 0, Qt.AlignHCenter)

            first_lidar_widget.setLayout(first_lidar_layout)
            image_widget.setLayout(image_layout)

            start_loading = time.time()
            lidar_scan = load_lidar_scan(os.path.join(walk_path, current_file), n_features=arguments.n_features)
            if arguments.normalize:
                lidar_scan[:,3] = lidar_scan[:,3]/255
            end_loading = time.time()

            start = time.time()
            dense_augmented_01 = haze_point_cloud(lidar_scan, B, arguments)
            end = time.time()

            dense_augmented_02 = haze_point_cloud(lidar_scan, C, arguments)
            dense_augmented_03 = haze_point_cloud(lidar_scan, D, arguments)
            dense_augmented_04 = haze_point_cloud(lidar_scan, E, arguments)
            dense_augmented_05 = haze_point_cloud(lidar_scan, F, arguments)

            cvl_augmented_01 = haze_point_cloud(lidar_scan, G, arguments)
            cvl_augmented_02 = haze_point_cloud(lidar_scan, H, arguments)
            cvl_augmented_03 = haze_point_cloud(lidar_scan, I, arguments)
            cvl_augmented_04 = haze_point_cloud(lidar_scan, J, arguments)
            cvl_augmented_05 = haze_point_cloud(lidar_scan, K, arguments)

            print('\nDENSE')

            for l in range(len(dense_randomizations)):
                print(f'{betas[l]} + {dense_randomizations[l].noise_mean} +/- {dense_randomizations[l].noise_std}')

            print('\nCVL')

            for l in range(len(cvl_randomizations)):
                print(f'{betas[l]} + {cvl_randomizations[l].noise_mean} +/- {cvl_randomizations[l].noise_std}')

            pts_3d = np.zeros((lidar_scan.shape[0], 5))
            pts_3d[:, 0:4] = lidar_scan

            dense_augmentations = [pts_3d,
                                   dense_augmented_01,
                                   dense_augmented_02,
                                   dense_augmented_03,
                                   dense_augmented_04,
                                   dense_augmented_05]

            cvl_augmentations = [pts_3d,
                                 cvl_augmented_01,
                                 cvl_augmented_02,
                                 cvl_augmented_03,
                                 cvl_augmented_04,
                                 cvl_augmented_05]

            print('')
            print(current_file)
            print(f'loading time: {(end_loading - start_loading) * 1000:.0f}ms')
            print(f'augmentation time: {(end - start) * 1000:.0f}ms')

            first_lidar_plots = []
            second_lidar_plots = []

            for j in range(len(first_lidar_widgets)):

                color = set_color(dense_augmentations[j])
                plot = gl.GLScatterPlotItem(pos=dense_augmentations[j][:,0:3], size=3, color=color)
                first_lidar_widgets[j].addItem(plot)

                first_lidar_plots.append(plot)

            for j in range(len(second_lidar_widgets)):

                color = set_color(cvl_augmentations[j])
                plot = gl.GLScatterPlotItem(pos=cvl_augmentations[j][:,0:3], size=3, color=color)
                second_lidar_widgets[j].addItem(plot)

                second_lidar_plots.append(plot)

            pos = QtGui.QVector3D(0, 0, 0)

            for widget in first_lidar_widgets:

                widget.setCameraPosition(pos=pos,
                                         distance=arguments.distance,
                                         azimuth=arguments.azimuth,
                                         elevation=arguments.elevation)

            for widget in second_lidar_widgets:

                widget.setCameraPosition(pos=pos,
                                         distance=arguments.distance,
                                         azimuth=arguments.azimuth,
                                         elevation=arguments.elevation)

            if arguments.first_lidar_monitor is not None:

                first_lidar_monitor = QDesktopWidget().screenGeometry(arguments.first_lidar_monitor)

                first_lidar_widget.move(first_lidar_monitor.left(), first_lidar_monitor.top())
                first_lidar_widget.setWindowTitle('DENSE')
                first_lidar_widget.update()
                first_lidar_widget.show()
                first_lidar_widget.showMaximized()

            if arguments.second_lidar_monitor is not None:

                second_lidar_monitor = QDesktopWidget().screenGeometry(arguments.second_lidar_monitor)

                second_lidar_widget.move(second_lidar_monitor.left(), second_lidar_monitor.top())
                second_lidar_widget.setWindowTitle('CVL')
                second_lidar_widget.update()
                second_lidar_widget.show()
                second_lidar_widget.showMaximized()

            if arguments.image_monitor is not None:

                image_monitor = QDesktopWidget().screenGeometry(arguments.image_monitor)

                image_widget.move(image_monitor.left(), image_monitor.top())
                image_widget.setWindowTitle(current_file)
                image_widget.update()
                image_widget.show()
                image_widget.showMaximized()

            app.exec_()

            # save_processed_image
            # first_lidar_widget.grabFrameBuffer().save(f'{current_file.split('.')[0]}.png')

            for k in range(len(first_lidar_widgets)):
                first_lidar_widgets[k].removeItem(first_lidar_plots[k])

            for k in range(len(second_lidar_widgets)):
                second_lidar_widgets[k].removeItem(second_lidar_plots[k])

            print('')
            print(f'distance:           {first_lidar_widget_01.opts["distance"]:.0f}')
            print(f'azimuth:            {first_lidar_widget_01.opts["azimuth"]:.0f}')
            print(f'elevation:          {first_lidar_widget_01.opts["elevation"]:.0f}')

            # update position in time
            for A in dense_randomizations:
                A.propagate_in_time(5)

            for A in cvl_randomizations:
                A.propagate_in_time(5)



if __name__ == '__main__':

    args = parsArgs()
    args.seed = 0

    hostname = socket.gethostname()

    if 'MacBook' in hostname or '.ee.ethz.ch' in hostname:

        bev_distance = 200
        bev_azimuth = 180
        bev_elevation = 90

        ego_distance = 2
        ego_azimuth = 180
        ego_elevation = 10

        near_distance = 35
        near_azimuth = 180
        near_elevation = 40

        far_distance = 86.8051380062
        far_azimuth = 180
        far_elevation = 40

        distance = 86.8051380062
        azimuth = 273
        elevation = -59

        near = [near_distance, near_azimuth, near_elevation]
        far = [far_distance, far_azimuth, far_elevation]

        args.scale = 1.8
        args.image_monitor = 1
        args.widescreen = True
        args.first_lidar_monitor = None
        args.second_lidar_monitor = 0

        dataset_folder = '/Users/Hahner/trace/datasets/KITTI/3D'

        args.distance, args.azimuth, args.elevation = near

    else:                   # assume CVL host

        bev_distance = 200
        bev_azimuth = 180
        bev_elevation = 90

        ego_distance = 2
        ego_azimuth = 180
        ego_elevation = 10

        near_distance = 40
        near_azimuth = 180
        near_elevation = 18

        far_distance = 90
        far_azimuth = 180
        far_elevation = 18

        distance = 86.8051380062
        azimuth = 273
        elevation = -59

        near = [near_distance, near_azimuth, near_elevation]
        far = [far_distance, far_azimuth, far_elevation]

        args.scale = 1.3
        args.image_monitor = 2
        args.widescreen = False
        args.first_lidar_monitor = 1
        args.second_lidar_monitor = 0

        dataset_folder = '/srv/beegfs-benderdata/scratch/tracezuerich/data/datasets'

        args.distance, args.azimuth, args.elevation = near

    dataset = 'DENSE'

    if dataset == 'KITTI':

        args.n_features = 4
        args.normalize = False
        args.root = 'KITTI/3D'
        args.lidar_folder = 'training/velodyne'

    else:   # assume DENSE

        args.n_features = 5
        args.normalize = True
        args.root = 'DENSE/SeeingThroughFog'
        args.lidar_folder = 'lidar_hdl64_strongest'

    src_folder = os.path.join(dataset_folder, args.root, args.lidar_folder)

    print('')
    print(f'searching for point clouds in    {src_folder}')

    main(src_folder, arguments=args)
