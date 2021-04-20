import os
import csv
import sys
import socket
import inspect
# sys.path.append('/scratch/fs2/Software/AdverseWeatherLabeling/lib')
# sys.path.append('/scratch/fs2/Software/frameworks/dense_label_tools')
# sys.path.append('/scratch/fs2/Software/frameworks/dense_label_tools/lib')

# Append path to |tools| module to |sys.path| list.
current_module_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
tools_directory = os.path.join(os.path.dirname(os.path.dirname(current_module_directory)))
sys.path.append(tools_directory)

from pathlib import Path

import pyqtgraph.opengl as gl
import matplotlib as mpl
import matplotlib.cm as cm
import argparse
from PyQt5 import QtGui, QtCore, uic
from PyQt5.QtWidgets import QDesktopWidget
import json
import numpy as np
import cv2
from tools.DatasetViewer.utils import convert_timestamp, colorize_pointcloud, get_time_difference
from tools.DatasetViewer.lib.read import load_calib_data, read_label
from tools.DatasetViewer.lib.visualization import draw_bbox2d_from_kitti, build_bbox3d_from_params, \
    project_points_to_2d, draw_bbox3d, project_3d_to_2d

from cvl.dense_dataset_utils import compare_points


                                #   R,   G,   B, alpha
COLORS = {'PassengerCar':        (  0, 255,   0, 255),  # green
          'Vehicle':             (  0, 100,   0, 255),  # dark green
          'LargeVehicle':        (  0,  50,   0, 255),  # darker green
          'Pedestrian':          (255,   0,   0, 255),  # red
          'Pedestrian_is_group': (255, 140,   0, 255),  # orange
          'RidableVehicle':      (255, 255,   0, 255),  # yellow
          'DontCare':            (100, 100, 100, 255),  # gray
          'Obstacle':            (255, 255, 255, 255)}  # white

MIN_HEIGHT = -1     # in m
MIN_DIST = 1.75     # in m


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--root_path', type=str, help='Path to Dataset root directory', default=None)
    parser.add_argument('-v', '--view_only', type=bool, default=True, help='Prevent Label Changes')
    parser.add_argument('-t', '--path_timestamps', type=str, default='./timestamps.json')
    parser.add_argument('-s', '--split', type=str, default='all')
    parser.add_argument('-f', '--fog_chamber', type=bool, default=False, help='load fog chamber data')
    parser.add_argument('-b', '--blend', type=bool, default=True, help='blend original and projection')
    parser.add_argument('-l', '--lidar_box', type=bool, default=True, help='draw lidar box')
    parser.add_argument('-e', '--ego_car_box', type=bool, default=True, help='draw ego vehicle box')
    parser.add_argument('-d', '--draw3Don2D', type=bool, default=False, help='if TRUE -> 3D box will be visualised '
                                                                             'on rgb and gated image')
    parser.add_argument('-g', '--draw_gated_boxes', type=bool, default=False, help='self explanatory')
    parser.add_argument('-c', '--color_feature', type=int, default=2, help='choose feature for colormap')
                        # 0: x, 1: y, 2: z, 3: distance, 4: intensity, 5: channel, 6: azimuth

    parser.add_argument('--full_screen', type=bool, default=True, help='show Dataset Viewer in full screen')
    parser.add_argument('--compare', help='compare and filter strongest != last', type=bool, default=False)
    parser.add_argument('--monitor', help='display number for visualization', type=int, default=0)
    parser.add_argument('--min_height', type=float, default=-np.infty, help='hide LiDAR points below this value')
    parser.add_argument('--min_dist', type=float, default=MIN_DIST, help='hide LiDAR points closer to this value')
    parser.add_argument('--point_size', type=float, default=5.5,
                        help='defines the size of the LiDAR point visualization')

    parser.add_argument('--username', type=str, default='admin',
                        help='Enter your username to recover and save the current index.')

    return parser.parse_args()


class DatasetViewer(QtGui.QMainWindow):

    def __init__(self, arguments, topics, timedelays, name):

        super(DatasetViewer, self).__init__()

        monitor = QDesktopWidget().screenGeometry(arguments.monitor)

        self.setGeometry(monitor)

        self.name = name
        self.topics = topics
        self.arguments = arguments
        self.rgb_default = topics['rgb'][0]
        self.gated_default = topics['gated'][0]
        self.lidar_default = topics['lidar'][0]
        self.lidar3d_default = topics['lidar3d'][0]
        self.can_speed_topic = 'can_body_basic'
        self.can_steering_angle_topic = 'can_body_chassis'
        self.can_light_sense_topic = 'can_body_lightsense'
        self.can_wiper_topic = 'can_body_wiper'
        self.road_friction_topic = 'road_friction'
        self.weather_topic = 'weather_station'
        self.label_topic = {'rgb': 'gt_labels/cam_left_labels_TMP',
                            'gated': 'gt_labels/gated_labels_TMP'}
        self.boxes = []  # Needed for 3d lidar boxes plot
        self.dir_labels = 'labeltool_labels'
        self.timedelays = timedelays
        self.filtered_points = None
        self.distant_points_iterator = 0

        self.distance_threshold = 10.0
        self.distance_step = 0.1

        self.arguments.min_height = MIN_HEIGHT
        self.height_step = 0.1

        self.min_channel = 0
        self.max_channel = -1
        self.MAX_CHANNEL = -1
        self.COLOR_FEATURES = 6
        self.life_is_easy = False
        self.color_feature = self.arguments.color_feature
        self.lidar_projection_available = not self.arguments.fog_chamber

        print('ROOT DIR: ' + str(self.arguments.root_path))

        if self.arguments.fog_chamber:

            self.recordings = sorted(os.listdir(os.path.join(self.arguments.root_path, self.rgb_default)))

        else:

            splits_folder = Path(os.getcwd()).parent.parent / 'splits'

            splits = sorted(os.listdir(splits_folder))

            assert f'{arguments.split}.txt' in splits, f'{arguments.split} is undefinded'

            recordings = []

            with open(f'{splits_folder / arguments.split}.txt') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    recordings.append(f'{row[0]}_{row[1]}.png')

            self.recordings = sorted(recordings)

        self.weather = ['clear', 'light_fog', 'dense_fog', 'rain', 'snow']
        self.daytimes = ['day', 'night']
        self.ratings = ['discard', 'dispensable', 'appropriate', 'very_interesting', 'interpolate']
        self.view_only = arguments.view_only

        QtGui.QShortcut(QtGui.QKeySequence('1'), self, self.decrease_min_channel)
        QtGui.QShortcut(QtGui.QKeySequence('2'), self, self.decrease_color)
        QtGui.QShortcut(QtGui.QKeySequence('3'), self, self.increase_min_channel)
        QtGui.QShortcut(QtGui.QKeySequence('4'), self, self.prev_sample)
        QtGui.QShortcut(QtGui.QKeySequence('5'), self, self.removeGroundButton_clicked)
        QtGui.QShortcut(QtGui.QKeySequence('6'), self, self.next_sample)
        QtGui.QShortcut(QtGui.QKeySequence('7'), self, self.decrease_max_channel)
        QtGui.QShortcut(QtGui.QKeySequence('8'), self, self.increase_color)
        QtGui.QShortcut(QtGui.QKeySequence('9'), self, self.increase_max_channel)

        QtGui.QShortcut(QtGui.QKeySequence('c'), self, self.decrease_color)
        QtGui.QShortcut(QtGui.QKeySequence('v'), self, self.increase_color)

        QtGui.QShortcut(QtGui.QKeySequence('n'), self, self.decrease_min_channel)
        QtGui.QShortcut(QtGui.QKeySequence('m'), self, self.increase_min_channel)

        QtGui.QShortcut(QtGui.QKeySequence('p'), self, self.iterate_distant_points)

        QtGui.QShortcut(QtGui.QKeySequence(','), self, self.decrease_max_channel)
        QtGui.QShortcut(QtGui.QKeySequence('.'), self, self.increase_max_channel)

        QtGui.QShortcut(QtGui.QKeySequence('f'), self, self.compare_strongest_and_last)
        QtGui.QShortcut(QtGui.QKeySequence('h'), self, self.removeGroundButton_clicked)

        QtGui.QShortcut(QtGui.QKeySequence('s'), self, self.decrease_height_threshold)
        QtGui.QShortcut(QtGui.QKeySequence('w'), self, self.increase_height_threshold)

        QtGui.QShortcut(QtGui.QKeySequence('a'), self, self.decrease_distance_threshold)
        QtGui.QShortcut(QtGui.QKeySequence('d'), self, self.increase_distance_threshold)

        QtGui.QShortcut(QtGui.QKeySequence('PgDown'), self, self.prev_sample)
        QtGui.QShortcut(QtGui.QKeySequence('PgUp'), self, self.next_sample)
        QtGui.QShortcut(QtGui.QKeySequence('Esc'), self, self.close)

        # QtGui.QShortcut(QtGui.QKeySequence('1'), self, self.set_bad_sensor)
        # QtGui.QShortcut(QtGui.QKeySequence('2'), self, self.set_no_objects)
        #
        # QtGui.QShortcut(QtGui.QKeySequence('q'), self, self.set_discard)
        # QtGui.QShortcut(QtGui.QKeySequence('w'), self, self.set_dispensable)
        # QtGui.QShortcut(QtGui.QKeySequence('e'), self, self.set_appropriate)
        # QtGui.QShortcut(QtGui.QKeySequence('r'), self, self.set_very_interesting)
        # QtGui.QShortcut(QtGui.QKeySequence('t'), self, self.set_interpolate)
        #
        # QtGui.QShortcut(QtGui.QKeySequence('a'), self, self.set_clear)
        # QtGui.QShortcut(QtGui.QKeySequence('s'), self, self.set_light_fog)
        # QtGui.QShortcut(QtGui.QKeySequence('d'), self, self.set_dense_fog)
        # QtGui.QShortcut(QtGui.QKeySequence('f'), self, self.set_rain)
        # QtGui.QShortcut(QtGui.QKeySequence('g'), self, self.set_snow)
        #
        # QtGui.QShortcut(QtGui.QKeySequence('y'), self, self.set_day)
        # QtGui.QShortcut(QtGui.QKeySequence('x'), self, self.set_night)
        #
        # QtGui.QShortcut(QtGui.QKeySequence('l'), self, self.editLabelsButton_clicked)

        self.get_current_index()
        self.create_label_folders()
        self.count_labels_first()

        self.labels_frozen = True
        self.initUI()

    def initUI(self):
        self.ui = uic.loadUi('src/DataViewerGUI.ui', self)

        if self.arguments.full_screen:
            self.showMaximized()

        self.w = gl.GLViewWidget()
        self.w.setCameraPosition(pos=QtGui.QVector3D(0, 0, 0), distance=10, azimuth=180, elevation=10)
        self.initComboBoxes()

        self.update_sample()
        self.show()

    def initComboBoxes(self):
        self.recordingComboBox.addItems(self.recordings)
        self.weatherComboBox.addItems(self.weather)
        self.daytimeComboBox.addItems(self.daytimes)
        self.ratingComboBox.addItems(self.ratings)
        self.rgbComboBox.addItems(self.topics['rgb'])
        self.rgb_topic = self.rgb_default
        self.gatedComboBox.addItems(self.topics['gated'])
        self.gated_topic = self.gated_default
        self.lidarComboBox.addItems(self.topics['lidar'])
        self.lidar_topic = self.lidar_default
        self.lidar3dComboBox.addItems(self.topics['lidar3d'])
        self.lidar3d_topic = self.lidar3d_default

    def update_sample(self):
        self.update_recordingComboBox()
        self.update_rgbComboBox()
        self.update_gatedComboBox()
        self.update_lidarComboBox()
        self.update_lidar3dComboBox()
        self.update_counter()
        self.update_can()
        self.update_image()
        self.update_labels()

        if self.labels_frozen:
            self.set_label_state(False)
        else:
            self.set_label_state(True)
            if self.badSensorCheckBox.isChecked():
                self.badSensorCheckBox_is_true()

    def update_recordingComboBox(self):
        cur_rec = self.current_index
        self.recordingComboBox.setCurrentIndex(cur_rec)

    def update_sampleComboBox(self, labels, label_idx):
        object_names = [i['identity'] + ',' + str(idx) for idx, i in enumerate(labels)]
        idxes = ['All']
        idxes.extend(object_names)
        self.box_idxes = idxes
        self.sampleComboBox.clear()
        self.sampleComboBox.addItems(idxes)
        self.sampleComboBox.setCurrentIndex(idxes.index(label_idx))

    def update_rgbComboBox(self):
        cur_rgb_topic = '{}'.format(self.rgbComboBox.currentText())
        rgb_types = self.topics['rgb']
        self.lidarComboBox.setCurrentIndex(rgb_types.index(cur_rgb_topic))

    def update_gatedComboBox(self):
        cur_gated_topic = '{}'.format(self.gatedComboBox.currentText())
        gated_types = self.topics['gated']

        self.gatedComboBox.setCurrentIndex(gated_types.index(cur_gated_topic))
        self.gated_topic = cur_gated_topic

    def update_lidarComboBox(self):
        cur_lidar_topic = '{}'.format(self.lidarComboBox.currentText())
        lidar_types = self.topics['lidar']
        self.lidarComboBox.setCurrentIndex(lidar_types.index(cur_lidar_topic))

    def update_lidar3dComboBox(self):
        cur_lidar3d_topic = '{}'.format(self.lidar3dComboBox.currentText())
        lidar3d_types = self.topics['lidar3d']
        self.lidar3dComboBox.setCurrentIndex(lidar3d_types.index(cur_lidar3d_topic))

    def update_counter(self):
        self.indexEdit.setText('{}'.format(self.current_index))
        self.progressEdit.setText('{}/{}'.format(self.count_labels(), len(self.recordings)))
        self.goToIndexEdit.setText('{}'.format(self.current_index))

    def update_can(self):

        if self.arguments.fog_chamber:

            self.update_meta_data()

        else:

            self.update_speed()
            self.update_angle()
            self.update_daytime()
            self.update_wiper()
            self.update_road_friction()
            self.update_weather()

    def update_meta_data(self):

        path = self.get_path('label')

        if path is None:

            self.speedLabel.setText('')
            self.speedEdit.setText('')

            self.roadFrictionLabel.setText('')
            self.roadFrictionEdit.setText('')

            self.dayNightLabel.setText('')
            self.daytimeEdit.setText('')

            self.angleLabel.setText('')
            self.angleEdit.setText('')

            self.outTempLabel.setText('')
            self.outTempEdit.setText('')

            self.outHumidityLabel.setText('')
            self.wiperLabel.setText('')
            self.dewpointLabel.setText('')

        else:

            with open(path) as f:
                meta_data = json.load(f)

            self.speedLabel.setText('Weather:')
            self.speedEdit.setText(f'{meta_data["weather"]}')

            self.roadFrictionLabel.setText('Visibility:')
            self.roadFrictionEdit.setText(f'{meta_data["metereological_visibility"]} m')

            self.dayNightLabel.setText('Rainfall Rate:')
            self.daytimeEdit.setText(f'{meta_data["rainfall_rate"]}')

            self.angleLabel.setText('Greenhouse:')
            self.angleEdit.setText(f'{meta_data["greenhouse_temperature"]:.2f}° C')

            self.outTempLabel.setText('Tunnel:')
            self.outTempEdit.setText(f'{meta_data["tunnel_temperature"]:.2f}° C')

            self.outHumidityLabel.setText('')
            self.wiperLabel.setText('')
            self.dewpointLabel.setText('')


    def update_speed(self):
        path = self.get_path('can_speed')
        try:
            with open(path) as f:
                can_speed = json.load(f)
            self.speedEdit.setText('{0:.2f} km/h'.format(can_speed['VehSpd_Disp']))

        except FileNotFoundError:
            self.speedEdit.setText('N/A')

    def update_angle(self):
        path = self.get_path('can_steering_angle')
        try:
            with open(path) as f:
                can_steering_angle = json.load(f)
            self.angleEdit.setText('{0:.2f} \xb0'.format(can_steering_angle['StWhl_Angl']))

        except FileNotFoundError:
            self.angleEdit.setText('N/A')

    def update_daytime(self):
        path = self.get_path('can_light_sense')
        try:
            with open(path) as f:
                can_light_sense = json.load(f)
            if can_light_sense['LgtSens_Night'] == 1:
                self.daytime = 'night'
            else:
                self.daytime = 'day'
            self.daytimeEdit.setText('{}'.format(self.daytime))

        except FileNotFoundError:
            self.daytimeEdit.setText('N/A')
            self.daytime = 'night'

    def update_wiper(self):
        path = self.get_path('can_wiper')
        try:
            with open(path) as f:
                can_wiper = json.load(f)
            self.wiperEdit.setText('{}'.format(can_wiper['Wpr_Stat']))

        except FileNotFoundError:
            self.wiperEdit.setText('N/A')

    def update_road_friction(self):
        path = self.get_path('road_friction')
        try:
            with open(path) as f:
                road_friction = json.load(f)
            self.roadFrictionEdit.setText('{}'.format(road_friction['surface_state_result']).lower())

        except FileNotFoundError:
            self.roadFrictionEdit.setText('N/A')

    def update_weather(self):
        path = self.get_path('weather')
        try:
            with open(path) as f:
                weather = json.load(f)
            self.outTempEdit.setText('{0:.2f} \xb0C'.format((weather['outTemp'] - 32.0) * 5 / 9))
            self.outHumidityEdit.setText('{0:.2f} %'.format(weather['outHumidity']))
            self.dewpointEdit.setText('{0:.2f} \xb0C'.format((weather['dewpoint'] - 32.0) * 5 / 9))

        except FileNotFoundError:
            self.outTempEdit.setText('N/A')
            self.outHumidityEdit.setText('N/A')
            self.dewpointEdit.setText('N/A')

    def update_image(self):
        self.update_calib()
        self.update_cmore_labels()
        self.update_rgb()
        self.update_gated()
        self.update_lidar3d()
        if self.lidar_projection_available:
            self.update_lidar()

    def update_image_boxes(self, idx):
        if idx is not None:
            object_idx = slice(max(0, idx - 1), idx)
        else:
            object_idx = slice(len(self.labels_rgb))
        self.update_rgb(label_idx=object_idx)
        self.update_gated(label_idx=object_idx)
        self.update_lidar3d(label_idx=object_idx)

    def update_cmore_labels(self):
        self.labels_rgb = self.read_labels('rgb', self.camera_to_velodyne_rgb)
        self.update_sampleComboBox(self.labels_rgb, 'All')
        self.labels_gated = self.read_labels('gated', self.camera_to_velodyne_gated)

    def update_calib(self):

        self.rgb_calib = self.read_calib('rgb')

        self.velodyne_to_camera_rgb = self.rgb_calib[0]
        self.camera_to_velodyne_rgb = self.rgb_calib[1]
        self.P_rgb = self.rgb_calib[2]
        self.R_rgb = self.rgb_calib[3]
        self.vtc_rgb = self.rgb_calib[4]
        self.radar_to_camera_rgb = self.rgb_calib[5]
        self.zero_to_camera_rgb = self.rgb_calib[6]

        self.gated_calib = self.read_calib('gated')

        self.velodyne_to_camera_gated = self.gated_calib[0]
        self.camera_to_velodyne_gated = self.gated_calib[1]
        self.P_gated = self.gated_calib[2]
        self.R_gated = self.gated_calib[3]
        self.vtc_gated = self.gated_calib[4]
        self.radar_to_camera_gated = self.gated_calib[5]
        self.zero_to_camera_gated = self.gated_calib[6]


    def update_rgb(self, label_idx=None):
        path = self.get_path('rgb')

        self.stereo_timestamp = self.get_timestamp_from_data_name(topic='rgb')
        if self.stereo_timestamp is not None:
            date = convert_timestamp(self.stereo_timestamp)
            self.timeRGBEdit.setText('{}'.format(date))
        else:
            self.timeRGBEdit.setText('No timestamp found!')

        image = cv2.imread(path, -1)
        image = self.draw_3d_labels_in_rgb(image, topic_type='rgb', label_idx=label_idx)
        self.displayImage(image, self.rgbLabel)

    def read_calib(self, topic_type):

        root = './calibs'
        tf_tree = 'calib_tf_tree_full.json'

        if topic_type == 'rgb':
            name_camera_calib = 'calib_cam_stereo_left.json'
        elif topic_type == 'gated':
            name_camera_calib = 'calib_gated_bwv.json'
        else:
            print('Unknown topic type in read calib!')

        return load_calib_data(root, name_camera_calib, tf_tree)

    def read_labels(self, topic_type, calib):
        label_path = os.path.join(self.arguments.root_path, self.label_topic[topic_type])
        recording = os.path.splitext(self.recordings[self.current_index])[
            0]  # here without '.txt' as it will be added in read_label function
        label_file = os.path.join(label_path, recording)
        label = read_label(label_file, label_path, camera_to_velodyne=calib)
        return label

    def draw_2d_labels_in_rgb(image, label):
        image = draw_bbox2d_from_kitti(image, label)
        return image

    def draw_3d_labels_in_rgb(self, image, topic_type, label_idx=None):
        corners_list = []

        if topic_type == 'rgb':
            label = self.labels_rgb
            zero_to_camera = self.zero_to_camera_rgb
            P = self.P_rgb
        elif topic_type == 'gated':
            label = self.labels_gated
            zero_to_camera = self.zero_to_camera_gated
            P = self.P_gated
        else:
            label, zero_to_camera, P = None, None, None
            print('Wrong topic_type draw_3d_labels_in_rgb!')

        if label_idx is None:
            label_idx = slice(len(label))

        for annotation in label[label_idx]:

            if annotation['identity'] in COLORS:
                color = COLORS[annotation['identity']]
            else:
                color = (255, 255, 255, 255)

            corners_list.append(build_bbox3d_from_params(annotation, zero_to_camera=zero_to_camera))

            if (topic_type == 'gated' and self.arguments.draw_gated_boxes) or topic_type != 'gated':
                image = draw_bbox2d_from_kitti(image, annotation, color=color[:-1][::-1])

        if self.arguments.draw3Don2D:

            for kitti_object in corners_list:
                bbox3d = project_points_to_2d(kitti_object, P)
                draw_bbox3d(image, bbox3d)

        return image

    def update_gated(self, label_idx=None):
        path = self.get_path('gated')
        # @ TODO check topic mapping
        if 'gated_full_rect8' != self.gated_topic:
            topic_id = self.gated_topic.split('_')[0][-1]
        else:
            topic_id = '4'
        mapping ={
            '0': '0',
            '1': '1',
            '2': '2',
            '4': 'full'
        }
        if topic_id not in mapping.keys():
            timestamp = None
        else:
            timestamp = self.get_timestamp_from_data_name(topic='gated%s'%(mapping[topic_id]))
        if timestamp is not None:
            time_diff = get_time_difference(self.stereo_timestamp, timestamp)
            self.timeGatedEdit.setText('{} ms'.format(time_diff))
        else:
            self.timeGatedEdit.setText('No timestamp found!')

        try:
            image = np.right_shift(cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), 2).astype(np.uint8)
            if self.arguments.fog_chamber:
                image = image*4             # 8 to 64bit
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image = self.draw_3d_labels_in_rgb(image_rgb, topic_type='gated', label_idx=label_idx)
            self.displayImage(image, self.gatedLabel)
        except:
            print('The gated_full_topic has not been recorded by the sensor in this case!')

    def update_lidar(self):
        path = self.get_path('lidar')

        timestamp = self.get_timestamp_from_data_name(topic='lidar')
        if timestamp is not None:
            time_diff = get_time_difference(self.stereo_timestamp, timestamp)
            self.timeLidarEdit.setText('{} ms'.format(time_diff))
        else:
            self.timeLidarEdit.setText('No timestamp found!')

        depth = np.load(path)['arr_0']
        image = colorize_pointcloud(depth)

        rgb_path = self.get_path('rgb')
        rgb_image = cv2.imread(rgb_path, -1)

        gated_path = self.get_path('gated')

        gated_image = np.right_shift(cv2.imread(gated_path,
                                                cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), 2).astype(np.uint8)

        gated_image = cv2.cvtColor(gated_image, cv2.COLOR_GRAY2RGB)

        if self.arguments.fog_chamber:
            image = image * 4  # 8 to 64bit

        if self.arguments.blend:

            if 'gated' in self.lidar_topic:
                image = cv2.add(image, gated_image)
            else:
                image = cv2.addWeighted(image, 0.7, rgb_image, 0.3, 0)

        # if self.arguments.compare:    # TODO: does not work yet
        #
        #     if 'gated' in self.lidar_topic:
        #         image = gated_image
        #         calib = self.gated_calib
        #     else:
        #         image = rgb_image
        #         calib = self.rgb_calib
        #
        #     if self.filtered_points is not None:
        #
        #         points_2d = project_3d_to_2d(self.filtered_points[:, :3].T, calib[4]).astype(int)
        #
        #         counter = 0
        #
        #         for i, point in enumerate(points_2d):
        #
        #             if point[0] < 0 or point[1] < 0:
        #                 continue
        #
        #             try:
        #                 image[point[1], point[0]] = [0,0,255]
        #                 counter += 1
        #             except IndexError:
        #                 pass
        #
        #         print(f'\n{counter}')

        self.displayImage(image, self.lidarLabel)

    def update_lidar3d(self, label_idx=None):

        path = self.get_path('lidar3d')

        timestamp = self.get_timestamp_from_data_name(topic='lidar')

        if timestamp is not None:
            time_diff = get_time_difference(self.stereo_timestamp, timestamp)
            self.timeLidar3dEdit.setText('{} ms'.format(time_diff))
        else:
            self.timeLidar3dEdit.setText('No timestamp found!')

        if self.arguments.compare:

            recording = self.recordings[self.current_index]
            filename = os.path.splitext(recording)[0] + '.bin'

            selected_lidar = '_'.join(self.lidar3d_topic.split('_')[:-1])

            path_last = os.path.join(self.arguments.root_path, f'{selected_lidar}_last', filename)
            path_strongest = os.path.join(self.arguments.root_path, f'{selected_lidar}_strongest', filename)

            pc_master, mask, num_last, num_strongest, diff = compare_points(path_last, path_strongest)

            initial_number_of_points = len(pc_master)

            pc = pc_master[mask]

            inverse_mask = [not value for value in mask]

            self.filtered_points = pc_master[inverse_mask]

            if diff > 0:
                sys.stdout.write(f'\r{len(self.filtered_points)} filtered points, '
                                 f'{num_last} last != {num_strongest} strongest, diff={diff}')
            else:
                sys.stdout.write(f'\r{len(self.filtered_points)} filtered points')

            sys.stdout.flush()

        else:

            sys.stdout.write(f'\r ')
            sys.stdout.flush()

            self.filtered_points = None

            pc = np.fromfile(path, dtype=np.float32)
            pc = pc.reshape((-1, 5))

            initial_number_of_points = len(pc)

        if self.filtered_points is None:
            self.LiDARcompareLabel.setText('')
            self.LiDARcompareEdit.setText(f'')
        else:
            self.LiDARcompareLabel.setText('filtered points:')
            self.LiDARcompareEdit.setText(f'{len(self.filtered_points)}')

        if self.MAX_CHANNEL == -1:
            self.MAX_CHANNEL = int(np.max(pc[:, 4]))

        if self.max_channel == -1:
            self.max_channel = self.MAX_CHANNEL

        self.LiDARchannelEdit.setText(f'[{self.min_channel+1}, {self.max_channel+1}]')
        self.HeightTresholdEdit.setText(f'{round(self.arguments.min_height, 2):.2f} m')
        self.DistanceTresholdEdit.setText(f'{round(self.distance_threshold, 2):.2f} m')

        if self.arguments.fog_chamber:

            # filter points that lie outside of the fog chamber dimensions
            noise_mask = pc[:, 0] > -10
            noise_mask = np.logical_and(noise_mask, pc[:, 1] < 4)
            noise_mask = np.logical_and(noise_mask, pc[:, 1] > -4)
            noise_mask = np.logical_and(noise_mask, pc[:, 2] > -2)

            pc = pc[noise_mask, :]

        self.LiDARcountEdit.setText(f'{pc.shape[0]:,}')

        dist = np.linalg.norm(pc[:, 0:3], axis=1)

        last_per_mille_dist = sorted(list(dist))[-int(len(dist) / 1000)]
        last_per_cent_dist = sorted(list(dist))[-int(len(dist) / 100)::]

        self.LiDARpermilleEdit.setText(f'{last_per_mille_dist*2:.2f} m')
        self.LiDARpercentEdit.setText(f'{np.median(last_per_cent_dist)*2:.2f} m')

        current_distance = {0: 0,
                            1: np.median(last_per_cent_dist),
                            2: last_per_mille_dist}

        current_label = {0: '                     ',
                         1: 'showing furthest 0,5%',
                         2: 'showing furthest 0,1%'}

        self.distantLabel.setText(current_label[self.distant_points_iterator])
        min_dist = max(self.arguments.min_dist, current_distance[self.distant_points_iterator])

        min_dist_mask = np.linalg.norm(pc[:, 0:3], axis=1) > min_dist
        pc = pc[min_dist_mask, :]

        max_dist_mask = np.linalg.norm(pc[:, 0:3], axis=1) < self.distance_threshold
        pc = pc[max_dist_mask, :]

        min_height_mask = pc[:, 2] > self.arguments.min_height
        pc = pc[min_height_mask, :]

        min_channel_mask = pc[:, 4] >= self.min_channel
        pc = pc[min_channel_mask, :]

        max_channel_mask = pc[:, 4] <= self.max_channel
        pc = pc[max_channel_mask, :]

        # remove lower half of the ego vehicle box (dimensions of W222 from wikipedia)
        l, w, h = 5.116, 1.899, 1.496

        x_mask_1 = -l / 2 < pc[:, 0]
        x_mask_2 = pc[:, 0] < l / 2
        x_mask = (x_mask_1 == 1) & (x_mask_2 == 1)

        y_mask_1 = -w / 2 < pc[:, 1]
        y_mask_2 = pc[:, 1] < w / 2
        y_mask = (y_mask_1 == 1) & (y_mask_2 == 1)

        z_mask_1 = -h < pc[:, 2]
        z_mask_2 = pc[:, 2] < -h / 2
        z_mask = (z_mask_1 == 1) & (z_mask_2 == 1)

        custom_mask = (x_mask == 1) & (y_mask == 1) & (z_mask == 1)
        custom_mask = ~custom_mask
        pc = pc[custom_mask, :]

        # create colormap
        if self.color_feature == 0:

            self.LiDARcolorEdit.setText('x')
            feature = pc[:, 0]
            min_value = np.min(feature)
            max_value = np.max(feature)

        elif self.color_feature == 1:

            self.LiDARcolorEdit.setText('y')
            feature = pc[:, 1]
            min_value = np.min(feature)
            max_value = np.max(feature)

        elif self.color_feature == 2:

            self.LiDARcolorEdit.setText('z')
            feature = pc[:, 2]
            min_value = -1.5
            max_value = 0.5

        elif self.color_feature == 3:

            self.LiDARcolorEdit.setText('distance')
            feature = np.linalg.norm(pc[:, 0:3], axis=1)
            min_value = np.min(feature)
            max_value = np.max(feature)

        elif self.color_feature == 4:

            self.LiDARcolorEdit.setText('intensity')
            feature = pc[:, 3]
            min_value = 0
            max_value = 255

        elif self.color_feature == 5:

            self.LiDARcolorEdit.setText('channel')
            feature = pc[:, 4]
            min_value = 0
            max_value = self.MAX_CHANNEL

        else:   # self.color_feature == 6:

            self.LiDARcolorEdit.setText('azimuth')
            feature = np.arctan2(pc[:, 1], pc[:, 0]) + np.pi
            min_value = 0
            max_value = 2*np.pi

        norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)

        if self.color_feature == 6:
            cmap = cm.hsv           # cyclic
        else:
            cmap = cm.jet           # sequential

        m = cm.ScalarMappable(norm=norm, cmap=cmap)

        colors = m.to_rgba(feature)
        colors[:, [2, 1, 0, 3]] = colors[:, [0, 1, 2, 3]]
        colors[:, 3] = 0.5

        # remove content of previous frame
        try:
            self.w.removeItem(self.plot)
            for box in self.boxes:
                self.w.removeItem(box)
        except Exception:
            pass

        self.boxes = []
        size = QtGui.QVector3D(1, 1, 1)

        if label_idx is None:
            label_idx = slice(len(self.labels_rgb))

        # create annotation boxes
        for annotation in self.labels_rgb[label_idx]:

            if annotation['identity'] in COLORS:
                color = COLORS[annotation['identity']]
            else:
                color = (255, 255, 255, 255)

            box = gl.GLBoxItem(size, color=color)
            box.setSize(annotation['length'], annotation['width'], annotation['height'])
            box.translate(-annotation['length'] / 2, -annotation['width'] / 2, -annotation['height'] / 2)
            box.rotate(angle=-annotation['rotz'] * 180 / 3.14159265359, x=0, y=0, z=1)
            box.rotate(angle=-annotation['roty'] * 180 / 3.14159265359, x=0, y=1, z=0)
            box.rotate(angle=-annotation['rotx'] * 180 / 3.14159265359, x=1, y=0, z=0)
            box.translate(0, 0, annotation['height'] / 2)
            box.translate(annotation['posx_lidar'], annotation['posy_lidar'], annotation['posz_lidar'])
            self.boxes.append(box)

        # add pointcloud
        self.plot = gl.GLScatterPlotItem(pos=np.asarray(pc[:, 0:3]), size=self.arguments.point_size, color=colors)
        self.w.addItem(self.plot)

        # add annotations
        for box in self.boxes:
            self.w.addItem(box)

        # add a grid to the floor
        if self.arguments.fog_chamber:
            g = gl.GLGridItem(QtGui.QVector3D(40,10,1))
            g.setSpacing(x=5, y=5, z=5)
            g.translate(10, 0, -1.8)
            self.w.addItem(g)

        if self.arguments.lidar_box:
            # add LiDAR sensor box (dimensions of velodyne spec sheet)
            l_l, w_l, h_l = 0.203, 0.203, 0.283
            lidar_sensor_box = gl.GLBoxItem(size=QtGui.QVector3D(l_l, w_l, h_l), color=(255, 255, 255, 255))
            lidar_sensor_box.translate(-l_l/2, -w_l/2, -h_l/2)
            self.w.addItem(lidar_sensor_box)

        if self.arguments.ego_car_box:
            # add ego vehicle box (dimensions of W222 from wikipedia)
            l_c, w_c, h_c = 5.116, 1.899, 1.496
            ego_car_box = gl.GLBoxItem(size=QtGui.QVector3D(l_c, w_c, h_c), color=(255, 255, 255, 255))
            ego_car_box.translate(-l_c/2, -w_c/2, -h_c - h_l/2)
            self.w.addItem(ego_car_box)

        self.lidar3dGridLayout.addWidget(self.w)

        self.NumberPointsEdit.setText(f'{len(pc)}/{initial_number_of_points}')

    def update_labels(self):
        labels = self.load_labels()

        if labels and not self.arguments.fog_chamber:
            self.badSensorCheckBox.setChecked(labels['bad_sensor'])
            self.labels_frozen = True
            if labels['bad_sensor']:
                self.badSensorCheckBox_is_true()
            else:
                self.noObjectsCheckBox.setChecked(labels['objects']['no_objects'])
                for weather in self.weather:
                    if labels['weather'][weather]:
                        weather_index = self.weather.index(weather)
                        self.weatherComboBox.setCurrentIndex(weather_index)
                        break
                for daytime in self.daytimes:
                    if labels['daytime'][daytime]:
                        daytime_index = self.daytimes.index(daytime)
                        self.daytimeComboBox.setCurrentIndex(daytime_index)
                        break
                for rating in self.ratings:
                    # If rating was not labeled before
                    try:
                        labels['rating'][rating]
                    except Exception:
                        self.labels_frozen = False
                        break
                    if labels['rating'][rating]:
                        rating_index = self.ratings.index(rating)
                        self.ratingComboBox.setCurrentIndex(rating_index)
                        break

        self.editLabelsButton.setEnabled(not self.labels_frozen)

    def load_labels(self):
        label_path = self.get_path('label')
        if label_path:
            with open(label_path) as infile:
                labels = json.load(infile)
            return labels
        else:
            return None

    def set_label_state(self, state):
        self.badSensorCheckBox.setEnabled(state)
        self.noObjectsCheckBox.setEnabled(state)
        self.weatherComboBox.setEnabled(state)
        self.daytimeComboBox.setEnabled(state)
        self.ratingComboBox.setEnabled(state)

    def set_init_label_values(self):
        self.badSensorCheckBox.setChecked(False)
        self.noObjectsCheckBox.setChecked(False)
        self.weatherComboBox.setCurrentIndex(0)
        self.daytimeComboBox.setCurrentIndex(0)
        self.ratingComboBox.setCurrentIndex(0)

    def editLabelsButton_clicked(self):
        if self.labels_frozen:
            if not self.badSensorCheckBox.isChecked():
                self.set_label_state(True)
            else:
                self.badSensorCheckBox.setEnabled(True)

    def displayImage(self, image, label):
        size = image.shape
        step = image.size / size[0]
        qformat = QtGui.QImage.Format_Indexed8

        if len(size) == 3:
            if size[2] == 4:
                qformat = QtGui.QImage.Format_RGBA8888
            else:
                qformat = QtGui.QImage.Format_RGB888

        img = QtGui.QImage(image, size[1], size[0], step, qformat)
        img = img.rgbSwapped()
        w = label.width()
        h = label.height()
        label.setPixmap(QtGui.QPixmap.fromImage(img).scaled(w, h, QtCore.Qt.KeepAspectRatio))

    def next_sample(self):
        if not self.view_only:
            self.save_labels()
        if self.current_index < len(self.recordings) - 1:
            self.current_index += 1
            self.update_sample()
        elif self.current_index == len(self.recordings) - 1:
            self.current_index = 0
            self.update_sample()

    def save_labels(self):
        label_path = self.create_label_path()
        recording = os.path.splitext(self.recordings[self.current_index])[0]

        self.samples_labeled.append(recording)
        labels = {}
        labels['bad_sensor'] = self.badSensorCheckBox.isChecked()
        labels['weather'] = {}
        labels['daytime'] = {}
        labels['objects'] = {}
        labels['rating'] = {}
        if not self.badSensorCheckBox.isChecked():
            labels['objects']['no_objects'] = self.noObjectsCheckBox.isChecked()

            weather = '{}'.format(self.weatherComboBox.currentText())
            index = self.weather.index(weather)
            for i, weather in enumerate(self.weather):
                if i == index:
                    labels['weather'][weather] = True
                else:
                    labels['weather'][weather] = False

            daytime = '{}'.format(self.daytimeComboBox.currentText())
            index = self.daytimes.index(daytime)
            for i, daytime in enumerate(self.daytimes):
                if i == index:
                    labels['daytime'][daytime] = True
                else:
                    labels['daytime'][daytime] = False

            rating = '{}'.format(self.ratingComboBox.currentText())
            index = self.ratings.index(rating)
            for i, rating in enumerate(self.ratings):
                if i == index:
                    labels['rating'][rating] = True
                else:
                    labels['rating'][rating] = False

        with open(label_path, 'w') as outfile:
            json.dump(labels, outfile)

    def create_label_path(self):
        recording = os.path.splitext(self.recordings[self.current_index])[0]
        return os.path.join(self.arguments.root_path, self.dir_labels, recording + '.json')

    def prev_sample(self):
        if not self.view_only:
            self.save_labels()
        if self.current_index > 0:
            self.current_index -= 1
            self.update_sample()
        elif self.current_index == 0:
            self.current_index = len(self.recordings) - 1
            self.update_sample()

    def get_current_index(self):
        index_file = self.get_index_file()
        if os.path.exists(index_file):
            with open(index_file) as infile:
                self.current_index = int(infile.read().split('\n')[0])
        else:
            self.current_index = 0

        if self.current_index > len(self.recordings):
            self.current_index = 0

    def get_index_file(self):
        return '.current_index_{}'.format(self.name)

    def count_labels(self):
        return len(set(self.samples_labeled).intersection(os.path.splitext(item)[0] for item in self.recordings))

    def count_labels_first(self):
        self.samples_labeled = []
        samples_labeled = [os.path.splitext(item)[0] for item in
                           os.listdir(os.path.join(self.arguments.root_path, self.dir_labels))]
        self.samples_labeled = self.samples_labeled + samples_labeled

    def create_label_folders(self):
        if not os.path.exists(os.path.join(self.arguments.root_path, self.dir_labels)):
            os.mkdir(os.path.join(self.arguments.root_path, self.dir_labels))

    def get_path(self, topic):
        recording = self.recordings[self.current_index]

        if topic == 'rgb':
            return os.path.join(self.arguments.root_path, self.rgb_topic, recording)
        if topic == 'gated':
            return os.path.join(self.arguments.root_path, self.gated_topic, recording)
        if topic == 'lidar':
            return os.path.join(self.arguments.root_path, self.lidar_topic, os.path.splitext(recording)[0] + '.npz')
        if topic == 'lidar3d':
            return os.path.join(self.arguments.root_path, self.lidar3d_topic, os.path.splitext(recording)[0] + '.bin')
        if topic == 'can_speed':
            try:
                return os.path.join(self.arguments.root_path, self.can_speed_topic, os.path.splitext(recording)[0] + '.json')
            except Exception:
                return None
        if topic == 'can_steering_angle':
            try:
                return os.path.join(self.arguments.root_path, self.can_steering_angle_topic,
                                    os.path.splitext(recording)[0] + '.json')
            except Exception:
                return None
        if topic == 'can_light_sense':
            try:
                return os.path.join(self.arguments.root_path, self.can_light_sense_topic, os.path.splitext(recording)[0] + '.json')
            except Exception:
                return None
        if topic == 'can_wiper':
            try:
                return os.path.join(self.arguments.root_path, self.can_wiper_topic, os.path.splitext(recording)[0] + '.json')
            except Exception:
                return None
        if topic == 'road_friction':
            try:
                return os.path.join(self.arguments.root_path, self.road_friction_topic, os.path.splitext(recording)[0] + '.json')
            except Exception:
                return None
        if topic == 'weather':
            try:
                return os.path.join(self.arguments.root_path, self.weather_topic, os.path.splitext(recording)[0] + '.json')
            except Exception:
                return None
        if topic == 'label':
            label_path = os.path.join(self.arguments.root_path, self.dir_labels, os.path.splitext(recording)[0] + '.json')
            if os.path.isfile(label_path):
                return label_path
            else:
                return None

    def get_timestamp_from_data_name(self, topic):
        recording = self.recordings[self.current_index]

        # get matching file name from AdverseWeather data
        matching_label_topic = self.timedelays.get(topic)
        old_data_name = matching_label_topic.get(recording.split('.png')[0], 'NoFrame')
        if old_data_name != 'NoFrame':
            old_data_name = os.path.splitext(old_data_name)[0]
            old_data_name = old_data_name.split('_')[1]
            stereo_timestamp = int(old_data_name)
            return stereo_timestamp

        else:
            return None

    def badSensorCheckBox_clicked(self):
        if self.badSensorCheckBox.isChecked():
            self.badSensorCheckBox_is_true()
        else:
            self.set_label_state(True)

    def badSensorCheckBox_is_true(self):
        self.set_init_label_values()
        self.set_label_state(False)
        self.badSensorCheckBox.setEnabled(True)
        self.badSensorCheckBox.setChecked(True)

    def editLabelsButton_clicked(self):
        if self.labels_frozen:
            if not self.badSensorCheckBox.isChecked():
                self.set_label_state(True)
            else:
                self.badSensorCheckBox.setEnabled(True)

    def resetViewButton_clicked(self):
        self.w.setCameraPosition(pos=QtGui.QVector3D(0, 0, 0), distance=10, azimuth=180, elevation=10)

    def decrease_color(self):

        self.color_feature = (self.color_feature - 1) % (self.COLOR_FEATURES + 1)
        self.update_lidar3d()

    def increase_color(self):

        self.color_feature = (self.color_feature + 1) % (self.COLOR_FEATURES + 1)
        self.update_lidar3d()

    def decrease_min_channel(self):

        if self.min_channel > 0:

            self.min_channel = (self.min_channel - 1) % (self.MAX_CHANNEL + 1)
            self.update_lidar3d()

    def increase_min_channel(self):

        if self.min_channel < self.max_channel:

            self.min_channel = (self.min_channel + 1) % (self.MAX_CHANNEL + 1)
            self.update_lidar3d()

    def decrease_max_channel(self):

        if self.max_channel > self.min_channel:

            self.max_channel = (self.max_channel - 1) % (self.MAX_CHANNEL + 1)
            self.update_lidar3d()

    def increase_max_channel(self):

        if self.max_channel < self.MAX_CHANNEL:

            self.max_channel = (self.max_channel + 1) % (self.MAX_CHANNEL + 1)
            self.update_lidar3d()

    def decrease_height_threshold(self):

        self.arguments.min_height = self.arguments.min_height - self.height_step
        self.update_lidar3d()

    def increase_height_threshold(self):

        self.arguments.min_height = self.arguments.min_height + self.height_step
        self.update_lidar3d()

    def decrease_distance_threshold(self):

        self.distance_threshold = self.distance_threshold - self.distance_step
        self.update_lidar3d()

    def increase_distance_threshold(self):

        self.distance_threshold = self.distance_threshold + self.distance_step
        self.update_lidar3d()

    def iterate_distant_points(self):

        self.distant_points_iterator = (self.distant_points_iterator + 1) % 3
        self.update_lidar3d()

    def compare_strongest_and_last(self):

        self.arguments.compare = not self.arguments.compare
        self.update_lidar3d()
        self.update_lidar()

    def removeGroundButton_clicked(self):

        if self.life_is_easy:   # make it harder
            self.arguments.min_height = -np.infty
            self.arguments.min_dist = 0
            self.removeGroundButton.setText('make life easier') # give choice to make it easier again
        else:                   # make it easier
            self.arguments.min_height = MIN_HEIGHT
            self.arguments.min_dist = MIN_DIST
            self.removeGroundButton.setText('make life harder') # give schoice to make it harder again

        self.update_lidar3d()
        self.life_is_easy = not self.life_is_easy

    def goToIndexEdit_returnPressed(self):
        try:
            self.current_index = int(self.goToIndexEdit.text())
        except ValueError:
            print('Inserted value is non-integer!')
            return

        if self.current_index >= 0 and self.current_index < len(self.recordings):
            self.update_sample()

    def recordingComboBox_activated(self):
        try:
            self.current_index = self.recordings.index(
                [sample for sample in self.recordings if '{}'.format(self.recordingComboBox.currentText()) in sample][
                    0])
        except Exception:
            return
        self.update_sample()

    def sampleComboBox_activated(self):
        current_text = self.sampleComboBox.currentText()
        if current_text == 'All':
            self.update_image_boxes(None)
        else:
            idx = self.box_idxes.index(current_text)
            self.update_image_boxes(idx)

    def rgbComboBox_activated(self):
        self.rgb_topic = self.rgbComboBox.currentText()
        self.update_image()

    def gatedComboBox_activated(self):
        self.gated_topic = self.gatedComboBox.currentText()
        self.update_gated()

    def lidarComboBox_activated(self):
        self.lidar_topic = self.lidarComboBox.currentText()
        if self.lidar_projection_available:
            self.update_lidar()

    def lidar3dComboBox_activated(self):
        self.lidar3d_topic = self.lidar3dComboBox.currentText()
        self.MAX_CHANNEL = -1
        self.max_channel = -1
        self.min_channel = 0
        self.update_lidar3d()

    def resizeEvent(self, event):
        try:
            self.update_rgb()
            self.update_gated()
            if self.lidar_projection_available:
                self.update_lidar()
        except Exception:
            pass

    def set_bad_sensor(self):
        if self.badSensorCheckBox.isEnabled():
            self.badSensorCheckBox.setChecked(not self.badSensorCheckBox.isChecked())
            self.badSensorCheckBox_clicked()

    def set_no_objects(self):
        if self.noObjectsCheckBox.isEnabled():
            self.noObjectsCheckBox.setChecked(not self.noObjectsCheckBox.isChecked())
            self.badSensorCheckBox_clicked()
            self.save_labels()

    def set_discard(self):
        if self.ratingComboBox.isEnabled():
            self.ratingComboBox.setCurrentIndex(self.ratings.index('discard'))
            self.badSensorCheckBox_clicked()
            self.save_labels()

    def set_dispensable(self):
        if self.ratingComboBox.isEnabled():
            self.ratingComboBox.setCurrentIndex(self.ratings.index('dispensable'))
            self.badSensorCheckBox_clicked()
            self.save_labels()

    def set_appropriate(self):
        if self.ratingComboBox.isEnabled():
            self.ratingComboBox.setCurrentIndex(self.ratings.index('appropriate'))
            self.badSensorCheckBox_clicked()
            self.save_labels()

    def set_very_interesting(self):
        if self.ratingComboBox.isEnabled():
            self.ratingComboBox.setCurrentIndex(self.ratings.index('very_interesting'))
            self.badSensorCheckBox_clicked()
            self.save_labels()

    def set_interpolate(self):
        if self.ratingComboBox.isEnabled():
            self.ratingComboBox.setCurrentIndex(self.ratings.index('interpolate'))
            self.badSensorCheckBox_clicked()
            self.save_labels()

    def set_clear(self):
        if self.weatherComboBox.isEnabled():
            self.weatherComboBox.setCurrentIndex(self.weather.index('clear'))
            self.badSensorCheckBox_clicked()

    def set_light_fog(self):
        if self.weatherComboBox.isEnabled():
            self.weatherComboBox.setCurrentIndex(self.weather.index('light_fog'))
            self.badSensorCheckBox_clicked()

    def set_dense_fog(self):
        if self.weatherComboBox.isEnabled():
            self.weatherComboBox.setCurrentIndex(self.weather.index('dense_fog'))
            self.badSensorCheckBox_clicked()

    def set_rain(self):
        if self.weatherComboBox.isEnabled():
            self.weatherComboBox.setCurrentIndex(self.weather.index('rain'))
            self.badSensorCheckBox_clicked()

    def set_snow(self):
        if self.weatherComboBox.isEnabled():
            self.weatherComboBox.setCurrentIndex(self.weather.index('snow'))
            self.badSensorCheckBox_clicked()

    def set_day(self):
        if self.daytimeComboBox.isEnabled():
            self.daytimeComboBox.setCurrentIndex(self.daytimes.index('day'))
            self.badSensorCheckBox_clicked()

    def set_night(self):
        if self.daytimeComboBox.isEnabled():
            self.daytimeComboBox.setCurrentIndex(self.daytimes.index('night'))
            self.badSensorCheckBox_clicked()

    def closeEvent(self, event):
        self.save_current_index()
        event.accept()

    def save_current_index(self):
        index_file = self.get_index_file()

        with open(index_file, 'w') as outfile:
            outfile.write('{}'.format(self.current_index))


def main(name):

    args = get_args()

    args.point_size = 3.0
    args.fog_chamber = False
    args.split = 'test_clear_day'

    hostname = socket.gethostname()

    if 'MacBook' in hostname or '.ee.ethz.ch' in hostname:

        if args.fog_chamber:

            args.root_path = '/Users/Hahner/trace/datasets/DENSE/FogchamberDataset'

        else:

            args.root_path = '/Users/Hahner/trace/datasets/DENSE/SeeingThroughFog'

    elif 'beast' in hostname:

        args.full_screen = False

        if args.fog_chamber:

            args.root_path = '/srv/beegfs-benderdata/scratch/tracezuerich/data/datasets/DENSE/FogchamberDataset'

        else:

            args.root_path = '/scratch_net/beast_second/mhahner/datasets/DENSE/SeeingThroughFog'

    else:   # assume CVL host

        args.monitor = 1

        if args.fog_chamber:

            args.root_path = '/srv/beegfs-benderdata/scratch/tracezuerich/data/datasets/DENSE/FogchamberDataset'

        else:

            args.root_path = '/srv/beegfs-benderdata/scratch/tracezuerich/data/datasets/DENSE/SeeingThroughFog'

    try:
        with open('topics.json') as file:
            topics = json.loads(file.read())
    except IOError:
        print('Problem when reading topic file!')

    if args.fog_chamber:
        topics['gated'] = ['gated_fog_rect8']
        topics['lidar3d'] = topics['lidar3d'][0:2]

    try:
        with open(args.path_timestamps) as file:
            timedelays = json.loads(file.read())
    except IOError:
        timedelays = None
        print('Problem when reading matching file for AdverseWeather2Algolux!')

    app = QtGui.QApplication(sys.argv)

    DatasetViewer(args, topics, timedelays, name)

    sys.exit(app.exec_())


if __name__ == '__main__':
    main('mhahner')