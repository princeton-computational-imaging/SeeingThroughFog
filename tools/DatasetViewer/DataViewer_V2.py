import sys
import os
# sys.path.append('/scratch/fs2/Software/AdverseWeatherLabeling/lib')
# sys.path.append('/scratch/fs2/Software/frameworks/dense_label_tools')
# sys.path.append('/scratch/fs2/Software/frameworks/dense_label_tools/lib')

import pyqtgraph.opengl as gl
import matplotlib as mpl
import matplotlib.cm as cm
import argparse
from PyQt5 import QtGui, QtCore, uic
import json
import numpy as np
import cv2
from datetime import datetime
from utils_DataViewer import convert_timestamp, colorize_pointcloud, get_time_difference
from lib.read import load_calib_data, read_label
from lib.visualization import draw_bbox2d_from_kitti, build_bbox3d_from_params, project_points_to_2d, draw_bbox3d


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', help='Path to Dataset root directory', required=True)
    parser.add_argument('--view_only', default=False, help='Prevent Label Changes')
    parser.add_argument('--path_timestamps', default='./timestamps.json', help='Prevent Label Changes')
    parser.add_argument('--username', default='admin', help='Enter your username to recover and save the current index.')
    return parser.parse_args()

class DatasetViewer(QtGui.QMainWindow):
    def __init__(self, root_dir, topics, timedelays, can_speed_topic, can_steering_angle_topic,
                 can_light_sense_topic, can_wiper_topic, road_friction_topic, weather_topic, label_topic, name,
                 view_only=False, key=None):
        super(DatasetViewer, self).__init__()

        self.root_dir = root_dir
        self.name = name
        self.topics = topics
        self.rgb_default = topics['rgb'][0]
        self.gated_default = topics['gated'][0]
        self.lidar_default = topics['lidar'][0]
        self.lidar3d_default = topics['lidar3d'][0]
        self.can_speed_topic = can_speed_topic
        self.can_steering_angle_topic = can_steering_angle_topic
        self.can_light_sense_topic = can_light_sense_topic
        self.can_wiper_topic = can_wiper_topic
        self.road_friction_topic = road_friction_topic
        self.weather_topic = weather_topic
        self.label_topic = label_topic
        self.boxes = []  # Needed for 3d lidar boxes plot
        self.dir_labels = 'labeltool_labels'
        self.timedelays = timedelays

        print('ROOT DIR: ' + str(self.root_dir))

        if key is not None:
            self.recordings = None
        else:
            self.recordings = sorted(os.listdir(os.path.join(self.root_dir, self.rgb_default)))

        self.weather = ['clear', 'light_fog', 'dense_fog', 'rain', 'snow']
        self.daytimes = ['day', 'night']
        self.ratings = ['discard', 'dispensable', 'appropriate', 'very_interesting', 'interpolate']
        self.view_only = view_only

        QtGui.QShortcut(QtGui.QKeySequence('PgDown'), self, self.prev_sample)
        QtGui.QShortcut(QtGui.QKeySequence('PgUp'), self, self.next_sample)
        QtGui.QShortcut(QtGui.QKeySequence('Esc'), self, self.close)

        QtGui.QShortcut(QtGui.QKeySequence('1'), self, self.set_bad_sensor)
        QtGui.QShortcut(QtGui.QKeySequence('2'), self, self.set_no_objects)

        QtGui.QShortcut(QtGui.QKeySequence('q'), self, self.set_discard)
        QtGui.QShortcut(QtGui.QKeySequence('w'), self, self.set_dispensable)
        QtGui.QShortcut(QtGui.QKeySequence('e'), self, self.set_appropriate)
        QtGui.QShortcut(QtGui.QKeySequence('r'), self, self.set_very_interesting)
        QtGui.QShortcut(QtGui.QKeySequence('t'), self, self.set_interpolate)

        QtGui.QShortcut(QtGui.QKeySequence('a'), self, self.set_clear)
        QtGui.QShortcut(QtGui.QKeySequence('s'), self, self.set_light_fog)
        QtGui.QShortcut(QtGui.QKeySequence('d'), self, self.set_dense_fog)
        QtGui.QShortcut(QtGui.QKeySequence('f'), self, self.set_rain)
        QtGui.QShortcut(QtGui.QKeySequence('g'), self, self.set_snow)

        QtGui.QShortcut(QtGui.QKeySequence('y'), self, self.set_day)
        QtGui.QShortcut(QtGui.QKeySequence('x'), self, self.set_night)

        QtGui.QShortcut(QtGui.QKeySequence('l'), self, self.editLabelsButton_clicked)

        self.get_current_index()
        self.create_label_folders()
        self.count_labels_first()

        self.labels_frozen = False
        self.initUI()

    def initUI(self):
        self.ui = uic.loadUi('src/DataViewerGUI.ui', self)
        # self.showMaximized()

        self.w = gl.GLViewWidget()
        self.w.setCameraPosition(pos=[0, 0, 0], distance=10, azimuth=180, elevation=10)
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
        self.update_speed()
        self.update_angle()
        self.update_daytime()
        self.update_wiper()
        self.update_road_friction()
        self.update_weather()

    def update_speed(self):
        path = self.get_path('can_speed')
        if path is None:
            self.speedEdit.setText('N/A')
        else:
            with open(path) as f:
                can_speed = json.load(f)
            self.speedEdit.setText('{0:.2f} km/h'.format(can_speed['VehSpd_Disp']))

    def update_angle(self):
        path = self.get_path('can_steering_angle')
        if path is None:
            self.angleEdit.setText('N/A')
        else:
            with open(path) as f:
                can_steering_angle = json.load(f)
            self.angleEdit.setText('{0:.2f} \xb0'.format(can_steering_angle['StWhl_Angl']))

    def update_daytime(self):
        path = self.get_path('can_light_sense')
        if path is None:
            self.daytimeEdit.setText('N/A')
            self.daytime = 'night'
        else:
            with open(path) as f:
                can_light_sense = json.load(f)
            if can_light_sense['LgtSens_Night'] == 1:
                self.daytime = 'night'
            else:
                self.daytime = 'day'
            self.daytimeEdit.setText('{}'.format(self.daytime))

    def update_wiper(self):
        path = self.get_path('can_wiper')
        if path is None:
            self.wiperEdit.setText('N/A')
        else:
            with open(path) as f:
                can_wiper = json.load(f)
            self.wiperEdit.setText('{}'.format(can_wiper['Wpr_Stat']))

    def update_road_friction(self):
        path = self.get_path('road_friction')
        if path is None or os.path.exists(path) is False:
            self.roadFrictionEdit.setText('N/A')
        else:
            with open(path) as f:
                road_friction = json.load(f)
            self.roadFrictionEdit.setText('{}'.format(road_friction['surface_state_result']).lower())

    def update_weather(self):
        path = self.get_path('weather')
        if path is None or os.path.exists(path) is False:
            self.outTempEdit.setText('N/A')
            self.outHumidityEdit.setText('N/A')
            self.dewpointEdit.setText('N/A')
        else:
            with open(path) as f:
                weather = json.load(f)
            self.outTempEdit.setText('{0:.2f} \xb0C'.format((weather['outTemp'] - 32.0)*5/9))
            self.outHumidityEdit.setText('{0:.2f} %'.format(weather['outHumidity']))
            self.dewpointEdit.setText('{0:.2f} \xb0C'.format((weather['dewpoint'] - 32.0)*5/9))

    def update_image(self):
        self.update_calib()
        self.update_cmore_labels()
        self.update_rgb()
        self.update_gated()
        self.update_lidar()
        self.update_lidar3d()
    
    def update_image_boxes(self, idx):
        if idx is not None:
            object_idx = slice(max(0, idx-1), idx)
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
        self.camera_to_velodyne_rgb, self.P_rgb, self.zero_to_camera_rgb = self.read_calib('rgb')
        self.camera_to_velodyne_gated, self.P_gated, self.zero_to_camera_gated = self.read_calib('gated')

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

        _, camera_to_velodyne, P, _, _, _, zero_to_camera = load_calib_data(root, name_camera_calib, tf_tree)
        return camera_to_velodyne, P, zero_to_camera

    def read_labels(self, topic_type, calib):
        label_path = os.path.join(self.root_dir, self.label_topic[topic_type])
        recording = os.path.splitext(self.recordings[self.current_index])[0]  # here without '.txt' as it will be added in read_label function
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
            print('Wrong topic_type draw_3d_labels_in_rgb!')

        if label_idx is None:
            label_idx = slice(len(label))

        for objects in label[label_idx]:
            corners_list.append(build_bbox3d_from_params(objects, zero_to_camera=zero_to_camera))
            image = draw_bbox2d_from_kitti(image, objects)

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
        self.displayImage(image, self.lidarLabel)

    def update_lidar3d(self, label_idx=None):
        path = self.get_path('lidar3d')

        timestamp = self.get_timestamp_from_data_name(topic='lidar')
        if timestamp is not None:
            time_diff = get_time_difference(self.stereo_timestamp, timestamp)
            self.timeLidar3dEdit.setText('{} ms'.format(time_diff))
        else:
            self.timeLidar3dEdit.setText('No timestamp found!')

        pc = np.fromfile(path, dtype=np.float32)
        try:
            pc = pc.reshape((-1, 5))
        except Exception:
            pc = pc.reshape((-1, 4))

        norm = mpl.colors.Normalize(vmin=3, vmax=80)
        cmap = cm.jet
        m = cm.ScalarMappable(norm=norm, cmap=cmap)

        if label_idx is None:
            label_idx = slice(len(self.labels_rgb))

        colors = m.to_rgba(np.linalg.norm(pc[:, 0:3], axis=1))
        colors[:, [2, 1, 0, 3]] = colors[:, [0, 1, 2, 3]]
        colors[:, 3] = 0.5

        try:
            self.w.removeItem(self.plot)
            for box in self.boxes:
                self.w.removeItem(box)
        except Exception:
            pass

        self.boxes = []
        size = QtGui.QVector3D(1, 1, 1)
        for objects in self.labels_rgb[label_idx]:
            box = gl.GLBoxItem(size, color=(255, 255, 255, 255))
            box.setSize(objects['length'], objects['width'], objects['height'])
            box.translate(-objects['length']/2, -objects['width']/2, -objects['height']/2)
            box.rotate(angle=-objects['rotz'] * 180 / 3.14159265359, x=0, y=0, z=1)
            box.rotate(angle=-objects['roty'] * 180 / 3.14159265359, x=0, y=1, z=0)
            box.rotate(angle=-objects['rotx'] * 180 / 3.14159265359, x=1, y=0, z=0)
            box.translate(0, 0, objects['height']/2)
            box.translate(objects['posx_lidar'], objects['posy_lidar'], objects['posz_lidar'])
            self.boxes.append(box)

        self.plot = gl.GLScatterPlotItem(pos=np.asarray(pc[:, 0:3]), size=5.5, color=colors)
        self.w.addItem(self.plot)
        for box in self.boxes:
            self.w.addItem(box)

        self.lidar3dGridLayout.addWidget(self.w)

    def update_labels(self):
        labels = self.load_labels()

        if labels:
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

        else:
            if not self.keepLabelsCheckBox.isChecked():
                self.set_init_label_values()

            self.labels_frozen = False

        self.editLabelsButton.setEnabled(self.labels_frozen)

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
        return os.path.join(self.root_dir, self.dir_labels, recording + '.json')

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
        samples_labeled = [os.path.splitext(item)[0] for item in os.listdir(os.path.join(self.root_dir, self.dir_labels))]
        self.samples_labeled = self.samples_labeled + samples_labeled

    def create_label_folders(self):
        if not os.path.exists(os.path.join(self.root_dir, self.dir_labels)):
            os.mkdir(os.path.join(self.root_dir, self.dir_labels))

    def get_path(self, topic):
        recording = self.recordings[self.current_index]

        if topic == 'rgb':
            return os.path.join(self.root_dir, self.rgb_topic, recording)
        if topic == 'gated':
            return os.path.join(self.root_dir, self.gated_topic, recording)
        if topic == 'lidar':
            return os.path.join(self.root_dir, self.lidar_topic, os.path.splitext(recording)[0] + '.npz')
        if topic == 'lidar3d':
            return os.path.join(self.root_dir, self.lidar3d_topic, os.path.splitext(recording)[0] + '.bin')
        if topic == 'can_speed':
            try:
                return os.path.join(self.root_dir, self.can_speed_topic, os.path.splitext(recording)[0] + '.json')
            except Exception:
                return None
        if topic == 'can_steering_angle':
            try:
                return os.path.join(self.root_dir, self.can_steering_angle_topic, os.path.splitext(recording)[0] + '.json')
            except Exception:
                return None
        if topic == 'can_light_sense':
            try:
                return os.path.join(self.root_dir, self.can_light_sense_topic, os.path.splitext(recording)[0] + '.json')
            except Exception:
                return None
        if topic == 'can_wiper':
            try:
                return os.path.join(self.root_dir, self.can_wiper_topic, os.path.splitext(recording)[0] + '.json')
            except Exception:
                return None
        if topic == 'road_friction':
            try:
                return os.path.join(self.root_dir, self.road_friction_topic, os.path.splitext(recording)[0] + '.json')
            except Exception:
                return None
        if topic == 'weather':
            try:
                return os.path.join(self.root_dir, self.weather_topic, os.path.splitext(recording)[0] + '.json')
            except Exception:
                return None
        if topic == 'label':
            label_path = os.path.join(self.root_dir, self.dir_labels, os.path.splitext(recording)[0] + '.json')
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
        self.w.setCameraPosition(pos=[0, 0, 0], distance=10, azimuth=180, elevation=10)

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
            self.current_index = self.recordings.index([sample for sample in self.recordings if '{}'.format(self.recordingComboBox.currentText()) in sample][0])
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
        self.update_lidar()

    def lidar3dComboBox_activated(self):
        self.lidar3d_topic = self.lidar3dComboBox.currentText()
        self.update_lidar3d()

    def resizeEvent(self, event):
        try:
            self.update_rgb()
            self.update_gated()
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

    root_dir = args.root_path


    try:
        with open('topics.json') as file:
            topics = json.loads(file.read())
    except IOError:
        print('Problem when reading topic file!')

    try:
        with open(args.path_timestamps) as file:
            timedelays = json.loads(file.read())
    except IOError:
        print('Problem when reading matching file for AdverseWeather2Algolux!')

    label_topics = {
        'rgb': 'gt_labels/cam_left_labels_TMP',
        'gated': 'gt_labels/gated_labels_TMP',
    }

    can_steering_angle_topic = 'filtered_relevant_can_data/can_body_chassis'
    can_speed_topic = 'filtered_relevant_can_data/can_body_basic'
    can_light_sense_topic = 'filtered_relevant_can_data/can_body_lightsense'
    can_wiper_topic = 'filtered_relevant_can_data/can_body_wiper'
    road_friction_topic = 'road_friction'
    weather_topic = 'weather_station'

    app = QtGui.QApplication(sys.argv)
    DatasetViewer(root_dir, topics, timedelays, can_speed_topic, can_steering_angle_topic,
                  can_light_sense_topic, can_wiper_topic, road_friction_topic, weather_topic, label_topics, name,
                  view_only=args.view_only)
    sys.exit(app.exec_())


if __name__ == '__main__':
    args=get_args()
    main(args)
