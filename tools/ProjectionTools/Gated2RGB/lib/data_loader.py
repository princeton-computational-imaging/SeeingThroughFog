import cv2
import numpy as np
import os
import json
from datetime import datetime


def load_image(filename, grayscale=False):
    image = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    if np.amax(image) > 255 and np.amax(image) <= 1023:
        image = np.right_shift(image, 2).astype(np.uint8)
    elif np.amax(image) > 1023 and np.amax(image) <= 4095:
        image = np.right_shift(image, 4).astype(np.uint8)

    if grayscale:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image


def load_vehicle_speed(root, sample):
    path = os.path.join(root,'filtered_relevant_can_data/can_body_basic', sample + '.json')
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return data['VehSpd_Disp']
    else:
        return 0

def load_stearing_ange(root, sample):
    path = os.path.join(root,'filtered_relevant_can_data/can_body_chassis', sample + '.json')
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)

        return data['StWhl_Angl']
    else:
        return 0


def load_time(sensor,sample):

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"../../../DatasetViewer/timestamps.json")) as f:
        data = json.load(f)
    timestamp = int(data[sensor][sample].split('_')[1])
    return convert_timestamp(timestamp), timestamp

def convert_timestamp(timestamp):
    dt = datetime.fromtimestamp(timestamp // 1000000000)
    s = dt.strftime('%Y-%m-%d %H:%M:%S')
    s += '.' + str(int(timestamp % 1000000000)).zfill(9)
    return s