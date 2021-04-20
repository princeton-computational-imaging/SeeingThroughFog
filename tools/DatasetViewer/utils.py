import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import cv2
from datetime import datetime
#from dense_tools.general.conversion import create_lut_from_kneepoints
import os
import glob
import json


def create_lut_from_kneepoints(kneepoints, bit_depth=16):
    lut_kneepoints = kneepoints[:]
    start_point = [0,0]
    end_point = [2**bit_depth, 2**bit_depth]
    lut_kneepoints.append(end_point)
    lut =  np.zeros((2**bit_depth,), dtype=np.uint16)


    counter = 0
    for idx, kneepoint in enumerate(lut_kneepoints):
        if counter == 0:
            first_point = start_point
            second_point = kneepoint
        else:
            first_point = lut_kneepoints[idx-1]
            second_point = lut_kneepoints[idx]
        counter += 1

        m = (second_point[1] - first_point[1])/(float(second_point[0] - first_point[0]))
        c = first_point[1] - m*first_point[0]

        lut[first_point[0]:second_point[0]] = np.floor(m*np.arange(first_point[0],second_point[0]) + c).astype(np.uint16)

    return lut


def colorize_pointcloud(depth, min_distance=3, max_distance=80, radius=3):
    norm = mpl.colors.Normalize(vmin=min_distance, vmax=max_distance)
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    pos = np.argwhere(depth > 0)

    pointcloud_color = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    for i in range(pos.shape[0]):
        color = tuple([int(255 * value) for value in m.to_rgba(depth[pos[i, 0], pos[i, 1]])[0:3]])
        cv2.circle(pointcloud_color, (pos[i, 1], pos[i, 0]), radius, (color[0], color[1], color[2]), -1)

    return pointcloud_color


def colorize_pointcloud_emphasize_clutter(depth, min_distance=3, max_distance=80, radius=3, threshold=15):
    norm = mpl.colors.Normalize(vmin=min_distance, vmax=max_distance)
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    pos = np.argwhere(depth > 0)

    pointcloud_color = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    for i in range(pos.shape[0]):
        color = tuple([int(255 * value) for value in m.to_rgba(depth[pos[i, 0], pos[i, 1]])[0:3]])
        if depth[pos[i, 0], pos[i, 1]] < threshold and pos[i, 0] < 550:
            r = 2 * radius
        else:
            r = radius
        cv2.circle(pointcloud_color, (pos[i, 1], pos[i, 0]), r, (color[0], color[1], color[2]), -1)

    return pointcloud_color


def colorize_depth(depth, min_distance=3, max_distance=80):
    norm = mpl.colors.Normalize(vmin=min_distance, vmax=max_distance)
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    depth_color = (255 * m.to_rgba(depth)[:, :, 0:3]).astype(np.uint8)
    depth_color[depth <= 0] = [0, 0, 0]
    depth_color[np.isnan(depth)] = [0, 0, 0]
    depth_color[depth == np.inf] = [0, 0, 0]

    return depth_color


def fb(x, bitdepth=16):
    return int(x*2**bitdepth)

def apply_clahe_8bit(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image[:, :, 0] = clahe.apply(image[:,:,0])
    clahe2 = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8, 8))
    image[:, :, 1] = clahe2.apply(image[:,:,1])
    image[:, :, 2] = clahe2.apply(image[:,:,2])
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

    return image


def image_processing(img, daytime, decomp):
    # This has to be initialized once
    bit_depth = 16
    kneepoints_day = [[fb(0.005), fb(0.05)], [fb(0.01), fb(0.2)], [fb(0.03), fb(0.35)], [fb(0.05), fb(0.4)], [fb(0.1), fb(0.5)], [fb(0.2), fb(0.7)], [fb(0.3), fb(0.8)], [fb(0.4), fb(0.9)], [fb(0.5), fb(0.98)]]
    kneepoints_night = [[fb(0.0025), fb(0.1)], [fb(0.005), fb(0.25)], [fb(0.01), fb(0.4)], [fb(0.1), fb(0.8)], [fb(0.2), fb(0.9)], [fb(0.3), fb(0.98)]]
    lut_night = create_lut_from_kneepoints(kneepoints_night, bit_depth=bit_depth)
    lut_day = create_lut_from_kneepoints(kneepoints_day, bit_depth=bit_depth)

    # decompand (create linear 16bit image from 12bit image)
    img = decomp.processImage(img)

    #8bit lookup table
    if daytime == 'day':
        img = lut_day[img]
    if daytime == 'night':
        img = lut_night[img]

    # debayer
    img = cv2.cvtColor(img, cv2.COLOR_BAYER_GB2BGR)
    img = np.right_shift(img, 8).astype(np.uint8)
    img = apply_clahe_8bit(img)

    return img


def get_daytime_from_can(root_dir, recording, sample):

    try:
        path = glob.glob(os.path.join(root_dir, recording, 'can/body/w222_body_can_2016_17a/LgtSens_State_AR', sample.split('_')[0] + '_*'))[0]
        with open(path) as f:
            can_light_sense = json.load(f)
        if can_light_sense['LgtSens_Night'] == 1:
            daytime = 'night'
        else:
            daytime = 'day'
    except Exception:
        daytime = 'night'

    return daytime


def convert_timestamp(timestamp):
    dt = datetime.fromtimestamp(timestamp // 1000000000)
    s = dt.strftime('%Y-%m-%d %H:%M:%S')
    s += '.' + str(int(timestamp % 1000000000)).zfill(9)
    return s


#es werden zwei timestamps uebergeben und es wird die Zeitdifference in ms berechnet
def get_time_difference(timestamp1, timestamp2):
    time_diff = np.absolute(timestamp1-timestamp2)      #in Nanosekunden
    return time_diff * 10**(-6)                         #in Millisekunden


def get_all_dataset_samples():
    all_samples = []
    with open(os.path.join(os.path.split(os.path.dirname(__file__))[0], 'adverse_files.txt'), 'r') as f:
        all_samples += f.read().splitlines()

    with open(os.path.join(os.path.split(os.path.dirname(__file__))[0], 'clear_files.txt'), 'r') as f:
        all_samples += f.read().splitlines()
    return all_samples


def get_additional_cmore_samples():
    all_samples = []
    with open(os.path.join(os.path.split(os.path.dirname(__file__))[0], 'additional_snow_files.txt'), 'r') as f:
        all_samples += f.read().splitlines()

    with open(os.path.join(os.path.split(os.path.dirname(__file__))[0], 'additional_clear_files.txt'), 'r') as f:
        all_samples += f.read().splitlines()
    return all_samples


def main():
    image_processing(1, 2)


if __name__ == '__main__':
    main()
