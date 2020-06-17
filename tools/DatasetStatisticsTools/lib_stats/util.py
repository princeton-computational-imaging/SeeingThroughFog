import os
import matplotlib.pyplot as plt
import csv
import numpy as np


def map_visible(value):
    if value == 'True':
        return 1
    elif value == 'False':
        return 0
    else:
        return -1

def map_unsure3dBox(valuevisible, valueBox):
    if valuevisible == 'True' and valueBox>=0:
        return 1
    elif valuevisible == 'False' and valueBox>=0:
        return 0
    else:
        return -1


def load_gt_obj(path, min_box_size=None):
    """ load bbox ground truth from files either via the provided label directory or list of label files"""
    files = os.listdir(path)
    files = [x for x in files if x.endswith('.txt')]
    _objects_all = {}
    if len(files) == 0:
        raise RuntimeError('error: no label files found in %s' % path)
    for label_file in files:
        objects_per_image = list()
        with open(os.path.join(path, label_file), 'r') as flabel:
            for kitti_properties in csv.reader(flabel, delimiter=' '):
                if len(kitti_properties) == 0:
                    # This can happen when you open an empty file
                    continue
                if len(kitti_properties) < 15:
                    raise ValueError('Invalid label format in "%s"'
                                     % os.path.join(self.label_dir, label_file))

                # load data
                # Cant be read as 2 name part is beeing interpret as truncation label.
                if kitti_properties[0] == 'traffic':
                    continue
                object_dict = {
                    'identity':     kitti_properties[0],
                    'truncated':    float(kitti_properties[1]),
                    'occlusion':    float(kitti_properties[2]),
                    'angle':        float(kitti_properties[3]),
                    'xleft':        int(round(float(kitti_properties[4]))),
                    'ytop':         int(round(float(kitti_properties[5]))),
                    'xright':       int(round(float(kitti_properties[6]))),
                    'ybottom':      int(round(float(kitti_properties[7]))),
                    '2dboxheight':  float(kitti_properties[7])-float(kitti_properties[5]),
                    'height':       float(kitti_properties[8]),
                    'width':        float(kitti_properties[9]),
                    'length':       float(kitti_properties[10]),
                    'posx':         float(kitti_properties[11]),
                    'posy':         float(kitti_properties[12]),
                    'posz':         float(kitti_properties[13]),
                    'orient3d':     float(kitti_properties[14]),
                    'rotx':         float(kitti_properties[15]),
                    'roty':         float(kitti_properties[16]),
                    'rotz':         float(kitti_properties[17]),
                    'score':        float(kitti_properties[18]),
                    'qx':           float(kitti_properties[19]),
                    'qy':           float(kitti_properties[20]),
                    'qz':           float(kitti_properties[21]),
                    'qw':           float(kitti_properties[22]),
                    'visibleRGB':   map_visible(kitti_properties[23]),
                    'visibleGated': map_visible(kitti_properties[24]),
                    'visibleLidar': map_visible(kitti_properties[25]),
                    'unsure':       map_visible(kitti_properties[26]),
                    'unsure3dBox':  map_unsure3dBox(kitti_properties[26], float(kitti_properties[11])),
                }
                # setting the object from the string

                objects_per_image.append(object_dict)
            key = os.path.splitext(label_file)[0]
            _objects_all[key] = objects_per_image
    return _objects_all



def create_statitics(labels, selected_files, keys, allowed_classes):
    # create empty statistics dict
    statistics = dict()
    for key in keys:
        statistics[key] = []

    for frame in selected_files:
        if frame not in labels:
            continue
        for single_annotation in labels[frame]:
            if single_annotation['identity'] in allowed_classes:
                for key in keys:
                    statistics[key].append(single_annotation[key])
    return statistics

def create_statitics_object_classes(labels, selected_files, label_file='undefined'):
    statistics = dict()
    for frame in selected_files:
        if frame not in labels:
            continue
        for single_annotation in labels[frame]:
            object_class = single_annotation['identity']
            saved_number = statistics.get(object_class, 0)
            saved_number += 1
            statistics[object_class]=saved_number

    write_csv_classes(statistics, label_file=label_file)
    return statistics

def read_split(path_object_detection_files, file):
    with open(os.path.join(path_object_detection_files, file), 'r') as infile:
        indexes_split = infile.readlines()
    indexes_split = [i.replace(',','_').split('\n')[0] for i in indexes_split]
    return indexes_split

def write_csv(name, bin_means, bin_edges, label_file='undefined'):
    with open('statistics_output/%s_%s.csv'%(label_file, name), 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        x_data = ['BinMeans']+bin_means
        y_data = ['BinEdges']+bin_edges
        for x,y in zip(x_data, y_data):
            spamwriter.writerow([x,y])


def write_csv_classes(statistics, label_file='undefined'):
    with open('statistics_output/classes_%s.csv'%(label_file), 'w', newline='') as csvfile:
        for key in statistics:
            writer = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([key]+[statistics[key]])

def illustrate_statistics(statistics, statistics_params, label_file='undefined', Viszualize=False):
    for key in statistics:
        bin_means, bin_edges = np.histogram(statistics[key], 100, range=statistics_params[key]['range'])
        if Viszualize:
            plt.hist(statistics[key], 100, range=statistics_params[key]['range'])
            plt.show()
        write_csv(key, bin_means.tolist(), bin_edges.tolist(), label_file=label_file)