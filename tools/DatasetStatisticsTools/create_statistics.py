import numpy as np
import argparse

from lib_stats.util import load_gt_obj, read_split, create_statitics_object_classes, create_statitics, illustrate_statistics


def parsArgs():
    parser = argparse.ArgumentParser(description='Build TF Records')
    parser.add_argument('--label_dir', '-r', help='Enter the raw data source folder')
    parser.add_argument('--split_dir', '-d', type=str, help='definde destination directory', default='../../splits')
    parser.add_argument('--dataset-id', '-id', type=str, help='defined dataset id')
    parser.add_argument('--file_list', '-f', help='Enter path to split files', default='DepthData')
    parser.add_argument('--dataset_type', '-t', help='Enter Dataset Type', default='FullSeeingThroughFogDataset')
    parser.add_argument('--batch_size', '-bs', type=int, help='Enter Batch Size per Record File', default=4)
    parser.add_argument('--num_threads', '-nt', type=int, help='Enter Number of Threads for parallel execution', default=1)
    parser.add_argument('--force_same_shape', '-fs', type=bool, help='Enforce same shape for all examples. Safety Feature not implemented', default=False)
    parser.add_argument('--stage', '-s', help='Stage (train, val, test)', default='train')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parsArgs()
    labels = load_gt_obj(args.label_dir)
    label_files = ['all']
    for label_file in label_files:
        seleced_files = read_split(args.split_dir, label_file + '.txt')
        print('Missing Labels', len(seleced_files - labels.keys()))
        print('Missing Labels', seleced_files - labels.keys())

        multiple_allowed_object_classes = [['Pedestrian'], ['PassengerCar']]

        create_statitics_object_classes(labels, seleced_files, label_file=label_file)

        for allowed_object_classes in multiple_allowed_object_classes:
            if allowed_object_classes[0] == 'Pedestrian':
                statistics_params = {
                    '2dboxheight': {'range': (0, 1000)},
                    'height': {'range': (0, 3)},
                    'width': {'range': (0, 3)},
                    'length': {'range': (0, 3)},
                    'posx': {'range': (0, 100)},
                    'posy': {'range': (0, 100)},
                    'posz': {'range': (0, 100)},
                    'rotx': {'range': (0, np.pi / 10)},
                    'roty': {'range': (0, np.pi / 10)},
                    'rotz': {'range': (0, np.pi)},
                    'visibleRGB': {'range': (-2, 2)},
                    'visibleGated': {'range': (-2, 2)},
                    'visibleLidar': {'range': (-2, 2)},
                    'unsure': {'range': (-2, 2)},
                    'unsure3dBox': {'range': (-2, 2)},
                    'truncated': {'range': (-1, 1)},
                }
            else:
                statistics_params = {
                    '2dboxheight': {'range': (0, 1000)},
                    'height': {'range': (0, 3)},
                    'width': {'range': (0, 5)},
                    'length': {'range': (0, 10)},
                    'posx': {'range': (0, 100)},
                    'posy': {'range': (0, 100)},
                    'posz': {'range': (0, 100)},
                    'rotx': {'range': (0, np.pi / 10)},
                    'roty': {'range': (0, np.pi / 10)},
                    'rotz': {'range': (0, np.pi)},
                    'visibleRGB': {'range': (-2, 2)},
                    'visibleGated': {'range': (-2, 2)},
                    'visibleLidar': {'range': (-2, 2)},
                    'unsure': {'range': (-2, 2)},
                    'unsure3dBox': {'range': (-2, 2)},
                    'truncated': {'range': (-1, 1)},
                }
            statistics = create_statitics(labels, seleced_files, statistics_params.keys(), allowed_object_classes)

            illustrate_statistics(statistics, statistics_params, label_file=label_file+'_'.join(allowed_object_classes))
