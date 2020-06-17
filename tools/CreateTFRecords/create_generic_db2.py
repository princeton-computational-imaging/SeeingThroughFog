#!/usr/bin/env python2
# Copyright (c) 2016-2017, Daimler AG.  All rights reserved.

import argparse
# Find the best implementation available
import logging
import os

from generic_tf_tools.tf_records import TFCreator
from generic_tf_tools.data2example import SwedenImagesv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(name='TfRecordsBuild')

def parsArgs():
    parser = argparse.ArgumentParser(description='Build TF Records')
    parser.add_argument('--source_dir', '-r', help='Enter the raw data source folder', default='')
    parser.add_argument('--dest_dir', '-d', type=str, help='definde destination directory')
    parser.add_argument('--dataset-id', '-id', type=str, help='defined dataset id')
    parser.add_argument('--file_list', '-f', help='Enter path to split files', default='DepthData')
    parser.add_argument('--dataset_type', '-t', help='Enter Dataset Type', default='FullSeeingThroughFogDataset')
    parser.add_argument('--batch_size', '-bs', type=int, help='Enter Batch Size per Record File', default=4)
    parser.add_argument('--num_threads', '-nt', type=int, help='Enter Number of Threads for parallel execution', default=1)
    parser.add_argument('--force_same_shape', '-fs', type=bool, help='Enforce same shape for all examples. Safety Feature not implemented', default=False)
    parser.add_argument('--stage', '-s', help='Stage (train, val, test)', default='train')
    args = parser.parse_args()
    global hazed
    return args


def create_generic_db(args):
    """
    Create a generic DB
    """

    # load dataset job
    dataset_dir = os.path.join(args.dest_dir, args.dataset_id)
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
        #raise IOError("Dataset dir %s does not exist" % dataset_dir)

    batch_size = args.batch_size
    num_threads = args.num_threads

    force_same_shape = args.force_same_shape

    with open(args.file_list, 'r') as f:
        entry_ids = f.readlines()
        entry_ids = [i.replace(',','_').split('\n')[0] for i in entry_ids]


    # create main DB creator object and execute main method

    records_dir = os.path.join(dataset_dir, args.stage)
    if not os.path.exists(records_dir):
        os.makedirs(records_dir)
    conversionClass = None
    if args.dataset_type == 'FullSeeingThroughFogDataset':
        conversionClass = SwedenImagesv2(source_dir=args.source_dir)
    else:
        logger.error('Wrong TF conversion Class specified')
        raise ValueError

    tf_creator = TFCreator(entry_ids,
                           args.stage,
                           args.source_dir,
                           records_dir,
                           batch_size,
                           num_threads,
                           conversionClass,
                           args.force_same_shape)
    tf_creator()

    logger.info('Generic TF-DB creation Done')
    logger.info('Created %s db for stage %s in %s' % ('features', args.stage, args.source_dir))


if __name__ == '__main__':

    args = parsArgs()

    try:
        create_generic_db(
            args
        )
    except Exception as e:
        logger.error('Failed DatasetBuild')
        raise

