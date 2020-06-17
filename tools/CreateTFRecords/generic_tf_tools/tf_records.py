import tensorflow as tf
import numpy as np
import logging

logger = logging.getLogger(name='TfRecordsBuild')


class TFCreator(object):


    def __init__(self, entry_ids,
                 stage,
                 source_dir,
                 dataset_dir,
                 batch_size,
                 num_threads,
                 conversionClass,
                 force_same_shape):

        # retrieve itemized list of entries
        self.force_same_shape = force_same_shape
        self.batch_size = batch_size
        self.stage = stage
        self.source_dir = source_dir
        self.num_threads = num_threads
        self.dataset_dir = dataset_dir
        self.entry_count = len(entry_ids)
        if self.num_threads <= 1:
            self.Parellize = False
        else:
            self.Parellize = True

        self.n_files = len(entry_ids)
        self.conversionClass = conversionClass
        # sort entry ids
        aix = sorted(range(len(entry_ids)), key=lambda k: entry_ids[k])
        entry_ids = [entry_ids[num] for num in aix]
        # create batches of entry id's
        self.reshaped_entry_ids = [entry_ids[idx * batch_size:idx * batch_size + batch_size] for idx in
                                   range(0, int(len(entry_ids) / batch_size))]
        self.reshaped_entry_ids.append(entry_ids[int(len(entry_ids) / batch_size) * batch_size:-1])

    def __call__(self, *args, **kwargs):
        label_shape, feature_shape, feature_sum = [None, None, None]
        if self.Parellize == False:
            label_shape, feature_shape, feature_sum = self.loop()
        elif self.Parellize == True:
            label_shape, feature_shape, feature_sum = self.parallellized_loop()

        logger.info('Found %d entries for stage %s' % (self.n_files, self.stage))

    def process_example(self, data_dict, tfrecord_writer):


        tf_train_example = self.conversionClass.create_example(data_dict)
        tfrecord_writer.write(tf_train_example.SerializeToString())

        return None, None, None

    def loop(self):
        label_shape, feature_shape, feature_sum = [None, None, None]
        for tfrecord_file_idx, batch in enumerate(self.reshaped_entry_ids):
            # print('tfrecord_file_idx, batch', tfrecord_file_idx, batch)
            tf_filename = self.conversionClass.get_output_filename(output_dir=self.dataset_dir,
                                                                   name=self.stage,
                                                                   idx=tfrecord_file_idx)

            label_shape, feature_shape, feature_sum = self.write_tf_records_file(tf_filename, batch, tfrecord_file_idx)

            logger.info('Processed %d/%d' % (tfrecord_file_idx + 1, len(self.reshaped_entry_ids)))

        return label_shape, feature_shape, feature_sum

    def procedure(self, j):  # just factoring out the
        tfrecord_file_idx = j
        batch = self.reshaped_entry_ids[tfrecord_file_idx]
        tf_filename = self.conversionClass.get_output_filename(output_dir=self.dataset_dir,
                                                               name=self.stage,
                                                               idx=tfrecord_file_idx)
        logger.info('Processed %d/%d' % (tfrecord_file_idx + 1, len(self.reshaped_entry_ids)))

        # call the calculation
        return self.write_tf_records_file(tf_filename, batch, j)

    def parallellized_loop(self):

        process_map = lambda x: self.procedure(x)

        import pathos.pools as pp
        pool = pp.ProcessPool(self.num_threads)

        a = pool.map(process_map, range(len(self.reshaped_entry_ids)))

        label_shape, feature_shape, feature_sum = a[0]

        return label_shape, feature_shape, feature_sum

    def write_tf_records_file(self, tf_filename, batch, j):
        feature_sum = None
        label_shape_out = None
        feature_shape_out = None
        feature_out = None

        with tf.io.TFRecordWriter(tf_filename) as tfrecord_writer:
            for idx, entry_id in enumerate(batch):
                total_id = j + idx

                feature = None
                data_file_idx = 0
                feature_shape = None
                numpy_lidar = None
                numpy_lidar_shape = None

                data = self.conversionClass.read_data(entry_id, total_id)

                label_shape_out, feature_shape_out, feature_out = self.process_example(data, tfrecord_writer)

        return label_shape_out, feature_shape_out, feature_out
