from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt


testing_dataset = [
    '/home/mikelf/Datasets/T-lessV2/shards/stn_test/tless_train-00000-of-00004-stn_75_75.tfrecords']


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        # features = tf.parse_single_example(
        #	serialized_example,
        # dense_keys=['image/image_raw', 'image/class/label', 'image/filename'],
        # Defaults are not specified since both keys are required.
        # dense_types=[tf.string, tf.int64, tf.string])

        features={

            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/index': tf.FixedLenFeature([], tf.int64),
            'image/filename': tf.FixedLenFeature([], tf.string),
        })

    # Convert from a scalar string tensor (whose single string has
    image = tf.image.decode_image(features['image/encoded'])

    image = tf.reshape(image, [227, 227, 3])
    image.set_shape([227, 227, 3])

    label = tf.cast(features['image/class/label'], tf.int32)
    index = tf.cast(features['image/index'], tf.int32)

    im_filename = tf.cast(features['image/filename'], tf.string)

    return image, label, im_filename, index

def main(unused_argv):

    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        filename_queue = tf.train.string_input_producer(testing_dataset)  # list of files to read
        image, label, image_name, index = read_and_decode(filename_queue)
        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        x = []

        # for i in range(1): #length of your filename list
        for i in range(20):

            image_array, label_str, index_str, filename_str = sess.run([image, label, index, image_name])

            plt.imshow(image_array)
            print(filename_str, label_str)
            plt.pause(0.01)

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    tf.app.run()


