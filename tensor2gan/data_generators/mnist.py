from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os

from tensor2gan.data_generators import utils
from tensor2gan.data_generators.generator import DataGenerator
from tensor2gan.utils import registry

from tensorflow.examples.tutorials.mnist import input_data

Dataset = tf.data.Dataset

DIR_NAME = "MNIST_DATA"

@registry.register_model
class GenerateMnist(DataGenerator):

    @property
    def num_classes(self):
        return 10

    @property
    def input_shape(self):
        return [28,28,1]
    
    def get_input_fn(self, batch_size, data_dir, train=True):
        """Create input pipeline. Returns input_fn, a callable 
        function that returns next element in iterator.
        """
        path = os.path.join(data_dir, DIR_NAME)
        _mnist = input_data.read_data_sets(path, one_hot=True)

        # Create a dataset tensor from the images and the labels
        images = _mnist.train.images.reshape([-1, 28, 28, 1])
        # add extra class; 0 class reserved for "fake" images
        labels = _mnist.train.labels
        fake_class = np.zeros((len(labels),1))
        labels = np.concatenate([fake_class, labels], axis=1).astype('float32')

        def input_fn():
            # TODO: add labels
            # dataset = Dataset.from_tensor_slices((images, labels))
            dataset = Dataset.from_tensor_slices(images)

            # Create batches of data
            dataset = dataset.batch(batch_size)
            iterator = dataset.make_one_shot_iterator()
            return iterator.get_next()

        return input_fn