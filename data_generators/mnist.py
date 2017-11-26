import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
Dataset = tf.data.Dataset

def generator(batch_size=32, train=True):
    def input_fn():
        _mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        # Create a dataset tensor from the images and the labels
        images = _mnist.train.images.reshape([-1, 28, 28, 1])
        # add extra class; 0 class reserved for "fake" images
        labels = _mnist.train.labels
        fake_class = np.zeros((len(labels),1))
        labels = np.concatenate([fake_class, labels], axis=1).astype('float32')
        dataset = Dataset.from_tensor_slices((images, labels))
        # Create batches of data
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    return input_fn
