import tensorflow as tf
import numpy as np
from data_generators.utils import maybe_download, prepare_dataset
import glob, os
import tarfile
import pickle

Dataset = tf.data.Dataset
num_classes = 11 # 10 classes + 1 for "fake"

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
DATA_DIR='./data/cifar10'
DATASET_FILE='cifar-10-python.tar.gz'

def read_batch(filepath):
    with open(filepath, mode='rb') as file:
        # In Python 3.X it is important to set the encoding,
        # otherwise an exception is raised here.
        data = pickle.load(file, encoding='bytes')

    images = np.array(data[b'data'], dtype=np.float32)
    images = np.reshape(images, (10000, 32, 32, 3))
    labels = np.array(data[b'labels'], dtype=np.float32)
    return images, labels
    
def read_datasets(train=True):
    # maybe download
    prepare_dataset(DATA_DIR, DATASET_FILE, DATA_URL)

    # prepare dataset
    train_files = glob.glob(DATA_DIR + '/cifar-10-batches-py/data_batch*')
    test_files = glob.glob(DATA_DIR + '/cifar-10-batches-py/test_batch')    
    files = train_files if train else test_files
    
    images = []
    labels = []
    for f in files:
        img, lbl = read_batch(f)
        images.append(img)
        labels.append(lbl)
    return images, labels

def _features(images, labels):
    images = tf.constant(np.concatenate(images))
    labels = np.concatenate(labels) + 1 # class=0 reserved for fake GAN images
    one_hot = tf.one_hot(np.array(labels), num_classes) 
    return images, one_hot

def generator(batch_size=32, train=True, shuffle=True):

    _images, _labels = read_datasets(train)

    def input_fn():
        images, labels = _features(_images, _labels)
        dataset = Dataset.from_tensor_slices((images, labels)) 
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()
        
    return input_fn