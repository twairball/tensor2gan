import tensorflow as tf
import numpy as np
import glob, os, pickle

from tensor2gan.data_generators import utils
from tensor2gan.data_generators.generator import DataGenerator

Dataset = tf.data.Dataset

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
DIR_NAME='cifar10'
DATASET_FILE='cifar-10-python.tar.gz'

def read_batch(filepath):
    """reads a cifar10 batch file"""
    with open(filepath, mode='rb') as file:
        # In Python 3.X it is important to set the encoding,
        # otherwise an exception is raised here.
        data = pickle.load(file, encoding='bytes')

    images = np.array(data[b'data'], dtype=np.float32)
    images = np.reshape(images, (10000, 32, 32, 3))
    labels = np.array(data[b'labels'], dtype=np.float32)
    return images, labels
    
def get_files(train=True, data_dir="./data"):
    """returns cifar10 batch files, e.g. data_batch_1"""
    # maybe download
    target_dir = os.path.join(data_dir, DIR_NAME)
    utils.prepare_dataset(target_dir, DATASET_FILE, DATA_URL)

    # prepare dataset
    train_files = glob.glob(target_dir + '/cifar-10-batches-py/data_batch*')
    test_files = glob.glob(target_dir + '/cifar-10-batches-py/test_batch')    
    files = train_files if train else test_files
    return files

def read_dataset(train=True, data_dir="./data"):
    """Returns (images, files) tuple of whole dataset"""
    files = get_files(train, data_dir)
    images = []
    labels = []
    for f in files:
        img, lbl = read_batch(f)
        images.append(img)
        labels.append(lbl)
    return np.concatenate(images), np.concatenate(labels).astype(int)


class GenerateCIFAR10(DataGenerator):
    
    @property
    def num_classes(self):
        return 10

    @property
    def input_shape(self):
        return [32,32,3]

    def get_record_filename(self, train):
        tag = "train" if train else "dev"
        return "cifar10_%s.tfrecords" % tag

    def prepare_data(self, data_dir, train):
        """Write dataset to TFRecords"""
        record_filepath = os.path.join(data_dir, self.get_record_filename(train))

        # write dataset to tfrecords if not exist
        if not utils.do_files_exist(record_filepath):
            tf.logging.info("Writing TFRecord to: %s" % record_filepath)
            images, labels = read_dataset(train, data_dir)
            utils.write_to_tf_records(record_filepath, images, labels)
        else:
            tf.logging.info("Skipping writing TFRecord, found: %s" % record_filepath)

    def get_input_fn(self, batch_size, data_dir, train=True):
        """Create input pipeline. Returns input_fn, a callable function that 
        returns next element in iterator.
        """
        self.prepare_data(data_dir, train)

        def parse_features(image, label):
            # reshape image
            image = tf.cast(tf.reshape(image, self.input_shape), tf.float32)
            label = tf.one_hot(label, self.num_classes)
            return image, label
            
        def input_fn():
            dataset = tf.data.TFRecordDataset([self.get_record_filename(train)])
            dataset = dataset.map(utils.parse_record)
            dataset = dataset.map(parse_features)
            dataset = dataset.shuffle(batch_size * 5).batch(batch_size).repeat()
            iterator = dataset.make_one_shot_iterator()
            return iterator.get_next()
        
        return input_fn