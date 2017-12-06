import tensorflow as tf
import numpy as np
import glob, os, pickle

from tensor2gan.data_generators import utils
from tensor2gan.data_generators.generator import DataGenerator

Dataset = tf.data.Dataset

POKEMON_URL='https://s3-us-west-2.amazonaws.com/twairball.datasets.pokemon/pokemon.tgz'
DIR_NAME='pokemon'
DATASET_FILE='pokemon.tgz'

def get_files(train=True, data_dir="./data"):
    """Returns Pokemon image files"""
    # maybe download
    target_dir = os.path.join(data_dir, DIR_NAME)
    utils.prepare_dataset(target_dir, DATASET_FILE, POKEMON_URL)

    # prepare dataset
    img_files = glob.glob('./pokemon' + '/*.png')
    train_files = img_files[:26000]
    test_files = img_files[26000:]
    
    files = train_files if train else test_files
    return files

def read_dataset(train=True, data_dir="./data"):
    files = get_files(train, data_dir)
    labels = np.ones(len(files))
    return files, labels

class GeneratePokemon(DataGenerator):
    
    @property
    def num_classes(self):
        return 1

    @property
    def input_shape(self):
        return [80,80,3]

    def get_record_filename(self, train):
        tag = "train" if train else "dev"
        return "pokemon_%s.tfrecords" % tag

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

        return record_filepath
    
    def get_input_fn(self, batch_size, data_dir="./data", 
        train=True):
        """Create input pipeline. Returns input_fn, a callable function that 
        returns next element in iterator.
        """
        record_filepath = self.prepare_data(data_dir, train)

        def parse_features(image, label):
            # reshape image
            image = tf.cast(tf.reshape(image, self.input_shape), tf.float32)
            label = tf.one_hot(label, self.num_classes)
            return image, label
            
        def input_fn():
            dataset = tf.data.TFRecordDataset([record_filepath])
            dataset = dataset.map(utils.parse_record)
            dataset = dataset.map(parse_features)
            dataset = dataset.shuffle(batch_size * 5).batch(batch_size).repeat()
            iterator = dataset.make_one_shot_iterator()
            return iterator.get_next()
        
        return input_fn
