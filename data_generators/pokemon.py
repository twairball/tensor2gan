import tensorflow as tf
import numpy as np
from .utils import maybe_download, prepare_dataset
import glob, os
import tarfile

Dataset = tf.data.Dataset
num_classes = 2

POKEMON_URL='https://s3-us-west-2.amazonaws.com/twairball.datasets.pokemon/pokemon.tgz'
DATA_DIR='./pokemon'
DATASET_FILE='pokemon.tgz'

def read_datasets(train=True):
    # maybe download
    prepare_dataset(DATA_DIR, DATASET_FILE, POKEMON_URL)

    # prepare dataset
    img_files = glob.glob('./pokemon' + '/*.png')
    train_files = img_files[:26000]
    test_files = img_files[26000:]
    
    files = train_files if train else test_files
    labels = np.ones(len(files))
    return files, labels

    # imgs = tf.constant(files)
    # labels = tf.ones(len(files), dtype=tf.int32)
    # return imgs, labels

def generator(batch_size=32, train=True, shuffle=True):
    
    imgs, labels = read_datasets(train)

    def input_parser(img_path, label):
        # convert the label to one-hot encoding
        one_hot = tf.one_hot(label, num_classes)
        # read the img from file
        img_file = tf.read_file(img_path)
        img_decoded = tf.image.decode_image(img_file, channels=3)
        return img_decoded, one_hot
        
    def input_fn():
        
        dataset = Dataset.from_tensor_slices((imgs, labels))        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=256)
        dataset = dataset.map(input_parser)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()

        return iterator.get_next()
        
    return input_fn