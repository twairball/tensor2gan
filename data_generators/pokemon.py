import tensorflow as tf
import numpy as np
from data_generators.utils import maybe_download, prepare_dataset, one_hot, do_files_exist
from data_generators.utils import write_to_tf_records, parse_record, get_image
import glob, os
import tarfile

Dataset = tf.data.Dataset
num_classes = 2

POKEMON_URL='https://s3-us-west-2.amazonaws.com/twairball.datasets.pokemon/pokemon.tgz'
DIR_NAME='./data/pokemon'
DATASET_FILE='pokemon.tgz'

def get_files(train=True, data_dir="./data"):
    """Returns Pokemon image files"""
    # maybe download
    target_dir = os.path.join(data_dir, DIR_NAME)
    prepare_dataset(target_dir, DATASET_FILE, POKEMON_URL)

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

def get_input_fn(batch_size=32,
    train=True,
    data_dir="./data",
    record_filename="pokemon_train.tfrecords"):
    """Write dataset to TFRecords, and create input pipeline
    Returns input_fn, a callable function that returns next element in iterator
    """
    image_shape=[80,80,3]
    num_classes=1
    record_filepath = os.path.join(data_dir, record_filename)

    # write dataset to tfrecords if not exist
    if not do_files_exist(record_filepath):
        tf.logging.info("Writing TFRecord to: %s" % record_filepath)
        files, labels = read_dataset(train, data_dir)
        write_to_tf_records(record_filepath, files, labels, get_image)
    else:
        tf.logging.info("Skipping writing TFRecord, found: %s" % record_filepath)

    def parse_features(image, label):
        # reshape image
        image = tf.cast(tf.reshape(image, image_shape), tf.float32)
        label = tf.one_hot(label, num_classes)
        return image, label
        
    def input_fn():
        dataset = tf.data.TFRecordDataset([record_filepath])
        dataset = dataset.map(parse_record)
        dataset = dataset.map(parse_features)
        dataset = dataset.shuffle(batch_size * 5).batch(batch_size).repeat()
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()
    
    return input_fn

# def generator(batch_size=32, train=True, shuffle=True):
    
#     imgs, labels = read_datasets(train)

#     def input_parser(img_path, label):
#         # convert the label to one-hot encoding
#         one_hot = tf.one_hot(label, num_classes)
#         # read the img from file
#         img_file = tf.read_file(img_path)
#         img_decoded = tf.image.decode_image(img_file, channels=3)
#         return img_decoded, one_hot
        
#     def input_fn():
        
#         dataset = Dataset.from_tensor_slices((imgs, labels))        
#         if shuffle:
#             dataset = dataset.shuffle(buffer_size=256)
#         dataset = dataset.map(input_parser)
#         dataset = dataset.batch(batch_size)

#         iterator = dataset.make_one_shot_iterator()

#         return iterator.get_next()
        
#     return input_fn