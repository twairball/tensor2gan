"""
Adaptation of tensor2tensor/data_generators/generator_utils.py

"""
import tensorflow as tf
import numpy as np
import os
import tarfile
import re
import gzip
import io
from collections import defaultdict

import requests
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import six.moves.urllib_request as urllib  # Imports urllib on Python2, urllib.request on Python3


## 
## Data Prepare
## 
def do_files_exist(filepaths):
    return not(False in [tf.gfile.Exists(f) for f in filepaths])

def download_report_hook(count, block_size, total_size):
    """Report hook for download progress.

    Args:
    count: current block number
    block_size: block size
    total_size: total size
    """
    percent = int(count * block_size * 100 / total_size)
    print("\r%d%%" % percent + " completed", end="\r")


# from tensor2tensor
def maybe_download(directory, filename, url):
    """Download filename from url unless it's already in directory.

    Args:
    directory: path to the directory that will be used.
    filename: name of the file to download to (do nothing if it already exists).
    url: URL to download from.

    Returns:
    The path to the downloaded file.
    """
    if not tf.gfile.Exists(directory):
        tf.logging.info("Creating directory %s" % directory)
        os.mkdir(directory)
    
    filepath = os.path.join(directory, filename)
    if not tf.gfile.Exists(filepath):
        tf.logging.info("Downloading %s to %s" % (url, filepath))
        inprogress_filepath = filepath + ".incomplete"
        inprogress_filepath, _ = urllib.urlretrieve(
            url, inprogress_filepath, reporthook=download_report_hook)
        # Print newline to clear the carriage return from the download progress
        print()
        tf.gfile.Rename(inprogress_filepath, filepath)
        statinfo = os.stat(filepath)
        tf.logging.info("Succesfully downloaded %s, %s bytes." % (filename,
                                                                statinfo.st_size))
    else:
        tf.logging.info("Not downloading, file already found: %s" % filepath)
    return filepath

def prepare_dataset(data_dir, filename, remote_url):
    # prepare data files
    filepath = os.path.join(data_dir, filename)
    if not tf.gfile.Exists(filepath):
        maybe_download(data_dir, filename, remote_url)
    read_type = "r:gz" if filename.endswith("gz") else "r"
    with tarfile.open(filepath, read_type) as corpus_tar:
        corpus_tar.extractall(data_dir)


def one_hot(x, num_classes):
    return np.eye(num_classes)[x]

## 
## TFRecord stuff
## 

def get_feats(image):
    """Convert image to binary features. 
    Returns image (uint8) and shape (int32) in binary
    """
    shape = np.array(image.shape, dtype=np.int32)
    img = image.astype(np.uint8)
    # convert image to raw data bytes in the array.
    feats = img.tobytes(), shape.tobytes() 
    return feats

def get_image(file):
    # read the img from file
    img_file = tf.read_file(file)
    img_decoded = tf.image.decode_image(img_file, channels=3)
    # Normalize [-1,1]
    img_decoded = tf.div(
        tf.subtract(tf.to_float(img_decoded), tf.constant(127.5)), 
        tf.constant(127.5))
    return img_decoded

def write_to_tf_records(record_filepath, images, labels, get_image_fn=None):
    """Write image data to TFRecord. 
    Args:
        record_filepath: string filepath to write .tfrecords file
        images: Array of image data, or list of filenames
        labels: Array of labels
        get_image_fn: If defined, will call this function on each element of 
            images. 

    Example: 
        # load image data into array and write to .tfrecords
        write_to_tf_records(filepath, images, labels)

        # pass list of image filepaths and read image with function
        write_to_tf_records(filepath, image_files, labels, get_image)
    """
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(record_filepath)
    
    for image, label in zip(images, labels):
        if get_image_fn:
            image = get_image_fn(image)
        binary_image, shape = get_feats(image)
        
        example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(label),
                'shape': _bytes_feature(shape),
                'image': _bytes_feature(binary_image)
                }))
        
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def parse_record(example):
    # label and image are stored as bytes but could be stored as 
    # int64 or float64 values in a serialized tf.Example protobuf.
    tfrecord_features = tf.parse_single_example(example,
                        features={
                            'label': tf.FixedLenFeature([], tf.int64),
                            'shape': tf.FixedLenFeature([], tf.string),
                            'image': tf.FixedLenFeature([], tf.string),
                        }, name='features')
    # image was saved as uint8, so we have to decode as uint8.
    image = tf.decode_raw(tfrecord_features['image'], tf.uint8)
    label = tfrecord_features['label']

    # shape is dynamic tensor -- can't reshape from this. 
#     shape = tf.decode_raw(tfrecord_features['shape'], tf.int32)
#     shape = tf.reshape(shape, [3])
    return image, label