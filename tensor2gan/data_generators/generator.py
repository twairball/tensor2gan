from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class DataGenerator(object):

    def __init__(self):
        pass

    @property
    def num_classes(self):
        raise NotImplementedError()

    @property
    def input_shape(self):
        raise NotImplementedError()

    def get_record_filename(self, train):
        raise NotImplementedError()
    
    def prepare_data(self, data_dir, train):
        raise NotImplementedError()

    def get_input_fn(self, batch_size, train):
        raise NotImplementedError()