from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class DataGenerator(object):
    """Generic Data Generator. 
    """
    def __init__(self):
        pass

    @property
    def num_classes(self):
        raise NotImplementedError()

    @property
    def input_shape(self):
        raise NotImplementedError()
    
    def get_input_fn(self, batch_size, data_dir, train):
        """Create input pipeline. Returns input_fn, a callable 
        function that returns next element in iterator.
        """
        raise NotImplementedError()