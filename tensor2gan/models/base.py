import tensorflow as tf

def convert2int(image):
    """ Transfrom from float tensor ([-1.,1.]) to int image ([0,255])
    """
    return tf.image.convert_image_dtype((image+1.0)/2.0, tf.uint8)

def batch_convert2int(images):
    return tf.map_fn(convert2int, images, dtype=tf.uint8)

class BaseGAN(object):

    def __init__(self, config):
        """Base GAN class, defines common props and methods to be subclassed
        Args:
            config: Object or hparams with model configurations
        Properties:
            optimizers: dict of Tensors
            losses: dict of Tensors
            outputs: dict of Tensors
        """
        self.build_model(config)
        self.optimizers = None
        self.losses = None
        self.outputs = None

    def build_model(self, config):
        raise NotImplementedError()

    def model(self, inputs):
        raise NotImplementedError()
    
    def gan_sample(self, z):
        raise NotImplementedError()    
