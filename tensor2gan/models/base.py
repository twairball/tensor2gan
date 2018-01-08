import tensorflow as tf

def create_optimizer(hparams, loss, variables):
    """TODO: add warmup steps for learning rate"""        
    return tf.train.AdamOptimizer(hparams.learning_rate, beta1=hparams.beta1) \
            .minimize(loss, var_list=variables)

def convert2int(image):
    """ Transfrom from float tensor ([-1.,1.]) to int image ([0,255])
    """
    return tf.image.convert_image_dtype((image+1.0)/2.0, tf.uint8)

def batch_convert2int(images):
    return tf.map_fn(convert2int, images, dtype=tf.uint8)

class BaseGAN:

    def __init__(self, config):
        """Base GAN class, defines common props and methods to be subclassed
        Args:
            config: Object or hparams with model configurations
        """
        self.build_model(config)

    @property
    def optimizers(self): pass

    @property
    def losses(self): pass

    @property
    def outputs(self): pass

    def build_model(self, config):
        raise NotImplementedError()

    def model(self, inputs):
        raise NotImplementedError()
    
    def gan_sample(self, z):
        raise NotImplementedError()    
