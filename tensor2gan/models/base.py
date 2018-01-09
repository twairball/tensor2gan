import tensorflow as tf

def create_optimizer(hparams, loss, variables):
    """TODO: add warmup steps for learning rate"""        
    return tf.train.AdamOptimizer(hparams.learning_rate, beta1=hparams.beta1) \
            .minimize(loss, var_list=variables)

def create_optimize_all(hparams, loss_variable_pairs, name='optimizers'):
    """
    Creates optimizers and merges all into single optimize op
    Args:
        hparams: Hparams object or namedtuples with key/values for 
            optimizer params e.g. learning_rate, beta1
        loss_variable_paris: Nested list or tuples with (loss, variables)
    Returns:
        tf.operation with merged optimizers
    """
    optimizers = [create_optimizer(hparams, loss, vars) for loss, vars in loss_variable_pairs]
    with tf.control_dependencies(optimizers):
        return tf.no_op(name=name)

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
