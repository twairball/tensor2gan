import tensorflow as tf
"""
hparams sets
"""

def dcgan_base():
    """Base set of hparams"""
    return tf.contrib.training.HParams(
        batch_size=64,
        z_dim=100,
        learning_rate=0.0001,
        beta1=0.5,
        clip_gradients=20.0,
        label_smoothing=0.9,
        gen_filters=1024,
        dis_filters=64,
    )

def wgan_base():
    hparams = dcgan_base()
    # TODO:
    return hparams

def sn_dcgan_base():
    hparams = dcgan_base()
    hparams.beta2=0.999
    return hparams