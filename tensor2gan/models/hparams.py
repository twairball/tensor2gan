import tensorflow as tf
"""
hparams sets
"""

def dcgan_base():
    """Base set of hparams"""
    return tf.contrib.training.HParams(
        batch_size=64,
        z_dim=100,
        gen_filters=1024,
        gen_learning_rate=0.0001,
        gen_adam_beta1=0.5,
        gen_loss_fn="modified_generator_loss",
        dis_filters=64,
        dis_learning_rate=0.0001,
        dis_adam_beta1=0.5,
        dis_loss_fn="modified_discriminator_loss",
    )

def wgan_base():
    hparams = dcgan_base()
    hparams.gen_loss_fn = "wasserstein_generator_loss"
    hparams.dis_loss_fn = "wasserstein_discriminator_loss"
    return 

def sn_dcgan_base():
    """Base set of hparams"""
    hparams = dcgan_base()
    hparams.gen_loss_fn = "softplus_generator_loss"
    hparams.dis_loss_fn = "softplus_discriminator_loss"
    return hparams