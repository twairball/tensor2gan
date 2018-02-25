from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
        clip_gradients=None,
        label_smoothing=None,
        gen_filters=1024,
        dis_filters=64,
    )

def dcgan_label_smoothing():
    """Base set of hparams"""
    hparams = dcgan_base()
    hparams.label_smoothing=0.9
    hparams.clip_gradients=20.0
    return hparams

def wgan_base():
    hparams = dcgan_base()
    hparams.learning_rate=5e-5
    return hparams

def sn_dcgan_base():
    hparams = dcgan_base()
    hparams.beta2=0.999
    return hparams

def wgan_gp():
    hparams = dcgan_base()
    hparams.learning_rate = 1e-4
    hparams.beta1 = 0.5
    hparams.beta2 = 0.9
    hparams.penalty = 10.0
    return hparams