import tensorflow as tf

from tensorflow.python.ops.losses import losses
from tensorflow.python.ops.losses import util
from tensorflow.python.summary import summary
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

from tensorflow.contrib.gan.python.losses.python.losses_impl import *

def softplus_discriminator_loss(
    discriminator_real_outputs,
    discriminator_gen_outputs,
    label_smoothing=0.25,
    real_weights=1.0,
    generated_weights=1.0,
    scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):
    """
    softplus 
    """
    with ops.name_scope(scope, 'discriminator_softplus_loss', (
        discriminator_real_outputs, discriminator_gen_outputs, real_weights,
        generated_weights, label_smoothing)) as scope:

        # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        loss_on_generated = tf.reduce_mean(tf.nn.softplus(discriminator_gen_outputs))
        loss_on_real = tf.reduce_mean(tf.nn.softplus(-discriminator_real_outputs))

        loss = loss_on_real + loss_on_generated
        util.add_loss(loss, loss_collection)

    if add_summaries:
        summary.scalar('discriminator_gen_softplus_loss', loss_on_generated)
        summary.scalar('discriminator_real_softplus_loss', loss_on_real)
        summary.scalar('discriminator_softplus_loss', loss)

    return loss


def softplus_generator_loss(
    discriminator_gen_outputs,
    label_smoothing=0.0,
    weights=1.0,
    scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
    add_summaries=False):

    with ops.name_scope(scope, 'generator_softplus_loss') as scope:
        loss = - minimax_discriminator_loss(
            array_ops.ones_like(discriminator_gen_outputs),
            discriminator_gen_outputs, label_smoothing, weights, weights, scope,
            loss_collection, reduction, add_summaries=False)

    if add_summaries:
        summary.scalar('generator_softplus_loss', loss)

    return loss
