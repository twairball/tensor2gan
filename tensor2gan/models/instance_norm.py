from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
Instance normalization layer
TODO: reference
"""
import tensorflow as tf

def instance_norm(input):
    """ Instance Normalization
    """
    with tf.variable_scope("instance_norm"):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], 
            initializer=tf.random_normal_initializer(
                mean=1.0, stddev=0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth],
            initializer=tf.constant_initializer(0.0))

        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset
