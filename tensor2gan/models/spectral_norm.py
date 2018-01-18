from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def l2_norm(x, epsilon=1e-12):
    return x / (tf.reduce_sum(x ** 2) ** 0.5 + epsilon)

def power_reduce(W, u, num_iters=1):
    W_shape = W.shape.as_list()
    W_reshaped = tf.reshape(W, [-1, W_shape[-1]])

    # index, u, v
    index = tf.constant(0)
    u = tf.get_variable("u", 
        shape=[1, W_shape[-1]], 
        initializer=tf.truncated_normal_initializer(), 
        trainable=False)
    # u = tf.Variable(tf.truncated_normal([1, W_shape[-1]]), trainable=False)
    # u = tf.truncated_normal([1, W_shape[-1]])
    v = tf.zeros(shape=[1, W_reshaped.shape.as_list()[0]], dtype=tf.float32)

    def condition(index, u, v):
        return tf.less(index, num_iters)
    
    def body(index, u, v):
        index_ = tf.add(index, 1)
        v_ = l2_norm(tf.matmul(u, tf.transpose(W_reshaped)))
        u_ = l2_norm(tf.matmul(v_, W_reshaped))
        return index_, u_, v_
    
    # return u, v
    return tf.while_loop(condition, body, (index, u, v))[1:]

def spectral_norm(W, num_iters=1):
    W_shape = W.shape.as_list()
    W_reshaped = tf.reshape(W, [-1, W_shape[-1]])

    u, v = power_reduce(W, num_iters)

    # sigma
    sigma = tf.matmul(tf.matmul(v, W_reshaped), tf.transpose(u))[0, 0]
    # W_bar
    W_bar = W_reshaped / sigma
    W_bar = tf.reshape(W_bar, W_shape)

    return W_bar