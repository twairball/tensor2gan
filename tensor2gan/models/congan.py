from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensor2gan.models.base import batch_convert2int
from tensor2gan.models.base import BaseGAN
from tensor2gan.utils import registry

@registry.register_model
class CONGAN(BaseGAN):
    """
    Continuous GAN

    https://github.com/Mylittlerapture/ConGAN/

    configs:
        input_shape: tuple or list for Discriminator input shape (or Generator output shape)
        gen_filters: int, Generator top layer filters
        dis_filters: int, Discriminator top layer filters
        label_smoothing: float, applied to d_loss_fake label smoothing
        clip_gradients: float, applied to D and G gradients
        learning_rate: float, Adam optimizer learning rate
        beta1: float, Adam optimizer momentum

    """

class Identifier():

    def __init__(self, name="Ident", z_dim=5):
        self.name = name
        self.reuse = False
        self.z_dim = z_dim

    def __call__(self, x, training=True):
        """
        Args:
            x: input tensor, typically real data with shape [b, h, w, c]
        Returns:
            output: tensor with shape = [None, z_dim]
        """

        with tf.variable_scope(self.name, reuse=self.reuse):
            f = tf.layers.dense(x, self.z_dim)
            f = tf.nn.tanh(f)

class Generator():

    def __init__(self, name="G", output_shape=(32,32,3)):
        self.name = name
        self.output_shape = output_shape
        self.reuse = False
    
    def __call__(self, x_pos, z_noise, channels=3, training=True):
        """
        Args:
            x_pos: tensor, with position mapping [-1, 1]. Has shape (Noise, 2)
            z_noise: latent space tensor, has shape (None, z_dim)
        Returns:
            output: tensor with shape = self.output_shape
        """

        def dense_block(inputs, units, alpha=0.2, name=None):
            _name = name if name else "dense%d" % units
            with tf.variable_scope(_name, reuse=self.reuse):
                g = tf.layers.dense(inputs, units)
                g = tf.nn.leaky_relu(g, alpha=alpha)
                return g
        
        def draw_output(inputs, channels=3, name="draw_output"):
            with tf.variable_scope(name, reuse=self.reuse):
                g = tf.layers.dense(inputs, channels)
                g = tf.nn.tanh(inputs)
                return g

        # model
        with tf.variable_scope(self.name, reuse=self.reuse):
            
            # position layer
            f_pos = dense_block(x_pos, 2, name="pos_mapping")
            
            # head layer
            head = tf.concat([f_pos, z_noise], axis=1)
            h = dense_block(head, 256)
            h = dense_block(h, 512)
            h = dense_block(h, 1024)
            h = dense_block(h, 1024)
            h = dense_block(h, 1024)
            h = dense_block(h, 1024)

            # draw layer
            draw = dense_block(head, 256, name="draw_256")
            draw = dense_block(draw, 32, name="draw_256")
            draw = draw_output(draw, channels=channels)
            
            # concat layer
            cnc_layer = tf.concat([head, f_pos, draw], axis=1)

            output = draw, cnc_layer

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return output


class Discriminator():
    
    def __init__(self, name="D"):
        self.name = name
        self.reuse = False

    def __call__(self, inputs, training=True):
        """
        Args:
            inputs: image tensor [batch, h, w, c]
        Returns:
            output: Tensor
            logit: Tensor, sigmoid logit
        """
        def dense_block(inputs, units, alpha=0.2, name=None):
            _name = name if name else "dense%d" % units
            with tf.variable_scope(_name, reuse=self.reuse):
                g = tf.layers.dense(inputs, units)
                g = tf.nn.leaky_relu(g, alpha=alpha)
                return g

        # model
        with tf.variable_scope(self.name, reuse=self.reuse):
            d = dense_block(inputs, 512)
            d = dense_block(d, 128)
            d = dense_block(d, 32)
            d = dense_block(d, 8)
            logit = tf.layers.dense(d, 1)
            output = tf.nn.sigmoid(logit)


        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return output, logit
