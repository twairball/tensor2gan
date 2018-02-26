from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensor2gan.models.base import batch_convert2int
from tensor2gan.models.base import BaseGAN
from tensor2gan.utils import registry

from tensor2gan.models.dcgan import DCGAN
from tensor2gan.models.dcgan import create_optimizer

@registry.register_model
class LSGAN(DCGAN):
    """
    LSGAN model

    "Least Squares Generative Adversarial Networks"    
    https://arxiv.org/abs/1611.04076

    configs:
        input_shape: tuple or list for Discriminator input shape (or Generator output shape)
        gen_filters: int, Generator top layer filters
        dis_filters: int, Discriminator top layer filters
        learning_rate: float, Adam optimizer learning rate
    """

    def model(self, inputs):
        """model function
        Args:
            inputs: [real_data_batch, z_noise] Tensors 
        Returns:
            losses: dict, {d_loss_real, d_loss_fake, g_loss}
            outputs: dict, {d_real, d_fake, fake_data}
            optimizers: dict, {d_optim, g_optim}
        """
        real_data, z = inputs

        # outputs
        fake_data = self.G(z)
        d_real, d_real_logits = self.D(real_data)
        d_fake, d_fake_logits = self.D(fake_data)

        self.outputs = dict(
            fake_data=fake_data, 
            d_real=d_real, 
            d_fake=d_fake
        )
        for key, val in self.outputs.items():
            tf.summary.histogram(key, val)

        # generated image
        tf.summary.image("G/generated", batch_convert2int(fake_data))

        # least squares error is also known as l2-norm 
        def least_squares(logits, labels):
            return tf.reduce_mean(tf.nn.l2_loss(logits - tf.ones_like(logits)))
        
        g_loss = least_squares(d_fake_logits, tf.ones_like(d_fake_logits))
        d_loss_real = least_squares(d_real_logits, tf.ones_like(d_real_logits))
        d_loss_fake = least_squares(d_fake_logits, tf.zeros_like(d_fake_logits))
        d_loss = d_loss_real + d_loss_fake
        self.losses = dict(
            d_loss_real=d_loss_real, 
            d_loss_fake=d_loss_fake, 
            d_loss=d_loss,
            g_loss=g_loss
        )
        for key, val in self.losses.items():
            tf.summary.scalar(key, val)

        # optimize
        lr = self.config.learning_rate
        beta1 = self.config.beta1
        beta2 = self.config.beta2

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            d_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2)\
                .minimize(
                    d_loss, 
                    global_step=tf.train.get_or_create_global_step(),
                    var_list=self.D.variables)

            g_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2)\
                .minimize(
                    g_loss, 
                    global_step=tf.train.get_or_create_global_step(),
                    var_list=self.G.variables)
        # group training ops
        self.optimizers = tf.group(d_optim, g_optim)

        return self.losses, self.outputs, self.optimizers

class Generator:
    
    def __init__(self, name="G", filters=256, output_shape=(32, 32, 3)):
        self.name = name
        self.filters = filters
        self.output_shape = output_shape
        self.reuse = False

    def __call__(self, z, training=True):
        """
        Args:
            z: noise tensor [None, z_dim]
        Returns:
            output: tensor with shape = self.output_shape
        """
        # output shapes: h/16, h/8, h/4, h/2, h
        h, w, c = self.output_shape
        h0, w0 = int(h/16), int(w/16)   # 2, 2
        
        
        # filters: 256, 128, 64,
        f0, f1, f2= self.filters, int(self.filters/2), int(self.filters/4)

        def linear_projection(z):
            # linear projection of z
            with tf.variable_scope("linear", reuse=self.reuse):
                g = tf.layers.dense(z, units=h0*w0*f0) # 2*2*256 = 1024
                g = tf.reshape(g, shape=[-1, h0, w0, f0])
                g = tf.layers.batch_normalization(g, training=training)
                g = tf.nn.leaky_relu(g, alpha=0.2)
                return g
                
        def deconv_block(x, filters, kernel_size=[3,3], strides=[1,2,2,1],  padding='SAME', name=None):
            _name = name or "deconv%d" % filters
            with tf.variable_scope(_name, reuse=self.reuse):
                g = tf.layers.conv2d_transpose(x, filters, kernel_size, strides=strides, padding=padding,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
                g = tf.layers.batch_normalization(g, training=training)
                g = tf.nn.relu(g)
                return g

        def last_deconv_block(x, filters, kernel_size=[3,3], strides=[1,1,1,1], padding='SAME'):
            with tf.variable_scope("deconv%d" % filters, reuse=self.reuse):
                g = tf.layers.conv2d_transpose(x, filters, kernel_size, strides=strides, padding=padding,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
                g = tf.nn.tanh(g)
                return g
            
        # model
        with tf.variable_scope(self.name, reuse=self.reuse):
            g = linear_projection(z)
            _name1 = "deconv%d_1" % f0
            _name2 = "deconv%d_2" % f0
            g = deconv_block(g, filters=f0, name=_name1) # 1024, 2x2
            g = deconv_block(g, filters=f0, name=_name2, strides=[1,1,1,1]) # 512, 4x4

            _name3 = "deconv%d_3" % f0
            _name4 = "deconv%d_4" % f0
            g = deconv_block(g, filters=f0, name=_name3) # 256, 8x8
            g = deconv_block(g, filters=f0, name=_name4, strides=[1,1,1,1]) # 128, 16x16

            g = deconv_block(g, filters=f1) # 128
            g = deconv_block(g, filters=f2) # 64, 
            output = last_deconv_block(g, filters=c) # 3, 32x32, tanh

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return output

class Discriminator:
    
    def __init__(self, name="D", filters=64):
        self.name = name
        self.filters = filters
        self.reuse = False

    def __call__(self, inputs, training=True):
        """
        Args:
            inputs: image tensor [batch, h, w, c]
        Returns:
            output: Tensor
            logit: Tensor, sigmoid logit
        """
        batch_size = inputs.get_shape()[0]
        # filters: 64, 128, 256, 512
        f0, f1, f2, f3 = self.filters, self.filters*2, self.filters*4, self.filters*8
                    
        def conv_block(x, filters, bn=True):
            with tf.variable_scope("conv%d" % filters, reuse=self.reuse):
                d = tf.layers.conv2d(x, filters, [5,5], strides=(2,2), padding='SAME',
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
                if bn:
                    d = tf.layers.batch_normalization(d, training=training)
                d = tf.nn.leaky_relu(d, alpha=0.2)
                return d
                            
        # model
        with tf.variable_scope(self.name, reuse=self.reuse):
            d = conv_block(inputs, f0, bn=False)
            d = conv_block(d, f1)
            d = conv_block(d, f2)
            d = conv_block(d, f3)
            d = tf.reshape(d, [batch_size, -1])
            logit = tf.layers.dense(d, 1) 
            output = tf.nn.sigmoid(logit) # prob(y|x)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return output, logit