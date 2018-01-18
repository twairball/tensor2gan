from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensor2gan.models.base import batch_convert2int
from tensor2gan.models.base import BaseGAN
from tensor2gan.utils import registry

@registry.register_model
class DCGAN(BaseGAN):
    """
    DCGAN model

    "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"
    https://arxiv.org/abs/1511.06434
    
    configs:
        input_shape: tuple or list for Discriminator input shape (or Generator output shape)
        gen_filters: int, Generator top layer filters
        dis_filters: int, Discriminator top layer filters
        label_smoothing: float, applied to d_loss_fake label smoothing
        clip_gradients: float, applied to D and G gradients
        learning_rate: float, Adam optimizer learning rate
        beta1: float, Adam optimizer momentum
    """
    def build_model(self, config):
        # params
        input_shape = config.input_shape
        g_filters = config.gen_filters
        d_filters = config.dis_filters

        self.config = config

        self.G = Generator(output_shape=input_shape, filters=g_filters)
        self.D = Discriminator(filters=d_filters)

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
        d_real = self.D(real_data)
        d_fake = self.D(fake_data)
        self.outputs = dict(
            fake_data=fake_data, 
            d_real=d_real, 
            d_fake=d_fake
        )
        for key, val in self.outputs.items():
            tf.summary.histogram(key, val)

        # generated image
        tf.summary.image("G/generated", batch_convert2int(fake_data))

        # losses -- Flip the Discriminator labels:
        # D(real) = 0, D(fake) = 1
        # We use label smoothing on D(fake)
        label_smoothing = self.config.label_smoothing
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.zeros_like(d_real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, 
            labels=label_smoothing * tf.ones_like(d_fake)))
        d_loss = d_loss_real + d_loss_fake 
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))
        self.losses = dict(
            d_loss_real=d_loss_real, 
            d_loss_fake=d_loss_fake, 
            d_loss=d_loss,
            g_loss=g_loss
        )
        for key, val in self.losses.items():
            tf.summary.scalar(key, val)

        # optimize
        if self.optimizers is None:
            lr = self.config.learning_rate
            beta1 = self.config.beta1
            clip_gradients = self.config.clip_gradients

            d_optim = create_optimizer(d_loss, self.D.variables, 
                lr=lr * 0.5, beta1=beta1, clip_gradients=clip_gradients)

            g_optim = create_optimizer(g_loss, self.G.variables, 
                lr=lr, beta1=beta1, clip_gradients=clip_gradients)

            with tf.control_dependencies([d_optim, g_optim]):
                self.optimizers = tf.no_op(name='optimizers')

        return self.losses, self.outputs, self.optimizers
        
    def gan_sample(self, z):       
        _train = self.G.training
        self.G.training = False  # set Generator to inference
        images = batch_convert2int(self.G(z))
        self.G.training = _train # restore
        return images


def create_optimizer(loss, var_list, lr=1e-3, beta1=0.5, clip_gradients=None):
    """Create optimizer with gradient clipping. Returns training op. 
    """
    optimizer = tf.train.AdamOptimizer(lr * 0.5, beta1=beta1)
    gradients, variables = zip(*optimizer.compute_gradients(loss, var_list=var_list))
    if clip_gradients:
        gradients, _ = tf.clip_by_global_norm(gradients, clip_gradients)
    op = optimizer.apply_gradients(zip(gradients, variables),
        global_step=tf.train.get_or_create_global_step())
    return op


class Generator:
    
    def __init__(self, name="G", training=True, filters=1024, output_shape=(32, 32, 3)):
        self.name = name
        self.filters = filters
        self.training = training
        self.output_shape = output_shape
        self.reuse = False

    def __call__(self, z):
        """
        Args:
            z: noise tensor [None, z_dim]
        Returns:
            output: tensor with shape = self.output_shape
        """
        # output shapes: h/16, h/8, h/4, h/2, h
        h, w, c = self.output_shape
        h0, w0 = int(h/16), int(w/16)   # 4, 4
        
        # filters: 1024, 512, 256, 128, c
        f0, f1, f2, f3, f4 = self.filters, int(self.filters/2), int(self.filters/4), int(self.filters/8), c

        def linear_projection(z):
            # linear projection of z
            with tf.variable_scope("linear", reuse=self.reuse):
                g = tf.layers.dense(z, units=h0*w0*f0)
                g = tf.reshape(g, shape=[-1, h0, w0, f0])
                g = tf.layers.batch_normalization(g, training=self.training)
                g = tf.nn.relu(g)
                return g
        
        def deconv_block(x, filters, kernel_size=[4,4], strides=(2,2)):
            with tf.variable_scope("deconv%d" % filters, reuse=self.reuse):
                g = tf.layers.conv2d_transpose(x, filters, kernel_size, strides=strides, padding='SAME')
                g = tf.layers.batch_normalization(g, training=self.training)
                g = tf.nn.relu(g)
                return g
        
        # model
        with tf.variable_scope(self.name, reuse=self.reuse):
            g = linear_projection(z)
            g = deconv_block(g, f1) # 1024, 4x4
            g = deconv_block(g, f2) # 512, 8x8
            g = deconv_block(g, f3) # 256, 16x16
            g = deconv_block(g, f4) # 3, 32x32
            output = tf.tanh(g) # activation for images

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return output

class Discriminator:
    
    def __init__(self, name="D", training=True, filters=64):
        self.name = name
        self.filters = filters
        self.training = training
        self.reuse = False

    def __call__(self, inputs):
        """
        Args:
            inputs: image tensor [batch, w, h, c]
        Returns:
            outputs: real/fake logit 
        """
        # filters: 64, 128, 256, 512
        f0, f1, f2, f3 = self.filters, self.filters*2, self.filters*4, self.filters*8
        
        def leaky_relu(x, leak=0.2, name='lrelu'):
            return tf.maximum(x, x * leak, name=name)
            
        def conv_block(x, filters):
            with tf.variable_scope("conv%d" % filters, reuse=self.reuse):
                d = tf.layers.conv2d(x, filters, [5,5], strides=(2,2), padding='SAME')
                d = tf.layers.batch_normalization(d, training=self.training)
                d = leaky_relu(d)
                return d
        
        def dense_block(x, filters=1024):
            with tf.variable_scope("dense", reuse=self.reuse):
                d = tf.layers.dense(x, filters)
                d = tf.layers.batch_normalization(d, training=self.training)
                d = leaky_relu(d)
                return d

        # model
        with tf.variable_scope(self.name, reuse=self.reuse):
            d = conv_block(inputs, f0)
            d = conv_block(d, f1)
            d = conv_block(d, f2)
            d = conv_block(d, f3)
            d = dense_block(d, 1024)
            d = tf.layers.dense(d, 1)
            output = tf.nn.sigmoid(d)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return output