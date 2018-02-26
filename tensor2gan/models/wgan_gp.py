from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensor2gan.models.base import batch_convert2int
from tensor2gan.models.base import BaseGAN
from tensor2gan.utils import registry

from tensor2gan.models.dcgan import DCGAN, Generator, Discriminator

@registry.register_model
class WGAN_GP(DCGAN):
    """
    WGAN-GP model

    "Improved Training of Wasserstein GANs"
    https://arxiv.org/abs/1704.00028

    configs:
        input_shape: tuple or list for Discriminator input shape (or Generator output shape)
        gen_filters: int, Generator top layer filters
        dis_filters: int, Discriminator top layer filters
        learning_rate: float, Adam optimizer learning rate
        penalty: float, gradient penalty factor. default=10.0
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

        # losses
        d_loss_real = tf.reduce_mean(d_real_logits)
        d_loss_fake = tf.scalar_mul(-1, tf.reduce_mean(d_fake_logits))
        d_loss = d_loss_real + d_loss_fake
        g_loss = tf.reduce_mean(d_fake)

        # x_hat is mixture of real/fake data
        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * real_data + (1 - epsilon) * fake_data
        d_hat = self.D(x_hat)

        # gradient penalty factor
        penalty = self.config.penalty
        gp = tf.gradients(d_hat, x_hat)[0]
        gp = tf.sqrt(tf.reduce_sum(tf.square(gp), axis=1))
        gp = tf.reduce_mean(tf.square(gp - 1.0) * penalty)
        d_loss = d_loss + gp

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