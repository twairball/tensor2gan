from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensor2gan.models.base import batch_convert2int
from tensor2gan.models.base import BaseGAN
from tensor2gan.utils import registry

from tensor2gan.models.dcgan import DCGAN, Generator, Discriminator

@registry.register_model
class WGAN(DCGAN):
    """
    WGAN model

    "Wasserstein GAN"
    https://arxiv.org/abs/1701.07875

    configs:
        input_shape: tuple or list for Discriminator input shape (or Generator output shape)
        gen_filters: int, Generator top layer filters
        dis_filters: int, Discriminator top layer filters
        learning_rate: float, RMSProp optimizer learning rate
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
        d_real, _ = self.D(real_data) # WGAN uses output, not logits.
        d_fake, _ = self.D(fake_data)
        self.outputs = dict(
            fake_data=fake_data, 
            d_real=d_real, 
            d_fake=d_fake
        )
        for key, val in self.outputs.items():
            tf.summary.histogram(key, val)

        # generated image
        tf.summary.image("G/generated", batch_convert2int(fake_data))

        # TODO: add apply_regularization to weights
        d_loss_real = tf.reduce_mean(d_real)
        d_loss_fake = tf.scalar_mul(-1, tf.reduce_mean(d_fake))
        d_loss = d_loss_real + d_loss_fake
        g_loss = tf.reduce_mean(d_fake)
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
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            d_optim = tf.train.RMSPropOptimizer(learning_rate=lr)\
                .minimize(d_loss, var_list=self.D.variables)
        
        # clip D values
        with tf.control_dependencies([d_optim]):
            d_clip = tf.group(*[v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.D.variables])

        with tf.control_dependencies([d_optim, d_clip]):
            g_optim = tf.train.RMSPropOptimizer(learning_rate=lr)\
                .minimize(g_loss, var_list=self.G.variables)

        # group training ops
        self.optimizers = tf.group(d_optim, g_optim)

        return self.losses, self.outputs, self.optimizers