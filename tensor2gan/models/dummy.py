from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensor2gan.models.base import batch_convert2int
from tensor2gan.models.base import BaseGAN
from tensor2gan.utils import registry

@registry.register_model
class Dummy(BaseGAN):
    """Dummy model for mock/testings
    """
    def build_model(self, config):
        self.config = config
        tf.logging.info("*** building dummy ***")

    def model(self, inputs):
        """Model function for training
        Args:
            inputs: [real_data_batch, z_noise] Tensors 
        Returns:
            losses: dict, {d_loss_real, d_loss_fake, g_loss}
            outputs: dict, {d_real, d_fake, fake_data}
            optimizers: dict, {d_optim, g_optim}
        """
        real_data, z = inputs
        tf.summary.image("real_data", batch_convert2int(real_data))

        # dummy output, just passes through noise.
        dummy = tf.reduce_mean(z)
        tf.summary.histogram("dummy", dummy)
        
        dummy_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=dummy, labels=tf.zeros_like(dummy))
        )
        tf.summary.scalar("dummy_loss", dummy_loss)

        # a dummy op that does nothing but increment global_step
        global_step = tf.train.get_or_create_global_step()
        op = tf.add(global_step, tf.constant(1, dtype=tf.int64), name="dummy_op")

        return dict(dummy_loss=dummy_loss), dict(dummy=dummy), dict(ops=op)

    def eval(self, z, add_summaries=True):
        pass