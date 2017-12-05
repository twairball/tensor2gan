import tensorflow as tf
import tensorflow.contrib.gan as tfgan
from tensor2gan.models.dcgan import DCGAN
from tensor2gan.models.spectral_norm import spectral_norm

class SN_DCGAN(DCGAN):
    """
    Spectral normalized DCGAN

    Useage:
        sn_dcgan = SN_DCGAN().gan_estimator
        sn_dcgan.train(input_fn)

    """
    def discriminator(self, filters=64, batch_size=32, training=True):
        
        def fn(inputs, conditioning):
            tf.logging.info("[disc] args: %s" % locals())
            
            # filters: 64, 128, 256, 512
            f0, f1, f2, f3 = filters, filters*2, filters*4, filters*8
            tf.logging.info("[disc] filters: %s" % [f0, f1, f2, f3])

            tf.logging.info("[discriminator] input shape: %s" % inputs.get_shape())
            tf.logging.info("[discriminator] batch_size: %s" % batch_size)
            
            def leaky_relu(x, leak=0.2, name='lrelu'):
                return tf.maximum(x, x * leak, name=name)
                
            def conv_block(x, filters):
                d = tf.layers.conv2d(x, filters, [5,5], strides=(2,2), padding='SAME', 
                    kernel_regularizer=spectral_norm)
                d = tf.layers.batch_normalization(d, training=training)
                d = leaky_relu(d)
                return d
            
            def dense_block(x, filters=1024):
                d = tf.layers.dense(x, filters, kernel_regularizer=spectral_norm)
                d = tf.layers.batch_normalization(d, training=training)
                d = leaky_relu(d)
                return d
            
            def classify_block(x, filters=1024):
                d = dense_block(x, filters)
                d = dense_block(d, 1)
                d = tf.nn.sigmoid(d) # classify as real or fake
                return d

            # model
            with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
                d = conv_block(inputs, f0)
                d = conv_block(d, f1)
                d = conv_block(d, f2)
                d = conv_block(d, f3)
                tf.logging.info("[disc] classify block, x: %s" % d)
                d = classify_block(d, 1024)

            return d
        
        return fn

