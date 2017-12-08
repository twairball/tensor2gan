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

    def generator(self, output_shape=(32,32,3), filters=512, training=True):
        
        def fn(z):
            print("[generator] args: ", locals())
            
            h, w, c = output_shape
            h0, w0 = int(h/8), int(w/8)   # 4, 4
            
            # filters: 512, 256, 128, 64, c
            f0, f1, f2, f3, f4 = filters, int(filters/2), int(filters/4), int(filters/8), c
            print("[generator] filters: ", [f0, f1, f2, f3, f4])
            
            def linear_projection(z, filters):
                # linear projection of z
                g = tf.layers.dense(z, units=h0*w0*f0)
                g = tf.reshape(g, shape=[-1, h0, w0, f0])
                g = tf.layers.batch_normalization(g, training=training)
                g = tf.nn.relu(g)
                return g
            
            def deconv_block(x, filters, kernel_size=[4,4], strides=(2,2)):
                g = tf.layers.conv2d_transpose(x, filters, kernel_size, strides=strides, padding='SAME')
                g = tf.layers.batch_normalization(g, training=training)
                g = tf.nn.relu(g)
                return g
            
            # model
            with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
                g = linear_projection(z, filters)
                g = deconv_block(g, f1) # 256, 8x8
                g = deconv_block(g, f2) # 128, 16x16
                g = deconv_block(g, f3) # 64, 32x32
                g = deconv_block(g, f4, kernel_size=[3,3], strides=(1,1)) # 3, 32x32
                g = tf.tanh(g) # activation for images
                tf.logging.info("[generator] output: %s" % g)
            return g

        return fn

    def discriminator(self, filters=64, batch_size=32, output_dim=1, training=True):
        
        def fn(inputs, conditioning):
            tf.logging.info("[disc] args: %s" % locals())
            
            # filters: 64, 128, 256, 512
            f0, f1, f2, f3 = filters, filters*2, filters*4, filters*8
            tf.logging.info("[disc] filters: %s" % [f0, f1, f2, f3])

            tf.logging.info("[discriminator] input shape: %s" % inputs.get_shape())
            tf.logging.info("[discriminator] batch_size: %s" % batch_size)
            
            def leaky_relu(x, leak=0.2, name='lrelu'):
                return tf.maximum(x, x * leak, name=name)
                
            def conv_block(x, filters, kernel_size=[5,5], strides=(2,2)):
                d = tf.layers.conv2d(x, filters, kernel_size , strides=strides, padding='SAME', 
                    kernel_regularizer=spectral_norm)
                # d = tf.layers.batch_normalization(d, training=training)
                d = leaky_relu(d)
                return d

            # model
            with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
                d = conv_block(inputs, f0, [3,3], (1,1))
                d = conv_block(d, f1, [4,4], (2,2))
                d = conv_block(d, f1, [3,3], (1,1))
                d = conv_block(d, f2, [4,4], (2,2))
                d = conv_block(d, f2, [3,3], (1,1))
                d = conv_block(d, f3, [4,4], (2,2))
                d = conv_block(d, f3, [3,3], (1,1))

                # single denseblock to output logits
                # matches original chainer
                d = tf.layers.dense(self.output_dim, kernel_regularizer=spectral_norm)
            return d
        
        return fn
