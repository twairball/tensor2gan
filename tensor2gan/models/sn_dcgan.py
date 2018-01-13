import tensorflow as tf
from tensor2gan.models.dcgan import DCGAN
from tensor2gan.models.spectral_norm import spectral_norm

class SN_DCGAN(DCGAN):
    """
    Spectral normalized DCGAN
    
    "Spectral Normalization for Generative Adversarial Networks" 
    https://openreview.net/forum?id=B1QRgziT-

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
        h, w, c = self.output_shape
        h0, w0 = int(h/8), int(w/8)   # 4, 4
        
        # filters: 512, 256, 128, 64, c
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
            g = deconv_block(g, f1) # 256, 8x8
            g = deconv_block(g, f2) # 128, 16x16
            g = deconv_block(g, f3) # 64, 32x32
            g = deconv_block(g, f4, kernel_size=[3,3], strides=(1,1)) # 3, 32x32
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
            
        def conv_block(x, filters, kernel_size=[5,5], strides=(2,2), name=None):
            _name = name if name else "conv%d" % filters
            with tf.variable_scope(_name, reuse=self.reuse):
                d = tf.layers.conv2d(x, filters, kernel_size , strides=strides, padding='SAME', 
                    kernel_regularizer=spectral_norm)
                d = leaky_relu(d)
                return d
        
        # model
        with tf.variable_scope(self.name, reuse=self.reuse):
            d = conv_block(inputs, f0, [3,3], (1,1), "conv%d" % f0)
            d = conv_block(d, f1, kernel_size=[4,4], strides=(2,2), name="conv%d_1" % f1)
            d = conv_block(d, f1, kernel_size=[3,3], strides=(1,1), name="conv%d_2" % f1)
            d = conv_block(d, f2, kernel_size=[4,4], strides=(2,2), name="conv%d_1" % f2)
            d = conv_block(d, f2, kernel_size=[3,3], strides=(1,1), name="conv%d_2" % f2)
            d = conv_block(d, f3, kernel_size=[4,4], strides=(2,2), name="conv%d_1" % f3)
            d = conv_block(d, f3, kernel_size=[3,3], strides=(1,1), name="conv%d_2" % f3)

            # single denseblock to output logits
            # matches original chainer
            output = tf.layers.dense(d, 1, kernel_regularizer=spectral_norm)
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return output


