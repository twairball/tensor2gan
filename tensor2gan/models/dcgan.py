import tensorflow as tf
from .base import create_optimizer, batch_convert2int
from .base import BaseGAN

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

        def linear_projection(z, filters):
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
            g = linear_projection(z, self.filters)
            g = deconv_block(g, f1) # 1024, 4x4
            g = deconv_block(g, f2) # 512, 8x8
            g = deconv_block(g, f3) # 256, 16x16
            g = deconv_block(g, f4) # 3, 32x32
            output = tf.tanh(g) # activation for images
            # tf.logging.info("[generator] output: %s" % output)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return output

    def sample(self, input):
        image = batch_convert2int(self.__call__(input))
        image = tf.image.encode_jpeg(tf.squeeze(image, [0]))
        return image


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
            # tf.logging.info("[disc] classify block, x: %s" % d)
            d = dense_block(d, 1024)
            output = tf.layers.dense(d, 1)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return output


class DCGAN(BaseGAN):
    """
    DCGAN model
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
        for val, key in enumerate(self.outputs):
            tf.summary.histogram(key, val)

        # generated image
        tf.summary.image("G/generated", batch_convert2int(fake_data))

        # losses - discriminator
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))
        d_loss = d_loss_real + d_loss_fake 
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)))
        self.losses = dict(
            d_loss_real=d_loss_real, 
            d_loss_fake=d_loss_fake, 
            d_loss=d_loss,
            g_loss=g_loss
        )
        for val, key in enumerate(self.losses):
            tf.summary.scalar(key, val)

        # optimizers
        if self.optimizers is None:
            d_optim = create_optimizer(self.config, d_loss, self.G.variables)
            g_optim = create_optimizer(self.config, g_loss, self.D.variables)
            self.optimizers = dict(
                d_optim=d_optim,
                g_optim=g_optim
            )
        
        return self.losses, self.outputs, self.optimizers
        
    def gan_sample(self, z):
        return self.G.sample(z)
