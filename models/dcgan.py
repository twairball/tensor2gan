import tensorflow as tf
import tensorflow.contrib.gan as tfgan

class DCGAN():
    """
    Create GAN Estimator

    Useage:
        dc_gan = DCGAN().gan_estimator
        dc_gan.train(input_fn)

    """
    def __init__(self, model_dir='./train'):
        # TODO:
        generator_optimizer = tf.train.AdamOptimizer(0.1, 0.5)
        discriminator_optimizer = tf.train.AdamOptimizer(0.1, 0.5)

        generator_fn = self.generator()
        discriminator_fn = self.discriminator()

        # Create GAN estimator.
        self.gan_estimator = tfgan.estimator.GANEstimator(
            model_dir,
            generator_fn=generator_fn,
            discriminator_fn=discriminator_fn,
            generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
            discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer
        )

    def generator(self, output_shape=(32,32,3), filters=1024, training=True):
        
        def fn(z):
            print("[generator] args: ", locals())
            
            # output shapes: h/16, h/8, h/4, h/2, h
            h, w, c = output_shape
            h0, w0 = int(h/16), int(w/16) 
            
            # filters: 1024, 512, 256, 128, c
            f0, f1, f2, f3, f4 = filters, int(filters/2), int(filters/4), int(filters/8), c
            print("[generator] filters: ", [f0, f1, f2, f3, f4])
            
            def linear_projection(z, filters):
                # linear projection of z
                g = tf.layers.dense(z, units=h0*w0*f0)
                g = tf.reshape(g, shape=[-1, h0, w0, f0])
                g = tf.layers.batch_normalization(g, training=training)
                g = tf.nn.relu(g)
                return g
            
            def deconv_block(x, filters):
                g = tf.layers.conv2d_transpose(x, filters, [5,5], strides=(2,2), padding='SAME')
                g = tf.layers.batch_normalization(g, training=training)
                g = tf.nn.relu(g)
                return g
            
            # model
            g = linear_projection(z, filters)
            g = deconv_block(g, f1)
            g = deconv_block(g, f2)
            g = deconv_block(g, f3)
            g = deconv_block(g, f4)
            g = tf.tanh(g) # activation for images
            print("[generator] output: ", g)
            return g

        return fn

    def discriminator(self, filters=64, training=True):
        
        def fn(inputs, conditioning):
            print("[discriminator] args: ", locals())
            
            # filters: 64, 128, 256, 512
            f0, f1, f2, f3 = filters, filters*2, filters*4, filters*8
            print("[discriminator] filters: ", [f0, f1, f2, f3])
        #     batch_size = inputs.get_shape()[0].value
        #     batch_size = tf.shape(inputs[0])[0]
            batch_size=32 # TODO: get batch_size dynamically
            print("[discriminator] input shape: ", inputs.get_shape())
            print("[discriminator] batch_size: ", batch_size)
            
            def leaky_relu(x, leak=0.2, name='lrelu'):
                return tf.maximum(x, x * leak, name=name)
                
            def conv_block(x, filters):
                d = tf.layers.conv2d(x, filters, [5,5], strides=(2,2), padding='SAME')
                d = tf.layers.batch_normalization(d, training=training)
                d = leaky_relu(d)
                return d
            
            def dense_block(x, filters=1024):
                d = tf.layers.dense(x, filters)
                d = tf.layers.batch_normalization(d, training=training)
                d = leaky_relu(d)
                return d
            
            def classify_block(x, filters=1024):
                print("[d] classify block: x: ", x)
        #         d = tf.reshape(x, [batch_size, -1])
        #         print("[d] classify block: d: ", d)
                d = dense_block(x, filters)
                d = dense_block(d, 1)
                d = tf.nn.sigmoid(d) # classify as real or fake
                return d

            # model    
            d = conv_block(inputs, f0)
            d = conv_block(d, f1)
            d = conv_block(d, f2)
            d = conv_block(d, f3)
            d = classify_block(d, 1024)
            return d
        
        return fn

