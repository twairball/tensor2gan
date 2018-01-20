from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
from PIL import Image

from tensorflow.python.training.session_run_hook import SessionRunArgs

# def _gallery(input_array, ncols=6):
#     """stack array of images into grid"""
#     array = np.asarray(input_array)
#     nindex, height, width, intensity = array.shape
#     nrows = nindex//ncols
#     assert nindex == nrows*ncols
#     # want result.shape = (height*nrows, width*ncols, intensity)
#     result = (array.reshape(nrows, ncols, height, width, intensity)
#               .swapaxes(1,2)
#               .reshape(height*nrows, width*ncols, intensity))
#     return result

def gallery(images, ncols=6):
    """stack array of images into grid"""
    num_images, h, w, c = tf.shape(images)
    nrows = num_images//ncols
    
    result = tf.reshape(images, [nrows, ncols, h, w, c])
    result = tf.transpose(result, [0, 2, 1, 3])
    result = tf.reshape(result, [h * nrows, w * ncols, c])
    return result

# def save_image(tensor, image_path):
#     """writes tensor as jpeg to file"""
#     enc = tf.image.encode_jpeg(tensor)
#     fwrite = tf.write_file(image_path, enc)
#     return fwrite


class SaveImageHook(tf.train.SessionRunHook):
    """Saves images to disk every N local steps or every N secs."""
    
    def __init__(self, tensor, save_steps=100, save_secs=None,
        save_num=10, image_dir=None):
        """
        Args:
            tensor: image tensor with shape [batch, height, width, channel]
            save_steps: int, save every N steps. Default 100. 
            save_secs: int, save every N secs
            save_num: int, number of images to save
            image_dir: string, path to write output images

        Raises:
            ValueError: One of `save_steps` or `save_secs` should be set.
        """
        tf.logging.info("Created SaveImageHook for %s" % image_dir)
        if save_steps is not None and save_secs is not None:
            raise ValueError("You cannot provide both save_steps and save_secs")
        self.image_dir = image_dir
        self.save_num = save_num
        self.tensor = tensor
        self.timer = tf.train.SecondOrStepTimer(every_secs=save_secs,
            every_steps=save_steps)

    def begin(self):
        self._global_step_tensor = tf.train.get_or_create_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use CheckpointSaverHook.")
        
    def before_run(self, run_context):
        return SessionRunArgs([self._global_step_tensor, self.tensor])
    
    def after_run(self, run_context, run_values):
        stale_global_step, images = run_values.results
        global_step = stale_global_step + 1
        if self.timer.should_trigger_for_step(global_step):
            tf.logging.info("[HOOK] step=%d, Saving images" % global_step)
            self.timer.update_last_triggered_step(global_step)
            self.save(images, global_step)

    def save(self, images, step):
        for i in range(self.save_num):
            path = os.path.join(self.image_dir, "img_%d_%d.jpg" % (step, i))

            # if mnist, we squeeze to dims (h,w) and change mode to grayscale
            img = images[i].squeeze()
            mode = "L" if len(img.shape) == 2 else None

            im = Image.fromarray(np.uint8((img + 1) * 127.5), mode=mode)
            im.save(path)

