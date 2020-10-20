import tensorflow as tf
from keras.callbacks import Callback

import ntpath
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
from scipy import ndimage
import random
import io
import math


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


class TensorBoardImage(Callback):
    def __init__(self, image_batch, image_dir, log_dir, f=lambda x:x):
        super().__init__()
        self.writer = tf.summary.create_file_writer(log_dir)
        self.image_batch = image_batch[0]
        self.labels = image_batch[1]
        self.g_size = math.ceil(math.sqrt(len(self.image_batch)))
        self.f = f
        print("Creating grid of:", self.g_size)

    def predict_image_batch(self):
        """ Loop through the dataframe, print images with their crops
        """
        fig = plt.figure(figsize=(15,15))
        predicts = self.model.predict(self.image_batch)

        for n in range(len(self.image_batch)):
            ax = plt.subplot(self.g_size, self.g_size, n+1)

            im = self.image_batch[n]

            #ime =  #np.expand_dims(im, axis=0)
            cr = predicts[n]  #self.model.predict(ime)
            c = cr[0:4]
            r = cr[4]

            # HACK: apply function to plto correctly:
            c = [self.f(p) for p in c]

            im = np.clip(ndimage.rotate(im, r), 0, 1)
            imw, imh = im.shape[0:2]
            plt.imshow(im)

            plt.hlines([c[0]*imh, c[2]*imh], 0, imw, colors=['r', 'g'])
            plt.vlines([c[1]*imw, c[3]*imw], 0, imh, colors=['r', 'g'])

            plt.axis('off')
        return plot_to_image(fig)

    def on_epoch_end(self, epoch, logs={}):
        filename = 'cropper'
        # Load image
        img_orig = self.predict_image_batch()
        #img_orig = np.expand_dims(img_orig, axis=0)

        with self.writer.as_default():
            tf.summary.image(name=filename+"/Prediction", data=img_orig, step=epoch)
        self.writer.flush()
        
    def close(self):
        self.writer.close()
