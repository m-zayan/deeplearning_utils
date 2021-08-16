import numpy as np

import tensorflow as tf

from tensorflow.keras import backend as K

import tensorflow_addons as tfa

from ..ops.eval import random_shape2d, xsigma

__all__ = ['RandomTransformation', 'ImageGenerator']


# @tf.function - candidate
class RandomTransformation:

    def __init__(self):

        self.__td__ = dict()

        self.name = None

        self.__build__()

    @staticmethod
    def tft_same(image):

        return image

    @staticmethod
    def tft_flip_left_right(image, seed: int = None):

        return tf.image.random_flip_left_right(image, seed)

    @staticmethod
    def tft_flip_up_down(image, seed: int = None):

        return tf.image.random_flip_up_down(image, seed)

    @staticmethod
    def tft_gaussian_filter(image, seed: int = None):

        size = random_shape2d(1, 9, 1, 9, seed)

        return tfa.image.gaussian_filter2d(image, filter_shape=size)

    @staticmethod
    def tft_gaussian_noise(image, seed: int = None):

        shape = K.shape(image)

        return image + tf.random.normal(shape, mean=0.0, stddev=xsigma(shape), seed=seed)

    def __build__(self):

        self.name = []

        i = 0

        for key in RandomTransformation.__dict__.keys():

            if 'tft_' in key:
                self.name.append(key)

                self.__td__[i] = getattr(RandomTransformation, key)

                i += 1

    def __call__(self, *args):

        ti = np.random.randint(0, len(self))

        outputs = tf.stack(args, axis=0)
        outputs = tf.map_fn(self.__td__[ti], outputs)
        outputs = tf.squeeze(outputs)

        return outputs

    def __len__(self):

        return len(self.__td__)


class ImageGenerator(tf.keras.layers.Layer):

    def __init__(self, training=True, *args, **kwargs):

        super(ImageGenerator, self).__init__(*args, **kwargs)

        self.tft = RandomTransformation()

        self.training = training

    def call(self, inputs,  *args, **kwargs):

        if self.training:

            outputs = self.tft(inputs)

            outputs = tf.ensure_shape(outputs, inputs.shape)

            return outputs

        else:

            return inputs
