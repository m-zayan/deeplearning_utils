from typing import List

import tensorflow as tf
from tensorflow.keras.layers import Layer


class WeightedAverage(Layer):

    def __init__(self, avg_weights: List[float], *args, **kwargs):

        self.avg_weights = tf.cast(avg_weights, dtype=tf.float32)[None, :]

        if tf.math.reduce_sum(self.avg_weights) != 1.0:

            raise ValueError('...')

        super(WeightedAverage, self).__init__(*args, **kwargs)

    def call(self, inputs, training=None):

        outputs = self.avg_weights * tf.stack(inputs, axis=0)

        return tf.math.reduce_sum(outputs, axis=0)
