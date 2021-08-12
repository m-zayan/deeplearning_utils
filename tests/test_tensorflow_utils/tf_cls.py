import tensorflow as tf

from ...tf.cls.tfdict import TFDict


@tf.function
def tfd_test():

    def tf_key(shape):

        return tf.random.uniform(shape, minval=0, maxval=10, dtype=tf.int32)

    tfd = TFDict()

    i = tf_key((7, 7))
    j = tf_key((7, 7))

    value = tf.random.normal(shape=(3, 3))

    tfd.add(key=i.ref(), value=value)

    out = tf.concat([tfd.get(i.ref()), tfd.get(j.ref())], axis=0)

    return out
