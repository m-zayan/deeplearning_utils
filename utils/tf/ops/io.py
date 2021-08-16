import tensorflow as tf

__all__ = ['download']


def download(fname, url, extract=True):

    ddir = tf.keras.utils.get_file(fname=fname, origin=url, extract=extract)

    return ddir
