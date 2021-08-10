import tensorflow as tf


def download(fname, url, extract=True):

    ddir = tf.keras.utils.get_file(fname=fname, origin=url, extract=extract)

    return ddir
