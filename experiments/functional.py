from tensorflow.keras import backend, losses

__all__ = ['reconstruction_loss', 'cross_entropy_loss']


def reconstruction_loss(x_true, x_pred):

    loss = 0.5 * (x_pred - x_true) ** 2
    loss = backend.sum(loss, axis=-1)

    return backend.mean(loss)


def cross_entropy_loss(y_true, y_pred):

    loss = losses.binary_crossentropy(y_true, y_pred)

    return backend.mean(loss)
