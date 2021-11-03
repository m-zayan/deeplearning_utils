from tensorflow.keras import backend, losses

__all__ = ['euclidean_norm_loss', 'cross_entropy_loss']


def euclidean_norm_loss(y_true, y_pred):

    loss = 0.5 * (y_true - y_pred) ** 2

    return backend.sum(loss)


def cross_entropy_loss(y_true, y_pred):

    loss = losses.binary_crossentropy(y_true, y_pred)

    return backend.mean(loss)
