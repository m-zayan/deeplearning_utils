import tensorflow as tf
from tensorflow.keras import backend as K

__all__ = ['iou']


def iou(y_true, y_pred):

    """
    Parameters
    ----------
    y_true: np.ndarray | tf.Tensor

        rank: Any
        format: [x_min, y_min, x_max, y_max],

        shape = [Any, Any, ..., 4]

    y_pred: np.ndarray | tf.Tensor

        same as y_true
    """

    bbox = tf.concat([y_true[..., None], y_pred[..., None]], axis=-1)

    x_bbox = tf.concat([bbox[..., 0, :], bbox[..., 2, :]], axis=-1)
    y_bbox = tf.concat([bbox[..., 1, :], bbox[..., 3, :]], axis=-1)

    x_mn = tf.math.reduce_max(x_bbox, axis=-1)
    y_mn = tf.math.reduce_max(y_bbox, axis=-1)

    x_mx = tf.math.reduce_min(x_bbox, axis=-1)
    y_mx = tf.math.reduce_min(y_bbox, axis=-1)

    inter_area = K.maximum(0.0, x_mx - x_mn + 1) * K.maximum(0.0, y_mx - y_mn + 1)

    area_0 = (y_true[..., 2] - y_true[..., 0] + 1) * (y_true[..., 3] - y_true[..., 1] + 1)
    area_1 = (y_pred[..., 2] - y_pred[..., 0] + 1) * (y_pred[..., 3] - y_pred[..., 1] + 1)

    score = inter_area / (area_0 + area_1 - inter_area)

    return score

