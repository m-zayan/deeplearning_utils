import numpy as np

__all__ = ['mean_average_precision', 'iou']


def mean_average_precision(y_pred: np.ndarray, query_label: np.ndarray):

    a = np.zeros((len(y_pred),), dtype='float32')
    b = np.arange(1, len(y_pred) + 1, dtype='float32')

    idx = np.where(y_pred == query_label)[0]

    if len(idx) == 0:

        return 0.0

    a[idx] = 1.0
    a = np.cumsum(a) * a

    idx_2 = np.where(a != 0)

    score = a[idx_2] * (1.0 / b[idx])

    return score.mean()


def iou(true, pred):

    """
    Parameters
    ----------
    true: np.ndarray
        [x_min, y_min, x_max, y_max]

    pred: np.ndarray
        [x_min, y_min, x_max, y_max]
    """

    x_mn = np.maximum(true[0], pred[0])
    y_mn = np.maximum(true[1], pred[1])

    x_mx = np.minimum(true[2], pred[2])
    y_mx = np.minimum(true[3], pred[3])

    inter_area = np.maximum(0.0, x_mx - x_mn + 1) * np.maximum(0.0, y_mx - y_mn + 1)

    area_0 = (true[2] - true[0] + 1) * (true[3] - true[1] + 1)
    area_1 = (pred[2] - pred[0] + 1) * (pred[3] - pred[1] + 1)

    iou_val = inter_area / (area_0 + area_1 - inter_area)

    return iou_val
