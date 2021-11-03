from typing import Union

import tensorflow as tf
from tensorflow.keras import backend as K

__all__ = ['xsigma', 'random_shape2d', 'clip_for_logits', 'random_proba', 'nindex', 'sample_chain',
           'grid_mask_indices', 'nd_non_max_suppression', 'nd_non_max_suppression', 'KFold']


# @tf.function - candidate
def grid_mask_indices(grid_h, grid_w):

    """
    Parameters
    ----------
    grid_h: int

        grid height

    grid_w: int

        grid width

    Returns
    -------
    2D Tensor [grid mask indices]
    """

    mask = tf.range(grid_h * grid_w)
    mask = tf.reshape(mask, shape=(grid_h, grid_w))

    return mask


# @tf.function - candidate
def non_max_suppression(boxes, scores, max_output_size: int,
                        iou_threshold: float = 0.5, score_threshold: float = float('-inf')):

    """
    https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression
    """
    # n_boxes x 4, n_boxes x 1

    indices = tf.image.non_max_suppression(boxes=boxes, scores=scores, max_output_size=max_output_size,
                                           iou_threshold=iou_threshold, score_threshold=score_threshold)

    return indices


# @tf.function - candidate
def nd_non_max_suppression(boxes, scores, max_output_size: int,
                           iou_threshold: float = 0.5, score_threshold: float = float('-inf'),
                           mask_indices=None):

    """
    Parameters
    ----------

    boxes: np.ndarray | tf.Tensor

        rank: 3
        shape: [grid_h * grid_w, n_boxes, 4]

    scores: np.ndarray | tf.Tensor

        rank: 2
        shape: [grid_h * grid_w, n_boxes]

    max_output_size: int

    iou_threshold: float, default = 0.5

    score_threshold: float, default = float('-inf')

    mask_indices: np.ndarray | tf.Tensor,

        e.g., tf.ops.eval.grid_mask_indices(....)

    Returns
    -------
    tuple: (choices, obj_cells)
        where:

            choices: list

                selected boxes indices

            obj_cells: list

                The corresponding, grid cell (y, x), for each index
    """

    choices = []
    obj_cells = []

    def body(i, j):

        indices = non_max_suppression(boxes[i], scores[i], max_output_size,
                                      iou_threshold, score_threshold)

        if len(indices):

            choices.append(indices)

            cells = tf.where(mask_indices == i)
            cells = tf.cast(cells, dtype='float32')

            obj_cells.append(cells)

            j += 1

        return i + 1, j

    _, _ = tf.while_loop(lambda i, j: i < len(scores), body, [0, 0])

    return choices, obj_cells


# @tf.function - candidate
def xsigma(shape):

    size = tf.math.reduce_sum(shape)

    return tf.sqrt(2.0 / float(size))


# @tf.function - candidate
def random_shape2d(minh: int, maxh: int, minw: int, maxw: int, seed: int = None):

    h = tf.random.uniform(shape=(), minval=minh, maxval=maxh, dtype=tf.int32, seed=seed)
    w = tf.random.uniform(shape=(), minval=minw, maxval=maxw, dtype=tf.int32, seed=seed)

    return h, w


# @tf.function - candidate
def clip_for_logits(proba):

    mn = K.epsilon()
    mx = 1.0 - mn

    return K.clip(proba, min_value=mn, max_value=mx)


# @tf.function - candidate
def random_proba(shape, clipindex=None, clipproba=None, seed=None):

    logits = tf.random.normal(shape=shape, mean=0.0, stddev=xsigma(shape), dtype=tf.float32, seed=seed)

    if clipindex is not None:

        clipproba = clip_for_logits(clipproba)

        cliplogits = tf.math.log(clipproba) - tf.math.log(1.0 - clipproba)

        logits = tf.tensor_scatter_nd_update(logits, [clipindex], [cliplogits])

    proba = tf.math.softmax(logits)

    return proba


# @tf.function - candidate
def nindex(ubound, indices):

    def _nindex(elems):

        if elems[0] > elems[1]:

            return elems[1] % elems[0], 0

        return elems[1], 0

    ubound = tf.cast(ubound, dtype=tf.int32)
    indices = tf.cast(indices, dtype=tf.int32)

    ind, _ = tf.map_fn(_nindex, (ubound, indices))

    return ind


# @tf.function - candidate
def sample_chain(n, chain, low=(0, 0), high=(-1, -1), plow=0.05, phigh=0.95, categorical=False):

    low = nindex((chain, n), low)
    high = nindex((chain, n), high)

    proba = random_proba((chain, n), clipindex=[low, high], clipproba=[plow, phigh])

    if categorical:

        indices = tf.random.categorical(tf.math.log(proba), 1, dtype=tf.int32)
        indices = tf.squeeze(indices)

        return indices

    return proba


class KFold:

    """
    Implements K-Fold cross validation for (tf.data.*).
    """

    def __init__(self, n_splits: int):

        # Number of folds
        self.n_splits = n_splits

    def split(self, tf_data: Union[tf.data.Dataset, tf.data.TFRecordDataset], size: int):

        """
        Parameters
        ----------

        tf_data: tf.data.Dataset

        size: int

            tf.data.Dataset or tf.data.TFRecordDataset, full size.

            For tf.data.Dataset, you could get, the full size using:

                - tf.data.experimental.cardinality(tf_data)

        Returns
        -------
        tuple: (kfold_train, kfold_val)
            where:

                kfold_train and kfold_val: are dictionaries, {key: value} - {int: tf.data.*},

                    ex. for training set at key=0 - kfold_train[0],
                        the corresponding validation set is - kfold_val[0].
        """

        step = size // self.n_splits

        kfold_train = dict()
        kfold_val = dict()

        for i in range(self.n_splits - 1):

            """
            For training the required sequence is (take ---> skip ----> take), 
            For validation the required sequence is (skip ----> take a step)
            """

            take_i = step * (i + 1)
            skip_i = step * (i + 2)

            # train
            kfold_train[i] = tf_data.take(take_i)
            kfold_train[i] = kfold_train[i].concatenate(tf_data.skip(skip_i))

            # validation
            kfold_val[i] = tf_data.skip(take_i)
            kfold_val[i] = kfold_val[i].take(step)

        # train at nth-split
        kfold_train[(self.n_splits - 1)] = tf_data.skip(step)

        # validation at nth-split
        kfold_val[(self.n_splits - 1)] = tf_data.take(step)

        return kfold_train, kfold_val
