import tensorflow as tf

from tensorflow.keras import backend


def mask_by_zero_var(tensor, axis=-1, threshold=1e-3):

    area = tf.math.reduce_variance(tensor, axis=axis)

    mask = tf.less_equal(area, threshold)

    return mask


def mask_by_zero_sum(tensor, axis=-1, threshold=1e-7):

    area = tf.reduce_sum(tensor, axis=axis)

    mask = tf.less_equal(area, threshold)
    mask = tf.logical_or(mask, tf.less(area, 0.0))

    return mask


def mask_by_max(tensor, return_max=False):

    max_values = tf.reduce_max(tensor, axis=-1)

    mask = tf.equal(tensor, max_values[..., None])

    if return_max:

        return mask, max_values

    return mask


def mask_by_value(tensor, value):

    mask = tf.not_equal(tf.squeeze(tensor, axis=-1), value)

    return mask


def boolean_mask(y_true, y_pred, mask):

    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)

    return y_true, y_pred


def trim_invalid_detections(y_true, y_pred, bbox_true, bbox_pred, return_condition=False):

    condition_0 = mask_by_value(y_true, value=-1.0)
    condition_1 = mask_by_zero_sum(bbox_pred, axis=-1)

    condition = tf.logical_and(condition_0, condition_1)

    y_true, y_pred = boolean_mask(y_true, y_pred, condition)
    bbox_true, regions_boxes = boolean_mask(bbox_true, bbox_pred, condition)

    if return_condition:

        return (y_true, y_pred), (bbox_true, regions_boxes), condition

    return (y_true, y_pred), (bbox_true, regions_boxes)


def select_max_score_boxes(y_pred, bbox_pred, return_scores=False):

    output_shape = bbox_pred.shape[:2] + (4, )

    condition, scores = mask_by_max(y_pred, return_max=True)

    bbox_pred = tf.boolean_mask(bbox_pred, condition)
    bbox_pred = tf.reshape(bbox_pred, output_shape)

    if return_scores:

        return bbox_pred, scores

    return bbox_pred


def suppress_invalid_detections(scores, boxes, max_output_size,
                                score_threshold=float('-inf'), iou_threshold=0.5):

    selected_indices = []

    def suppress_step(i):

        iselected_indices = \
            tf.image.non_max_suppression(boxes[i], scores[i], max_output_size, iou_threshold, score_threshold)

        if len(iselected_indices):

            selected_indices.append(iselected_indices)

        return [i + 1]

    _ = tf.while_loop(lambda i: i < len(scores), suppress_step, [0])

    return selected_indices


def gather_selected(tensor, selected_indices):

    selected_values = []

    def gather_step(i):

        iselected_values = tf.gather(tensor[i], selected_indices[i])

        selected_values.append(iselected_values)

        return [i + 1]

    _ = tf.while_loop(lambda i: i < len(selected_indices), gather_step, [0])

    return selected_values


def as_binary(y_true, keep_invalid=True):

    binary_true = tf.greater(y_true, 0.0)
    binary_true = tf.cast(binary_true, dtype=tf.float32)

    if keep_invalid:

        invalid = tf.equal(y_true, -1.0)
        invalid = -1.0 * tf.cast(invalid, dtype=tf.float32)

        return binary_true + invalid

    return binary_true


def smooth_l1_loss(y_true, y_pred):

    if tf.size(y_true) == 0:

        return tf.constant(0.0, dtype=tf.float32)

    loss = backend.abs(y_true - y_pred)

    mask = tf.less(loss, 1.0)

    loss_0 = tf.boolean_mask(loss, mask)
    loss_0 = backend.sum(0.5 * loss_0 ** 2)

    loss_1 = tf.boolean_mask(loss, tf.logical_not(mask))
    loss_1 = backend.sum(loss_1 - 0.5)

    loss = loss_0 + loss_1

    return loss


def sparse_categorical_crossentropy(y_true, y_pred):

    if tf.size(y_true) == 0:

        return tf.constant(0.0, dtype=tf.float32)

    loss = backend.sparse_categorical_crossentropy(y_true, y_pred)

    return backend.mean(loss)


def binary_crossentropy(y_true, y_pred):

    if tf.size(y_true) == 0:

        return tf.constant(0.0, dtype=tf.float32)

    loss = backend.binary_crossentropy(y_true, y_pred)

    return backend.mean(loss)
