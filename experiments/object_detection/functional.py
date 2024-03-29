import tensorflow as tf

from tensorflow.keras import backend

from utils.tf.ops.metrics import iou


def pad_for_op(super_tensor, tensor, padding_value):

    if tf.size(super_tensor) <= tf.size(tensor):

        return tensor

    shape = tf.shape(tensor)

    size = tf.shape(super_tensor)[0] - shape[0]

    pad = tf.broadcast_to([padding_value], shape=(size, *shape[1:]))

    tensor = tf.concat([tensor, pad], axis=0)

    return tensor


def mask_by_zero_var(tensor, axis=-1, threshold=1e-3, logical_not=True):

    area = tf.math.reduce_variance(tensor, axis=axis)

    mask = tf.less_equal(area, threshold)

    if logical_not:

        mask = tf.logical_not(mask)

    return mask


def mask_by_zero_sum(tensor, axis=-1, threshold=1e-7, logical_not=True):

    area = tf.reduce_sum(tensor, axis=axis)

    mask = tf.less_equal(tf.abs(area), threshold)

    if logical_not:

        mask = tf.logical_not(mask)

    return mask


def mask_by_max(tensor, return_max=False, logical_not=False):

    max_values = tf.reduce_max(tensor, axis=-1)

    mask = tf.equal(tensor, max_values[..., None])

    if logical_not:

        mask = tf.logical_not(mask)

    if return_max:

        return mask, max_values

    return mask


def mask_by_value(tensor, value, logical_not=False):

    mask = tf.equal(tf.squeeze(tensor, axis=-1), value)

    if logical_not:

        mask = tf.logical_not(mask)

    return mask


def boolean_mask(y_true, y_pred, mask):

    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)

    return y_true, y_pred


def trim_invalid_detections(y_true, y_pred, bbox_true, bbox_pred, return_condition=False):

    condition_0 = mask_by_value(y_true, value=-1.0, logical_not=True)
    condition_1 = mask_by_zero_sum(bbox_pred, axis=-1, logical_not=True)

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

    steps = tf.shape(scores)[0]

    selected_indices = []

    def suppress_step(i):

        iselected_indices = \
            tf.image.non_max_suppression(boxes[i], scores[i], max_output_size, iou_threshold, score_threshold)

        if iselected_indices.shape[0]:

            selected_indices.append(iselected_indices)

        return [i + 1]

    _ = tf.while_loop(lambda i: i < steps, suppress_step, [0])

    return selected_indices


def gather_selected(tensor, selected_indices, same_padding=False, padding_value=0.0):

    shape = tf.shape(tensor)

    padding_value = tf.cast(padding_value, dtype=tensor.dtype)

    pad = tf.broadcast_to([padding_value], shape=shape[1:])

    selected_values = []

    def gather_step(i):

        iselected_values = tf.gather(tensor[i], selected_indices[i])

        # padding
        if same_padding:

            iselected_values = tf.tensor_scatter_nd_update(pad, selected_indices[i][:, None], iselected_values)

        selected_values.append(iselected_values)

        return [i + 1]

    _ = tf.while_loop(lambda i: i < len(selected_indices), gather_step, [0])

    if same_padding:

        return tf.stack(selected_values, axis=0)

    else:

        return selected_values


def suppress_selection_contradictions(selected_indices0, selected_indices1):

    if len(selected_indices0) != len(selected_indices1):

        raise ValueError('...')

    padding_value0 = tf.cast(-1, dtype=tf.int32)
    padding_value1 = tf.cast(-2, dtype=tf.int32)

    selected_indices = []

    def suppress_step(i):

        iselected_indices0 = pad_for_op(selected_indices1[i], selected_indices0[i], padding_value0)
        iselected_indices1 = pad_for_op(selected_indices0[i], selected_indices1[i], padding_value1)

        iselected_indices = tf.sets.intersection(iselected_indices0[None, :], iselected_indices1[None, :])

        selected_indices.append(iselected_indices.values)

        return [i + 1]

    _ = tf.while_loop(lambda i: i < len(selected_indices0), suppress_step, [0])

    return selected_indices


def suppress_selection_matching(selected_indices0, selected_indices1):

    if len(selected_indices0) != len(selected_indices1):

        raise ValueError('...')

    padding_value0 = tf.cast(-1, dtype=tf.int32)
    padding_value1 = tf.cast(-2, dtype=tf.int32)

    selected_indices = []

    def suppress_step(i):

        iselected_indices0 = pad_for_op(selected_indices1[i], selected_indices0[i], padding_value0)
        iselected_indices1 = pad_for_op(selected_indices0[i], selected_indices1[i], padding_value1)

        iselected_indices = tf.sets.difference(iselected_indices0[None, :], iselected_indices1[None, :])

        selected_indices.append(iselected_indices.values)

        return [i + 1]

    _ = tf.while_loop(lambda i: i < len(selected_indices0), suppress_step, [0])

    return selected_indices


def shift_indices(indices, start):

    shifted_indices = []

    def shift_step(i):

        iindices = indices[i] + start

        shifted_indices.append(iindices)

        return [i + 1]

    _ = tf.while_loop(lambda i: i < len(indices), shift_step, [0])

    return shifted_indices


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


def pruning_score(y_true, y_pred, score_threshold: float = float('-inf')):

    scores = tf.reduce_max(y_pred, axis=-1)

    y_pred = backend.argmax(y_pred, axis=-1)

    y_true = tf.cast(tf.squeeze(y_true), dtype=tf.int32)
    y_pred = tf.cast(y_pred, dtype=tf.int32)

    condition_0 = tf.equal(y_true, y_pred)
    condition_1 = tf.greater(scores, score_threshold)

    pruning_tensor = tf.logical_and(condition_0, condition_1)
    pruning_tensor = tf.cast(pruning_tensor, dtype=tf.float32)

    return pruning_tensor


def compute_bbox_iou(y_true, y_pred, bbox_true, bbox_pred,
                     score_threshold: float = float('-inf'), factor: float = 1.0):

    if tf.size(y_true) == 0:

        return tf.constant(0.0, dtype=tf.float32)

    pruning_tensor = pruning_score(y_true, y_pred, score_threshold)

    score = iou(bbox_true, bbox_pred)

    return factor * backend.mean(score * pruning_tensor)


def compute_masks_iou(y_true, y_pred, masks_true, masks_pred,
                      score_threshold: float = float('-inf'), mask_threshold: float = 0.5,
                      iou_threshold: float = 0.5, factor: float = 1.0):

    if tf.size(y_true) == 0:

        return tf.constant(0.0, dtype=tf.float32)

    pruning_tensor = pruning_score(y_true, y_pred, score_threshold)

    masks_true = tf.cast(masks_true, dtype=tf.bool)
    masks_pred = tf.greater(masks_pred, mask_threshold)

    def compute_step(elems):

        mask_true, mask_pred = elems

        intersection = tf.logical_and(mask_true, mask_pred)
        intersection = tf.cast(intersection, dtype=tf.float32)
        intersection = tf.reduce_sum(intersection)

        union = tf.logical_or(mask_true, mask_pred)
        union = tf.cast(union, dtype=tf.float32)
        union = tf.reduce_sum(union)

        iscore = tf.divide(intersection, union)

        iscore = tf.cond(tf.greater(iscore, iou_threshold), lambda: iscore, lambda: 0.0)

        return iscore

    score = tf.map_fn(compute_step, (masks_true, masks_pred), fn_output_signature=tf.float32)

    return factor * backend.mean(score * pruning_tensor)
