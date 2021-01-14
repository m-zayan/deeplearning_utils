import tensorflow as tf


def get_grid_mask(grid_h, grid_w):

    """
    Parameters
    ----------
    grid_h: int

        grid height

    grid_w: int

        grid width

    Returns
    -------
    2D Tensor [grid mask]
    """

    grid_mask = tf.range(grid_h * grid_w)
    grid_mask = tf.reshape(grid_mask, shape=(grid_h, grid_w))

    return grid_mask


def non_max_suppression(boxes, scores, max_output_size: int,
                        iou_threshold: float = 0.5, score_threshold: float = float('-inf')):

    """
    https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression
    """
    # n_boxes x 4, n_boxes x 1

    indices = tf.image.non_max_suppression(boxes=boxes, scores=scores, max_output_size=max_output_size,
                                           iou_threshold=iou_threshold, score_threshold=score_threshold)

    return indices


def nd_non_max_suppression(boxes, scores, max_output_size: int,
                           iou_threshold: float = 0.5, score_threshold: float = float('-inf'),
                           grid_mask=None):

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

    grid_mask: np.ndarray | tf.Tensor,

        ex. tf.model_utils.ops.get_grid_mask(....)

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

            cells = tf.where(grid_mask == i)
            cells = tf.cast(cells, dtype='float32')

            obj_cells.append(cells)

            j += 1

        return i + 1, j

    _, _ = tf.while_loop(lambda i, j: i < len(scores), body, [0, 0])

    return choices, obj_cells
