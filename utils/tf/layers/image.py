from typing import Tuple

import tensorflow as tf

from tensorflow.keras.layers import Layer


class NonMaxSuppression(Layer):

    def __init__(self, max_num_boxes_per_class: int, max_num_boxes: int, iou_threshold: float = 0.5,
                 score_threshold: float = float('-inf'), pad_per_class: bool = False, clip_boxes: bool = True,
                 *args, **kwargs):

        self.max_num_boxes_per_class = max_num_boxes_per_class
        self.max_num_boxes = max_num_boxes

        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

        self.pad_per_class = pad_per_class
        self.clip_boxes = clip_boxes

        super(NonMaxSuppression, self).__init__(*args, **kwargs)

    def call(self, inputs, training=None):

        scores = inputs[0]
        boxes = inputs[1]

        output_boxes, output_scores, output_classes, valid_counts = tf.image.\
            combined_non_max_suppression(boxes, scores, self.max_num_boxes_per_class, self.max_num_boxes,
                                         self.iou_threshold, self.score_threshold, self.pad_per_class, self.clip_boxes)

        return output_scores, output_boxes, output_classes, valid_counts


class RoIAlign(Layer):

    def __init__(self, crop_size: Tuple[int, int], method: str = 'bilinear', boxes_normalized: bool = True,
                 *args, **kwargs):

        self.crop_size = crop_size

        self.method = method

        self.boxes_normalized = boxes_normalized

        super(RoIAlign, self).__init__(*args, **kwargs)

    def map_crop_and_resize(self, inputs):

        images = inputs[0]
        boxes = inputs[1]

        batch_size = tf.shape(boxes)[0]

        def crop_and_resize(_boxes):

            boxes_indices = tf.range(0, tf.shape(inputs[0])[0])

            output = tf.image.crop_and_resize(images, _boxes, boxes_indices, self.crop_size, self.method,
                                              self.boxes_normalized)

            return output

        boxes = tf.reshape(boxes, (batch_size, -1, 4))
        boxes = tf.transpose(boxes, perm=[1, 0, 2])

        outputs = tf.map_fn(crop_and_resize, boxes, fn_output_signature=tf.float32)

        outputs = tf.transpose(outputs, perm=[1, 2, 3, 0, 4])

        return outputs

    def call(self, inputs, training=None):

        return self.map_crop_and_resize(inputs)
