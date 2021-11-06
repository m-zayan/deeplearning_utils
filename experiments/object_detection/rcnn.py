from typing import Tuple, List

import tensorflow as tf

from tensorflow.keras import backend, optimizers

from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Dense, LayerNormalization, \
    Flatten, Reshape, Activation, TimeDistributed, UpSampling2D

from tensorflow.keras.models import Model

from utils.tf.layers.image import RoIAlign

from . import functional

__all__ = ['BaseRCNN', 'SeparableMaskRCNN', 'MaskRCNN']


class BaseRCNN(Model):

    def __init__(self, inputs: Input, encoder, rpn: List[Model], region_size: Tuple[int, int],
                 num_classes, num_boxes, **kwargs):

        """ Faster R-CNN + RoIAlign + Some Assumptions """

        # ============================================================================================

        self.region_size = region_size

        self.num_classes = num_classes
        self.num_boxes = num_boxes

        # ============================================================================================

        regions_score = []
        regions_boxes = []

        # List <---> [batch_size, height, width, depth]
        inception = encoder(inputs)

        # ============================================================================================

        for i in range(len(inception)):

            # score: [batch_size, num_boxes, num_classes], boxes: [batch_size, max_num_boxes, 4]
            iregions_score, iregions_boxes = rpn[i](inception[i])

            iregions_score = \
                Reshape((-1, self.num_classes), name=f'inception_{i}_regions_scores')(iregions_score)

            iregions_boxes = \
                Reshape((-1, 4), name=f'inception_{i}_regions_boxes')(iregions_boxes)

            regions_score.append(iregions_score)
            regions_boxes.append(iregions_boxes)

        # ============================================================================================

        outputs = [regions_score, regions_boxes]

        # ============================================================================================

        super(BaseRCNN, self).__init__(inputs=inputs, outputs=outputs, **kwargs)

        # ============================================================================================

        self.max_num_boxes_per_inception = []

        self.roi_align: List[RoIAlign] = []

        for i in range(len(inception)):

            self.roi_align.append(RoIAlign(crop_size=self.region_size, method='bilinear', boxes_normalized=True))

            self.roi_align[i].build([regions_score[i].shape, regions_boxes[i].shape])

        # ============================================================================================

    def get_config(self):

        return super(BaseRCNN, self).get_config()

    def call(self, inputs, training=None, mask=None):

        return super(BaseRCNN, self).call(inputs, training, mask)

    def train_step(self, data):

        images, y_true, bbox_true = data

        with tf.GradientTape() as tape:

            regions_score, regions_boxes = self(images, training=False)

            regions_score = tf.concat(regions_score, axis=1)
            regions_boxes = tf.concat(regions_boxes, axis=1)

            # ==================================================================================================

            (y_true, regions_score), (bbox_true, regions_boxes) = \
                functional.trim_invalid_detections(y_true, regions_score, bbox_true, regions_boxes)

            # ==================================================================================================

            rpn_cls_loss = functional.sparse_categorical_crossentropy(y_true, regions_score)
            rpn_loc_loss = functional.smooth_l1_loss(bbox_true, regions_boxes)

            loss = rpn_cls_loss + rpn_loc_loss

        if loss != 0.0:

            # ============================================================================================

            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

            # ============================================================================================

            # update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            # ============================================================================================

        return {'rpn_cls_loss': rpn_cls_loss, 'rpn_loc_loss': rpn_loc_loss}

    def test_step(self, data):

        images, y_true, bbox_true = data

        # ==================================================================================================

        regions_score, regions_boxes = self(images, training=False)

        # ==================================================================================================

        (y_true, regions_score), (bbox_true, regions_boxes) = \
            functional.trim_invalid_detections(y_true, regions_score, bbox_true, regions_boxes)

        # ==================================================================================================

        rpn_cls_loss = functional.sparse_categorical_crossentropy(y_true, regions_score)
        rpn_loc_loss = functional.smooth_l1_loss(bbox_true, regions_boxes)

        # ==================================================================================================

        return {'rpn_cls_loss': rpn_cls_loss, 'rpn_loc_loss': rpn_loc_loss}

    def predict_regions_per_batch(self, images, max_num_regions=None):

        batch_size = tf.shape(images)[0]

        regions = []
        selected_indices = []

        # List <---> [batch_size, height, width, depth]
        inception = self.layers[1](images)

        # ============================================================================================

        for i in range(len(inception)):

            # score: [batch_size, num_boxes, num_classes], boxes: [batch_size, max_num_boxes, 4]
            iregions_score, iregions_boxes = self.layers[i + 2](inception[i])

            # ============================================================================================

            iregions_score = tf.reduce_max(iregions_score, axis=-1)

            iregions_boxes = tf.reshape(iregions_boxes, (batch_size, -1, 4))

            # ============================================================================================

            if max_num_regions is None:

                max_num_regions = tf.shape(iregions_boxes)[1]

            # non max suppression
            iselected_indices = functional.suppress_invalid_detections(iregions_score, iregions_boxes, max_num_regions)

            # boxes: [batch_size, max_num_boxes, 4] <---> maybe include zero paddings
            iregions_boxes = functional.gather_selected(iregions_boxes, iselected_indices, same_padding=True)

            # ============================================================================================

            # regions: [batch_size, crop_height, crop_width, num_regions, depth] <---> Proceed with zero paddings
            iregions = self.roi_align[i]([inception[i], iregions_boxes])

            # ============================================================================================

            # regions: [batch_size, num_regions, crop_height, crop_width, depth] <---> Proceed with zero paddings
            iregions = tf.transpose(iregions, perm=[0, 3, 1, 2, 4])

            regions.append(iregions)
            selected_indices.append(iselected_indices)

        return regions, selected_indices

    @staticmethod
    def downsampling_block(inputs, filters, kernel_size, activation='relu', padding='same', use_maxpool=True):

        z = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=1)(inputs)
        z = LayerNormalization()(z)
        z = Activation(activation)(z)

        z = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=1)(z)
        z = LayerNormalization()(z)
        z = Activation(activation)(z)

        if use_maxpool:

            z = MaxPooling2D(pool_size=(2, 2))(z)

        return z

    @staticmethod
    def upsampling_block(inputs, filters, kernel_size, activation='relu', padding='same'):

        z = TimeDistributed(Conv2DTranspose(filters=filters, kernel_size=kernel_size, padding=padding))(inputs)
        z = TimeDistributed(LayerNormalization())(z)
        z = TimeDistributed(Activation(activation))(z)

        z = TimeDistributed(Conv2DTranspose(filters=filters, kernel_size=kernel_size, padding=padding))(z)
        z = TimeDistributed(LayerNormalization())(z)
        z = TimeDistributed(Activation(activation))(z)

        z = TimeDistributed(UpSampling2D(size=(2, 2)))(z)

        return z


class SeparableMaskRCNN:

    def __init__(self, input_shape, filters_per_level, kernel_size_per_level, region_size=(7, 7),
                 cls_projection_size=512, masks_projection_size=64, num_classes=1000, num_boxes=5,
                 use_box_per_class=True, use_rpn_multiclass=False):

        # ============================================================================================

        if len(filters_per_level) != len(kernel_size_per_level):

            raise ValueError('...')

        # ============================================================================================

        self.num_levels = len(filters_per_level)

        self.filters_per_level = filters_per_level
        self.kernel_size_per_level = kernel_size_per_level

        self.region_size = region_size

        self.cls_projection_size = cls_projection_size
        self.masks_projection_size = masks_projection_size

        self.num_classes = num_classes
        self.num_boxes = num_boxes

        self.use_box_per_class = use_box_per_class

        self.use_rpn_multiclass = use_rpn_multiclass

        # ============================================================================================

        if self.use_rpn_multiclass:

            self.rpn = self.build_base_rcnn(input_shape=input_shape)

        else:

            self.rpn = self.build_base_rcnn(input_shape=input_shape, num_classes=2)

        self.classification_networks = []
        self.segmentation_networks = []

        for i in range(self.num_levels):

            self.classification_networks.append(self.build_classification_network(inception_index=i))
            self.segmentation_networks.append(self.build_segmentation_network(inception_index=i))

        # ============================================================================================

        self.rpn_optimizer = None
        self.cls_optimizer = None
        self.masks_optimizer = None

        # ============================================================================================

    # have to be replaced by a pretrained backbone later via subclassing (e.g. ResNet)
    def build_encoder(self, input_shape):

        inputs = Input(shape=input_shape)

        z = BaseRCNN.downsampling_block(inputs, 8, (3, 3))
        z = BaseRCNN.downsampling_block(z, 16, (3, 3))
        z = BaseRCNN.downsampling_block(z, 32, (3, 3))

        inception = []

        for j in range(self.num_levels):

            z = BaseRCNN.downsampling_block(z, self.filters_per_level[j], self.kernel_size_per_level[j])

            inception.append(z)

        encoder = Model(inputs, inception)

        return encoder

    def build_rpn(self, input_shape, num_classes=None, num_boxes=None):

        if num_classes is None:

            num_classes = self.num_classes

        if num_boxes is None:

            num_boxes = self.num_boxes

        inputs = Input(shape=input_shape)

        scores = Conv2D(num_boxes * num_classes, kernel_size=(1, 1))(inputs)
        scores = Reshape((-1, num_classes))(scores)
        scores = TimeDistributed(Activation('softmax'))(scores)

        boxes = Conv2D(num_boxes * 4, kernel_size=(1, 1))(inputs)
        boxes = Reshape((-1, 4))(boxes)

        rpn = Model(inputs, [scores, boxes])

        return rpn

    def build_base_rcnn(self, input_shape, num_classes=None, num_boxes=None):

        if num_classes is None:

            num_classes = self.num_classes

        if num_boxes is None:

            num_boxes = self.num_boxes

        inputs = Input(shape=input_shape)

        encoder = self.build_encoder(input_shape=input_shape)

        rpn_per_inception = []

        for i in range(len(encoder.outputs)):

            shape = encoder.output_shape[i][1:]

            rpn_per_inception.append(self.build_rpn(shape, num_classes, num_boxes))

        base_rcnn = BaseRCNN(inputs=inputs, encoder=encoder, rpn=rpn_per_inception, region_size=self.region_size,
                             num_classes=num_classes, num_boxes=num_boxes)

        return base_rcnn

    def build_classification_network(self, inception_index):

        depth = self.filters_per_level[inception_index]

        inputs = Input(shape=(None, self.region_size[0], self.region_size[1], depth))

        z = TimeDistributed(Flatten())(inputs)

        z = TimeDistributed(Dense(self.cls_projection_size))(z)
        z = TimeDistributed(LayerNormalization())(z)
        z = TimeDistributed(Activation('relu'))(z)

        z = TimeDistributed(Dense(self.cls_projection_size))(z)
        z = TimeDistributed(LayerNormalization())(z)
        z = TimeDistributed(Activation('relu'))(z)

        rcnn_logits = TimeDistributed(Dense(self.num_classes), name='rcnn_logits')(z)
        rcnn_prob = TimeDistributed(Activation('softmax'), name='rcnn_prob')(rcnn_logits)

        if self.use_box_per_class:

            rcnn_bbox = \
                TimeDistributed(Dense(4 * self.num_classes))(z)

            rcnn_bbox = Reshape((-1, self.num_classes, 4), name='rcnn_bbox')(rcnn_bbox)

        else:

            rcnn_bbox = \
                TimeDistributed(Dense(4), name='rcnn_bbox')(z)

        classification_network = Model(inputs, [rcnn_prob, rcnn_bbox])

        return classification_network

    def build_segmentation_network(self, inception_index):

        depth = self.filters_per_level[inception_index]

        inputs = Input(shape=[None, self.region_size[0], self.region_size[1], depth])

        z = BaseRCNN.upsampling_block(inputs, filters=self.masks_projection_size, kernel_size=(3, 3))
        z = BaseRCNN.upsampling_block(z, filters=self.masks_projection_size, kernel_size=(3, 3))

        masks = TimeDistributed(Conv2DTranspose(filters=1, kernel_size=(1, 1), padding='valid'))(z)
        masks = Activation('sigmoid')(masks)

        segmentation_network = Model(inputs, masks)

        return segmentation_network

    def compile(self, rpn_optimizer, cls_optimizer, masks_optimizer):

        self.rpn.compile(optimizer=rpn_optimizer)

        for i in range(self.num_levels):

            self.classification_networks[i].compile(optimizer=cls_optimizer)
            self.segmentation_networks[i].compile(optimizer=masks_optimizer)

    def summary(self):

        self.rpn.summary()

        for i in range(self.num_levels):

            self.classification_networks[i].summary()
            self.segmentation_networks[i].summary()

    def inception_size(self):

        size = []

        shapes = self.rpn.layers[1].output_shape

        for i in range(self.num_levels):

            size.append(shapes[i][1:3] + (self.num_boxes, ))

        return size

    def num_predictions(self):

        count = 0

        for i in range(self.num_levels):

            count += self.rpn.output_shape[0][i][1]

        return count

    def max_num_objects(self):

        return self.num_predictions() // self.num_boxes

    def train_rpn_step(self, data):

        images, y_true, bbox_true = data

        if self.use_rpn_multiclass:

            # update rpn network
            metrics_dict = self.rpn.train_step((images, y_true, bbox_true))

        else:

            binary_true = functional.as_binary(y_true)

            # update rpn network
            metrics_dict = self.rpn.train_step((images, binary_true, bbox_true))

        return metrics_dict

    def train_step(self, data, update_rpn=False):

        images, y_true, bbox_true, masks_true = data

        # ============================================================================================

        if update_rpn:

            self.train_rpn_step((images, y_true, bbox_true))

        # ============================================================================================

        regions, selected_indices = self.rpn.predict_regions_per_batch(images)

        # ============================================================================================

        start = 0
        end = regions[0].shape[1]

        # ============================================================================================

        metrics_dict = {'loss_cls': 0.0, 'loss_loc': 0.0, 'loss_seg': 0.0,
                        'bbox_iou': 0.0, 'masks_iou': 0.0}

        factor = 1.0 / float(self.num_levels)

        # ============================================================================================

        for i in range(self.num_levels):

            iy_true = y_true[:, start:end]
            ibbox_true = bbox_true[:, start:end]
            imasks_true = masks_true[:, start:end]

            iy_true = functional.gather_selected(iy_true, selected_indices[i], same_padding=True, padding_value=-1.0)
            ibbox_true = functional.gather_selected(ibbox_true, selected_indices[i], same_padding=True)
            imasks_true = functional.gather_selected(imasks_true, selected_indices[i], same_padding=True)

            with tf.GradientTape() as tape:

                y_pred, bbox_pred = self.classification_networks[i](regions[i], training=True)

                if self.use_box_per_class:

                    bbox_pred = functional.select_max_score_boxes(y_pred, bbox_pred)

                # ==================================================================================================

                (iy_true, y_pred), (ibbox_true, bbox_pred), condition =\
                    functional.trim_invalid_detections(iy_true, y_pred, ibbox_true, bbox_pred, return_condition=True)

                # ==================================================================================================

                iloss_cls = functional.sparse_categorical_crossentropy(iy_true, y_pred)
                iloss_loc = functional.smooth_l1_loss(ibbox_true, bbox_pred)

                iloss = iloss_cls + iloss_loc

                metrics_dict['loss_cls'] += iloss_cls
                metrics_dict['loss_loc'] += iloss_loc

            if iloss != 0.0:

                # ============================================================================================

                trainable_vars = self.classification_networks[i].trainable_variables
                gradients = tape.gradient(iloss, trainable_vars)

                self.classification_networks[i].optimizer.apply_gradients(zip(gradients, trainable_vars))

                # ============================================================================================

            with tf.GradientTape() as tape:

                masks_pred = self.segmentation_networks[i](regions[i], training=True)

                # trim invalid detections for segmentation
                imasks_true, masks_pred = functional.boolean_mask(imasks_true, masks_pred, condition)

                iloss_seg = functional.binary_crossentropy(imasks_true, masks_pred)

                metrics_dict['loss_seg'] += iloss_seg

            if iloss_seg != 0.0:

                # ============================================================================================

                trainable_vars = self.segmentation_networks[i].trainable_variables
                gradients = tape.gradient(iloss_seg, trainable_vars)

                self.segmentation_networks[i].optimizer.apply_gradients(zip(gradients, trainable_vars))

                # ============================================================================================

            start = end

            if i + 1 < len(regions):

                end += regions[i + 1].shape[1]

            # ============================================================================================

            metrics_dict['bbox_iou'] += \
                functional.compute_bbox_iou(iy_true, y_pred, ibbox_true, bbox_pred, factor=factor)

            metrics_dict['masks_iou'] += \
                functional.compute_masks_iou(iy_true, y_pred, imasks_true, masks_pred, factor=factor)

            # ============================================================================================

        return metrics_dict

    def test_step(self, data):

        images, y_true, bbox_true, masks_true = data

        # ============================================================================================

        regions, selected_indices = self.rpn.predict_regions_per_batch(images)

        # ============================================================================================

        start = 0
        end = regions[0].shape[1]

        # ============================================================================================

        metrics_dict = {'loss_cls': 0.0, 'loss_loc': 0.0, 'loss_seg': 0.0,
                        'bbox_iou': 0.0, 'masks_iou': 0.0}

        factor = 1.0 / float(self.num_levels)

        # ============================================================================================

        for i in range(self.num_levels):

            iy_true = y_true[:, start:end]
            ibbox_true = bbox_true[:, start:end]
            imasks_true = masks_true[:, start:end]

            iy_true = functional.gather_selected(iy_true, selected_indices[i], same_padding=True, padding_value=-1.0)
            ibbox_true = functional.gather_selected(ibbox_true, selected_indices[i], same_padding=True)
            imasks_true = functional.gather_selected(imasks_true, selected_indices[i], same_padding=True)

            # ==================================================================================================

            y_pred, bbox_pred = self.classification_networks[i](regions[i], training=True)

            if self.use_box_per_class:

                bbox_pred = functional.select_max_score_boxes(y_pred, bbox_pred)

            # ==================================================================================================

            (iy_true, y_pred), (ibbox_true, bbox_pred), condition =\
                functional.trim_invalid_detections(iy_true, y_pred, ibbox_true, bbox_pred, return_condition=True)

            # ==================================================================================================

            iloss_cls = functional.sparse_categorical_crossentropy(iy_true, y_pred)
            iloss_loc = functional.smooth_l1_loss(ibbox_true, bbox_pred)

            metrics_dict['loss_cls'] += iloss_cls
            metrics_dict['loss_loc'] += iloss_loc

            # ==================================================================================================

            masks_pred = self.segmentation_networks[i](regions[i], training=True)

            # trim invalid detections for segmentation
            imasks_true, masks_pred = functional.boolean_mask(imasks_true, masks_pred, condition)

            iloss_seg = functional.binary_crossentropy(imasks_true, masks_pred)

            metrics_dict['loss_seg'] += iloss_seg

            # ============================================================================================

            start = end

            if i + 1 < len(regions):

                end += regions[i + 1].shape[1]

            # ============================================================================================

            metrics_dict['bbox_iou'] += \
                functional.compute_bbox_iou(iy_true, y_pred, ibbox_true, bbox_pred, factor=factor)

            metrics_dict['masks_iou'] += \
                functional.compute_masks_iou(iy_true, y_pred, imasks_true, masks_pred, factor=factor)

            # ============================================================================================

        return metrics_dict

    def detect_disjoint_step(self, images, max_output_size):

        detections_scores = []
        detections_boxes = []
        detections_masks = []

        detections_classes = []

        # ============================================================================================

        regions, regions_indices = self.rpn.predict_regions_per_batch(images)

        # ============================================================================================

        for i in range(self.num_levels):

            y_pred, bbox_pred = self.classification_networks[i](regions[i], training=False)
            masks_pred = self.segmentation_networks[i](regions[i], training=False)

            iclasses = backend.argmax(y_pred, axis=-1)

            if self.use_box_per_class:

                iboxes, iscores = functional.select_max_score_boxes(y_pred, bbox_pred, return_scores=True)

            else:

                iboxes, iscores = bbox_pred, tf.reduce_max(y_pred, axis=-1)

            iselected_indices = functional.suppress_invalid_detections(iscores, iboxes, max_output_size)

            iselected_indices = \
                functional.suppress_selection_contradictions(regions_indices[i], iselected_indices)

            iscores = functional.gather_selected(iscores, iselected_indices)
            iboxes = functional.gather_selected(iboxes, iselected_indices)
            imasks = functional.gather_selected(masks_pred, iselected_indices)
            iclasses = functional.gather_selected(iclasses, iselected_indices)

            detections_scores.append(iscores)
            detections_boxes.append(iboxes)
            detections_masks.append(imasks)
            detections_classes.append(iclasses)

        return detections_scores, detections_classes, detections_boxes, detections_masks

    def detect_joint_step(self, images, max_output_size):

        # ============================================================================================

        regions, regions_indices = self.rpn.predict_regions_per_batch(images)

        scores = []
        boxes = []
        masks = []

        classes = []

        # ============================================================================================

        for i in range(self.num_levels):

            y_pred, bbox_pred = self.classification_networks[i](regions[i], training=False)
            masks_pred = self.segmentation_networks[i](regions[i], training=False)

            iclasses = backend.argmax(y_pred, axis=-1)

            if self.use_box_per_class:

                iboxes, iscores = functional.select_max_score_boxes(y_pred, bbox_pred, return_scores=True)

            else:

                iboxes, iscores = bbox_pred, tf.reduce_max(y_pred, axis=-1)

            scores.append(iscores)
            boxes.append(iboxes)
            masks.append(masks_pred)
            classes.append(iclasses)

        # ============================================================================================

        scores = tf.concat(scores, axis=1)
        boxes = tf.concat(boxes, axis=1)
        masks = tf.concat(masks, axis=1)
        classes = tf.concat(classes, axis=1)

        # ============================================================================================

        # shift regions_indices

        shifted_indices = []

        for i in range(1, self.num_levels):

            start = regions[i].shape[1]

            regions_indices[i] = functional.shift_indices(regions_indices[i], start)

        regions_indices = list(zip(*regions_indices))

        for i in range(len(regions_indices)):

            shifted_indices.append([])

            for j in range(len(regions_indices[i])):

                shifted_indices[i] += list(regions_indices[i][j])

        for i in range(len(shifted_indices)):

            shifted_indices[i] = tf.cast(shifted_indices[i], dtype=tf.int32)

        # ============================================================================================

        selected_indices = functional.suppress_invalid_detections(scores, boxes, max_output_size)
        selected_indices = functional.suppress_selection_contradictions(shifted_indices, selected_indices)

        detections_scores = functional.gather_selected(scores, selected_indices)
        detections_boxes = functional.gather_selected(boxes, selected_indices)
        detections_masks = functional.gather_selected(masks, selected_indices)
        detections_classes = functional.gather_selected(classes, selected_indices)

        return detections_scores, detections_classes, detections_boxes, detections_masks


class MaskRCNN(Model):

    def __init__(self, input_shape, filters_per_level, kernel_size_per_level, region_size=(7, 7),
                 cls_projection_size=512, masks_projection_size=64, num_classes=1000, num_boxes=5,
                 use_box_per_class=True, use_rpn_multiclass=False):

        self.separable_mrcnn = SeparableMaskRCNN(input_shape=input_shape,
                                                 filters_per_level=filters_per_level,
                                                 kernel_size_per_level=kernel_size_per_level,
                                                 region_size=region_size,
                                                 cls_projection_size=cls_projection_size,
                                                 masks_projection_size=masks_projection_size,
                                                 num_classes=num_classes,
                                                 num_boxes=num_boxes,
                                                 use_box_per_class=use_box_per_class,
                                                 use_rpn_multiclass=use_rpn_multiclass)

        self._rpn_status = 'active'
        self._mrcnn_status = 'active'

        super(MaskRCNN, self).__init__(inputs=[], outputs=[])

    def get_config(self):

        return super(MaskRCNN, self).get_config()

    def summary(self):

        self.separable_mrcnn.summary()

    def call(self, data, training=None):

        raise NotImplementedError('...')

    def compile(self, optimizer='sgd', rpn_optimizer=None, cls_optimizer=None, masks_optimizer=None):

        if rpn_optimizer is None:

            rpn_optimizer = optimizers.Adam(learning_rate=1e-3)

        if cls_optimizer is None:

            cls_optimizer = optimizers.Adam(learning_rate=1e-3)

        if masks_optimizer is None:

            masks_optimizer = optimizers.Adam(learning_rate=1e-3)

        self.separable_mrcnn.compile(rpn_optimizer, cls_optimizer, masks_optimizer)

        super(MaskRCNN, self).compile(optimizer=optimizer, run_eagerly=True)

    def max_num_objects(self):

        return self.separable_mrcnn.max_num_objects()

    def num_predictions(self):

        return self.separable_mrcnn.num_predictions()

    def inception_size(self):

        return self.separable_mrcnn.inception_size()

    def set_rpn_status(self, status: str = 'active'):

        if status not in ['active', 'disabled']:

            raise ValueError('...')

        self._rpn_status = status

    def set_mrcnn_status(self, status: str = 'active'):

        if status not in ['active', 'disabled']:

            raise ValueError('...')

        self._mrcnn_status = status

    def train_step(self, data):

        if self._mrcnn_status == 'active':

            return self.separable_mrcnn.train_step(data, update_rpn=(self._rpn_status == 'active'))

        else:

            return self.separable_mrcnn.train_rpn_step(data[:3])

    def test_step(self, data):

        return self.separable_mrcnn.test_step(data)

    def predict_on_batch(self, images, max_output_size, disjoint=True):

        if disjoint:

            return self.separable_mrcnn.detect_disjoint_step(images, max_output_size=max_output_size)

        else:

            return self.separable_mrcnn.detect_joint_step(images, max_output_size=max_output_size)
