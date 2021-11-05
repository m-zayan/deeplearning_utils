import tensorflow as tf

from tensorflow.keras import optimizers

from experiments.object_detection.rcnn import SeparableMaskRCNN


def dummy_dataset(input_shape, num_predictions, num_classes):

    images = tf.random.normal(shape=(2, *input_shape), dtype=tf.float32)

    # sparse: [2, num_predictions, num_classes]
    y_true = tf.random.uniform(shape=(2, num_predictions, 1), minval=-1, maxval=num_classes, dtype=tf.int32)
    y_true = tf.cast(y_true, dtype=tf.float32)

    # sparse: [2, num_predictions, num_classes, 4]
    bbox_true = tf.random.normal(shape=(2, num_predictions, 4), dtype=tf.float32) * 0.01

    masks_true = tf.random.uniform(shape=(2, num_predictions, 28, 28, 1), minval=0, maxval=2, dtype=tf.int32)
    masks_true = tf.cast(masks_true, dtype=tf.float32)

    return images, y_true, bbox_true, masks_true


def test_build_mrcnn(input_shape, num_classes):

    mrcnn = SeparableMaskRCNN(input_shape=input_shape, filters_per_level=[32, 64, 128],
                              kernel_size_per_level=[(7, 7), (5, 5), (3, 3)],
                              num_classes=num_classes, use_box_per_class=True, use_rpn_multiclass=False)

    rpn_optimizer = optimizers.Adam(learning_rate=1e-3)
    cls_optimizer = optimizers.Adam(learning_rate=1e-3)
    masks_optimizer = optimizers.Adam(learning_rate=1e-3)

    mrcnn.compile(rpn_optimizer, cls_optimizer, masks_optimizer)

    mrcnn.summary()

    num_predictions = mrcnn.num_predictions()

    print(num_predictions, mrcnn.max_num_objects())

    data = dummy_dataset(input_shape, num_predictions, num_classes)

    loss = mrcnn.train_step(data)

    print(loss)

    any_size = 1000

    # detections = mrcnn.detect_disjoint_step(data[0], max_output_size=any_size)
    #
    # print(len(detections[0][0][0]))

    detections = mrcnn.detect_joint_step(data[0], max_output_size=any_size)

    print(len(detections[0][0]))


test_build_mrcnn(input_shape=(224, 224, 3), num_classes=100)
