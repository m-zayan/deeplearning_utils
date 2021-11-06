import tensorflow as tf

from experiments.object_detection.rcnn import MaskRCNN


def dummy_dataset(num_test_cases, input_shape, num_predictions, num_classes):

    images = tf.random.normal(shape=(num_test_cases, *input_shape), dtype=tf.float32)

    # sparse: [num_test_cases, num_predictions, num_classes]
    y_true = tf.random.uniform(shape=(num_test_cases, num_predictions, 1),
                               minval=-1, maxval=num_classes, dtype=tf.int32)

    y_true = tf.cast(y_true, dtype=tf.float32)

    # sparse: [num_test_cases, num_predictions, num_classes, 4]
    bbox_true = tf.random.normal(shape=(num_test_cases, num_predictions, 4), dtype=tf.float32) * 0.01

    masks_true = tf.random.uniform(shape=(num_test_cases, num_predictions, 28, 28, 1),
                                   minval=0, maxval=2, dtype=tf.int32)

    masks_true = tf.cast(masks_true, dtype=tf.float32)

    return images, y_true, bbox_true, masks_true


def test_build_mrcnn(num_test_cases, input_shape, num_classes):

    mrcnn = MaskRCNN(input_shape=input_shape, filters_per_level=[32, 64, 128],
                     kernel_size_per_level=[(7, 7), (5, 5), (3, 3)],
                     num_classes=num_classes, use_box_per_class=True, use_rpn_multiclass=False)

    mrcnn.compile()

    mrcnn.summary()

    print(mrcnn.inception_size())

    num_predictions = mrcnn.num_predictions()

    print(num_predictions, mrcnn.max_num_objects())

    data = dummy_dataset(num_test_cases, input_shape, num_predictions, num_classes)

    tf_data = tf.data.Dataset.from_tensor_slices(data)
    tf_data = tf_data.batch(1)

    _ = mrcnn.fit(tf_data, epochs=2)

    # loss = mrcnn.train_step(data)
    #
    # print(loss)
    #
    # print(mrcnn.test_step(data))

    any_size = 1000

    # detections = mrcnn.detect_disjoint_step(data[0], max_output_size=any_size)
    #
    # print(len(detections[0][0][0]))

    detections = mrcnn.predict_on_batch(data[0], max_output_size=any_size)

    print(len(detections[0][0]))


test_build_mrcnn(num_test_cases=2, input_shape=(224, 224, 3), num_classes=100)
