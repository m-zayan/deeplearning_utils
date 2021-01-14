from typing import Union
import tensorflow as tf

__all__ = ['KFold']


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
