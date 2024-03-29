from typing import Union, Callable, Dict

import warnings

from tqdm import tqdm

import cv2

import tensorflow as tf
import pandas as pd

from ...ops.io import imread
from ...ops.reshape import split

__all__ = ['TfRecordWriter', 'TfRecordReader', 'download']


def download(fname, url, extract=True):

    ddir = tf.keras.utils.get_file(fname=fname, origin=url, extract=extract)

    return ddir


class TfRecordWriter:

    def __init__(self, shape: Union[tuple, list], n_records: int, image_ext: str = '.jpg', quality: int = 95,
                 grayscale=False, failure_ignore: bool = False, cache_warnings: bool = False):

        """
        Parameters
        ----------

        shape: tuple

        n_records: int

        image_ext: str
            default = '.jpg'

        quality: int
             Quality of the compression from 0 to 100, (higher is better and slower).

        failure_ignore: bool
            if True, then Get a warning message in case of
            failed reading image or writing any feature of an example,
            instead of raising ValueError,
            ex. skip reading a corrupted image file and continue writing TFRecord,

            default = False

        cache_warnings: bool
            if True, cache warnings messages, as dict,
            index of the failure example as a key, and causes  (error: failure message) as value,

            cache, failure_examples attribute

            default = False
        """

        self.shape = shape
        self.n_records = n_records
        self.image_ext = image_ext
        self.quality = quality
        self.grayscale = grayscale
        self.failure_ignore = failure_ignore
        self.cache_warnings = cache_warnings
        self.failure_examples = None

    def _check_ext(self):

        if self.image_ext[0] != '.':

            ext = '.'
            ext += self.image_ext
            self.image_ext = ext

    @staticmethod
    def _bytes_feature(value):

        """Returns a bytes_list from a string / byte."""

        if isinstance(value, type(tf.constant(0))):

            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.

        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):

        """Returns a float_list from a float / double."""

        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _int64_feature(value):

        """Returns an int64_list from a bool / enum / int / uint."""

        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _as_feature_type(self, f_dtype, value):

        if isinstance(f_dtype, str):

            if f_dtype == 'int':

                return TfRecordWriter._int64_feature(value)

            elif f_dtype == 'float':

                return TfRecordWriter._float_feature(value)

            elif f_dtype == 'bytes':

                return TfRecordWriter._bytes_feature(value)

            elif f_dtype == 'str':

                return TfRecordWriter._bytes_feature(value.encode())

            elif 'custom_' in f_dtype:

                return TfRecordWriter._bytes_feature(self.custom_type_to_bytes(value))

            else:

                raise ValueError('Value of Type : ' + f_dtype + ', is not supported, try "custom_type"')
        else:

            if isinstance(f_dtype(), int):

                return TfRecordWriter._int64_feature(value)

            elif isinstance(f_dtype(), float):

                return TfRecordWriter._float_feature(value)

            elif isinstance(f_dtype(), bytes):

                return TfRecordWriter._bytes_feature(value)

            elif isinstance(f_dtype(), str):

                return TfRecordWriter._bytes_feature(value.encode())

            else:

                raise ValueError('Value of Type : ' + str(f_dtype)[8:-2] + ', is not supported')

    def _serialize_example(self, loc: int, row: dict, dtypes: dict, image_key: str,
                           from_dir: str, has_ext: bool, func: Dict[str, Callable]):

        try:

            example = {}

            for key, value in row.items():

                _value = value

                if key == image_key:

                    if from_dir is not None:

                        _value = from_dir + value

                    if not has_ext:

                        _value += self.image_ext

                    if func is not None and image_key in func.keys():

                        _value = self.image_from_dir(_value)
                        _value = func[image_key](_value)

                        if _value is None:
                            raise ValueError(f'Function, func[{image_key}], returns None')

                        _value = self.image_to_bytes(_value)

                    else:

                        _value = self.bytes_from_dir(_value)

                example[key] = self._as_feature_type(dtypes[key], _value)

            example = tf.train.Example(features=tf.train.Features(feature=example))

        except Exception as error:

            if self.failure_ignore is True:

                if self.cache_warnings is True:

                    self.failure_examples[loc] = repr(error)

                else:

                    warnings.warn(f'Warning: {repr(error)}')

                return None

            else:

                raise ValueError(f'An error has occurred, Failed writing an example:'
                                 f'\n, {error}, to ignore failure state:\n'
                                 f'\tset TfRecordWriter(...., failure_ignore=True)')
        else:

            return example.SerializeToString()

    def image_from_dir(self, path):

        try:

            img = imread(path, cvt=False, grayscale=self.grayscale, size=self.shape)

        except Exception as error:

            raise ValueError('Failed to read an image\n, image_path :' + path + '\n' + str(error))

        else:

            return img

    def image_to_bytes(self, img):

        encoded_img = cv2.imencode(self.image_ext, img, (cv2.IMWRITE_JPEG_QUALITY, self.quality))[1].tostring()

        return encoded_img

    def bytes_from_dir(self, path):

        img = self.image_from_dir(path)
        encoded_img = self.image_to_bytes(img)

        return encoded_img

    def custom_type_to_bytes(self, value):

        raise NotImplementedError('You should Implement your custom type bytes conversion\n'
                                  f'{"=" * 75}\n'
                                  'Example\n'
                                  '---------\n'
                                  'class Writer(TfRecordWriter):\n\n'
                                  '\tdef custom_type_to_bytes(self, value: numpy.ndarray):\n\n'
                                  '\t\treturn value.tobytes(order=None)')

    def from_dataframe(self, dataframe: pd.DataFrame, dtypes: Union[list, dict],
                       image_key: str = 'image', pref_fname: str = 'train',
                       from_dir: str = None, to_dir='./', has_ext: bool = False,
                       func: Dict[str, Callable] = None, inplace: bool = False):

        """
        Parameters
        ----------

        dataframe: pd.DataFrame

        dtypes: Union[list, dict]
            Encoding type for each dataframe column,
            utils primitive types, [bytes, int, float, str], image column must be equal (bytes),
            dtypes could be ordered list or dictionary of column name as a key and type as a value,
            type could be as string or type instance object, ex. {'image': 'bytes'} or {'image': bytes}

        image_key: str
            dataframe images directory column name

        pref_fname: str
            TfRecord file name prefix, pref_fname='train' - (ex. 'train_*.tfrec')

        from_dir: str
            images directory, default = None

        to_dir: str
            writing directory

        has_ext: bool
            should set, has_ext = true, if dataframe images column (path info) not include the image extension

        func: Dict[str, Callable]
            Preprocessing function/s to be applied to a specific feature,
            the function should return a modified value, if the key is an image_key, then
            expect a function which get pixel values (np.ndarray), and
            returns a modified value, default = None

        inplace: bool
            In case of func parameter is not None, If False,
            apply func[column_name] using a copy of the dataframe, to a column
            if it's not an image_key column, otherwise, apply func inplace.

            default = False

        Returns
        -------
        """

        _dtypes = {}
        _dataframe = None

        size = len(dataframe)
        keys = list(dataframe.keys())

        self.failure_examples = {}

        if len(dtypes) != len(keys):

            raise ValueError('Failed match, No. of dataframe columns (keys) with dtypes : len(dtypes) ! =  len(keys)')

        if isinstance(dtypes, list):

            for i in range(len(keys)):

                _dtypes[keys[i]] = dtypes[i]

        elif isinstance(dtypes, dict):

            _dtypes = dtypes

        else:

            ValueError('dtypes, must be type of : list or dict not ' + f'<{str(dtypes)[8:-2]}>')

        if from_dir is not None and from_dir[-1] != '/':

            from_dir += '/'

        if to_dir[-1] == '/':

            to_dir = to_dir[:-1]

        self._check_ext()

        if func is not None and inplace is False:

            _dataframe = dataframe.copy()

        else:

            _dataframe = dataframe

        if func is not None:

            for func_key in func.keys():

                if func_key != image_key:

                    if func_key in _dataframe.keys():

                        _dataframe[func_key] = _dataframe[func_key].apply(func[func_key])

                    else:

                        raise ValueError(f'{func_key} column does not exist in dataframe, func[{func_key}]')

        for i, (start, end) in enumerate(split(size, self.n_records)):

            _path = f'{to_dir}/{pref_fname}_{i}.tfrec'

            with tf.io.TFRecordWriter(_path) as writer:

                progress_bar = tqdm(range(start, end))

                for j in progress_bar:

                    example = self._serialize_example(j, _dataframe.iloc[j].to_dict(),
                                                      _dtypes, image_key, from_dir, has_ext, func)

                    if example is not None:

                        writer.write(example)

                    if self.cache_warnings is True:

                        progress_bar.set_description('Failure Count = ' + str(len(self.failure_examples)))

    def from_directory(self, from_dir: str, query: str = None, image_key: str = 'image'):

        raise NotImplementedError('...')


class TfRecordReader:

    def __init__(self, features_dtype: dict, image_key: str,
                 tfrecord_shape: Union[tuple, list], shape: Union[tuple, list] = None,
                 channels: int = 3, func: Dict[str, Callable] = None):
        """
        features_dtype: dict
            TfRecord features types, features_dtype is a dictionary of column name as a key and type as a value,
            type as string, could be,  - ['int8', 'int16', ..., 'float16', 'float32', ..., 'str', 'bytes']

        image_key: str

        tfrecord_shape: Union[tuple, list]
            TfRecord file default image shape, (height, width, channels)

        shape: Union[tuple, list]
            Encoding shape, (height, width), default = None

        channels: int
            default = 3

        func: Dict[str, Callable]
            Preprocessing function/s to be applied to a specific feature,
            the function should return a modified value, default = None
        """

        self.tfrecord_shape = tfrecord_shape
        self.shape = shape
        self.channels = channels
        self.features_dtype = features_dtype
        self.image_key = image_key
        self.func = func

    def _decode(self, encoded_image):

        image = tf.io.decode_jpeg(encoded_image, channels=self.channels)
        image = tf.reshape(image, self.tfrecord_shape)

        if self.shape is not None:

            image = tf.image.resize(image, self.shape)

        else:

            image = tf.cast(image, dtype=tf.float32)

        return image

    def decode_custom_type(self, value):

        raise NotImplementedError('You should Implement your custom type bytes decoding\n'
                                  f'{"=" * 75}\n'
                                  'Example\n'
                                  '---------\n'
                                  'class Reader(TfRecordReader):\n\n'
                                  '\tdef decode_custom_type(self, bytes_str: string):\n\n'
                                  '\t\tvalue = tf.io.decode_raw(bytes_str, out_type="int32")\n'
                                  '\t\tvalue = tf.cast(tvalue, dtype="float32")\n\n'
                                  '\t\treturn value')

    @staticmethod
    def _get_feature_type(f_dtype):

        if isinstance(f_dtype, str):

            if f_dtype == 'int8':

                return tf.io.FixedLenFeature([], tf.int8)

            elif f_dtype == 'int16':

                return tf.io.FixedLenFeature([], tf.int16)

            elif f_dtype == 'int32':

                return tf.io.FixedLenFeature([], tf.int32)

            elif f_dtype == 'int64':
                return tf.io.FixedLenFeature([], tf.int64)

            elif f_dtype == 'float16':

                return tf.io.FixedLenFeature([], tf.float16)

            elif f_dtype == 'float32':

                return tf.io.FixedLenFeature([], tf.float32)

            elif f_dtype == 'float64':

                return tf.io.FixedLenFeature([], tf.float64)

            elif f_dtype == 'bytes' or f_dtype == 'str' or 'custom_' in f_dtype:

                return tf.io.FixedLenFeature([], tf.string)

            else:

                raise ValueError('Value of Type : ' + f_dtype + ', is not supported')
        else:

            raise ValueError('Value of Type : ' + str(f_dtype)[8:-2] + ', is not supported')

    def read_tfrecord(self, example):

        """Parse features a given `example`.

        Returns
        -------
        tuple: of features value
        """

        features = {}

        for key in self.features_dtype:

            features[key] = TfRecordReader._get_feature_type(self.features_dtype[key])

        # parser
        example = tf.io.parse_single_example(example, features)

        keys = list(features.keys())
        values = [None] * len(features)

        for i in range(len(keys)):

            values[i] = example[keys[i]]

            if self.image_key == keys[i]:

                values[i] = self._decode(values[i])

            elif 'custom_' in self.features_dtype[keys[i]]:

                values[i] = self.decode_custom_type(values[i])

            if self.func is not None:

                if keys[i] in self.func:

                    values[i] = self.func[keys[i]](values[i])

        return tuple(values)
