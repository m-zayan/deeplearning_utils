from typing import Callable, Dict, Any

import numpy as np

from . import _abstract

from ...ops.reshape import aligned_with


def __np_as_type__(dtype):

    def to_nbits(func: Callable):

        def inner(*args, **kwargs) -> np.ndarray:

            return func(*args, **kwargs).astype(dtype)

        return inner

    return to_nbits


class Meta(_abstract.Meta):

    @staticmethod
    def __multi_level_map__(ref: dict, _id: str, nlevels: int, sep: str = '_',
                            _reversed=True, start=None, end=None, extra_value='') -> None:

        str_list = _id.split(sep)

        if _reversed:

            str_list = str_list[::-1]

        if start is None:

            start = 0

        if end is None:

            end = len(str_list)

        str_list = str_list[start:end]

        str_list.append(extra_value)

        n = min(len(str_list), nlevels)

        def new_key(_ref, key, value):

            if key not in _ref:

                _ref[key] = value

        def normalize_ref(_ref, i=1):

            if i >= n:

                return

            k1 = str_list[i - 1]
            k2 = str_list[i]

            if i == n - 1:

                new_key(_ref[k1], k2, [])

                return

            new_key(_ref[k1], k2, {})

            normalize_ref(_ref[k1], i + 1)

        def ref_value(_ref, i=0):

            if i >= n:

                return

            key = str_list[i]

            if i == n - 1:

                value = sep.join(str_list[i + 1:])

                _ref[key].append(value)

                return

            ref_value(_ref[key], i + 1)

        if n > 0:

            new_key(ref, str_list[0], {})

        normalize_ref(ref, 1)
        ref_value(ref, 0)

    @staticmethod
    def multi_level_id(id_list, nlevels) -> Dict[Any, Any]:

        ref = {}

        for i, iid in enumerate(id_list):

            Meta.__multi_level_map__(ref, iid, nlevels, start=1, end=None, extra_value=str(i))

        return ref

    @staticmethod
    def ids_xy_format(data_ids, ann_ids):

        return data_ids, aligned_with(data_ids, ann_ids, ann_ids)


class Annotation:

    @staticmethod
    def __digits_count__(a):

        if a == 0:

            return 1

        return int(np.log10(a) + 1)

    @staticmethod
    @__np_as_type__(dtype=np.int32)
    def cv_load_fix(image):

        min_value = image.min()

        if min_value == 0:

            min_value = 1e-45

        return image / min_value

    @staticmethod
    def segmentation_level(pixel_value):

        ndigits = Annotation.__digits_count__(pixel_value)

        if ndigits <= 2:

            return 0

        elif ndigits <= 5:

            return 1

        elif ndigits <= 7:

            return 2

        else:

            raise ValueError('Invalid Annotation, ndigits > 7')

    @staticmethod
    def mask_segmentation_level(image, dtype=np.int32):
        mask = np.zeros_like(image, dtype=dtype)

        mview1d = mask.ravel()
        iview1d = image.ravel()

        size = len(iview1d)

        for i in range(size):

            mview1d[i] = Annotation.segmentation_level(iview1d[i])

        return mask
