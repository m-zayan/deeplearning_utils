from typing import Union, Tuple, List, Any

from io import BytesIO

from PIL import Image

import numpy as np

import cv2

__all__ = ['write_image', 'imread', 'cv_decode_image', 'pil_decode_image',
           'save_as_npy', 'load_npy', 'save_as_npz', 'load_npz', 'rle_encode', 'rle_decode']


def write_image(path, image, *args, **kwargs):

    cv2.imwrite(path, image, *args, **kwargs)


def imread(path: str, cvt: bool = False, grayscale: bool = False,
           size: Union[Tuple, List] = None, interpolation: Any = cv2.INTER_AREA) -> np.ndarray:

    flag = cv2.IMREAD_UNCHANGED

    if grayscale:

        flag = cv2.IMREAD_GRAYSCALE

    img = cv2.imread(path, flag)

    if cvt:

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if size:

        img = cv2.resize(img, dsize=size, interpolation=interpolation)

    return img


def cv_decode_image(dbytes: bytes, **kwargs) -> np.ndarray:

    dtype = kwargs.get('dtype', np.uint8)

    image = np.frombuffer(dbytes, dtype)
    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)

    return image


def pil_decode_image(dbytes: bytes) -> np.ndarray:

    image = Image.open(BytesIO(dbytes))

    return np.array(image)


# noinspection PyTypeChecker
def save_as_npy(filename, *ndarr, **kwargs):

    info = {'narr': len(ndarr), }

    with open(f'{filename}.npy', 'wb') as buffer:

        # info
        np.save(buffer, np.array([info]))

        for i in range(len(ndarr)):

            np.save(buffer, ndarr[i], **kwargs)


# noinspection PyTypeChecker
def load_npy(filename, **kwargs):

    data = []

    with open(f'{filename}.npy', 'rb') as buffer:

        info = np.load(buffer, **kwargs)[0]

        for i in range(info['narr']):

            data.append(np.load(buffer, **kwargs))

    return data


def save_as_npz(filename, **kwargs):

    np.savez(f'{filename}.npz', **kwargs)


def load_npz(filename, as_dict=False, **kwargs):

    if as_dict:

        return dict(np.load(f'{filename}.npz', **kwargs))

    return np.load(f'{filename}.npz', **kwargs)


def rle_encode(mask: np.ndarray, mask_size: int) -> str:

    """ run-length-encoding (RLE)"""

    if mask.size != mask_size:

        raise ValueError(f'Invalid mask, mask_size={mask.size}')

    values = mask.ravel().astype('uint8')
    values = np.concatenate([[0], values, [0]])

    indices = np.where(values[1:] != values[:-1])[0] + 1
    indices[1::2] -= indices[::2]

    encoded = ' '.join(indices.astype('str'))

    return encoded


def rle_decode(rle_code: str, size: tuple):

    """ run-length-decoding (RLE)"""

    sequence = rle_code.split()

    starts, lengths = [np.asarray(x, dtype=np.int32) for x in (sequence[0:][::2], sequence[1:][::2])]

    starts -= 1

    ends = starts + lengths

    mask = np.zeros(size[0] * size[1], dtype=np.uint8)

    for l, r in zip(starts, ends):

        mask[l:r] = 1

    return mask.reshape(size)
