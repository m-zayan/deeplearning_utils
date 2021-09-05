from typing import Dict, Any

from tqdm import tqdm

import numpy as np

from tensorflow.keras import datasets

from ..external.common import OS

from ..ops.io import load_npz

__all__ = ['num_of_batches', 'npz_batch', 'npz_nbatch']


def num_of_batches(src: str, dname: str) -> int:

    src = OS.realpath(src)

    path = OS.join(src, dname)

    return len(OS.listdir(path))


def npz_batch(src: str, batch_num: int, dname: str, prefix: str) -> Dict[str, np.ndarray]:

    src = OS.realpath(src)

    fname = f'{prefix}_{batch_num}'

    path = OS.join(src, dname) + f'/{fname}'

    data = load_npz(path, as_dict=True, allow_pickle=True)

    return data


def npz_nbatch(src: str, start: int, end: int, dname: str, prefix: str) -> Dict[str, np.ndarray]:

    src = OS.realpath(src)

    data = npz_batch(src, start, dname, prefix)

    for i in tqdm(range(start + 1, end)):

        datai = npz_batch(src, i, dname, prefix)

        for key in datai:

            data[key] = np.concatenate([data[key], datai[key]], axis=0)

    return data


def meta(src: str, dname: str, meta_fname='meta') -> Dict[str, Any]:

    src = OS.realpath(src)

    path = OS.join(src, dname, meta_fname)

    _meta = load_npz(path, as_dict=True, allow_pickle=True)

    return _meta


def mnist(as_1d=True, rescale=255.0):

    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= rescale
    x_test /= rescale

    if as_1d:

        x_train = x_train.reshape(-1, 784)
        x_test = x_test.reshape(-1, 784)

    return (x_train, y_train), (x_test, y_test)
