from typing import Tuple

import glob

import numpy as np

import cv2

from tqdm import tqdm

from ..external.common import OS, Logger

from ..tf.ops.io import download

from ..ops.io import imread, save_as_npz

from ..ops.reshape_utils import batch

from ..ops.random import aligned_shuffle

__all__ = ['get']


def get(dest: str, shape: Tuple[int, int] = (224, 224), batch_size: int = 64,
        dname: str = 'lfw', prefix: str = 'data', shuffle: bool = False,
        random_state: int = None, **kwargs) -> str:

    # ---------------------------------------------------------------------

    # http://vis-www.cs.umass.edu/lfw

    interpolation = kwargs.get('interpolation', cv2.INTER_AREA)

    # ---------------------------------------------------------------------

    lfw_url = 'http://vis-www.cs.umass.edu/lfw/lfw.tgz'

    ddir = download('lfw.zip', lfw_url)

    ddir, _ = OS.split(ddir)
    ddir = OS.realpath(ddir)
    ddir = OS.join(ddir, 'lfw')

    # ---------------------------------------------------------------------

    image_path = glob.glob(ddir + '/*/*')

    if shuffle:

        aligned_shuffle([image_path], random_state=random_state)

    # ---------------------------------------------------------------------

    size = len(image_path)

    Logger.set_line(length=60)
    Logger.info({'directory': ddir,
                 'size': size})
    Logger.set_line(length=60)

    # ---------------------------------------------------------------------

    def lfw_load():

        for start, end in batch(size, batch_size):

            _x = []
            _y = []

            for _i in tqdm(range(start, end)):

                img = imread(image_path[_i], cvt=True, grayscale=False,
                             size=shape, interpolation=interpolation)

                class_name = OS.splitdir(image_path[_i], -2)

                _x.append(img)
                _y.append(class_name)

            _data = {'x': np.array(_x),
                     'y': np.array(_y)}

            yield _data

    # ---------------------------------------------------------------------

    dest = OS.realpath(dest)
    
    newdir = OS.join(dest, dname)

    if not OS.dir_exists(newdir):

        OS.make_dir(newdir)

    # ---------------------------------------------------------------------

    lfw_data = lfw_load()

    for i, data in enumerate(lfw_data):

        fname = f'{prefix}_{i}'

        path = OS.join(newdir, fname)

        save_as_npz(path, **data)

    # ---------------------------------------------------------------------

    return newdir
