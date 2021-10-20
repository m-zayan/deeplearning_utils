from typing import Tuple

import glob

import numpy as np

import pandas as pd

import cv2

from tqdm import tqdm

from ..external.common import OS, Reader, Logger

from ..tf.ops.io import download

from ..ops.io import imread, save_as_npz

from ..ops.reshape import batch

from ..ops.random import aligned_shuffle

__all__ = ['get']


def get(dest: str, shape: Tuple[int, int] = (224, 224), batch_size: int = 64,
        dname: str = 'kvasir', prefix: str = 'data', shuffle: bool = False,
        random_state: int = None, **kwargs) -> str:

    # ---------------------------------------------------------------------

    # https://datasets.simula.no/kvasir-seg/

    image_interpolation = kwargs.get('image_interpolation', cv2.INTER_AREA)

    mask_interpolation = kwargs.get('mask_interpolation', cv2.INTER_NEAREST)

    grayscale_mask = kwargs.get('grayscale_mask', False)

    # ---------------------------------------------------------------------

    kvasir_url = 'https://datasets.simula.no/kvasir-seg/Kvasir-SEG.zip'

    ddir = download('kvasir', kvasir_url)

    # ---------------------------------------------------------------------

    ddir, _ = OS.split(ddir)
    ddir = OS.realpath(ddir)
    ddir = OS.join(ddir, 'Kvasir-SEG')

    # ---------------------------------------------------------------------

    image_path = glob.glob(ddir + '/images/*')
    mask_path = glob.glob(ddir + '/masks/*')

    if shuffle:

        aligned_shuffle([image_path, mask_path], random_state=random_state)

    # ---------------------------------------------------------------------

    meta_path = OS.join(ddir, 'kavsir_bboxes.json')

    # ---------------------------------------------------------------------

    size = len(image_path)

    Logger.set_line(length=60)
    Logger.info({'directory': ddir,
                 'size': size})
    Logger.set_line(length=60)

    # ---------------------------------------------------------------------

    def kvasir_validate():

        if size != len(mask_path):

            raise ValueError('...')

        for _i in range(size):

            f0 = image_path[_i].split('/')[-1]
            f1 = mask_path[_i].split('/')[-1]

            if f0 != f1:

                raise ValueError('...')

    # ---------------------------------------------------------------------

    def kvasir_load():

        metadata = Reader.json_to_dict(meta_path)
        metadata = pd.DataFrame(metadata)

        for start, end in batch(size, batch_size):

            _x = []
            _y = []

            meta = []

            for _i in tqdm(range(start, end)):

                img = imread(image_path[_i], cvt=True, grayscale=False,
                             size=shape, interpolation=image_interpolation)

                msk = imread(mask_path[_i], cvt=False, grayscale=grayscale_mask,
                             size=shape, interpolation=mask_interpolation)

                info = metadata[OS.filename(image_path[_i])].values

                _x.append(img)
                _y.append(msk)

                meta.append(info)

            _data = {'x': np.array(_x),
                     'y': np.array(_y),
                     'meta': np.array(meta)}

            yield _data

    # ---------------------------------------------------------------------

    kvasir_validate()

    # ---------------------------------------------------------------------

    dest = OS.realpath(dest)
    
    newdir = OS.join(dest, dname)

    # ---------------------------------------------------------------------

    if not OS.dir_exists(newdir):

        OS.make_dir(newdir)

    # ---------------------------------------------------------------------

    kvasir_data = kvasir_load()

    for i, data in enumerate(kvasir_data):

        fname = f'{prefix}_{i}'

        path = OS.join(newdir, fname)

        save_as_npz(path, **data)

    # ---------------------------------------------------------------------

    return newdir
