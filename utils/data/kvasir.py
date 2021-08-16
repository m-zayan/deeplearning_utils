import glob
import os

import numpy as np
import pandas as pd

from tqdm import tqdm

from ..external.common import OS, Reader, Logger

from ..tf.ops.io import download

from ..ops.io import imread, save_as_npz

from ..ops.reshape_utils import batch

from ..ops.random import aligned_shuffle

__all__ = ['get']


def get(dest, shape=(224, 224), batch_size=64, dname='kvasir', prefix='data', shuffle=False, random_state=None):

    # https://datasets.simula.no/kvasir-seg/

    kvasir_url = 'https://datasets.simula.no/kvasir-seg/Kvasir-SEG.zip'

    ddir = download('kvasir', kvasir_url)
    
    ddir, _ = OS.split(ddir)
    ddir = OS.realpath(ddir)
    ddir = OS.join(ddir, 'Kvasir-SEG')

    image_path = glob.glob(ddir + '/images/*')
    mask_path = glob.glob(ddir + '/masks/*')

    if shuffle:

        aligned_shuffle([image_path, mask_path], random_state=random_state)

    meta_path = OS.join(ddir, 'kavsir_bboxes.json')

    size = len(image_path)

    Logger.info({'directory': ddir, 
                 'size': size})

    def kvasir_validate():

        if size != len(mask_path):

            raise ValueError('...')

        for _i in range(size):

            f0 = image_path[_i].split('/')[-1]
            f1 = mask_path[_i].split('/')[-1]

            if f0 != f1:

                raise ValueError('...')

    def kvasir_load():

        metadata = Reader.json_to_dict(meta_path)
        metadata = pd.DataFrame(metadata)

        for start, end in batch(size, batch_size):

            _x = []
            _y = []

            meta = []

            for _i in tqdm(range(start, end)):

                img = imread(image_path[_i], cvt=True, grayscale=False, size=shape)
                msk = imread(mask_path[_i], cvt=False, grayscale=True, size=shape)

                info = metadata[OS.filename(image_path[_i])].values

                _x.append(img)
                _y.append(msk)

                meta.append(info)

            _data = {'x': np.array(_x),
                     'y': np.array(_y),
                     'meta': np.array(meta)}

            yield _data

    kvasir_validate()

    dest = OS.realpath(dest)
    
    newdir = OS.join(dest, dname)

    if not OS.dir_exists(newdir):

        os.mkdir(newdir)

    kvasir_data = kvasir_load()

    for i, data in enumerate(kvasir_data):

        fname = f'{prefix}_{i}'

        path = os.path.join(newdir, fname)

        save_as_npz(path, **data)

    return newdir
