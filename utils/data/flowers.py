from typing import Tuple, Dict

from copy import deepcopy

import glob

import numpy as np

import cv2

import scipy.io

from tqdm import tqdm

from ..external.common import OS, Logger

from ..tf.ops.io import download

from ..ops.io import imread, save_as_npz

from ..ops.reshape_utils import batch, aligned_with

from ..ops.random import aligned_shuffle

from . import load

__all__ = ['get', 'get_image_labels', 'get_data_splits']


class Info:

    images = {'name': 'flowers', 'fname': '102flowers.tgz',
              'dname': 'jpg', 'size': '328 MB', 'type': 'image'}

    segments = {'name': 'flowers_segments', 'fname': '102segmentations.tgz',
                'dname': 'segmim', 'size': '194 MB', 'type': 'image'}

    labels = {'name': 'image_labels', 'fname': 'imagelabels.mat',
              'size': '512 B', 'type': '.mat'}

    data_splits = {'name': 'ids', 'fname': 'setid.mat',
                   'size': '14 KB', 'type': '.mat'}

    @staticmethod
    def get_alignment_meta(src: str, dname: str):

        alignment_meta = load.meta(src, dname)['alignment_meta']

        return alignment_meta

    @staticmethod
    def parse_id(path: str) -> int:

        return int(OS.filename(path).split('_')[1])


def download_request(info, key='dname') -> str:

    fname = info['fname']

    url = f'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/{fname}'

    # ---------------------------------------------------------------------

    ddir = download(fname, url)

    # ---------------------------------------------------------------------

    ddir, _ = OS.split(ddir)
    ddir = OS.realpath(ddir)
    ddir = OS.join(ddir, info[key])

    # ---------------------------------------------------------------------

    return ddir


def get_image_labels() -> np.ndarray:

    ddir = download_request(Info.labels, 'fname')

    return scipy.io.loadmat(ddir)['labels']


def get_data_splits() -> Dict[str, np.ndarray]:

    ddir = download_request(Info.data_splits, 'fname')

    meta = scipy.io.loadmat(ddir)['labels']

    splits = {'train_ids': meta['trnid'],
              'test_ids': meta['tstid'],
              'val_ids': meta['valid']}

    return splits


def get(info: dict, dest: str, shape: Tuple[int, int] = (224, 224), batch_size: int = 64,
        dname: str = 'flowers', prefix: str = 'data', shuffle: bool = False,
        random_state: int = None, **kwargs) -> str:

    # ---------------------------------------------------------------------

    # https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

    interpolation = kwargs.get('interpolation', cv2.INTER_AREA)

    grayscale = kwargs.get('grayscale', False)

    meta_fname = kwargs.get('meta_fname', 'meta')

    # .load_alignment_meta(...)
    alignment_meta = kwargs.get('alignment_meta', [])

    # ---------------------------------------------------------------------

    meta = {'alignment_meta': None}

    # ---------------------------------------------------------------------

    ddir = download_request(info)

    # ---------------------------------------------------------------------

    image_path = glob.glob(ddir + '/*')

    if shuffle:

        aligned_shuffle([image_path], random_state=random_state)

    # ---------------------------------------------------------------------

    ids = []

    for path in image_path:

        # subclassing Info, must be sufficient, to change the alignment criteria
        ids.append(Info.parse_id(path))

    # ---------------------------------------------------------------------

    meta['alignment_meta'] = deepcopy(ids)

    # ---------------------------------------------------------------------

    if len(alignment_meta) == len(ids):

        image_path = aligned_with(alignment_meta, ids, image_path)
        ids = aligned_with(alignment_meta, ids, ids)

    elif len(alignment_meta) > 0:

        Logger.set_line(length=60)
        Logger.fail('Aligning aborted!')
        Logger.set_line(length=60)

        ok = Logger.wait_key('Would you like to continue without considering the alignment?',
                             'y',  ['n', 'y'])

        if not ok:

            return ddir

    # ---------------------------------------------------------------------

    size = len(image_path)

    Logger.set_line(length=60)
    Logger.info({'directory': ddir,
                 'size': size})
    Logger.set_line(length=60)

    # ---------------------------------------------------------------------

    def flowers_load():

        for start, end in batch(size, batch_size):

            _x = []

            for _i in tqdm(range(start, end)):

                img = imread(image_path[_i], cvt=True, grayscale=grayscale,
                             size=shape, interpolation=interpolation)

                _x.append(img)

            _data = {'x': np.array(_x),
                     'id': np.array(ids[start:end])}

            yield _data

    # ---------------------------------------------------------------------

    dest = OS.realpath(dest)

    newdir = OS.join(dest, dname)

    if not OS.dir_exists(newdir):

        OS.make_dir(newdir)

    # ---------------------------------------------------------------------

    path = OS.join(newdir, meta_fname)

    save_as_npz(path, **meta)

    # ---------------------------------------------------------------------

    flowers_data = flowers_load()

    for i, data in enumerate(flowers_data):

        fname = f'{prefix}_{i}'

        path = OS.join(newdir, fname)

        save_as_npz(path, **data)

    # ---------------------------------------------------------------------

    return newdir
