from typing import Tuple, Dict, Union, Any

from copy import deepcopy

import numpy as np

import cv2

from tqdm import tqdm

from ..external import handlers

if not handlers.__selenium_installed__():

    raise EnvironmentError('Dependencies are not installed, consider using:\n\t\t'
                           'handlers.install_dependencies(...)')


from ..external.common import OS, IO, Logger

from ..external.handlers.requests import Chrome

from ..ops.io import pil_decode_image, cv_decode_image, save_as_npz

from ..ops.reshape import batch, aligned_with

from ..ops.random import aligned_shuffle

from . import load

__all__ = ['Info', 'get', 'cityscapes_download']


class Info:

    panoptic_parts = {'packid': 35, 'name': 'panoptic_parts', 'size': '465 MB', 'type': 'images',
                      'split': ['train', 'val'], 'fname': 'gtFinePanopticParts_trainval.zip'}

    left_view = {'packid': 3, 'name': 'left_8_bit', 'size': '11 GB', 'type': 'images',
                 'split': ['train', 'test', 'val'], 'fname': 'leftImg8bit_trainvaltest.zip'}

    right_view = {'packid': 5, 'name': 'right_8_bit', 'size': '11 GB', 'type': 'images',
                  'split': ['train', 'test', 'val'], 'fname': 'rightImg8bit_trainvaltest.zip'}

    valid_ext = ['.png', '.jpg', '.jpeg', '.tif']

    cache_dir = './cache/cityscapes'

    @staticmethod
    def get_alignment_meta(src: str, dname: str):

        alignment_meta = load.meta(src, dname)['alignment_meta'].item()

        for key in alignment_meta:

            alignment_meta[key] = np.array(alignment_meta[key])

        return alignment_meta

    @staticmethod
    def parse_id(_id):

        return '_'.join(_id.split('_')[:-1])

    @staticmethod
    def parse_ids(id_list):

        return np.array(list(map(Info.parse_id, id_list)))

    @staticmethod
    def parse_split(path, **kwargs):

        index = kwargs.get('index', 1)

        split_type = OS.splitdir(path, index)

        return split_type


def cityscapes_download(username, password, packid, **kwargs) -> str:

    # ----------------------------------------------------

    load_timeout = kwargs.get('load_timeout', 60)
    logs_delay = kwargs.get('logs_delay', 5)

    # -----------------------------------------------------

    # https://www.cityscapes-dataset.com/downloads/

    # ---------------------------------------------------------

    cache_dir = OS.realpath(Info.cache_dir)

    if not OS.dir_exists(cache_dir):

        OS.make_dirs(cache_dir)

    # ---------------------------------------------------------

    chrome = Chrome()
    chrome.set_download_directory(cache_dir, extend_session=True)

    # ---------------------------------------------------------

    url = 'https://www.cityscapes-dataset.com/login/'

    chrome.login(url, username, password, load_timeout=load_timeout)

    # ---------------------------------------------------------

    durl = f'https://www.cityscapes-dataset.com/file-handling/?packageID={packid}'

    # ---------------------------------------------------------

    cache_dir = chrome.download(url=durl, cache_dir=cache_dir, load_timeout=load_timeout, logs_delay=logs_delay)

    # ---------------------------------------------------------

    return cache_dir


def get(username: str, password: str, info: Dict[Any, Any], dest: str, shape: Union[None, Tuple[int, int]] = None,
        batch_size: Dict[str, int] = None, dname: str = 'cityscapes', prefix: str = 'data',
        shuffle: bool = False, random_state: int = None, **kwargs) -> str:

    # ---------------------------------------------------------------------

    # https://www.cityscapes-dataset.com

    default_size = {'train': 64, 'test': 64, 'val': 64}

    # e.g. Info.panoptic_parts <--> a reasonable choice might be - cv2.INTER_NEAREST
    interpolation = kwargs.get('interpolation', cv2.INTER_AREA)

    cv_decode = kwargs.get('cv_decode', False)

    meta_fname = kwargs.get('meta_fname', 'meta')

    # load.alignment_meta(...)
    alignment_meta = kwargs.get('alignment_meta', {})

    # -------------------------------------,--------------------------------

    meta = {'alignment_meta': None}

    # -------------------------------------,--------------------------------

    if batch_size is None:

        batch_size = default_size

    else:

        default_size.update(batch_size)

        batch_size = default_size

    # ---------------------------------------------------------------------

    cache_dir = cityscapes_download(username=username, password=password, packid=info['packid'], **kwargs)

    # ---------------------------------------------------------------------

    zip_path = OS.join(cache_dir, info['fname'])

    if not OS.file_exists(zip_path):

        raise ValueError(f'data-file: {zip_path},\n\t does not exist, '
                         f'consider using a unique cache directory')

    # ---------------------------------------------------------------------

    data_file, finfo_list = IO.zipfile_read(zipfile_path=zip_path, include_dirs=False, ext_list=Info.valid_ext)

    # ---------------------------------------------------------------------

    if shuffle:

        aligned_shuffle([finfo_list], random_state=random_state)

    # ---------------------------------------------------------------------

    size = len(finfo_list)

    path_mp = {}
    ids_mp = {}

    for i in range(size):

        fname = finfo_list[i].filename

        fst = Info.parse_split(fname)

        if fst not in path_mp:

            path_mp[fst] = []
            ids_mp[fst] = []

        path_mp[fst].append(fname)

        # subclassing Info, must be sufficient to change the alignment criteria
        ids_mp[fst].append(Info.parse_id(OS.filename(fname)))

    # ---------------------------------------------------------------------

    # new --> alignment_meta
    meta['alignment_meta'] = deepcopy(ids_mp)

    # ---------------------------------------------------------------------

    # old --> alignment_meta
    for key in alignment_meta:

        if key in ids_mp:

            if len(ids_mp[key]) == len(alignment_meta[key]):

                path_mp[key] = aligned_with(alignment_meta[key], ids_mp[key], path_mp[key])
                ids_mp[key] = aligned_with(alignment_meta[key], ids_mp[key], ids_mp[key])

            elif len(alignment_meta) > 0:

                Logger.set_line(length=60)
                Logger.fail('Aligning aborted!')
                Logger.set_line(length=60)

                ok = Logger.wait_key('Would you like to continue without considering the alignment?',
                                     'y', ['n', 'y'])

                if not ok:

                    return cache_dir

                else:

                    # joke! <--> should be erased later
                    Logger.warning('Ah, thanks I was not sure if it\'s gonna works properly :)')

    # ---------------------------------------------------------------------

    Logger.set_line(length=60)
    Logger.info({'directory': cache_dir,
                 'size': size})
    Logger.set_line(length=60)

    # ---------------------------------------------------------------------

    def cityscapes_load(_split_type):

        n_examples = len(path_mp[_split_type])
        bsize = batch_size[_split_type]

        for start, end in batch(n_examples, bsize):

            _data: Dict[str, Union[list, np.ndarray]] = {'x': [],
                                                         'id': ids_mp[_split_type][start:end],
                                                         'size': []}

            for _j in tqdm(range(start, end)):

                _path = path_mp[_split_type][_j]

                # -----------------------------

                _bytes = data_file.read(_path)

                _x = None

                if cv_decode:

                    _x = cv_decode_image(_bytes)

                else:

                    _x = pil_decode_image(_bytes)

                # -----------------------------

                _size = _x.shape[:2]

                if shape is not None:

                    _x = cv2.resize(_x, shape, interpolation=interpolation)

                _data['x'].append(_x)
                _data['size'].append(_size)

            for _key in _data:

                _data[_key] = np.array(_data[_key])

            yield _data

    # ---------------------------------------------------------------------

    dest = OS.realpath(dest)

    newdir = OS.join(dest, dname)

    # ---------------------------------------------------------------------

    if not OS.dir_exists(newdir):

        OS.make_dir(newdir)

    for split in path_mp:

        split_dir = OS.join(newdir, split)

        if (not OS.dir_exists(split_dir)) and (len(path_mp[split]) > 0):

            OS.make_dir(split_dir)

    # ---------------------------------------------------------------------

    path = OS.join(newdir, meta_fname)

    save_as_npz(path, **meta)

    # ---------------------------------------------------------------------

    for split in path_mp:

        cityscapes_data = cityscapes_load(split)

        split_dir = OS.join(newdir, split)

        for i, data in enumerate(cityscapes_data):

            fname = f'{prefix}_{i}'

            path = OS.join(split_dir, fname)

            save_as_npz(path, **data)

    # ---------------------------------------------------------------------

    return newdir
