from typing import Tuple, Dict, Callable, Union, Any

from threading import Thread

import numpy as np

import cv2

from tqdm import tqdm

from ..external import handlers

if not handlers.__selenium_installed__():

    raise EnvironmentError('Dependencies are not installed, consider using:\n\t\t'
                           'handlers.install_dependencies(...)')


from ..external.common import OS, IO, Logger, Time, StatusWatch

from ..external.handlers.requests import Chrome

from ..ops.io import pil_decode_image, cv_decode_image, save_as_npz

from ..ops.reshape_utils import batch

from ..ops.random import aligned_shuffle


class Info:

    panoptic_parts = {'packid': 35, 'name': 'panoptic_parts', 'size': '465 MB', 'type': 'annotation',
                      'split': ['train', 'val'], 'fname': 'gtFinePanopticParts_trainval.zip'}

    left_view = {'packid': 3, 'name': 'left_8_bit', 'size': '11 GB', 'type': 'images',
                 'split': ['train', 'test', 'val'], 'fname': 'leftImg8bit_trainvaltest.zip'}

    right_view = {'packid': 5, 'name': 'right_8_bit', 'size': '11 GB', 'type': 'images',
                  'split': ['train', 'test', 'val'], 'fname': 'rightImg8bit_trainvaltest.zip'}

    valid_ext = ['.png', '.jpg', '.jpeg', '.tif']

    cache_dir = './cache/cityscapes'


def parse_split(path, **kwargs):

    index = kwargs.get('index', 1)

    split_type = OS.splitdir(path, index)

    return split_type


def download_request(username, password, packid, load_timeout) -> Tuple[str, Callable]:

    # ---------------------------------------------------------

    # https://www.cityscapes-dataset.com/downloads/

    # ---------------------------------------------------------

    cache_dir = OS.realpath(Info.cache_dir)

    if not OS.dir_exists(cache_dir):

        OS.make_dirs(cache_dir)

    # ---------------------------------------------------------

    chrome = Chrome()

    chrome.set_download_directory(cache_dir)

    chrome.build()

    # ---------------------------------------------------------

    url = 'https://www.cityscapes-dataset.com/login/'

    chrome.login(url, username, password, load_timeout=load_timeout)

    # ---------------------------------------------------------

    def get_file() -> None:

        durl = f'https://www.cityscapes-dataset.com/file-handling/?packageID={packid}'

        chrome.get(durl, load_timeout=load_timeout)

    # ---------------------------------------------------------

    return cache_dir, get_file


def watch_download(cache_dir) -> None:

    Logger.set_line(length=60)

    on_watch = StatusWatch.download(cache_dir)

    ongoing, current_size = True, 0.0

    while ongoing:

        ongoing, current_size = on_watch(int)

        if ongoing:

            Logger.info_r(current_size)

    Logger.info('\n', end='')
    Logger.set_line(length=60)


def cityscapes_download(username, password, packid, **kwargs) -> str:

    # ----------------------------------------------------

    load_timeout = kwargs.get('load_timeout', 60)
    logs_delay = kwargs.get('logs_delay', 5)

    # -----------------------------------------------------

    cache_dir, get_file = download_request(username, password, packid, load_timeout)

    if len(OS.listdir(cache_dir)):

        Logger.set_line(length=60)
        Logger.fail('A non-empty caching directory is not supported')
        Logger.set_line(length=60)
        Logger.warning('Download aborted!')
        Logger.set_line(length=60)

        return cache_dir

    # -----------------------------------------------------

    download_th = Thread(target=get_file, args=())
    watch_th = Thread(target=watch_download, args=(cache_dir,))

    # -----------------------------------------------------

    download_th.start()

    Time.sleep(max(5, logs_delay))

    watch_th.start()

    # -----------------------------------------------------

    download_th.join()
    watch_th.join()

    # -----------------------------------------------------

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

    # ---------------------------------------------------------------------

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

    for i in range(size):

        fname = finfo_list[i].filename

        fst = parse_split(fname)

        if fst not in path_mp:

            path_mp[fst] = []

        path_mp[fst].append(fname)

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

            _data: Dict[str, Union[list, np.ndarray]] = {'x': [], 'id': [], 'size': []}

            for _j in tqdm(range(start, end)):

                _path = path_mp[_split_type][_j]

                # -----------------------------

                _id = OS.filename(_path)

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
                _data['id'].append(_id)
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

    for split in path_mp:

        cityscapes_data = cityscapes_load(split)

        split_dir = OS.join(newdir, split)

        for i, data in enumerate(cityscapes_data):

            fname = f'{prefix}_{i}'

            path = OS.join(split_dir, fname)

            save_as_npz(path, **data)

    # ---------------------------------------------------------------------

    return newdir
