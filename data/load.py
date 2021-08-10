from tqdm import tqdm

import numpy as np

from ..external.common import OS

from ..ops.io import load_npz


def npz_batch(src, batch_num, dname, prefix):

    src = OS.realpath(src)

    fname = f'{prefix}_{batch_num}'

    path = OS.join(src, dname) + f'/{fname}'

    data = dict(load_npz(path, allow_pickle=True))

    return data


def npz_nbatch(src, start, end, dname, prefix):

    src = OS.realpath(src)

    data = npz_batch(src, start, dname, prefix)

    for i in tqdm(range(start + 1, end)):

        datai = npz_batch(src, i, dname, prefix)

        for key in datai:

            data[key] = np.concatenate([data[key], datai[key]], axis=0)

    return data
