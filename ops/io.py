import numpy as np

import cv2


def imread(path, cvt=False, grayscale=False, size=None):

    flag = cv2.IMREAD_UNCHANGED

    if grayscale:

        flag = cv2.IMREAD_GRAYSCALE

    img = cv2.imread(path, flag)

    if cvt:

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if size:

        img = cv2.resize(img, dsize=size)

    return img


# noinspection PyTypeChecker
def save_as_npy(filename, *args):

    info = {'narr': len(args), }

    with open(f'{filename}.npy', 'wb') as buffer:

        # info
        np.save(buffer, np.array([info]))

        for i in range(len(args)):

            np.save(buffer, args[i])


# noinspection PyTypeChecker
def load_npy(filename, allow_pickle=True):

    data = []

    with open(f'{filename}.npy', 'rb') as buffer:

        info = np.load(buffer, allow_pickle=allow_pickle)[0]

        for i in range(info['narr']):

            data.append(np.load(buffer, allow_pickle=allow_pickle))

    return data


def save_as_npz(filename, **kwargs):

    np.savez(f'{filename}.npz', **kwargs)


def load_npz(filename, allow_pickle=True):

    return np.load(f'{filename}.npz', allow_pickle=allow_pickle)
