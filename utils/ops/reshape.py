import numpy as np

__all__ = ['batch', 'split', 'nd_pad', 'concat_vertical',
           'aligned_with', 'images_to_grid', 'grid_ground_truth']


def batch(size: int, batch_size: int):

    n = size // batch_size

    if size % batch_size:

        n += 1

    for i in range(n):

        start = batch_size * i

        end = batch_size * (i + 1)
        end = min(end, size)

        yield start, end


def split(size: int, n: int):

    batch_size = size // n

    if size % n:

        batch_size += 1

    for i in range(n):
        
        start = batch_size * i

        end = batch_size * (i + 1)
        end = min(end, size)

        yield start, end


def nd_pad(img: np.ndarray, pad_width: int, pad_value: float):

    w, h, c = img.shape

    w += 2 * pad_width
    h += 2 * pad_width

    padded_img = np.zeros((w, h, c), dtype='float32')

    for i in range(c):

        padded_img[..., i] = np.pad(img[..., i], pad_width=pad_width,
                                    mode='constant', constant_values=pad_value)

    return padded_img


def images_to_grid(images: np.ndarray, nrows: int, ncols: int, pad_width: int, pad_value: float = 35.0):

    m, h, w, c = images.shape

    if nrows * ncols != m:
        raise ValueError(' nrows * ncols != No. images')

    h += 2 * pad_width
    w += 2 * pad_width

    shape = (nrows * h, ncols * w, c)
    grid_img = np.ones(shape, dtype='float32')

    for i in range(nrows):

        for j in range(ncols):

            idx = i * ncols + j

            s_h, e_h = i * h, (i + 1) * h
            s_w, e_w = j * w, (j + 1) * w

            pad_img = nd_pad(images[idx], pad_width, pad_value)

            grid_img[s_h:e_h, s_w:e_w, :] = pad_img

    return grid_img


def grid_ground_truth(grid_img: np.ndarray, images: np.ndarray, pad_width: int,
                      sep_width: int, pad_value_1: float = 35.0, pad_value_2: float = 1.0):

    h, w, c = grid_img.shape

    m, ih, iw, ic = images.shape

    iw += 2 * pad_width
    ih += 2 * pad_width

    if m * ih != h or w % iw != 0:

        raise ValueError('Inconsistent number of rows or cols, grid_img..., images=...')

    if c != ic:

        raise ValueError('number of channels must be the same,\
                      grid_img.shape[-1] != image.shape[-1]')

    w += 2 * sep_width

    shape = (h, w + iw, c)
    new_grid_img = np.ones(shape, dtype='float32') + pad_value_2

    gs_h, gs_w = 0, iw + (2 * sep_width)
    new_grid_img[gs_h:, gs_w:] = grid_img

    for i in range(m):

        s_h, e_h = i * ih, (i + 1) * ih

        pad_img = nd_pad(images[i], pad_width, pad_value_1)

        new_grid_img[s_h:e_h, :iw, :] = pad_img

    return new_grid_img


def aligned_with(fids, sids, select=None, dtype=np.uint32):

    size = len(sids)

    aligned_sid = np.zeros((size,), dtype=dtype)

    for i, sid in enumerate(sids):

        index = np.where(fids == sid)[0]

        if len(index) == 1:

            aligned_sid[index[0]] = i

    if select is not None:

        return np.array(select)[aligned_sid]

    return aligned_sid


def concat_vertical(*a) -> np.ndarray:

    output = np.stack(a, axis=0)
    output = np.transpose(output, axes=[1, 0, *range(2, output.ndim)])
    output = output.reshape(-1, *output.shape[2:])

    return output
