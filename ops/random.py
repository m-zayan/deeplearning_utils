import numpy as np

__all__ = ['random_indices', 'gaussian_noise']


def random_indices(n: int, start: int, end: int, step=1, replace=False, random_state=None):

    random = np.random.RandomState(random_state)

    if (end - start) / step < n:

        raise ValueError('invalid range, range must satisfies: (end - start) / step >= n')

    indices = np.arange(start, end, step)
    indices = random.choice(indices, size=n, replace=replace)

    return indices


def gaussian_noise(x, mean=0.0, sigma=1.0, random_state=None):

    random = np.random.RandomState(random_state)

    scale = (x.max() - x.min())

    mean = mean * 0.9 + 0.1 * scale
    sigma = sigma * 0.8 + 0.2 * scale

    noise = mean + sigma * random.randn(*x.shape).astype('float32')

    noisy_x = x + noise

    return noisy_x
