import numpy as np

__all__ = ['random_indices', 'gaussian_noise']


def random_indices(n: int, start: int, end: int, step=1, replace=False, random_state=None):

    random = np.random.RandomState(random_state)

    if (end - start) / step < n:

        raise ValueError('invalid range, range must satisfies: (end - start) / step >= n')

    indices = np.arange(start, end, step)
    indices = random.choice(indices, size=n, replace=replace)

    return indices


def gaussian_noise(x, mu=0.0, sigma=1.0, mu_rate=0.9, sigma_rate=0.8, random_state=None):

    random = np.random.RandomState(random_state)

    scale = (x.max() - x.min())

    mean = mu * mu_rate + (1.0 - mu_rate) * scale
    sigma = sigma * sigma_rate + (1.0 - sigma_rate) * scale

    noise = mean + sigma * random.randn(*x.shape).astype('float32')

    noisy_x = x + noise

    return noisy_x
