import numpy as np

from ..external.common import Sys

__all__ = ['random_indices', 'gaussian_noise', 'aligned_shuffle']


def random_indices(n: int, start: int, end: int, step=1, replace=False, random_state=None):

    random = np.random.RandomState(random_state)

    if (end - start) / step < n:

        raise ValueError('invalid range, range must satisfies: (end - start) / step >= n')

    indices = np.arange(start, end, step)
    indices = random.choice(indices, size=n, replace=replace)

    return indices


# noinspection PyArgumentList
def gaussian_noise(x: np.ndarray, mu: float = 0.0, sigma: float = 1.0, mu_rate: float = 0.9, sigma_rate: float = 0.8,
                   random_state: int = None) -> np.ndarray:

    random = np.random.RandomState(random_state)

    scale = (x.max(axis=-1, keepdims=True) - x.min(axis=-1, keepdims=True))

    mean = mu * mu_rate + (1.0 - mu_rate) * scale
    sigma = sigma * sigma_rate + (1.0 - sigma_rate) * scale

    noise = mean + sigma * random.randn(*x.shape).astype('float32')

    noisy_x = x + noise

    return noisy_x


def aligned_shuffle(args: list, random_state=None) -> None:

    if random_state is None:

        random_state = np.random.randint(0, Sys.max(32), (1, ))

    random = np.random.RandomState(random_state)

    for i in range(len(args)):

        random.shuffle(args[i])
        random.seed(random_state)
