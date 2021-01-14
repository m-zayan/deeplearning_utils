import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE

from .random import random_indices
from .reshape_utils import images_to_grid, grid_ground_truth

__all__ = ['plot_random', 'plot', 'plot_latent', 'grid_plot']


def plot_random(x: np.ndarray, y: np.ndarray, nrows: int = 6, ncols: int = 18, figsize: tuple = None,
                random_state: int = None):

    if len(x) != len(y):

        raise ValueError('len(x) != len(y)')

    m = len(x)
    n = nrows * ncols

    if m < n:

        raise ValueError('len(x) != (nrows * ncols)')

    if figsize is None:

        figsize = (ncols * x.shape[2] * 0.1, nrows * x.shape[1] * 0.1)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axs = axs.flatten()

    indices = random_indices(n, start=0, end=m, step=1, replace=False, random_state=random_state)

    images = x[indices, ..., 0]
    labels = y[indices]

    for i in range(len(axs)):

        axs[i].imshow(images[i])
        axs[i].set_title(f'{labels[i]}')
        axs[i].grid('off')
        axs[i].axis('off')

    return fig


def plot(x: np.ndarray, y: np.ndarray, nrows: int = 6, ncols: int = 18, figsize: tuple = None):

    if figsize is None:

        figsize = (ncols * x.shape[2] * 0.1, nrows * x.shape[1] * 0.1)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axs = axs.flatten()

    for i in range(len(axs)):

        axs[i].imshow(x[i].squeeze())
        axs[i].set_title(f'{y[i]}')
        axs[i].grid('off')
        axs[i].axis('off')

    return fig


def plot_latent(latent: np.ndarray, labels: np.ndarray = None, n_iter: int = 1000,
                figsize: tuple = (15, 10), random_state: int = None):

    n_colors = 1

    if labels is not None:

        n_colors = len(np.unique(labels))

    embedded = TSNE(n_components=2, n_iter=n_iter, random_state=random_state).fit_transform(latent)

    fig, ax = plt.subplots(figsize=figsize)

    palette = sns.color_palette("tab10", n_colors=n_colors)
    sns.scatterplot(x=embedded[:, 0], y=embedded[:, 1], hue=labels, palette=palette, ax=ax)

    return fig


def grid_plot(images: np.ndarray, nrows: int, ncols: int, pad: int = 5,
              ground_truth: np.ndarray = None, sep_width: int = None, sep_value: float = 255.0,
              figsize: tuple = (25, 10)):

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    grid_img = images_to_grid(images, nrows, ncols, pad)

    if ground_truth is not None:

        if not sep_width:

            sep_width = pad

        grid_img = grid_ground_truth(grid_img, ground_truth, pad,
                                     sep_width, pad_value_2=sep_value)

    grid_img = grid_img.squeeze()

    ax.imshow(grid_img)
    ax.axis('off')
