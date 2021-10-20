from typing import Callable

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from .random import random_indices
from .reshape import images_to_grid, grid_ground_truth

from skimage import color

__all__ = ['plot_random', 'plot', 'plot_latent',
           'grid_plot', 'label_to_rgb', 'nbins_cmap']


def plot_random(x: np.ndarray, y: np.ndarray = None, nrows: int = 6, ncols: int = 18, figsize: tuple = None,
                random_state: int = None):

    if y is not None and len(x) != len(y):

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
    labels = None

    if y is not None:

        labels = y[indices]

    for i in range(len(axs)):

        axs[i].imshow(images[i])

        if y is not None:

            axs[i].set_title(f'{labels[i]}')

        axs[i].grid(None)
        axs[i].axis('off')

    return fig, axs


def plot(x: np.ndarray, y: np.ndarray = None, nrows: int = 6, ncols: int = 18, figsize: tuple = None):

    if figsize is None:

        figsize = (ncols * x.shape[2] * 0.1, nrows * x.shape[1] * 0.1)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axs = axs.flatten()

    for i in range(len(axs)):

        axs[i].imshow(x[i].squeeze())

        if y is not None:

            axs[i].set_title(f'{y[i]}')

        axs[i].grid(None)
        axs[i].axis('off')

    return fig, axs


def nbins_cmap(n, name='tab10'):

    cmap = plt.cm.get_cmap(name)

    colors = np.linspace(0, 1, n)

    colors = cmap(colors)

    return colors


def plot_latent(latent2d: np.ndarray, labels: np.ndarray = None,
                figsize: tuple = (10, 8), marker='o', marker_size=20.0, **kwargs):

    # -----------------------------------------------------------------------------------

    n_classes = 1

    if labels is not None:

        classes = np.unique(labels)

        n_classes = len(classes)

    fig, ax = plt.subplots(figsize=figsize)

    palette = sns.color_palette('tab10', n_colors=n_classes)

    ax = sns.scatterplot(x=latent2d[:, 0], y=latent2d[:, 1], hue=labels, palette=palette,
                         ax=ax, s=marker_size, marker=marker)

    # -----------------------------------------------------------------------------------

    grid = kwargs.pop('grid', None)

    anchor_legend = kwargs.pop('anchor_legend', (0.0, -0.05))

    loc_legend = kwargs.pop('loc_legend', 'lower left')
    ncols_legend = kwargs.pop('ncols_legend', n_classes)

    axis_off = kwargs.pop('axis_off', True)

    ax.legend(loc=loc_legend, ncol=ncols_legend, bbox_to_anchor=anchor_legend)

    ax.grid(grid)

    if axis_off:

        fig.gca().set_axis_off()

    return fig, ax


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


# =====================================================================================================================

label_to_rgb: Callable = color.label2rgb
