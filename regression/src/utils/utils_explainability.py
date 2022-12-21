# utility functions for explainability

import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker


def normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def convert_to_viz_format(input):
    visualization_image = torch.squeeze(input).permute(1, 2, 0)
    return visualization_image.cpu().detach().numpy()


def plot_palette(pal, size=1):
    n = len(pal)
    f, ax = plt.subplots(1, 1, figsize=(n * size, size))
    ax.imshow(np.arange(n).reshape(1, n),
              cmap=mpl.colors.ListedColormap(list(pal)),
              interpolation="nearest", aspect="auto")
    ax.set_xticks(np.arange(n) - .5)
    ax.set_yticks([-.5, .5])
    # Ensure nice border between colors
    ax.set_xticklabels(["" for _ in range(n)])
    # The proper way to set no ticks
    ax.yaxis.set_major_locator(ticker.NullLocator())

    return f, ax