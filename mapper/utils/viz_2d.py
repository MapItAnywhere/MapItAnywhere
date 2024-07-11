# Copyright (c) Meta Platforms, Inc. and affiliates.

# Adapted from Hierarchical-Localization, Paul-Edouard Sarlin, ETH Zurich
# https://github.com/cvg/Hierarchical-Localization/blob/master/hloc/utils/viz.py
# Released under the Apache License 2.0

import numpy as np
import torch 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def features_to_RGB(*Fs, masks=None, skip=1):
    """Project a list of d-dimensional feature maps to RGB colors using PCA."""
    from sklearn.decomposition import PCA

    def normalize(x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    if masks is not None:
        assert len(Fs) == len(masks)

    flatten = []
    for i, F in enumerate(Fs):
        c, h, w = F.shape
        F = np.rollaxis(F, 0, 3)
        F_flat = F.reshape(-1, c)
        if masks is not None and masks[i] is not None:
            mask = masks[i]
            assert mask.shape == F.shape[:2]
            F_flat = F_flat[mask.reshape(-1)]
        flatten.append(F_flat)
    flatten = np.concatenate(flatten, axis=0)
    flatten = normalize(flatten)

    pca = PCA(n_components=3)
    if skip > 1:
        pca.fit(flatten[::skip])
        flatten = pca.transform(flatten)
    else:
        flatten = pca.fit_transform(flatten)
    flatten = (normalize(flatten) + 1) / 2

    Fs_rgb = []
    for i, F in enumerate(Fs):
        h, w = F.shape[-2:]
        if masks is None or masks[i] is None:
            F_rgb, flatten = np.split(flatten, [h * w], axis=0)
            F_rgb = F_rgb.reshape((h, w, 3))
        else:
            F_rgb = np.zeros((h, w, 3))
            indices = np.where(masks[i])
            F_rgb[indices], flatten = np.split(flatten, [len(indices[0])], axis=0)
            F_rgb = np.concatenate([F_rgb, masks[i][..., None]], axis=-1)
        Fs_rgb.append(F_rgb)
    assert flatten.shape[0] == 0, flatten.shape
    return Fs_rgb


def one_hot_argmax_to_rgb(y, num_class):
    '''
    Args:
        probs: (B, C, H, W)
        num_class: int
        0: road 0
1: crossing 1
2: explicit_pedestrian 2
4: building 
6: terrain
7: parking `

    '''


    class_colors = {
        'road': (68, 68, 68),           # 0: Black
        'crossing': (244, 162, 97),     # 1; Red
        'explicit_pedestrian': (233, 196, 106),  # 2: Yellow
        # 'explicit_void': (128, 128, 128),      # 3: White
        'building': (231, 111, 81),   # 5: Magenta
        'terrain': (42, 157, 143),    # 7: Cyan
        'parking': (204, 204, 204),  # 8: Dark Grey
        'predicted_void': (255, 255, 255)
    }
    class_colors = class_colors.values()
    class_colors = [torch.tensor(x).float() for x in class_colors]

    argmaxed = torch.argmax((y > 0.25).float(), dim=1) # Take argmax
    argmaxed[torch.all(y <= 0.25, dim=1)] = num_class
    # print(argmaxed.shape)

    seg_rgb = torch.ones(
        (
            argmaxed.shape[0],
            3,
            argmaxed.shape[1],
            argmaxed.shape[2],
        )
    ) * 256
    for i in range(num_class + 1):
        seg_rgb[:, 0, :, :][argmaxed == i] = class_colors[i][0]
        seg_rgb[:, 1, :, :][argmaxed == i] = class_colors[i][1]
        seg_rgb[:, 2, :, :][argmaxed == i] = class_colors[i][2]

    return seg_rgb

def plot_images(imgs, titles=None, cmaps="gray", dpi=100, pad=0.5, adaptive=True):
    """Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        adaptive: whether the figure size should fit the image aspect ratios.
    """
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    if adaptive:
        ratios = [i.shape[1] / i.shape[0] for i in imgs]  # W / H
    else:
        ratios = [4 / 3] * n
    figsize = [sum(ratios) * 4.5, 4.5]
    fig, ax = plt.subplots(
        1, n, figsize=figsize, dpi=dpi, gridspec_kw={"width_ratios": ratios}
    )
    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])
    
    # Create legend
    class_colors = {
        'Road': (68, 68, 68),           # 0: Black
        'Crossing': (244, 162, 97),     # 1; Red
        'Sidewalk': (233, 196, 106),  # 2: Yellow
        'Building': (231, 111, 81),   # 5: Magenta
        'Terrain': (42, 157, 143),    # 7: Cyan
        'Parking': (204, 204, 204),  # 8: Dark Grey
    }
    patches = [mpatches.Patch(color=[c/255.0 for c in color], label=label) for label, color in class_colors.items()]
    plt.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

    fig.tight_layout(pad=pad)